"""Run an evaluation over a split and write a reproducible report (Markdown + JSON).

For every song the analysis layer runs twice with the SAME config:
    oracle = analyze(reference chords)      -> isolates analysis-layer errors
    system = analyze(recognised chords)     -> end-to-end
Reported per song and aggregated:
  * chord metrics (mir_eval, or chart symbol-error rates + approximate aligned scores)
  * key: global key of oracle / system (and mir_eval key score if meta.toml gives a key)
  * roman-numeral agreement system vs oracle (time-weighted)
  * events: P/R/F1 of oracle and system against gold events (if events.json exists),
    error attribution, and system-vs-oracle agreement (always available)
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mir_eval

from .. import __version__
from ..analysis import analyze
from ..config import load_config
from ..schema import AnalysisResult, RecognitionResult
from .chords import chart_aligned, mir_eval_scores, symbol_error_rates, top1_intervals
from .dataset import Reference, discover, load_estimate
from .events import Span, attribute, gold_spans, prf
from .simulate import SimConfig, simulate

# Event types scored by default (all analysis-layer event types).
EVENT_TYPES = ("ii_V_I", "ii_V", "secondary_dominant", "secondary_leading_tone", "tritone_sub",
               "borrowed_chord", "aeolian_cadence", "deceptive_cadence", "modulation")


def _spans(res: AnalysisResult, seg_time: list[tuple[float, float]] | None = None) -> list[Span]:
    out = []
    for e in res.events:
        if seg_time is None:
            start, end = e.start, e.end
        else:
            ts = [seg_time[i] for i in e.segment_indices if seg_time[i] is not None]
            if not ts:
                continue
            start, end = min(t[0] for t in ts), max(t[1] for t in ts)
            if e.type == "modulation":
                end = start = seg_time[e.segment_indices[-1]][0]
        out.append(Span(e.type, start, end, e.label, e.confidence))
    return out


def _numeral_agreement(oracle: AnalysisResult, system: AnalysisResult, emap: list[int | None] | None) -> float | None:
    """Time-weighted fraction of the system timeline whose roman numeral equals the oracle's."""
    num = den = 0.0
    if emap is not None:
        for j, s in enumerate(system.segments):
            i = emap[j]
            if i is None or s.roman is None:
                continue
            w = s.end - s.start
            den += w
            o = oracle.segments[i]
            num += w * (o.roman is not None and o.roman.display == s.roman.display)
        return num / den if den else None
    # lab: sweep over both segmentations
    oi = 0
    for s in system.segments:
        if s.roman is None:
            continue
        while oi < len(oracle.segments) and oracle.segments[oi].end <= s.start:
            oi += 1
        k = oi
        while k < len(oracle.segments) and oracle.segments[k].start < s.end:
            o = oracle.segments[k]
            w = min(o.end, s.end) - max(o.start, s.start)
            if w > 0 and o.roman is not None:
                den += w
                num += w * (o.roman.display == s.roman.display)
            k += 1
    return num / den if den else None


def evaluate_song(ref: Reference, est: RecognitionResult, cfg: dict) -> dict[str, Any]:
    oracle = analyze(ref.recognition, config=cfg)
    system = analyze(est, config=cfg)
    out: dict[str, Any] = {"song_id": ref.song_id, "kind": ref.kind, "time_unit": ref.time_unit}

    if ref.kind == "lab":
        ref_iv, ref_lab = ref.intervals_labels()
        est_iv, est_lab = top1_intervals(est)
        chord = mir_eval_scores(ref_iv, ref_lab, est_iv, est_lab)
        chord.update(symbol_error_rates([s.chord_harte for s in oracle.segments],
                                        [s.chord_harte for s in system.segments]))
        chord["duration"] = ref_iv[-1][1] - ref_iv[0][0]
        sys_spans = _spans(system)
        emap = None
    else:
        chord, emap = chart_aligned(oracle, system)
        chord.update(symbol_error_rates([s.chord_harte for s in oracle.segments],
                                        [s.chord_harte for s in system.segments]))
        chord["duration"] = oracle.segments[-1].end - oracle.segments[0].start
        seg_time = [(oracle.segments[i].start, oracle.segments[i].end) if i is not None else None for i in emap]
        sys_spans = _spans(system, seg_time)
    chord["n_ref_chords"] = len(oracle.segments)
    out["chord"] = chord

    gk = ref.meta.get("key")
    out["key"] = {
        "reference": gk,
        "oracle": oracle.global_key.key.label, "system": system.global_key.key.label,
        "oracle_score": mir_eval.key.weighted_score(gk, oracle.global_key.key.label) if gk else None,
        "system_score": mir_eval.key.weighted_score(gk, system.global_key.key.label) if gk else None,
        "system_matches_oracle": oracle.global_key.key.label == system.global_key.key.label,
    }
    out["numeral_agreement"] = _numeral_agreement(oracle, system, emap)

    oracle_spans = _spans(oracle)
    out["events_agreement"] = prf(oracle_spans, sys_spans)
    if ref.events is not None:
        gold = gold_spans(ref.events)
        out["events_oracle"] = prf(gold, oracle_spans)
        out["events_system"] = prf(gold, sys_spans)
        out["attribution"] = attribute(gold, oracle_spans, sys_spans)
    out["warnings"] = {"oracle": oracle.warnings, "system": system.warnings}
    out["n_low_confidence_segments"] = sum(s.low_confidence for s in system.segments)
    return out


# ------------------------------------------------------------------------------ aggregate

def _sum_counts(songs: list[dict], key: str) -> dict[str, Any] | None:
    have = [s[key] for s in songs if key in s]
    if not have:
        return None
    types = sorted({t for h in have for t in h["by_type"]})

    def agg(rows: list[dict]) -> dict[str, Any]:
        tp = sum(r["tp"] for r in rows)
        ng = sum(r["n_gold"] for r in rows)
        npred = sum(r["n_pred"] for r in rows)
        p = tp / npred if npred else None
        r = tp / ng if ng else None
        f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
        return {"tp": tp, "n_gold": ng, "n_pred": npred, "precision": p, "recall": r, "f1": f}

    return {"songs": len(have), "overall": agg([h["overall"] for h in have]),
            "by_type": {t: agg([h["by_type"][t] for h in have if t in h["by_type"]]) for t in types}}


def aggregate(songs: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_songs": len(songs)}
    lab = [s for s in songs if s["kind"] == "lab"]
    if lab:
        tot = sum(s["chord"]["duration"] for s in lab)
        out["chord_mir_eval"] = {k: sum(s["chord"][k] * s["chord"]["duration"] for s in lab) / tot
                                 for k in ("root", "majmin", "sevenths", "mirex", "seg") if k in lab[0]["chord"]}
    n = sum(s["chord"]["n_ref_chords"] for s in songs)
    if songs:
        out["chord_symbol_error"] = {k: sum(s["chord"][k] * s["chord"]["n_ref_chords"] for s in songs) / n
                                     for k in ("cer_root", "cer_majmin", "cer_sevenths")}
    nums = [s["numeral_agreement"] for s in songs if s["numeral_agreement"] is not None]
    out["numeral_agreement_mean"] = sum(nums) / len(nums) if nums else None
    out["global_key_matches_oracle"] = sum(s["key"]["system_matches_oracle"] for s in songs)
    for k in ("events_agreement", "events_oracle", "events_system"):
        out[k] = _sum_counts(songs, k)
    attr: dict[str, int] = {}
    for s in songs:
        for c, v in s.get("attribution", {}).get("counts", {}).items():
            attr[c] = attr.get(c, 0) + v
    out["attribution"] = attr or None
    return out


# ------------------------------------------------------------------------------ provenance

def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                                    text=True).stdout.strip())
        return {"commit": commit or None, "dirty": dirty}
    except OSError:
        return {"commit": None, "dirty": None}


def provenance(cfg: dict, split: str, estimates: str) -> dict[str, Any]:
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
    return {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "harmonia_version": __version__, "git": _git_state(Path(__file__).resolve().parents[2]),
        "config_sha256_12": cfg_hash, "split": split, "estimates": estimates,
        "python": platform.python_version(), "mir_eval": mir_eval.__version__,
    }


# ------------------------------------------------------------------------------ entry points

def run(data_root: Path, split: str, *, estimates_dir: Path | None = None, sim: SimConfig | None = None,
        cfg: dict | None = None) -> dict[str, Any]:
    if (estimates_dir is None) == (sim is None):
        raise ValueError("give exactly one of estimates_dir or sim")
    cfg = cfg or load_config()
    refs = discover(data_root, split)
    songs, skipped = [], []
    for ref in refs:
        if sim is not None:
            iv, labels = ref.intervals_labels()
            est = simulate(iv, labels, SimConfig(**{**asdict(sim), "seed": sim.seed + _stable_hash(ref.song_id)}),
                           time_unit=ref.time_unit)
        else:
            cand = [estimates_dir / f"{ref.song_id}{ext}" for ext in (".json", ".lab")]
            path = next((p for p in cand if p.is_file()), None)
            if path is None:
                skipped.append({"song_id": ref.song_id, "reason": "no estimate file"})
                continue
            est = load_estimate(path)
        songs.append(evaluate_song(ref, est, cfg))
    src = f"simulated {asdict(sim)}" if sim is not None else str(estimates_dir)
    return {"provenance": provenance(cfg, split, src), "aggregate": aggregate(songs),
            "songs": songs, "skipped": skipped}


def _stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def sweep(data_root: Path, split: str, levels: list[tuple[float, float]], seed: int = 0,
          jitter: float = 0.1, cfg: dict | None = None) -> dict[str, Any]:
    """Robustness of the analysis layer vs simulated recognition error."""
    rows = []
    for re_, qe in levels:
        r = run(data_root, split, sim=SimConfig(re_, qe, jitter, seed), cfg=cfg)
        a = r["aggregate"]
        rows.append({"root_error": re_, "quality_error": qe,
                     "chord_majmin": (a.get("chord_mir_eval") or {}).get("majmin"),
                     "cer_majmin": (a.get("chord_symbol_error") or {}).get("cer_majmin"),
                     "numeral_agreement": a["numeral_agreement_mean"],
                     "events_vs_oracle_f1": (a["events_agreement"] or {}).get("overall", {}).get("f1"),
                     "events_vs_gold_f1": ((a["events_system"] or {}).get("overall") or {}).get("f1")})
    return {"provenance": provenance(cfg or load_config(), split, f"simulated sweep seed={seed} jitter={jitter}"),
            "rows": rows}


# ------------------------------------------------------------------------------ report

def _f(x: Any) -> str:
    return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def to_markdown(rep: dict[str, Any]) -> str:
    p = rep["provenance"]
    L = [f"# Harmonia evaluation — split `{p['split']}`", "",
         f"- estimates: {p['estimates']}",
         f"- git {(_f(p['git']['commit']) or '')[:10]}{' (dirty)' if p['git']['dirty'] else ''}, "
         f"config {p['config_sha256_12']}, harmonia {p['harmonia_version']}, mir_eval {p['mir_eval']}, "
         f"python {p['python']}, {p['timestamp']}", ""]
    if "rows" in rep:  # sweep
        L += ["## Robustness sweep (simulated recognition errors)", "",
              "| root err | quality err | majmin acc | CER majmin | numeral agree | events F1 vs oracle | events F1 vs gold |",
              "|---|---|---|---|---|---|---|"]
        for r in rep["rows"]:
            L.append(f"| {r['root_error']} | {r['quality_error']} | {_f(r['chord_majmin'])} | {_f(r['cer_majmin'])} | "
                     f"{_f(r['numeral_agreement'])} | {_f(r['events_vs_oracle_f1'])} | {_f(r['events_vs_gold_f1'])} |")
        return "\n".join(L) + "\n"
    a = rep["aggregate"]
    L += [f"## Aggregate ({a['n_songs']} songs)", ""]
    if a.get("chord_mir_eval"):
        L += ["Chord recognition (mir_eval, duration-weighted, time-aligned refs):", "",
              "| " + " | ".join(a["chord_mir_eval"]) + " |", "|" + "---|" * len(a["chord_mir_eval"]),
              "| " + " | ".join(_f(v) for v in a["chord_mir_eval"].values()) + " |", ""]
    if a.get("chord_symbol_error"):
        L += ["Chord-symbol error rate (edit distance / #ref chords; lower is better):", "",
              "| root | majmin | sevenths |", "|---|---|---|",
              "| " + " | ".join(_f(v) for v in a["chord_symbol_error"].values()) + " |", ""]
    L += [f"Roman-numeral agreement system vs oracle: {_f(a['numeral_agreement_mean'])}; "
          f"global key equals oracle in {a['global_key_matches_oracle']}/{a['n_songs']} songs.", ""]
    for key, title in (("events_oracle", "Events — ORACLE (reference chords) vs gold → analysis layer alone"),
                       ("events_system", "Events — SYSTEM vs gold → end to end"),
                       ("events_agreement", "Events — SYSTEM vs ORACLE → impact of recognition errors")):
        ev = a.get(key)
        if not ev:
            continue
        L += [f"### {title}", "", "| type | gold | pred | TP | P | R | F1 |", "|---|---|---|---|---|---|---|"]
        o = ev["overall"]
        L.append(f"| **all** | {o['n_gold']} | {o['n_pred']} | {o['tp']} | {_f(o['precision'])} | "
                 f"{_f(o['recall'])} | {_f(o['f1'])} |")
        for t, r in ev["by_type"].items():
            L.append(f"| {t} | {r['n_gold']} | {r['n_pred']} | {r['tp']} | {_f(r['precision'])} | "
                     f"{_f(r['recall'])} | {_f(r['f1'])} |")
        L.append("")
    if a.get("attribution"):
        L += ["### Error attribution", "", "| category | count |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in sorted(a["attribution"].items())]
        L += ["", "analysis_* = error already present with reference chords (fix rules / key model); "
                  "recognition_* = error caused by recognised chords (fix front end).", ""]
    L += ["## Per song", "", "| song | kind | majmin / CER-majmin | key oracle → system | numeral agree | "
          "events F1 sys vs gold | vs oracle |", "|---|---|---|---|---|---|---|"]
    for s in rep["songs"]:
        c = s["chord"]
        chord = f"{_f(c.get('majmin', c.get('aligned_majmin')))} / {_f(c['cer_majmin'])}"
        L.append(f"| {s['song_id']} | {s['kind']} | {chord} | {s['key']['oracle']} → {s['key']['system']} | "
                 f"{_f(s['numeral_agreement'])} | {_f(s.get('events_system', {}).get('overall', {}).get('f1'))} | "
                 f"{_f(s['events_agreement']['overall']['f1'])} |")
    for sk in rep.get("skipped", []):
        L.append(f"| {sk['song_id']} | skipped: {sk['reason']} | | | | | |")
    return "\n".join(L) + "\n"


def write_report(rep: dict[str, Any], out_dir: Path, name: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = rep["provenance"]["timestamp"].replace(":", "").replace("-", "")
    jp = out_dir / f"{name}_{stamp}.json"
    mp = out_dir / f"{name}_{stamp}.md"
    jp.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    mp.write_text(to_markdown(rep), encoding="utf-8")
    return jp, mp
