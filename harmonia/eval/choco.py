"""Symbolic validation of the ANALYSIS layer against expert annotations in ChoCo.

Uses reference chords as input (no audio), so it measures the analysis layer alone, against
labels written by other people (not by this project):
  * rock-corpus (de Clercq & Temperley, jams-converted): Harte chords + expert roman
    numerals (incl. applied chords V/x, viio/x) + keys.  Time = beats (cumulative durations).
  * isophonics: Harte chords + time-varying keys with mode.  Time = seconds.
Metrics
  key            duration-weighted mir_eval.key.weighted_score (exact 1, fifth .5, relative .3,
                 parallel .2) of our local key vs the reference key at every instant; plus
                 tonic-only accuracy (rock-corpus keys carry no reliable mode).
  numeral_degree rock-corpus only: fraction of chord spans whose root scale degree (semitones
                 above the tonic, from our local key) equals the expert numeral's degree.
  applied        rock-corpus only: chord-span precision / recall of applied dominant /
                 leading-tone readings (expert "V…/x", "vii…/x", x ≠ I) vs our functions
                 ("V…/x", "vii…/x" from secondary_dominant / secondary_leading_tone / ii–V events);
                 target_agreement = among hits, same target scale degree.
Split: stable hash of the song id → dev (60 %) / test (40 %). Tune on dev only.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mir_eval

from ..analysis import analyze
from ..config import load_config
from ..schema import AnalysisResult, ChordCandidate, Frame, RecognitionResult
from ..theory.chord import ChordParseError, parse_chord
from ..theory.key import parse_key

CHOCO_ROOT = Path(__file__).resolve().parents[2] / "data/external/choco/smashub-choco-f7dd3ee/partitions"
_NUMERAL_DEG = {"i": 0, "ii": 2, "iii": 4, "iv": 5, "v": 7, "vi": 9, "vii": 11}
_ROMAN_RE = re.compile(r"^(?P<acc>[b#♭♯]*)(?P<num>vii|VII|vi|VI|iv|IV|v|V|iii|III|ii|II|i|I)(?P<rest>[^/]*)(?:/(?P<target>.+))?$")


@dataclass
class Span:
    start: float
    end: float
    value: str


@dataclass
class Song:
    id: str
    partition: str
    rec: RecognitionResult
    keys: list[Span]
    romans: list[Span]


def split_of(song_id: str) -> str:
    return "dev" if int(hashlib.sha256(song_id.encode()).hexdigest()[:8], 16) % 100 < 60 else "test"


def roman_degree(numeral: str) -> int | None:
    """Semitones above the tonic of an expert numeral's root ('bVII7' -> 10, '#iv' -> 6)."""
    m = _ROMAN_RE.match(numeral.strip())
    if not m:
        return None
    acc = m.group("acc")
    shift = acc.count("#") + acc.count("♯") - acc.count("b") - acc.count("♭")
    return (_NUMERAL_DEG[m.group("num").lower()] + shift) % 12


def applied_target(numeral: str) -> int | None:
    """Target degree of an applied dominant / leading-tone numeral, else None."""
    m = _ROMAN_RE.match(numeral.strip().replace("♭", "b").replace("♯", "#"))
    if not m or not m.group("target"):
        return None
    if m.group("num").lower() not in ("v", "vii") or m.group("acc"):
        return None
    t = roman_degree(m.group("target"))
    return None if t in (None, 0) else t


def _key_label(v: str) -> str | None:
    """'Bb:major' / 'A:minor' / 'F' / 'A:mix' -> mir_eval-style 'Bb major' (None if unusable)."""
    v = v.strip()
    if v in ("N", "", "X"):
        return None
    tonic, _, mode = v.partition(":")
    mode = {"": "major", "major": "major", "maj": "major", "minor": "minor", "min": "minor"}.get(mode)
    if mode is None:
        return None  # modal labels (mix, dor, ...) are not scored
    try:
        k = parse_key(f"{tonic} {mode}")
    except ValueError:
        return None
    return k.label


def load_song(path: Path, partition: str) -> Song | None:
    d = json.loads(path.read_text(encoding="utf-8"))
    ann = {a["namespace"]: a["data"] for a in d["annotations"]}
    chords = ann.get("chord_harte") or ann.get("chord")
    if not chords:
        return None
    cumulative = partition == "rock-corpus"  # 'bar.beat' times; durations are beats
    frames: list[Frame] = []
    t = 0.0
    starts: list[float] = []
    for x in chords:
        start = t if cumulative else float(x["time"])
        dur = float(x["duration"])
        label = x["value"]
        try:
            parse_chord(label)
        except ChordParseError:
            label = "X"
        frames.append(Frame(start, dur, [ChordCandidate(label, 1.0)], beats=dur if cumulative else None))
        starts.append(float(x["time"]))
        t += dur

    def remap(data: list[dict]) -> list[Span]:
        out = []
        for x in data:
            if cumulative:  # map 'bar.beat' time onto the cumulative timeline via chord starts
                i = min(range(len(starts)), key=lambda j: abs(starts[j] - float(x["time"])))
                s = frames[i].time
                out.append(Span(s, s + float(x["duration"]), x["value"]))
            else:
                out.append(Span(float(x["time"]), float(x["time"]) + float(x["duration"]), x["value"]))
        return out

    keys = remap(ann.get("key_mode", []))
    romans = remap(ann["chord_roman"]) if "chord_roman" in ann else []
    if cumulative and romans:  # roman spans correspond 1:1 to chord spans
        romans = [Span(f.time, f.time + f.duration, r.value.split(":", 1)[-1]) for f, r in zip(frames, romans)]
    rec = RecognitionResult(frames=frames, time_unit="beat" if cumulative else "second",
                            source={"type": "choco", "partition": partition, "path": str(path)})
    return Song(path.stem, partition, rec, keys, romans)


def iter_songs(partitions: list[str], split: str | None) -> list[Song]:
    out = []
    for p in partitions:
        sub = "jams-converted" if p == "rock-corpus" else "jams"
        for f in sorted((CHOCO_ROOT / p / "choco" / sub).glob("*.jams")):
            if split and split_of(f.stem) != split:
                continue
            s = load_song(f, p)
            if s is not None:
                out.append(s)
    return out


def _seg_at(res: AnalysisResult, t: float):
    for s in res.segments:
        if s.start <= t < s.end:
            return s
    return None


def evaluate_song(song: Song, cfg: dict) -> dict[str, Any]:
    res = analyze(song.rec, config=cfg)
    out: dict[str, Any] = {"id": song.id, "partition": song.partition}
    # key: sample at segment level, weight by overlap with each reference key span
    num = den = tonic_hit = 0.0
    for r in song.keys:
        ref = _key_label(r.value)
        if ref is None:
            continue
        for s in res.segments:
            w = min(s.end, r.end) - max(s.start, r.start)
            if w <= 0 or s.key is None or s.chord_harte in ("N", "X"):
                continue
            den += w
            num += w * mir_eval.key.weighted_score(ref, s.key.label)
            tonic_hit += w * (parse_key(ref).tonic == parse_key(s.key.label).tonic)
    out["key_weight"] = den
    out["key_score"] = num / den if den else None
    out["tonic_acc"] = tonic_hit / den if den else None
    out["global_key"] = res.global_key.key.label
    if song.romans:
        hits = n = 0.0
        gold_applied = pred_applied = tp_r = tp_p = target_ok = 0
        for r in song.romans:
            deg = roman_degree(r.value)
            s = _seg_at(res, (r.start + r.end) / 2)
            if deg is None or s is None or s.roman is None:
                continue
            n += 1
            hits += (s.roman.degree == deg)
        pred_spans = []
        for s in res.segments:
            for f in s.functions:
                base = f.split("/")[0].replace("♭", "b")
                if "/" in f and (base.startswith("V") or base.startswith("vii")):
                    tgt = roman_degree(f.split("/", 1)[1])
                    pred_spans.append((s.start, s.end, tgt))
                    break
        gold_spans = [(r.start, r.end, applied_target(r.value)) for r in song.romans if applied_target(r.value) is not None]
        for gs, ge, gt in gold_spans:
            hit = [p for p in pred_spans if min(p[1], ge) - max(p[0], gs) > 0]
            if hit:
                tp_r += 1
                target_ok += any(p[2] == gt for p in hit)
        for ps, pe, _ in pred_spans:
            tp_p += any(min(pe, ge) - max(ps, gs) > 0 for gs, ge, _ in gold_spans)
        gold_applied, pred_applied = len(gold_spans), len(pred_spans)
        out.update({"numeral_n": n, "numeral_hits": hits, "applied_gold": gold_applied, "applied_pred": pred_applied,
                    "applied_tp_recall": tp_r, "applied_tp_precision": tp_p, "applied_target_ok": target_ok})
    return out


def aggregate(rows: list[dict]) -> dict[str, Any]:
    agg: dict[str, Any] = {"songs": len(rows)}
    kw = sum(r["key_weight"] for r in rows if r["key_score"] is not None)
    if kw:
        agg["key_score"] = sum(r["key_score"] * r["key_weight"] for r in rows if r["key_score"] is not None) / kw
        agg["tonic_acc"] = sum(r["tonic_acc"] * r["key_weight"] for r in rows if r["tonic_acc"] is not None) / kw
    rr = [r for r in rows if "numeral_n" in r]
    if rr:
        n = sum(r["numeral_n"] for r in rr)
        agg["numeral_degree_acc"] = sum(r["numeral_hits"] for r in rr) / n if n else None
        g = sum(r["applied_gold"] for r in rr)
        p = sum(r["applied_pred"] for r in rr)
        tr = sum(r["applied_tp_recall"] for r in rr)
        tp = sum(r["applied_tp_precision"] for r in rr)
        agg["applied"] = {"gold": g, "pred": p, "recall": tr / g if g else None, "precision": tp / p if p else None,
                          "target_agreement": sum(r["applied_target_ok"] for r in rr) / tr if tr else None}
    return agg


def run(partitions: list[str], split: str, cfg: dict | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    out: dict[str, Any] = {"split": split, "partitions": {}}
    for p in partitions:
        rows = [evaluate_song(s, cfg) for s in iter_songs([p], split)]
        out["partitions"][p] = {"aggregate": aggregate(rows), "songs": rows}
    return out
