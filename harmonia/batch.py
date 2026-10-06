"""Batch: transcribe + analyse every audio file under a directory; write per-song JSON and a
summary table. ``harmonia batch data/external/user_music -o outputs/runs/<name>``"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

from .analysis import analyze
from .config import load_config
from .frontend.pipeline import AUDIO_SUFFIXES, transcribe


def run_batch(src: Path, out: Path, cfg_path: str | None = None) -> Path:
    cfg = load_config(cfg_path)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
    rows = []
    for i, f in enumerate(files, 1):
        t0 = time.time()
        print(f"[{i}/{len(files)}] {f.name}", file=sys.stderr)
        try:
            rec = transcribe(f, cfg)
            res = analyze(rec, config=cfg)
        except Exception as e:  # report, never swallow
            rows.append({"file": str(f.relative_to(src)), "error": repr(e)})
            print(f"   FAILED: {e!r}", file=sys.stderr)
            continue
        stem = f.stem
        (out / f"{stem}.recognition.json").write_text(json.dumps(rec.to_dict(), ensure_ascii=False), encoding="utf-8")
        (out / f"{stem}.analysis.json").write_text(res.to_json(indent=None), encoding="utf-8")
        m = rec.source.get("meter") or {}
        segs = [s for s in res.segments if s.chord_harte not in ("N", "X")]
        dur = sum(s.end - s.start for s in segs) or 1.0
        rows.append({
            "file": str(f.relative_to(src)), "seconds": round(time.time() - t0, 1),
            "bpm": m.get("bpm"), "meter": m.get("time_signature"), "bars": m.get("n_bars"),
            "tempo_cv": m.get("tempo_cv"), "irregular_bars": len(m.get("irregular_bars", [])),
            "global_key": res.global_key.key.label, "key_p": res.global_key.key.prob,
            "key_ambiguous": res.global_key.ambiguous, "key_regions": [r.key.label for r in res.key_regions],
            "segments": len(segs),
            "low_conf_time": round(sum(s.end - s.start for s in segs if s.low_confidence) / dur, 3),
            "events": dict(Counter(e.type for e in res.events)),
            "progressions": dict(Counter(p.name for p in res.progressions)),
        })
    (out / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["| song | BPM | meter | bars | global key | key regions | chords | low-conf time | events | named progressions |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['file']} | FAILED: {r['error']} |||||||||")
            continue
        ev = ", ".join(f"{k}×{v}" for k, v in sorted(r["events"].items()))
        pr = ", ".join(f"{k}×{v}" for k, v in r["progressions"].items()) or "—"
        regions = " → ".join(dict.fromkeys(r["key_regions"]))
        lines.append(f"| {Path(r['file']).stem} | {r['bpm']} | {r['meter']} | {r['bars']} | {r['global_key']} "
                     f"({r['key_p']:.2f}{', ambiguous' if r['key_ambiguous'] else ''}) | {regions} | {r['segments']} | "
                     f"{r['low_conf_time']:.0%} | {ev} | {pr} |")
    md = out / "summary.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md
