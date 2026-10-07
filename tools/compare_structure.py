"""Compare structure option A (our sections, from `harmonia batch` recognitions) with
SongFormer output (option C) — agreement, not accuracy: neither is ground truth.

    .venv/bin/python tools/compare_structure.py outputs/songformer/runs/gpu_v1 [--runs outputs/runs]

Per song, SongFormer is treated as the reference for mir_eval:
  boundary F@3s     mir_eval.segment.detection, ±3 s window (boundaries only)
  pairwise F        mir_eval.segment.pairwise (same-label grouping, names ignored)
  label agreement   share of time where the (normalised) labels are equal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mir_eval
import numpy as np

NORMALISE = {"prechorus": "pre-chorus", "pre-chorus": "pre-chorus", "silence": "other"}


def norm(label: str) -> str:
    return NORMALISE.get(label, label)


def intervals(sections: list[dict], end: float | None = None) -> tuple[np.ndarray, list[str]]:
    secs = [s for s in sections if s["end"] > s["start"]]
    iv = np.array([[s["start"], s["end"]] for s in secs], dtype=float)
    if end is not None and len(iv):
        iv[-1, 1] = max(iv[-1, 1], end)
    return iv, [norm(s["label"]) for s in secs]


def label_agreement(a_iv, a_lab, b_iv, b_lab) -> float:
    tot = agree = 0.0
    for (s, e), la in zip(a_iv, a_lab):
        for (t, u), lb in zip(b_iv, b_lab):
            w = min(e, u) - max(s, t)
            if w > 0:
                tot += w
                agree += w * (la == lb)
    return agree / tot if tot else float("nan")


def our_sections(runs: Path) -> dict[str, list[dict]]:
    """audio stem -> option-A sections from the newest batch recognition that has structure."""
    out: dict[str, tuple[float, list[dict]]] = {}
    for f in runs.glob("*/*.recognition.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        st = (d.get("source") or {}).get("structure")
        if not st or not st.get("sections"):
            continue
        stem = Path(d["source"]["path"]).stem
        if stem not in out or f.stat().st_mtime > out[stem][0]:
            out[stem] = (f.stat().st_mtime, st["sections"])
    return {k: v[1] for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("songformer_dir")
    ap.add_argument("--runs", default="outputs/runs")
    args = ap.parse_args()
    ours = our_sections(Path(args.runs))
    rows = []
    for f in sorted(Path(args.songformer_dir).glob("*.songformer.json")):
        c = json.loads(f.read_text(encoding="utf-8"))
        stem = f.name[: -len(".songformer.json")]
        if stem not in ours:
            print(f"{stem}: no option-A sections found under {args.runs}", file=sys.stderr)
            continue
        c_iv, c_lab = intervals(c["sections"])
        a_iv, a_lab = intervals(ours[stem], end=c_iv[-1, 1] if len(c_iv) else None)
        a_iv, a_lab = mir_eval.util.adjust_intervals(a_iv, a_lab, t_min=0.0, t_max=c_iv[-1, 1])
        _, _, bf = mir_eval.segment.detection(c_iv, a_iv, window=3.0)
        _, _, pf = mir_eval.segment.pairwise(c_iv, c_lab, a_iv, a_lab)
        la = label_agreement(c_iv, c_lab, a_iv, a_lab)
        rows.append((stem, bf, pf, la, len(c_lab), len(a_lab), c.get("seconds")))
    if not rows:
        print("nothing to compare")
        return 1
    print("| song | boundary F@3s | pairwise F | label agreement | #sections C / A | C time (s) |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f} | {r[4]} / {r[5]} | {r[6]} |")
    m = np.mean([[r[1], r[2], r[3]] for r in rows], axis=0)
    print(f"| **mean ({len(rows)})** | {m[0]:.3f} | {m[1]:.3f} | {m[2]:.3f} | | |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
