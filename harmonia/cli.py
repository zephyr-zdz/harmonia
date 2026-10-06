"""Command line: ``python -m harmonia analyze "| C | Am | F | G |" [--key "C major"] [--json]``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analysis import analyze
from .config import load_config
from .io import lab_to_recognition, parse_progression
from .schema import AnalysisResult, RecognitionResult


def _load_input(src: str, beats_per_bar: int) -> RecognitionResult:
    p = Path(src)
    if src == "-":
        return parse_progression(sys.stdin.read(), beats_per_bar)
    if p.is_file():
        if p.suffix == ".lab":
            return lab_to_recognition(p)
        if p.suffix == ".json":
            return RecognitionResult.from_dict(json.loads(p.read_text(encoding="utf-8")))
        return parse_progression(p.read_text(encoding="utf-8"), beats_per_bar)
    return parse_progression(src, beats_per_bar)


def format_table(res: AnalysisResult) -> str:
    gk = res.global_key
    lines = [f"Global key: {gk.key.label} (p={gk.key.prob:.2f}, {gk.source})"
             + (f"  alternatives: " + ", ".join(f"{a.label} {a.prob:.2f}" for a in gk.alternatives[:2])
                if gk.alternatives else "")
             + ("  [AMBIGUOUS]" if gk.ambiguous else "")]
    lines.append(f"{'#':>3} {'time':>7} {'chord':<12} {'key':<10} {'numeral':<10} {'function':<12} {'conf':>5}  events")
    for s in res.segments:
        lines.append(
            f"{s.index:>3} {s.start:>7.2f} {s.chord:<12} {(s.key.label if s.key else '-'):<10} "
            f"{(s.roman.display if s.roman else '-'):<10} {', '.join(s.functions):<12} "
            f"{(s.roman.confidence if s.roman else s.confidence):>5.2f}{' !' if s.low_confidence else '  '}"
            f"{' '.join(s.event_ids)}")
    lines.append("")
    lines.append("Events:")
    for e in res.events:
        flag = " [LOW CONFIDENCE]" if e.low_confidence else ""
        lines.append(f"  {e.id:<5} {e.type:<23} {e.label:<40} conf={e.confidence:.2f}{flag}")
        lines.append(f"        {' → '.join(e.chords)}   ({' → '.join(e.functions)}) in {e.key}")
        for ev in e.evidence:
            lines.append(f"          · {ev.kind}: {ev.detail} [×{ev.factor:.2f}]")
        for alt in e.alternatives:
            lines.append(f"          ~ alternative: {alt['label']} ({alt['type']}, conf={alt['confidence']:.2f})")
    if res.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  ! {w}" for w in res.warnings)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="harmonia")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="analyse a chord progression (text, file, .lab, or recognition .json)")
    a.add_argument("input", help="progression text, a file path, or '-' for stdin")
    a.add_argument("--key", help="fix the key instead of estimating it, e.g. 'C major', 'Am'")
    a.add_argument("--config", help="TOML file deep-merged over the default config")
    a.add_argument("--beats-per-bar", type=int, default=4)
    a.add_argument("--json", action="store_true", help="print the AnalysisResult JSON")
    a.add_argument("-o", "--output", help="write the AnalysisResult JSON to this path")
    args = ap.parse_args(argv)

    rec = _load_input(args.input, args.beats_per_bar)
    res = analyze(rec, config=load_config(args.config), key=args.key)
    if args.output:
        Path(args.output).write_text(res.to_json(), encoding="utf-8")
    print(res.to_json() if args.json else format_table(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
