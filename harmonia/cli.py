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

AUDIO_SUFFIXES = {".mp3", ".m4a", ".flac", ".wav", ".aiff", ".aif", ".ogg", ".opus"}


def _load_input(src: str, beats_per_bar: int, cfg: dict | None = None) -> RecognitionResult:
    p = Path(src)
    if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES:
        from .frontend.pipeline import transcribe  # optional 'audio' extra
        return transcribe(p, cfg)
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


SWEEP_LEVELS = [(0.0, 0.0), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4)]


def _run_eval(args: argparse.Namespace) -> int:
    from .eval.runner import run, sweep, to_markdown, write_report  # optional dependency (mir_eval)
    from .eval.simulate import SimConfig

    cfg = load_config(args.config)
    data = Path(args.data)
    if args.sweep:
        rep = sweep(data, args.split, SWEEP_LEVELS, seed=args.seed, jitter=args.jitter, cfg=cfg)
        name = f"{args.split}_sweep"
    elif args.simulate:
        rep = run(data, args.split, sim=SimConfig(args.root_error, args.quality_error, args.jitter, args.seed), cfg=cfg)
        name = f"{args.split}_simulated"
    else:
        rep = run(data, args.split, estimates_dir=Path(args.estimates), cfg=cfg)
        name = f"{args.split}_{Path(args.estimates).name}"
    if "aggregate" in rep and rep["aggregate"]["n_songs"] == 0:
        print(f"no songs evaluated in {data / args.split}", file=sys.stderr)
    jp, mp = write_report(rep, Path(args.out), name)
    print(to_markdown(rep))
    print(f"report: {mp}\n        {jp}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="harmonia")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="analyse a chord progression (text, file, .lab, or recognition .json)")
    a.set_defaults(cmd="analyze")
    a.add_argument("input", help="progression text, a file path, or '-' for stdin")
    a.add_argument("--key", help="fix the key instead of estimating it, e.g. 'C major', 'Am'")
    a.add_argument("--config", help="TOML file deep-merged over the default config")
    a.add_argument("--beats-per-bar", type=int, default=4)
    a.add_argument("--json", action="store_true", help="print the AnalysisResult JSON")
    a.add_argument("-o", "--output", help="write the AnalysisResult JSON to this path")
    a.add_argument("--save-recognition", help="audio input: also write the RecognitionResult JSON here")
    t = sub.add_parser("transcribe", help="audio -> RecognitionResult JSON (needs the 'audio' extra)")
    t.set_defaults(cmd="transcribe")
    t.add_argument("input", help="audio file (mp3 / m4a / flac / wav ...)")
    t.add_argument("-o", "--output", help="write JSON here instead of stdout")
    t.add_argument("--config", help="TOML file deep-merged over the default config")
    e = sub.add_parser("eval", help="evaluate on data/eval/<split> (needs the 'eval' extra: mir_eval)")
    e.add_argument("--split", default="dev", choices=["dev", "test"])
    e.add_argument("--data", default="data/eval", help="evaluation root")
    src = e.add_mutually_exclusive_group(required=True)
    src.add_argument("--estimates", help="dir with <song_id>.json (RecognitionResult) or <song_id>.lab")
    src.add_argument("--simulate", action="store_true", help="use the simulated recogniser")
    src.add_argument("--sweep", action="store_true", help="robustness sweep over simulated error rates")
    e.add_argument("--root-error", type=float, default=0.1)
    e.add_argument("--quality-error", type=float, default=0.15)
    e.add_argument("--jitter", type=float, default=0.1, help="boundary jitter (std, in reference time units)")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--config", help="TOML file deep-merged over the default config")
    e.add_argument("--out", default="outputs/eval", help="report directory")
    u = sub.add_parser("ui", help="local web UI: timeline, playback, chord editing (127.0.0.1)")
    u.set_defaults(cmd="ui")
    u.add_argument("input", nargs="*", help="song library: dirs and/or files (audio, progression .txt, .lab, "
                   "*.recognition.json); default: [ui].library in the config")
    u.add_argument("--port", type=int, default=8765)
    u.add_argument("--key", help="fix the key, e.g. 'C major'")
    u.add_argument("--config", help="TOML file deep-merged over the default config")
    u.add_argument("--no-browser", action="store_true")
    rp = sub.add_parser("report", help="section-by-section harmonic summary (markdown)")
    rp.set_defaults(cmd="report")
    rp.add_argument("input", nargs="+", help="recognition .json, audio, .lab or progression text files")
    rp.add_argument("-o", "--output", help="write markdown here instead of stdout")
    rp.add_argument("--config", help="TOML file deep-merged over the default config")
    bt = sub.add_parser("batch", help="transcribe + analyse every audio file under a directory")
    bt.set_defaults(cmd="batch")
    bt.add_argument("input", help="directory of audio files (searched recursively)")
    bt.add_argument("-o", "--output", required=True, help="output directory")
    bt.add_argument("--config", help="TOML file deep-merged over the default config")
    args = ap.parse_args(argv)

    if args.cmd == "batch":
        from .batch import run_batch
        md = run_batch(Path(args.input), Path(args.output), args.config)
        print(md.read_text(encoding="utf-8"))
        return 0
    if args.cmd == "ui":
        from .ui.server import serve
        serve(args.input or None, port=args.port, key=args.key, config=args.config, open_browser=not args.no_browser)
        return 0
    if args.cmd == "eval":
        return _run_eval(args)
    if args.cmd == "report":
        from .report import summarise, to_markdown
        cfg = load_config(args.config)
        parts = []
        for src in args.input:
            rec = _load_input(src, 4, cfg)
            res = analyze(rec, config=cfg)
            name = Path(src).name.replace(".recognition.json", "")
            parts.append(to_markdown(name, rec, res, summarise(rec, res)))
        text = "\n".join(parts)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            print(f"wrote {args.output}", file=sys.stderr)
        else:
            print(text)
        return 0

    cfg = load_config(args.config)
    if args.cmd == "transcribe":
        from .frontend.pipeline import transcribe
        rec = transcribe(args.input, cfg)
        text = json.dumps(rec.to_dict(), ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            print(f"wrote {args.output} ({len(rec.frames)} beats)", file=sys.stderr)
        else:
            print(text)
        return 0
    rec = _load_input(args.input, args.beats_per_bar, cfg)
    if args.save_recognition:
        Path(args.save_recognition).write_text(json.dumps(rec.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    res = analyze(rec, config=cfg, key=args.key)
    if args.output:
        Path(args.output).write_text(res.to_json(), encoding="utf-8")
    print(res.to_json() if args.json else format_table(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
