"""Calibrate [frontend.decode] on data/eval/dev (NEVER on test).

    python -m harmonia.frontend.calibrate --out outputs/calibration/<name> [--workers 4]

1. Network features (beats + lv-chordia frame evidence) are extracted once per dev song and
   cached in outputs/cache/features/ (they do not depend on the decode parameters).
2. Every grid point re-decodes all dev songs and scores them:
   chord: mir_eval root / majmin / sevenths (duration-weighted over songs);
   key:   local key of the analysis of the decoded chords vs keys.lab (if present).
3. Objective, fixed before looking at results: mean(majmin, sevenths), subject to the local-key
   score not dropping more than KEY_TOLERANCE below the current default config.
Writes results.json + results.md (all grid points, sorted) with provenance.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from ..config import load_config

GRID = {
    "obs_weight": [0.5, 1.0, 1.5, 2.0],
    "change_penalty": [1.0, 2.0, 3.0, 4.0],
    "change_extra_offbeat": [0.5, 1.5, 3.0],
    "seventh_bias": [0.0, 0.5, 1.0, 2.0],
}
KEY_TOLERANCE = 0.01
CACHE = Path("outputs/cache/features")


def _feature_path(audio: Path, cfg: dict) -> Path:
    st = audio.stat()
    h = hashlib.sha256(json.dumps([str(audio.resolve()), st.st_size, st.st_mtime, cfg["frontend"]["chords"],
                                   cfg["frontend"]["beats"]], sort_keys=True).encode()).hexdigest()[:12]
    return CACHE / f"{audio.stem}.{h}.pkl"


def features(audio: Path, cfg: dict):
    path = _feature_path(audio, cfg)
    if path.is_file():
        return pickle.loads(path.read_bytes())
    from .pipeline import extract_features
    feats = extract_features(audio, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(feats))
    return feats


_STATE: dict[str, Any] = {}


def _init(songs: list[tuple[str, Path]], cfg: dict) -> None:
    from ..eval.dataset import discover
    refs = {r.song_id: r for r in discover(Path("data/eval"), "dev")}
    _STATE["cfg"] = cfg
    _STATE["songs"] = [(refs[sid], features(audio, cfg)) for sid, audio in songs]


def score(params: dict[str, float]) -> dict[str, Any]:
    from ..analysis import analyze
    from ..eval.chords import mir_eval_scores, top1_intervals
    from ..eval.runner import local_key_scores
    from .pipeline import from_features
    cfg = copy.deepcopy(_STATE["cfg"])
    cfg["frontend"]["decode"].update(params)
    acc = {"root": 0.0, "majmin": 0.0, "sevenths": 0.0, "seg": 0.0}
    dur = kdur = kw = 0.0
    for ref, feats in _STATE["songs"]:
        est = from_features(*feats, cfg)
        ref_iv, ref_lab = ref.intervals_labels()
        est_iv, est_lab = top1_intervals(est)
        m = mir_eval_scores(ref_iv, ref_lab, est_iv, est_lab)
        d = ref_iv[-1][1] - ref_iv[0][0]
        for k in acc:
            acc[k] += m[k] * d
        dur += d
        if ref.key_spans:
            k = local_key_scores(analyze(est, config=cfg), ref.key_spans)
            if k:
                kw += k["weighted"] * k["duration"]
                kdur += k["duration"]
    out = {k: v / dur for k, v in acc.items()}
    out["local_key"] = kw / kdur if kdur else None
    out["objective"] = (out["majmin"] + out["sevenths"]) / 2
    return {"params": params, **out}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="harmonia.frontend.calibrate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--grid", help="JSON object overriding GRID, e.g. '{\"change_penalty\": [0.5, 1.0]}'")
    args = ap.parse_args(argv)
    grid = dict(GRID, **json.loads(args.grid)) if args.grid else GRID
    from ..eval.dataset import discover
    from ..eval.runner import provenance
    cfg = load_config()
    refs = discover(Path("data/eval"), "dev")
    songs = [(r.song_id, Path(r.meta["audio"])) for r in refs if r.meta.get("audio") and r.kind == "lab"]
    if not songs:
        raise SystemExit("no dev songs with meta.toml 'audio' and chords.lab")
    for sid, audio in songs:  # extract (or load) features once, sequentially
        print(f"features: {sid}", flush=True)
        features(audio, cfg)
    default = {k: cfg["frontend"]["decode"].get(k, 0.0) for k in grid}
    points = [default] + [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(songs, cfg)) as ex:
        rows = []
        for i, r in enumerate(ex.map(score, points)):
            rows.append(r)
            print(f"[{i + 1}/{len(points)}] {r['params']} obj={r['objective']:.4f} majmin={r['majmin']:.4f} "
                  f"sev={r['sevenths']:.4f} key={r['local_key']}", flush=True)
    base = rows[0]
    ok = [r for r in rows[1:] if base["local_key"] is None or r["local_key"] is None
          or r["local_key"] >= base["local_key"] - KEY_TOLERANCE]
    best = max(ok, key=lambda r: r["objective"]) if ok else base
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"provenance": provenance(cfg, "dev", f"calibration grid {grid}"), "songs": [s for s, _ in songs],
           "objective": "mean(majmin, sevenths) s.t. local_key >= default - %.2f" % KEY_TOLERANCE,
           "default": base, "best": best, "rows": sorted(rows, key=lambda r: -r["objective"])}
    (out / "results.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    L = ["# Decoder calibration (dev)", "", f"objective: {rep['objective']}", f"songs: {', '.join(rep['songs'])}", "",
         "| " + " | ".join(grid) + " | root | majmin | sevenths | seg | local key | objective |",
         "|" + "---|" * (len(grid) + 6)]
    for r in [base, best] + rep["rows"][:30]:
        L.append("| " + " | ".join(str(r["params"][k]) for k in grid) + " | " + " | ".join(
            f"{r[k]:.4f}" if r[k] is not None else "—" for k in ("root", "majmin", "sevenths", "seg", "local_key",
                                                                   "objective")) + " |")
    (out / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"default: {base}\nbest:    {best}\nreport: {out / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
