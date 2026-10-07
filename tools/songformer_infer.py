"""Run SongFormer (option C, docs/structure_options.md) in its OWN environment and write
sections as JSON. Not imported by harmonia (different Python / torch); harmonia reads the JSON.

    outputs/songformer/venv/bin/python tools/songformer_infer.py \
        --model outputs/songformer/model --out outputs/songformer/runs/<name> AUDIO [AUDIO ...]

Writes <out>/<audio stem>.songformer.json = {"audio", "device", "seconds", "sections": [{start, end, label}]}.
Model: HuggingFace ASLP-lab/SongFormer @ a75880ed (code reviewed 2026-10-07; weights: MuQ part
CC-BY-NC 4.0 → non-commercial use only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local SongFormer snapshot directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", help="cpu | mps")
    ap.add_argument("audio", nargs="+")
    args = ap.parse_args()

    model_dir = str(Path(args.model).resolve())
    sys.path.insert(0, model_dir)
    os.environ["SONGFORMER_LOCAL_DIR"] = model_dir
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # everything is local; never fetch code at run time

    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, low_cpu_mem_usage=False,
                                      local_files_only=True)
    model.to(args.device)
    model.eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for a in args.audio:
        t0 = time.time()
        with torch.no_grad():
            sections = model(str(Path(a).resolve()))
        dt = time.time() - t0
        rec = {"audio": str(Path(a).resolve()), "device": args.device, "seconds": round(dt, 1),
               "sections": [{"start": float(s["start"]), "end": float(s["end"]), "label": s["label"]}
                            for s in sections]}
        (out / f"{Path(a).stem}.songformer.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")
        print(f"{Path(a).name}: {len(sections)} sections in {dt:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
