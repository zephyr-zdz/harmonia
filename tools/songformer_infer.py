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
    ap.add_argument("--low-mem", action="store_true", help="load weights without an extra full copy")
    ap.add_argument("--bf16", action="store_true", help="CPU bfloat16 autocast (halves activation memory; "
                                                        "check sections against fp32 before trusting it)")
    ap.add_argument("--list", help="text file with one audio path per line (in addition to positional paths)")
    ap.add_argument("--force", action="store_true", help="recompute songs whose JSON already exists")
    ap.add_argument("audio", nargs="*")
    args = ap.parse_args()
    if args.list:
        args.audio += [l.strip() for l in Path(args.list).read_text(encoding="utf-8").splitlines()
                       if l.strip() and not l.startswith("#")]
    if not args.audio:
        ap.error("no audio given")

    model_dir = str(Path(args.model).resolve())
    sys.path.insert(0, model_dir)
    os.environ["SONGFORMER_LOCAL_DIR"] = model_dir
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # everything is local; never fetch code at run time

    import contextlib
    import types

    import torch
    from transformers import AutoModel

    # --- CPU / low-memory adaptations (verified 2026-10-07, see docs/structure_options.md) ---
    # 1. MusicFM's fused-attention conformer imports transformers.deepspeed (moved to
    #    transformers.integrations in the pinned 4.51): provide the old name.
    from transformers.integrations import is_deepspeed_zero3_enabled
    shim = types.ModuleType("transformers.deepspeed")
    shim.is_deepspeed_zero3_enabled = is_deepspeed_zero3_enabled
    sys.modules["transformers.deepspeed"] = shim
    # 2. Their attention forces the CUDA flash kernel; let PyTorch pick a CPU kernel instead.
    torch.backends.cuda.sdp_kernel = lambda **kw: contextlib.nullcontext()

    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, low_cpu_mem_usage=args.low_mem,
                                      local_files_only=True)
    # 3. MusicFM is built with transformers' plain conformer, whose attention materialises
    #    T×T matrices (T ≈ 10.5k frames for the 420 s window → GBs per layer). Swap in its
    #    fused-attention twin (same weights; outputs equal to 1e-6 on a 2-layer check).
    sys.path.append(os.path.join(model_dir, "musicfm"))   # after loading: its "model" package would shadow model.py
    from modules.flash_conformer import Wav2Vec2ConformerEncoder as FusedEncoder
    old = model.musicfm.conformer
    fused = FusedEncoder(old.config)
    sd = {k.replace("parametrizations.weight.original0", "weight_g")
           .replace("parametrizations.weight.original1", "weight_v"): v for k, v in old.state_dict().items()}
    fused.load_state_dict(sd, strict=True)
    model.musicfm.conformer = fused
    del old, sd
    import gc
    gc.collect()
    # 4. Fused attention applies dropout_p even in eval mode (upstream behaviour): disable it
    #    for deterministic inference. [deviation from upstream, recorded]
    for m in model.modules():
        if hasattr(m, "dropout_p"):
            m.dropout_p = 0.0
    model.to(args.device)
    model.eval()
    print(f"model ready on {args.device} (torch {torch.__version__}"
          + (f", {torch.cuda.get_device_name(0)}" if args.device.startswith("cuda") else "") + ")", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for a in args.audio:
        dst = out / f"{Path(a).stem}.songformer.json"
        if dst.is_file() and not args.force:
            print(f"{Path(a).name}: exists, skipped", flush=True)
            continue
        t0 = time.time()
        amp = torch.autocast("cpu", dtype=torch.bfloat16) if args.bf16 else contextlib.nullcontext()
        with torch.no_grad(), amp:
            sections = model(str(Path(a).resolve()))
        dt = time.time() - t0
        rec = {"audio": str(Path(a).resolve()), "device": args.device, "bf16": args.bf16, "seconds": round(dt, 1),
               "sections": [{"start": float(s["start"]), "end": float(s["end"]), "label": s["label"]}
                            for s in sections]}
        dst.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{Path(a).name}: {len(sections)} sections in {dt:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
