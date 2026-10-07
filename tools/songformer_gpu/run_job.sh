#!/usr/bin/env bash
# Job body: run SongFormer on every audio file in LIST, one model load for all songs.
# Called by submit.sh (any scheduler) or directly on a GPU node.
#
#   bash run_job.sh ENV_DIR MODEL_DIR LIST OUT_DIR
#
# Resumable: songs whose JSON exists in OUT_DIR are skipped.
set -euo pipefail
ENV_DIR="$1"; MODEL_DIR="$2"; LIST="$3"; OUT_DIR="$4"
BUNDLE="$(cd "$(dirname "$LIST")" && pwd)"   # not dirname $0: Slurm runs a spooled copy of this script
mkdir -p "$OUT_DIR"
cd "$BUNDLE"   # list entries are relative to the bundle directory

echo "host $(hostname)  start $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv 2>/dev/null || echo "nvidia-smi unavailable"
export HF_HUB_OFFLINE=1   # everything was fetched by setup_env.sh; never fetch code inside the job

"$ENV_DIR/bin/python" -W ignore "$BUNDLE/songformer_infer.py" \
    --model "$MODEL_DIR" --out "$OUT_DIR" --device cuda --list "$LIST" 2>&1 | tee -a "$OUT_DIR/run.log"

# provenance next to the results
{
  echo "finished $(date -Is) on $(hostname)"
  "$ENV_DIR/bin/python" -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
} > "$OUT_DIR/PROVENANCE.txt"
ls "$OUT_DIR"/*.songformer.json | wc -l | xargs echo "songs with results:"
