#!/usr/bin/env bash
# Create the SongFormer environment on the GPU host (Linux, NVIDIA RTX 6000 Pro Blackwell).
#
#   bash setup_env.sh [ENV_DIR] [MODEL_DIR]
#
# Blackwell (sm_120) needs PyTorch >= 2.7 built for CUDA 12.8 — SongFormer's own pin
# (torch 2.4.0) cannot run on this GPU. Everything else follows SongFormer's requirements.txt
# (inference subset). Uses uv if present, else python3 -m venv + pip.
# Downloads: PyTorch cu128 wheels (~3 GB) + SongFormer weights (2.76 GB, HuggingFace
# ASLP-lab/SongFormer, pinned revision) + a 2 KB config — skip the weights with
# MODEL_DIR=/path/to/copied/model if you copied outputs/songformer/model from the Mac.
set -euo pipefail

ENV_DIR="${1:-$PWD/sf_env}"
MODEL_DIR="${2:-$PWD/sf_model}"
REVISION=a75880ed1b7375ac71860ec6c4fc9c899cf99515   # code reviewed 2026-10-07
TORCH_INDEX=https://download.pytorch.org/whl/cu128

PKGS=(
  "transformers==4.51.1" "huggingface-hub==0.30.1" "safetensors==0.5.3" "accelerate==1.5.2"
  "numpy==1.26.4" "scipy>=1.11" "librosa==0.11.0" "soundfile==0.13.1" "einops==0.8.1"
  "x-transformers==2.4.14" "ema-pytorch==0.7.7" "loguru==0.7.3" "omegaconf==2.3.0"
  "msaf==0.1.80" "mir_eval==0.8.2" "jams==0.3.4" "muq==0.1.0" "nnAudio==0.3.3" "tqdm"
  "setuptools<80"
)

if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.11 "$ENV_DIR"
  uv pip install --python "$ENV_DIR/bin/python" --index-url "$TORCH_INDEX" "torch==2.7.1" "torchaudio==2.7.1"
  uv pip install --python "$ENV_DIR/bin/python" "${PKGS[@]}"
else
  python3 -m venv "$ENV_DIR"   # needs Python >= 3.10
  "$ENV_DIR/bin/pip" install --upgrade pip
  "$ENV_DIR/bin/pip" install --index-url "$TORCH_INDEX" "torch==2.7.1" "torchaudio==2.7.1"
  "$ENV_DIR/bin/pip" install "${PKGS[@]}"
fi

PY="$ENV_DIR/bin/python"
if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
  "$PY" - "$MODEL_DIR" "$REVISION" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download("ASLP-lab/SongFormer", revision=sys.argv[2], local_dir=sys.argv[1],
                  ignore_patterns=["SongFormer.pt", "SongFormer.safetensors", "musicfm/figs/*"])
EOF
fi
# MusicFM builds its encoder from this config (no weights)
"$PY" -c "from huggingface_hub import hf_hub_download; hf_hub_download('facebook/wav2vec2-conformer-rope-large-960h-ft', 'config.json')"

"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
    print("arch list", torch.cuda.get_arch_list())
EOF
echo "OK: env=$ENV_DIR model=$MODEL_DIR"
