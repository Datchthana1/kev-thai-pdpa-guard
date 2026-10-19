#!/usr/bin/env bash
# Recreate the project environment (Linux, macOS or Windows Git Bash). Needs Python 3.12/3.13 and an NVIDIA GPU.
#   PYTHON=/path/to/python3.13 bash scripts/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3.13}
command -v "$PY" >/dev/null 2>&1 || PY=python
KEV_COMMIT=1f04cfa28e6a708485b28a9585010b9e436b776e   # jaredpalmer/kev revision this project was built against

# official Kev source (trainer, model, API) at the pinned commit
if [ ! -d third_party/kev ]; then
  git clone https://github.com/jaredpalmer/kev.git third_party/kev
  git -C third_party/kev checkout -q "$KEV_COMMIT"
fi

"$PY" -m pip install -q uv
"$PY" -m uv venv .venv --python "$PY"
VPY=.venv/Scripts/python.exe; [ -x "$VPY" ] || VPY=.venv/bin/python
# torch 2.8.0 CUDA 12.8 (Kev pins torch>=2.6,<2.9), then everything else
"$PY" -m uv pip install --python "$VPY" torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m uv pip install --python "$VPY" -r requirements.txt
"$PY" -m uv pip install --python "$VPY" -e third_party/kev
# fast Gated-DeltaNet kernels (~10x faster training); on Windows triton comes from triton-windows
if [ "$(uname -o 2>/dev/null)" = "Msys" ]; then TRITON="triton-windows<3.5"; else TRITON="triton"; fi
"$PY" -m uv pip install --python "$VPY" "$TRITON" flash-linear-attention optuna
"$VPY" -c "import torch, transformers, peft, bitsandbytes, kev, fla; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
