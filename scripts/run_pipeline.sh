#!/usr/bin/env bash
# Full GPU pipeline: QLoRA fine-tune Kev-4B, then evaluate three systems on the same test set and build the comparison.
# Usage: bash scripts/run_pipeline.sh [skip_train]
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-D:/hf_cache} PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=.venv/bin/python

if [ "${1:-}" != "skip_train" ]; then
  echo "== train =="
  $PY scripts/train.py --data data/processed/kev/train.jsonl --dev data/processed/kev/dev.jsonl --dev_limit 150 \
      --init_from jaredpalmer/kev-4b --load_4bit 1 --epochs 1 --lr 2e-5 --batch 1 --accum 8 --max_state 1024 \
      --shared_prefix 1 --length_sort 1 --seed 1 --out runs/pdpa-4b
fi

echo "== eval kev-pdpa =="
$PY scripts/evaluate.py --system kev-pdpa --run runs/pdpa-4b --device cuda --load_4bit 1
echo "== eval kev-base =="
$PY scripts/evaluate.py --system kev-base --device cuda --load_4bit 1
echo "== eval openthai =="
$PY scripts/evaluate.py --system openthai --device cuda
echo "== compare =="
$PY scripts/compare.py
echo "PIPELINE DONE"
