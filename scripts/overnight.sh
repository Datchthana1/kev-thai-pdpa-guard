#!/usr/bin/env bash
# Overnight chain: wait for the v1 pipeline, run the Optuna search, retrain a full epoch with the best params (v2) and
# evaluate v2 on the same test set. Pushing v2 is a separate, manual decision (only if it beats v1).
# Usage: bash scripts/overnight.sh > runs/overnight.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-D:/hf_cache} PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe

echo "== waiting for v1 pipeline ($(date +%H:%M)) =="
until grep -q "PIPELINE DONE" runs/pipeline.log 2>/dev/null; do
  if grep -qE "Traceback|OOM" runs/pipeline.log 2>/dev/null; then echo "v1 pipeline failed; stopping"; exit 1; fi
  sleep 60
done

echo "== optuna ($(date +%H:%M)) =="
$PY scripts/tune_optuna.py --trials 12 --trial_steps 40

echo "== retrain v2 with best params ($(date +%H:%M)) =="
mapfile -t BEST < <($PY -c "import json;[print(x) for x in json.load(open('results/optuna/best_params.json'))['train_flags']]" | tr -d '\r')
echo "best flags: ${BEST[*]}"
$PY scripts/train.py --data data/processed/kev/train.jsonl --dev data/processed/kev/dev.jsonl --dev_limit 150 \
    --init_from jaredpalmer/kev-4b --load_4bit 1 --epochs 1 --batch 1 --max_state 1024 \
    --shared_prefix 1 --length_sort 1 --seed 1 --out runs/pdpa-4b-v2 "${BEST[@]}"

echo "== eval v2 ($(date +%H:%M)) =="
$PY scripts/evaluate.py --system kev-pdpa --run runs/pdpa-4b-v2 --device cuda --load_4bit 1 --out results/eval_kev-pdpa-v2.json
echo "OVERNIGHT DONE ($(date +%H:%M))"
