#!/bin/bash
# Train v16 (single-process; NO resilient retry wrapper — those spawned dup
# writers per the overnight process notes). Same recipe as v15@release.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
.venv/bin/python -m src.train_semantic_endpoint \
    --data data/semantic_endpoint_v16/data.pt \
    --steps 12000 --bsz 8 --eval-every 1500 --eval-holdout-size 130 \
    --score-spec v9 --snapshot-evals \
    --no-wd-on-embeddings --lr-schedule cosine --lr 2e-4 --lr-min 1e-5 \
    --checkpoint-dir checkpoints/semantic_endpoint_v16_es --early-stop-patience 100 --resume
