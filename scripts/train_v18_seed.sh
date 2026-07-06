#!/bin/bash
# v18 recipe under an alternate seed for the robustness table.
set -uo pipefail; cd "$(dirname "$0")/.."; export PYTHONUNBUFFERED=1
SEED=$1
.venv/bin/python -m src.train_semantic_endpoint \
    --data data/semantic_endpoint_v18/data.pt \
    --steps 12000 --bsz 8 --eval-every 1500 --eval-holdout-size 130 \
    --score-spec v9 --snapshot-evals --no-wd-on-embeddings \
    --lr-schedule cosine --lr 2e-4 --lr-min 1e-5 --lora-r 32 --lora-alpha 64 --seed $SEED \
    --checkpoint-dir checkpoints/semantic_endpoint_v18_seed${SEED}_es --early-stop-patience 100 --resume
