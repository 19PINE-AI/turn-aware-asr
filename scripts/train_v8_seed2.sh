#!/bin/bash
set -uo pipefail; cd "$(dirname "$0")/.."; export PYTHONUNBUFFERED=1
.venv/bin/python -m src.train_semantic_endpoint \
    --data data/semantic_endpoint_v8/data.pt \
    --steps 18000 --bsz 8 --eval-every 1500 --eval-holdout-size 60 \
    --score-spec legacy --seed 2 --snapshot-evals --early-stop-patience 100 \
    --checkpoint-dir checkpoints/semantic_endpoint_v8_seed2_es --resume
