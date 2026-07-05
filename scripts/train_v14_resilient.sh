#!/bin/bash
# v14 = v12 data balance (best unified) + recipe fixes (wd off embeddings,
# cosine LR). Clean isolated A/B of the recipe fixes on identical data to v12.
set -uo pipefail
cd "$(dirname "$0")/.."
CKPT_DIR=checkpoints/semantic_endpoint_v14_es
mkdir -p "$CKPT_DIR"
for attempt in $(seq 1 30); do
    echo "=== attempt $attempt $(date -Is) ==="
    .venv/bin/python -m src.train_semantic_endpoint \
        --data data/semantic_endpoint_v12/data.pt \
        --steps 12000 --bsz 8 --eval-every 1500 --eval-holdout-size 130 \
        --score-spec v9 --snapshot-evals \
        --no-wd-on-embeddings --lr-schedule cosine --lr 2e-4 --lr-min 1e-5 \
        --checkpoint-dir "$CKPT_DIR" --early-stop-patience 100 --resume
    rc=$?
    if [ $rc -eq 0 ]; then echo "=== completed cleanly $(date -Is) ==="; exit 0; fi
    echo "=== exited rc=$rc; waiting 120 s ==="; sleep 120
done
echo "=== giving up ==="; exit 1
