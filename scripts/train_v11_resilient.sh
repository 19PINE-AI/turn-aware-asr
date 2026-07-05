#!/bin/bash
# v11 endpoint fine-tune with OOM-resilient restart (shared-GPU co-tenant).
# Same recipe as the causal (v9) run; only the data and holdout size differ.
set -uo pipefail
cd "$(dirname "$0")/.."

CKPT_DIR=checkpoints/semantic_endpoint_v11_es
mkdir -p "$CKPT_DIR"

for attempt in $(seq 1 30); do
    echo "=== attempt $attempt $(date -Is) ==="
    .venv/bin/python -m src.train_semantic_endpoint \
        --data data/semantic_endpoint_v11/data.pt \
        --steps 15000 --bsz 8 --eval-every 1500 --eval-holdout-size 130 \
        --score-spec v9 \
        --checkpoint-dir "$CKPT_DIR" \
        --early-stop-patience 4 \
        --resume
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "=== training completed cleanly $(date -Is) ==="
        exit 0
    fi
    echo "=== exited rc=$rc; waiting 120 s for GPU pressure to ease ==="
    sleep 120
done
echo "=== giving up after 30 attempts ==="
exit 1
