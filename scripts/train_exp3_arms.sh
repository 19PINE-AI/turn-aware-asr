#!/bin/bash
# exp-3: train the ablation arms + a matched baseline (full v9 pool), same modern
# recipe as the release (score-spec v9, cosine, wd-fix), single-process serial.
set -uo pipefail; cd "$(dirname "$0")/.."; export PYTHONUNBUFFERED=1
train () { # $1=data $2=ckptdir
  .venv/bin/python -m src.train_semantic_endpoint \
    --data "$1" --steps 12000 --bsz 8 --eval-every 1500 --eval-holdout-size 130 \
    --score-spec v9 --snapshot-evals --no-wd-on-embeddings \
    --lr-schedule cosine --lr 2e-4 --lr-min 1e-5 \
    --checkpoint-dir "$2" --early-stop-patience 100 --resume
}
echo "=== baseline (full v9) $(date -Is) ===" && train data/semantic_endpoint_v9/data.pt checkpoints/abl_base_es
echo "=== arm A -minimal-pair $(date -Is) ===" && train data/semantic_endpoint_abl_A/data.pt checkpoints/abl_A_es
echo "=== arm B -silence $(date -Is) ===" && train data/semantic_endpoint_abl_B/data.pt checkpoints/abl_B_es
echo "=== arm C -pause-pair $(date -Is) ===" && train data/semantic_endpoint_abl_C/data.pt checkpoints/abl_C_es
echo "=== DONE exp3 arms $(date -Is) ==="
