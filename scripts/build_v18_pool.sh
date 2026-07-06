#!/bin/bash
# v18 = v17 recipe (20% replay, rank-32) with a GENTLER natural-biasing
# counterfactual to recover uplift: conflict-only SINGLE distractor entity
# (--n-distractor-hot 1, not full lists) + 2:1 relevant:distractor ratio
# (1000 relevant : 500 distractor) to bias toward entity emission.
set -euo pipefail; cd "$(dirname "$0")/.."; PY=.venv/bin/python; export PYTHONUNBUFFERED=1
echo "=== [1/2] replay n_asr=4000 (~20%) $(date -Is) ==="
$PY eval/build_v15_asr_replay.py --n-asr 4000 --out data/semantic_endpoint_v18
echo "=== [2/2] gentler biasing: 1000 relevant : 500 distractor, 1 conflict entity $(date -Is) ==="
$PY eval/build_natural_biasing.py --pool data/semantic_endpoint_v18/data.pt \
    --n-distractor 500 --n-relevant 1000 --n-distractor-hot 1
echo "=== v18 pool ready $(date -Is) ===" && cat data/semantic_endpoint_v18/biasing_meta.json
