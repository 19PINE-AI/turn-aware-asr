#!/bin/bash
# v17 = best-of-both: moderated replay (~20%) + REBALANCED 1:1 natural biasing
# (recover uplift) — paired with a rank-32 adapter at train time (recover the
# endpointing precision that high replay costs at rank-16). CPU/IO only.
set -euo pipefail; cd "$(dirname "$0")/.."; PY=.venv/bin/python; export PYTHONUNBUFFERED=1
echo "=== [1/2] replay dose n_asr=4000 (~20%) $(date -Is) ==="
$PY eval/build_v15_asr_replay.py --n-asr 4000 --out data/semantic_endpoint_v17
echo "=== [2/2] natural biasing 1:1 (800 distractor + 800 relevant) $(date -Is) ==="
$PY eval/build_natural_biasing.py --pool data/semantic_endpoint_v17/data.pt \
    --n-distractor 800 --n-relevant 800
echo "=== v17 pool ready $(date -Is) ===" && cat data/semantic_endpoint_v17/biasing_meta.json
