#!/bin/bash
# Build the v16 unified pool = v12 base + larger ASR-replay (gap 4) +
# natural-speech biasing counterfactuals (gap 1). CPU/IO only (no GPU).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONUNBUFFERED=1

echo "=== [1/2] ASR-replay dose (n_asr=5000, ~25% replay) $(date -Is) ==="
$PY eval/build_v15_asr_replay.py --n-asr 5000 --out data/semantic_endpoint_v16

echo "=== [2/2] natural-speech biasing counterfactuals $(date -Is) ==="
$PY eval/build_natural_biasing.py \
    --pool data/semantic_endpoint_v16/data.pt \
    --n-distractor 1200 --n-relevant 600

echo "=== v16 pool ready $(date -Is) ==="
ls -la data/semantic_endpoint_v16/data.pt
cat data/semantic_endpoint_v16/biasing_meta.json
