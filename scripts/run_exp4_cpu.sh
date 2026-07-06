#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/.."
export EXT_MODEL_DIR=data/external_models
for adapter in smart_turn livekit_turn; do
  echo "=== exp4 $adapter dev25 $(date -Is) ==="
  .venv_ext_smart_turn/bin/python -m eval.external.harness \
      --adapter $adapter --n-stretches 25 --seed 0 \
      --out research/95-exp4-${adapter}-dev25.json 2>&1 | tail -3
done
echo "=== DONE exp4 CPU $(date -Is) ==="
