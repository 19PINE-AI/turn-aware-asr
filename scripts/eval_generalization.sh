#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONUNBUFFERED=1
CKPT=checkpoints/semantic_endpoint_v15_es/unified_release.pt
for shape in card phone13 addr; do
  echo "=== generalization: $shape $(date -Is) ==="
  $PY -m eval.dictation_probe_eval --checkpoint "$CKPT" \
      --probes data/probes_${shape} --skip-b \
      --out research/93-gen-${shape}-release.json
done
echo "=== DONE generalization $(date -Is) ==="
