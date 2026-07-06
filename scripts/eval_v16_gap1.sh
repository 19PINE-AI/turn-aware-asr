#!/bin/bash
# gap-1 + behavior validation for a v16 snapshot (transformers inference; safe
# alongside training). WER (vLLM) is run separately once the GPU frees.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONUNBUFFERED=1
STEP=$1
SNAP=checkpoints/semantic_endpoint_v16_es/snap_step${STEP}.pt
MERGED=checkpoints/merged/v16-step${STEP}

echo "=== merge $(date -Is) ===" && $PY -m scripts.merge_endpoint_lora --checkpoint "$SNAP" --out-dir "$MERGED"
echo "=== big dictation+spelled probe $(date -Is) ===" && $PY -m eval.dictation_probe_eval --checkpoint "$SNAP" --probes data/probes_big --out research/90-probe-v16-step${STEP}-big.json 2>&1 | tail -2
echo "=== earnings22 biasing v16 (gap 1) $(date -Is) ===" && $PY -m eval.earnings22_biasing --model "$MERGED" --n-utts 300 --shuffle --entity-backend llm --out research/91-biasing-v16-step${STEP}.json 2>&1 | tail -3
echo "=== earnings22 biasing RELEASE re-baseline on GPU (fair compare) $(date -Is) ===" && $PY -m eval.earnings22_biasing --model checkpoints/merged/qwen3-asr-0.6b-endpoint-unified --n-utts 300 --shuffle --entity-backend llm --out research/91-biasing-release-gpu.json 2>&1 | tail -3
echo "=== DONE gap1 eval step ${STEP} $(date -Is) ==="
