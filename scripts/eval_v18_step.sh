#!/bin/bash
set -uo pipefail; cd "$(dirname "$0")/.."; PY=.venv/bin/python; export PYTHONUNBUFFERED=1
STEP=$1; SNAP=checkpoints/semantic_endpoint_v18_es/snap_step${STEP}.pt; MERGED=checkpoints/merged/v18-step${STEP}
echo "=== merge $STEP $(date -Is) ===" && $PY -m scripts.merge_endpoint_lora --checkpoint "$SNAP" --out-dir "$MERGED" 2>&1 | tail -1
echo "=== big probe $(date -Is) ===" && $PY -m eval.dictation_probe_eval --checkpoint "$SNAP" --probes data/probes_big --out research/119-probe-v18-${STEP}.json 2>&1 | tail -2
echo "=== earnings22 $(date -Is) ===" && $PY -m eval.earnings22_biasing --model "$MERGED" --n-utts 300 --shuffle --entity-backend llm --out research/120-biasing-v18-${STEP}.json 2>&1 | tail -2
echo "=== WER $(date -Is) ===" && VLLM_USE_FLASHINFER_SAMPLER=0 /home/ubuntu/vllm023-venv/bin/python -m eval.offline_wer_ls --model "$MERGED" --out research/121-wer-v18-${STEP}.json 2>&1 | grep -E "WER="
echo "=== DONE v18 step ${STEP} $(date -Is) ==="
