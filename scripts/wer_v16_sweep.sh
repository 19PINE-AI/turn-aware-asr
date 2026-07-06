#!/bin/bash
set -uo pipefail; cd "$(dirname "$0")/.."; export PYTHONUNBUFFERED=1
for step in 7500 9000 12000; do
  if [ "$step" = "12000" ]; then SNAP=checkpoints/semantic_endpoint_v16_es/step12000.pt; else SNAP=checkpoints/semantic_endpoint_v16_es/snap_step${step}.pt; fi
  MERGED=checkpoints/merged/v16-step${step}
  echo "=== merge+WER $step $(date -Is) ==="
  .venv/bin/python -m scripts.merge_endpoint_lora --checkpoint "$SNAP" --out-dir "$MERGED" 2>&1 | tail -1
  VLLM_USE_FLASHINFER_SAMPLER=0 /home/ubuntu/vllm023-venv/bin/python -m eval.offline_wer_ls --model "$MERGED" --out research/89-wer-v16-${step}.json 2>&1 | grep -E "WER=" 
done
echo "=== DONE v16 WER sweep $(date -Is) ==="
