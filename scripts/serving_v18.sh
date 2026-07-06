#!/bin/bash
set -uo pipefail; cd "$(dirname "$0")/.."; export PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=/home/ubuntu/streaming-vad-asr
PY=/home/ubuntu/sglang-venv/bin/python
MODEL=checkpoints/merged/qwen3-asr-0.6b-endpoint-unified
echo "=== E3 per-chunk latency (v18 release) $(date -Is) ==="
$PY worker_integration/e3_replay_vllm.py --model-dir "$MODEL" --energy-gate --n-stretches 25 --out research/132-e3-serving-v18.json 2>&1 | grep -E "E3 vLLM|median|P95" | tail -3
echo "=== E5 concurrency (v18 release) $(date -Is) ==="
$PY worker_integration/e5_concurrency.py --model-dir "$MODEL" --out research/133-e5-concurrency-v18.json 2>&1 | grep -E "E5|n_star|within|P95|concurren" | tail -6
echo "=== DONE serving v18 $(date -Is) ==="
