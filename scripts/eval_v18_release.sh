#!/bin/bash
set -uo pipefail; cd "$(dirname "$0")/.."; PY=.venv/bin/python; export PYTHONUNBUFFERED=1
SNAP=checkpoints/semantic_endpoint_v18_es/snap_step9000.pt
echo "=== 25-stretch dev replay (gated) $(date -Is) ===" && $PY -m eval.streaming_replay_eval --checkpoint "$SNAP" --energy-gate --n-stretches 25 --seed 0 --out research/124-replay-v18-dev25.json 2>&1 | tail -2
echo "=== 25-stretch confirm h=1 $(date -Is) ===" && $PY -m eval.streaming_replay_eval --checkpoint "$SNAP" --energy-gate --confirm-silent-chunks 1 --n-stretches 25 --seed 0 --out research/124-replay-v18-dev25-h1.json 2>&1 | tail -2
echo "=== 100-stretch big replay (gated) $(date -Is) ===" && $PY -m eval.streaming_replay_eval --checkpoint "$SNAP" --energy-gate --n-stretches 100 --seed 0 --out research/125-replay-v18-big.json 2>&1 | tail -2
echo "=== streaming latency $(date -Is) ===" && $PY -m eval.streaming_latency --checkpoint "$SNAP" --out research/126-latency-v18.json 2>&1 | tail -3 || echo "(latency eval args differ; skip)"
echo "=== DONE v18 release evals $(date -Is) ==="
