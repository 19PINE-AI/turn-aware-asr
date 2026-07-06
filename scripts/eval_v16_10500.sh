#!/bin/bash
set -uo pipefail; cd "$(dirname "$0")/.."; PY=.venv/bin/python; export PYTHONUNBUFFERED=1
echo "=== big probe 10500 $(date -Is) ===" && $PY -m eval.dictation_probe_eval --checkpoint checkpoints/semantic_endpoint_v16_es/snap_step10500.pt --probes data/probes_big --out research/90-probe-v16-step10500-big.json 2>&1 | tail -2
echo "=== earnings22 10500 $(date -Is) ===" && $PY -m eval.earnings22_biasing --model checkpoints/merged/v16-step10500 --n-utts 300 --shuffle --entity-backend llm --out research/91-biasing-v16-step10500.json 2>&1 | tail -3
echo "=== DONE 10500 evals $(date -Is) ==="
