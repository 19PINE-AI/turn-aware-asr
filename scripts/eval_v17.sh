#!/bin/bash
# Full v17 eval: probe sweep -> pick gate-passers -> WER + earnings + big-probe.
set -uo pipefail; cd "$(dirname "$0")/.."; PY=.venv/bin/python; export PYTHONUNBUFFERED=1
echo "=== v17 probe sweep $(date -Is) ==="
$PY -m eval.select_checkpoint --glob 'checkpoints/semantic_endpoint_v17_es/snap_step*.pt' \
    --probes data/probes --out research/110-v17-probe-sweep.json 2>&1 | tail -8
echo "=== v17 sweep done; eval STEP passed as arg below ==="
