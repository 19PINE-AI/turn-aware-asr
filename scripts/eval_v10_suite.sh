#!/bin/bash
# Post-training v10 evaluation: dictation probes + replay-benchmark regression.
set -uo pipefail
cd "$(dirname "$0")/.."

CKPT=checkpoints/semantic_endpoint_v10_es/best.pt

echo "=== [1/2] dictation probes (research/75) ==="
.venv/bin/python eval/dictation_probe_eval.py \
    --checkpoint "$CKPT" \
    --out research/75-dictation-probe-v10.json || exit 1

echo "=== [2/2] replay benchmark regression (dev set, gate) ==="
.venv/bin/python -m eval.streaming_replay_eval \
    --checkpoint "$CKPT" \
    --energy-gate \
    --progress-file research/75-replay-v10gate.progress \
    --out research/75-replay-v10gate.json || exit 1

echo "=== summaries ==="
.venv/bin/python - <<'EOF'
import json
p = json.load(open("research/75-dictation-probe-v10.json"))
print("digit:", json.dumps(p["digit"]["summary"]))
print("spelled:", json.dumps(p["spelled"]["summary"]))
r = json.load(open("research/75-replay-v10gate.json"))
print("replay:", json.dumps(r["summary"]))
EOF
