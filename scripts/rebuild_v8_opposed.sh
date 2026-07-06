#!/bin/bash
# Gap 3: regenerate the opposed-pools (v8) data chain from raw sources, so a
# second training seed can confirm the oscillation fingerprint reproduces.
# Chain: v2(_simple) -> v3 -> v4 -> v5 -> v6 -> v8. All deterministically seeded.
# The historical single-run curve is retained at
# checkpoints/semantic_endpoint_v8_es/eval_log.json (do NOT overwrite it).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONUNBUFFERED=1

echo "=== [1/6] v2 (base-model transcription of LibriSpeech) $(date -Is) ==="
# write directly to the _simple path that build_v3 reads (avoids the historical rename)
$PY -m eval.semantic_endpoint_data_v2 \
    --n-single 500 --n-double 1500 --n-disfluency 500 \
    --out data/semantic_endpoint_v2_simple/data.pt \
    --transcripts-cache data/semantic_endpoint_v2_simple/base_transcripts.json

echo "=== [2/6] v3 (truncated + AMI turn pairs) $(date -Is) ==="
$PY -m eval.build_v3_training_data --base-data data/semantic_endpoint_v2_simple/data.pt

echo "=== [3/6] v4 $(date -Is) ===" && $PY -m eval.build_v4_training_data
echo "=== [4/6] v5 $(date -Is) ===" && $PY -m eval.build_v5_training_data
echo "=== [5/6] v6 $(date -Is) ===" && $PY -m eval.build_v6_training_data
echo "=== [6/6] v8 opposed-pools (fresh disjoint AMI no_fire) $(date -Is) ==="
$PY -m eval.build_v8_training_data --n-no-fire 1000

echo "=== DONE data chain $(date -Is) ==="
ls -la data/semantic_endpoint_v8/data.pt
