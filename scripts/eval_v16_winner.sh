#!/bin/bash
# Full eval of a chosen v16 snapshot: merge -> offline WER (gap 4) ->
# big dictation+spelled probe -> Earnings-22 biasing (gap 1) -> streaming replay.
# Usage: scripts/eval_v16_winner.sh <STEP>   e.g. scripts/eval_v16_winner.sh 6000
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONUNBUFFERED=1
STEP=$1
SNAP=checkpoints/semantic_endpoint_v16_es/snap_step${STEP}.pt
MERGED=checkpoints/merged/v16-step${STEP}

echo "=== merge $SNAP $(date -Is) ==="
$PY -m scripts.merge_endpoint_lora --checkpoint "$SNAP" --out-dir "$MERGED"

echo "=== offline WER (gap 4) $(date -Is) ==="
$PY -m eval.offline_wer_ls --model "$MERGED" --out research/89-wer-v16-step${STEP}.json

echo "=== big dictation + spelled probe $(date -Is) ==="
$PY -m eval.dictation_probe_eval --checkpoint "$SNAP" \
    --probes data/probes_big --out research/90-probe-v16-step${STEP}-big.json

echo "=== Earnings-22 biasing (gap 1) $(date -Is) ==="
$PY -m eval.earnings22_biasing --model "$MERGED" --n-utts 300 --shuffle \
    --entity-backend llm --out research/91-biasing-v16-step${STEP}.json

echo "=== streaming replay (conv recall / false / latency) $(date -Is) ==="
$PY -m eval.streaming_replay_eval --checkpoint "$SNAP" \
    --out research/92-replay-v16-step${STEP}.json 2>/dev/null || \
    echo "(streaming_replay_eval needs its default args; run manually if it errors)"

echo "=== DONE v16 step ${STEP} eval $(date -Is) ==="
