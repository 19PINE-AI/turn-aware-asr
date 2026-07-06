#!/bin/bash
# Exp-3 schema-ablation replay scoring: 4 arms x 3 configs = 12 runs.
set -u
cd /home/ubuntu/streaming-vad-asr
PY=.venv/bin/python
SPLIT=data/semantic_endpoint_v3/meeting_split.json
N=25
PROG=/tmp/claude-1000/-home-ubuntu-streaming-vad-asr/dcc4cf15-de08-41a4-b6ba-62d660562903/scratchpad
LOG=$PROG/exp3.log

declare -A CKPT=(
  [base]=checkpoints/abl_base_es/best.pt
  [A]=checkpoints/abl_A_es/best.pt
  [B]=checkpoints/abl_B_es/best.pt
  [C]=checkpoints/abl_C_es/best.pt
)

run () {
  local arm=$1 cfg=$2; shift 2
  local out=research/104-replay-${arm}-${cfg}.json
  local pf=$PROG/104-${arm}-${cfg}.progress
  if [ -f "$out" ]; then echo "[skip existing] $out" | tee -a "$LOG"; return; fi
  echo "=== $(date +%H:%M:%S) START arm=$arm cfg=$cfg $* ===" | tee -a "$LOG"
  $PY -m eval.streaming_replay_eval --checkpoint "${CKPT[$arm]}" \
      --split "$SPLIT" --n-stretches $N --out "$out" \
      --progress-file "$pf" "$@" >>"$LOG" 2>&1
  echo "=== $(date +%H:%M:%S) DONE arm=$arm cfg=$cfg rc=$? ===" | tee -a "$LOG"
}

for arm in base A B C; do
  run "$arm" gated    --energy-gate
  run "$arm" ungated
  run "$arm" confirm1 --energy-gate --confirm-silent-chunks 1
done
echo "ALL DONE $(date +%H:%M:%S)" | tee -a "$LOG"
