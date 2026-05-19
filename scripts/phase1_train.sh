#!/bin/bash
# Phase 1 — ASR foundation training on train-clean-100.
#
# Trains AuT-frozen + AuT.proj/ln_post unfrozen + Qwen3-0.6B fully trainable
# on 100 h of LibriSpeech for STEPS steps. Then evaluates on test-clean.
set -euo pipefail

STEPS="${STEPS:-30000}"
BSZ="${BSZ:-8}"
LR="${LR:-1e-4}"
WARMUP="${WARMUP:-1000}"
TRAIN_DATA="${TRAIN_DATA:-data/librispeech/train-clean-100/data.pt}"
EVAL_DATA="${EVAL_DATA:-data/librispeech/test-clean/data.pt}"
EVAL_MAX="${EVAL_MAX:-500}"
OUT_DIR="${OUT_DIR:-checkpoints/phase1}"

mkdir -p "$OUT_DIR"

echo "===== Phase 1: $STEPS steps × bsz $BSZ × lr $LR ====="
PYTHONUNBUFFERED=1 .venv/bin/python -u -m src.train_phase0 \
    --arm aut_frozen \
    --unfreeze-aut-proj \
    --data "$TRAIN_DATA" \
    --steps "$STEPS" --bsz "$BSZ" --lr "$LR" \
    --warmup-steps "$WARMUP" --log-every 200 \
    --save-every 2500 \
    --checkpoint-dir "$OUT_DIR"

ckpt="$OUT_DIR/aut_frozen_step${STEPS}.pt"
echo "===== Eval Phase 1 on $EVAL_MAX utterances ====="
PYTHONUNBUFFERED=1 .venv/bin/python -u -m eval.run_librispeech \
    --data "$EVAL_DATA" \
    --max-utterances "$EVAL_MAX" \
    --arm aut_frozen \
    --checkpoint "$ckpt" \
    --out "$OUT_DIR/eval.json"

echo "===== Phase 1 result ====="
python3 -c "import json; r=json.load(open('$OUT_DIR/eval.json')); print('WER:', r['wer_final'], '%')"
