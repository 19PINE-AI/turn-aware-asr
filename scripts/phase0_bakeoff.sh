#!/bin/bash
# Phase 0 bake-off launcher.
#
# Runs the three encoder arms sequentially on the same data slice with the
# same hyperparameters. Waits for GPU memory to free between runs.
#
# Usage:
#   bash scripts/phase0_bakeoff.sh data/librispeech/train-clean-100/data.pt
set -euo pipefail

DATA_PATH="${1:-data/librispeech/train-clean-100/data.pt}"
EVAL_DATA="${2:-data/librispeech/test-clean/data.pt}"
STEPS="${STEPS:-2000}"
BSZ="${BSZ:-4}"
LR="${LR:-5e-5}"
EVAL_MAX="${EVAL_MAX:-200}"
OUT_DIR="${OUT_DIR:-checkpoints/phase0-bakeoff}"
MIN_FREE_MB="${MIN_FREE_MB:-10000}"   # require ≥ 10 GB free before launch

wait_for_gpu() {
    while true; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        if [ "$free" -ge "$MIN_FREE_MB" ]; then
            echo "GPU has ${free} MB free; proceeding"
            break
        fi
        echo "GPU only has ${free} MB free; waiting 30 s…"
        sleep 30
    done
}

run_arm() {
    arm="$1"
    extra_flags="${2:-}"
    echo "===== Running arm: $arm $extra_flags ====="
    wait_for_gpu
    PYTHONUNBUFFERED=1 .venv/bin/python -u -m src.train_phase0 \
        --arm "$arm" \
        --data "$DATA_PATH" \
        --steps "$STEPS" --bsz "$BSZ" --lr "$LR" \
        --warmup-steps 100 --log-every 100 \
        --checkpoint-dir "$OUT_DIR/$arm" \
        $extra_flags
    ckpt="$OUT_DIR/$arm/${arm}_step${STEPS}.pt"
    echo "===== Eval $arm ====="
    wait_for_gpu
    PYTHONUNBUFFERED=1 .venv/bin/python -u -m eval.run_librispeech \
        --data "$EVAL_DATA" \
        --max-utterances "$EVAL_MAX" \
        --arm "$arm" \
        --checkpoint "$ckpt" \
        --out "$OUT_DIR/$arm/eval.json"
}

mkdir -p "$OUT_DIR"
run_arm aut_frozen
# AuT top-6 trainable requires more memory — let GPU settle
sleep 20
run_arm aut_unfrozen_top6
echo "===== Bake-off complete ====="
echo "Results:"
for arm in aut_frozen aut_unfrozen_top6; do
    if [ -f "$OUT_DIR/$arm/eval.json" ]; then
        wer=$(python3 -c "import json; print(json.load(open('$OUT_DIR/$arm/eval.json'))['wer_final'])")
        echo "  $arm: WER ${wer}%"
    fi
done
