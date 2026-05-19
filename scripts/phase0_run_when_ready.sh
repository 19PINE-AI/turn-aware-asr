#!/bin/bash
# Wait for train-clean-100 to be available, extract, prep, then run the
# Phase 0 bake-off with --unfreeze-aut-proj. Polls every 30 s.
set -euo pipefail

TAR=data/librispeech_raw/train-clean-100.tar.gz
EXPECTED_SIZE=6387309499
EXTRACTED_DIR=data/librispeech_raw/LibriSpeech/train-clean-100
PREP_OUT=data/librispeech/train-clean-100/data.pt

echo "[1/3] Waiting for $TAR to reach $EXPECTED_SIZE bytes…"
until [ -f "$TAR" ] && [ "$(stat -c %s "$TAR")" -ge "$EXPECTED_SIZE" ]; do
    sz=$(stat -c %s "$TAR" 2>/dev/null || echo 0)
    pct=$(( sz * 100 / EXPECTED_SIZE ))
    echo "  $((sz / 1024 / 1024)) MB / 6090 MB ($pct%)"
    sleep 30
done
echo "Download complete."

echo "[2/3] Verify and extract…"
if ! gzip -t "$TAR" 2>/dev/null; then
    echo "FATAL: $TAR is corrupted"; exit 1
fi
if [ ! -d "$EXTRACTED_DIR" ]; then
    tar -xzf "$TAR" -C data/librispeech_raw/
fi
echo "Extracted to $EXTRACTED_DIR"

echo "[3/3] Prep mels for first 5000 utterances (~17 h) — full 28k is overkill for Phase 0…"
.venv/bin/python -m scripts.prepare_librispeech \
    --split train-clean-100 \
    --raw-root data/librispeech_raw \
    --out-root data/librispeech \
    --max-utterances 5000

echo "Ready to launch bake-off."
echo "Run:  STEPS=3000 BSZ=4 LR=5e-5 bash scripts/phase0_bakeoff.sh data/librispeech/train-clean-100/data.pt"
