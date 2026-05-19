#!/bin/bash
# Direct LibriSpeech download from openslr.org. Bypasses HF datasets/torchcodec.
#
# Usage:
#   bash scripts/download_librispeech.sh test-clean
#   bash scripts/download_librispeech.sh train-clean-100
set -euo pipefail

split="${1:-test-clean}"
out_root="${2:-data/librispeech_raw}"
mkdir -p "$out_root"

case "$split" in
    dev-clean)        url="https://www.openslr.org/resources/12/dev-clean.tar.gz";;
    dev-other)        url="https://www.openslr.org/resources/12/dev-other.tar.gz";;
    test-clean)       url="https://www.openslr.org/resources/12/test-clean.tar.gz";;
    test-other)       url="https://www.openslr.org/resources/12/test-other.tar.gz";;
    train-clean-100)  url="https://www.openslr.org/resources/12/train-clean-100.tar.gz";;
    train-clean-360)  url="https://www.openslr.org/resources/12/train-clean-360.tar.gz";;
    train-other-500)  url="https://www.openslr.org/resources/12/train-other-500.tar.gz";;
    *) echo "Unknown split: $split"; exit 1;;
esac

tar_path="$out_root/$split.tar.gz"
echo "Downloading/resuming $split from $url …"
curl -fL -C - --retry 5 --retry-delay 10 --progress-bar -o "$tar_path" "$url"

if [ ! -d "$out_root/LibriSpeech/$split" ]; then
    echo "Extracting…"
    tar -xzf "$tar_path" -C "$out_root"
fi
echo "Ready: $out_root/LibriSpeech/$split"
