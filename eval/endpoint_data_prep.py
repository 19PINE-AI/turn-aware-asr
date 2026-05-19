"""Generate endpoint-detection training data using Qwen3-ForcedAligner.

For each LibriSpeech utterance:
  - Align audio to transcript → word-level timestamps
  - Audio runs 0 .. T_aud frames at 12.5 Hz
  - end_target[t] = 1 for frames in [last_word_end - 200ms, T_aud_end + 200ms]
  - eager_target[t] = 1 for frames in [last_word_start - 200ms, last_word_end + 200ms]

These labels enable a 0-delay BCE head on AuT output to predict endpoint
probability per 80 ms frame.

Also generates a synthetic "internal silence" test set by concatenating
utterance pairs with a 1.0 s pause between them. The endpoint detector
should NOT fire during the internal silence (since the speaker continues).

Output: data/endpoint/data.pt with keys
  - aut_frames: list of (T, 1024) tensors (AuT-encoded each utt)
  - end_targets: list of (T,) tensors
  - eager_targets: list of (T,) tensors
  - audio_len_frames: list of int
  - kind: list of str ("solo" or "concat")

Usage:
  python -m eval.endpoint_data_prep --max-utterances 100
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from eval.run_qwen3asr_pkg import iter_librispeech_files
from src.aut_encoder import load_aut_from_safetensors
from src.features import log_mel

logger = logging.getLogger(__name__)

FRAME_HZ = 12.5
FRAME_S = 1.0 / FRAME_HZ  # 80 ms per frame


def s_to_frame(s: float) -> int:
    return int(round(s * FRAME_HZ))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--split-dir", default="data/librispeech_raw/LibriSpeech/test-clean")
    p.add_argument("--max-utterances", type=int, default=100)
    p.add_argument("--n-concat-pairs", type=int, default=30,
                    help="Number of synthetic concatenated pairs to add")
    p.add_argument("--gap-s", type=float, default=1.0,
                    help="Inter-utt gap for concat pairs (in seconds)")
    p.add_argument("--eager-window-s", type=float, default=0.20)
    p.add_argument("--end-window-s", type=float, default=0.20)
    p.add_argument("--out", default="data/endpoint/data.pt")
    args = p.parse_args()

    logger.info("Loading Qwen3-ForcedAligner-0.6B…")
    from qwen_asr import Qwen3ForcedAligner
    aligner = Qwen3ForcedAligner.from_pretrained(
        "Qwen/Qwen3-ForcedAligner-0.6B",
        cache_dir="data/qwen3-forced-aligner-pkg",
    )

    logger.info("Loading AuT for feature extraction…")
    aut = load_aut_from_safetensors(
        "data/qwen3-asr-0.6b/model.safetensors",
        device="cuda",
        dtype=torch.bfloat16,
    )

    pairs = list(iter_librispeech_files(args.split_dir))[: args.max_utterances]
    logger.info("Aligning %d utterances", len(pairs))

    out = {"aut_frames": [], "end_targets": [], "eager_targets": [],
           "audio_len_frames": [], "kind": [], "ref": []}

    eager_w = max(1, s_to_frame(args.eager_window_s))
    end_w = max(1, s_to_frame(args.end_window_s))

    for flac, ref in tqdm(pairs, desc="solo"):
        audio, sr = sf.read(flac, dtype="float32")
        if sr != 16000:
            continue
        try:
            results = aligner.align(audio=(np.array(audio, dtype=np.float32), sr),
                                     text=ref, language="English")
        except Exception as e:
            logger.warning("align failed on %s: %s", flac, e)
            continue
        r = results[0]
        # Qwen3-ForcedAligner returns a ForcedAlignResult with .items, each
        # a ForcedAlignItem with .text, .start_time, .end_time.
        items = getattr(r, "items", None)
        if not items:
            continue
        last = items[-1]
        last_start = float(last.start_time)
        last_end = float(last.end_time)

        T_aud_post = s_to_frame(len(audio) / sr)
        mel = log_mel(torch.from_numpy(audio))
        with torch.no_grad():
            aut_out = aut(mel.unsqueeze(0).cuda().bfloat16())[0].cpu().float()
        T = aut_out.shape[0]

        end_target = torch.zeros(T, dtype=torch.float32)
        eager_target = torch.zeros(T, dtype=torch.float32)
        # End fires from last_word_end onwards (with ±window)
        ef = s_to_frame(last_end)
        lo, hi = max(0, ef - end_w), min(T, ef + end_w + 1)
        end_target[lo:hi] = 1.0
        # Also count the trailing audio as end
        end_target[ef:] = 1.0
        # Eager fires at the last word's interval
        sf_idx = s_to_frame(last_start)
        lo, hi = max(0, sf_idx - eager_w), min(T, ef + eager_w + 1)
        eager_target[lo:hi] = 1.0

        out["aut_frames"].append(aut_out)
        out["end_targets"].append(end_target)
        out["eager_targets"].append(eager_target)
        out["audio_len_frames"].append(T_aud_post)
        out["kind"].append("solo")
        out["ref"].append(ref)

    # Build synthetic concat pairs (utt_A + gap + utt_B). The "internal" gap
    # is NOT an endpoint — should be classified as non-end.
    logger.info("Building %d synthetic concat pairs (gap=%.1fs)",
                 args.n_concat_pairs, args.gap_s)
    rng = np.random.default_rng(0)
    indices = list(range(len(pairs)))
    n_done = 0
    for i in tqdm(range(args.n_concat_pairs), desc="concat"):
        a_idx = int(rng.choice(indices))
        b_idx = int(rng.choice(indices))
        if a_idx == b_idx:
            continue
        a_flac, a_ref = pairs[a_idx]
        b_flac, b_ref = pairs[b_idx]
        a_audio, sr = sf.read(a_flac, dtype="float32")
        b_audio, _ = sf.read(b_flac, dtype="float32")
        gap = np.zeros(int(args.gap_s * sr), dtype=np.float32)
        cat = np.concatenate([a_audio, gap, b_audio])
        # For the concat case, the "true" endpoint is only at the end of cat
        mel = log_mel(torch.from_numpy(cat))
        with torch.no_grad():
            aut_out = aut(mel.unsqueeze(0).cuda().bfloat16())[0].cpu().float()
        T = aut_out.shape[0]
        end_target = torch.zeros(T, dtype=torch.float32)
        eager_target = torch.zeros(T, dtype=torch.float32)
        # End fires only at the very end
        T_audio = s_to_frame(len(cat) / sr)
        ef = T_audio
        lo, hi = max(0, ef - end_w), min(T, ef + end_w + 1)
        end_target[lo:hi] = 1.0
        end_target[ef:] = 1.0
        # Eager: fire near last word; we approximate as the last 1s before T_audio.
        # No alignment for concat — this is a coarse heuristic.
        eager_lo = max(0, T_audio - s_to_frame(1.0))
        eager_target[eager_lo:T_audio] = 1.0
        out["aut_frames"].append(aut_out)
        out["end_targets"].append(end_target)
        out["eager_targets"].append(eager_target)
        out["audio_len_frames"].append(T_audio)
        out["kind"].append("concat")
        out["ref"].append(f"{a_ref} <gap> {b_ref}")
        n_done += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    logger.info("Saved %d examples to %s (solo=%d, concat=%d)",
                 len(out["aut_frames"]), args.out,
                 sum(1 for k in out["kind"] if k == "solo"),
                 sum(1 for k in out["kind"] if k == "concat"))


if __name__ == "__main__":
    main()
