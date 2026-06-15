"""Scaled endpoint data prep with proper post-pause labels.

Improvements over v1 (eval/endpoint_data_prep.py):
  1. Scales from 88 utts → ~2500 (1500 LS solo + 500 AMI solo + 500 concat)
  2. **Solo audio is padded with 0.3-0.8 s synthetic silence** so the
     model sees actual post-pause frames during training.
     end_target = 1 only AFTER last_word_end + 200 ms (post-pause),
     NOT at the word-end frame itself (that's eager).
  3. **AMI source** added so the head sees real conversational tail
     characteristics, not just clean-speech micro-silence.
  4. **Concat pairs use varied gaps** (0.5 - 2.5 s) so the model learns
     the false-endpoint discrimination more robustly.

Label schema per frame (12.5 Hz, 80 ms/frame):
  eager_target = 1 around last-word boundary (±200 ms)
  end_target   = 1 in the post-pause silence (last_word_end + 200ms
                  onwards, including the synthetic silence pad)
  end_target   = 0 during internal silences (concat pairs)

Output: data/endpoint_v2/data.pt
  - aut_frames: list[(T, 1024)]
  - end_targets, eager_targets: list[(T,)]
  - audio_len_frames: list[int]
  - kind: list[str]  ("solo_ls", "solo_ami", "concat")
  - last_word_end_frame: list[int|None]  (None for concat)
  - ref: list[str]

Usage:
    python -m eval.endpoint_data_prep_v2 \\
        --max-ls 1500 --max-ami 500 --n-concat 500
"""

from __future__ import annotations
import argparse
import hashlib
import io
import logging
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from tqdm import tqdm

from eval.run_qwen3asr_pkg import iter_librispeech_files
from src.aut_encoder import load_aut_from_safetensors
from src.features import log_mel

logger = logging.getLogger(__name__)

FRAME_HZ = 12.5
FRAME_S = 1.0 / FRAME_HZ
SR = 16000


def s_to_frame(s: float) -> int:
    return int(round(s * FRAME_HZ))


def meeting_in_train(meeting_id: str) -> bool:
    h = int(hashlib.md5(meeting_id.encode()).hexdigest(), 16)
    return (h % 3) in (0, 1)


def decode_ami_audio(blob: dict):
    if "array" in blob and blob["array"] is not None:
        return np.asarray(blob["array"], dtype=np.float32), int(blob.get("sampling_rate", SR))
    if "bytes" in blob:
        audio, sr = sf.read(io.BytesIO(blob["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        return audio.astype(np.float32), int(sr)
    return None, None


def encode_with_pad(audio: np.ndarray, pad_silence_s: float, aut, device) -> tuple[torch.Tensor, int]:
    """Return (aut_out (T,1024), audio_end_frame).

    The audio is padded with synthetic silence at the end. The
    audio_end_frame is the LAST frame containing real speech / the
    boundary where silence starts.
    """
    pad_n = int(pad_silence_s * SR)
    padded = np.concatenate([audio, np.zeros(pad_n, dtype=np.float32)])
    audio_end_frame = s_to_frame(len(audio) / SR)
    mel = log_mel(torch.from_numpy(padded))
    with torch.no_grad():
        aut_out = aut(mel.unsqueeze(0).to(device).bfloat16())[0].cpu().float()
    return aut_out, audio_end_frame


def build_solo_labels(T: int, last_word_end_s: float, audio_end_frame: int,
                       eager_w_s: float, end_post_pause_s: float,
                       last_word_start_s: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    eager_target = torch.zeros(T, dtype=torch.float32)
    end_target = torch.zeros(T, dtype=torch.float32)

    last_end_f = s_to_frame(last_word_end_s)
    if last_word_start_s is not None:
        last_start_f = s_to_frame(last_word_start_s)
    else:
        last_start_f = last_end_f

    # Eager: ±eager_w around last-word interval
    ew = max(1, s_to_frame(eager_w_s))
    lo, hi = max(0, last_start_f - ew), min(T, last_end_f + ew + 1)
    eager_target[lo:hi] = 1.0

    # End fires only AFTER last_word_end + end_post_pause_s
    end_lo = min(T, last_end_f + s_to_frame(end_post_pause_s))
    end_target[end_lo:] = 1.0

    return eager_target, end_target


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--ls-dir", default="data/librispeech_raw/LibriSpeech/test-clean")
    p.add_argument("--ami-dir", default="data/ami/ihm")
    p.add_argument("--max-ls", type=int, default=1500)
    p.add_argument("--max-ami", type=int, default=500)
    p.add_argument("--n-concat", type=int, default=500)
    p.add_argument("--solo-pad-min", type=float, default=0.3)
    p.add_argument("--solo-pad-max", type=float, default=0.8)
    p.add_argument("--gap-min", type=float, default=0.5)
    p.add_argument("--gap-max", type=float, default=2.5)
    p.add_argument("--eager-w-s", type=float, default=0.20)
    p.add_argument("--end-post-pause-s", type=float, default=0.20)
    p.add_argument("--out", default="data/endpoint_v2/data.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ami-max-dur", type=float, default=10.0)
    p.add_argument("--ami-min-dur", type=float, default=0.5)
    args = p.parse_args()

    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading Qwen3-ForcedAligner-0.6B…")
    from qwen_asr import Qwen3ForcedAligner
    aligner = Qwen3ForcedAligner.from_pretrained(
        "Qwen/Qwen3-ForcedAligner-0.6B",
        cache_dir="data/qwen3-forced-aligner-pkg",
    )

    logger.info("Loading AuT for feature extraction…")
    aut = load_aut_from_safetensors(
        "data/qwen3-asr-0.6b/model.safetensors",
        device=device,
        dtype=torch.bfloat16,
    )

    out = {"aut_frames": [], "end_targets": [], "eager_targets": [],
           "audio_len_frames": [], "kind": [], "ref": [],
           "last_word_end_frame": []}

    # ── 1. Solo LibriSpeech ────────────────────────────────────────────
    ls_pairs = list(iter_librispeech_files(args.ls_dir))[: args.max_ls]
    logger.info("Solo LibriSpeech: %d utterances", len(ls_pairs))
    n_ok = n_skip = 0
    for flac, ref in tqdm(ls_pairs, desc="solo_ls"):
        audio, sr = sf.read(flac, dtype="float32")
        if sr != SR or len(audio) == 0:
            n_skip += 1; continue
        try:
            results = aligner.align(audio=(np.array(audio, dtype=np.float32), sr),
                                     text=ref, language="English")
        except Exception as e:
            n_skip += 1
            if n_skip < 5: logger.warning("align failed: %s", e)
            continue
        if not results or not getattr(results[0], "items", None):
            n_skip += 1; continue
        items = results[0].items
        last = items[-1]
        last_end = float(last.end_time)
        last_start = float(last.start_time)

        pad_s = rng.uniform(args.solo_pad_min, args.solo_pad_max)
        aut_out, audio_end_frame = encode_with_pad(audio, pad_s, aut, device)
        T = aut_out.shape[0]
        eager_t, end_t = build_solo_labels(T, last_end, audio_end_frame,
                                            args.eager_w_s, args.end_post_pause_s,
                                            last_word_start_s=last_start)
        out["aut_frames"].append(aut_out)
        out["end_targets"].append(end_t)
        out["eager_targets"].append(eager_t)
        out["audio_len_frames"].append(audio_end_frame)
        out["kind"].append("solo_ls")
        out["ref"].append(ref)
        out["last_word_end_frame"].append(s_to_frame(last_end))
        n_ok += 1
    logger.info("solo_ls: %d kept, %d skipped", n_ok, n_skip)

    # ── 2. Solo AMI ──────────────────────────────────────────────────────
    ami_pool: list[dict] = []
    for pq_path in sorted(Path(args.ami_dir).glob("*.parquet")):
        t = pq.read_table(pq_path)
        for row in t.to_pylist():
            if not meeting_in_train(row["meeting_id"]):
                continue
            txt = (row["text"] or "").strip()
            if not txt:
                continue
            ami_pool.append(row)
            if len(ami_pool) >= args.max_ami * 4:
                break
        if len(ami_pool) >= args.max_ami * 4:
            break
    rng.shuffle(ami_pool)
    logger.info("Solo AMI: %d candidate rows", len(ami_pool))
    n_ok = n_skip = 0
    for row in tqdm(ami_pool, desc="solo_ami"):
        if n_ok >= args.max_ami:
            break
        audio, sr = decode_ami_audio(row["audio"])
        if audio is None or sr != SR:
            n_skip += 1; continue
        dur = len(audio) / SR
        if dur < args.ami_min_dur or dur > args.ami_max_dur:
            n_skip += 1; continue
        ref = row["text"].strip()
        try:
            results = aligner.align(audio=(np.array(audio, dtype=np.float32), sr),
                                     text=ref, language="English")
        except Exception:
            n_skip += 1; continue
        if not results or not getattr(results[0], "items", None):
            n_skip += 1; continue
        items = results[0].items
        last = items[-1]
        last_end = float(last.end_time)
        last_start = float(last.start_time)
        pad_s = rng.uniform(args.solo_pad_min, args.solo_pad_max)
        aut_out, audio_end_frame = encode_with_pad(audio, pad_s, aut, device)
        T = aut_out.shape[0]
        eager_t, end_t = build_solo_labels(T, last_end, audio_end_frame,
                                            args.eager_w_s, args.end_post_pause_s,
                                            last_word_start_s=last_start)
        out["aut_frames"].append(aut_out)
        out["end_targets"].append(end_t)
        out["eager_targets"].append(eager_t)
        out["audio_len_frames"].append(audio_end_frame)
        out["kind"].append("solo_ami")
        out["ref"].append(ref)
        out["last_word_end_frame"].append(s_to_frame(last_end))
        n_ok += 1
    logger.info("solo_ami: %d kept, %d skipped", n_ok, n_skip)

    # ── 3. Concat pairs (LS+LS, varied gap) ───────────────────────────
    logger.info("Concat pairs: %d", args.n_concat)
    n_done = n_skip = 0
    indices = list(range(len(ls_pairs)))
    while n_done < args.n_concat and n_skip < args.n_concat * 10:
        a_idx = rng.choice(indices)
        b_idx = rng.choice(indices)
        if a_idx == b_idx:
            n_skip += 1; continue
        a_flac, a_ref = ls_pairs[a_idx]
        b_flac, b_ref = ls_pairs[b_idx]
        a_audio, sr = sf.read(a_flac, dtype="float32")
        b_audio, _ = sf.read(b_flac, dtype="float32")
        gap_s = rng.uniform(args.gap_min, args.gap_max)
        gap = np.zeros(int(gap_s * SR), dtype=np.float32)
        cat = np.concatenate([a_audio, gap, b_audio])
        try:
            results = aligner.align(audio=(np.array(cat, dtype=np.float32), sr),
                                     text=f"{a_ref} {b_ref}", language="English")
        except Exception:
            n_skip += 1; continue
        if not results or not getattr(results[0], "items", None):
            n_skip += 1; continue
        items = results[0].items
        last = items[-1]
        last_end = float(last.end_time)
        last_start = float(last.start_time)

        pad_s = rng.uniform(args.solo_pad_min, args.solo_pad_max)
        aut_out, audio_end_frame = encode_with_pad(cat, pad_s, aut, device)
        T = aut_out.shape[0]
        # End fires only AFTER the final word of the concatenated pair
        eager_t, end_t = build_solo_labels(T, last_end, audio_end_frame,
                                            args.eager_w_s, args.end_post_pause_s,
                                            last_word_start_s=last_start)
        # The internal gap region: ensure end_target = 0 there.
        # last_word of utt_A finished at len(a_audio)/sr; gap starts there.
        gap_start_s = len(a_audio) / sr
        gap_end_s = gap_start_s + gap_s
        gap_lo = s_to_frame(gap_start_s)
        gap_hi = min(T, s_to_frame(gap_end_s) + 1)
        end_t[gap_lo:gap_hi] = 0.0  # explicit no-end during internal silence
        out["aut_frames"].append(aut_out)
        out["end_targets"].append(end_t)
        out["eager_targets"].append(eager_t)
        out["audio_len_frames"].append(audio_end_frame)
        out["kind"].append("concat")
        out["ref"].append(f"{a_ref} <gap_{gap_s:.1f}s> {b_ref}")
        out["last_word_end_frame"].append(s_to_frame(last_end))
        n_done += 1
    logger.info("concat: %d kept, %d skipped", n_done, n_skip)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    logger.info("Saved %d examples to %s (solo_ls=%d, solo_ami=%d, concat=%d)",
                 len(out["aut_frames"]), args.out,
                 sum(1 for k in out["kind"] if k == "solo_ls"),
                 sum(1 for k in out["kind"] if k == "solo_ami"),
                 sum(1 for k in out["kind"] if k == "concat"))


if __name__ == "__main__":
    main()
