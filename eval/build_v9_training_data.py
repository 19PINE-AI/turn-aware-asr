"""Build v9 training data — one causal label rule (design: research/59).

Fire marker `<EAGER_END_SPEECH><END_SPEECH>` appears at a point in the
target iff, at that point in the AUDIO: (1) speech so far is semantically
complete AND (2) >= 0.3 s of silence has elapsed. Both observable from the
prefix — no label ever depends on future audio.

Schemas:
  single_sil      utt (tail-trimmed) + sil 0.3-1.2 s          ->  "T M"
  complete_nosil  SAME utts as single_sil, tail <= 0.1 s      ->  "T"
  truncated       LS cut mid-speech 30-80 %                   ->  partial T
  pair_fire       A(complete) + gap 0.3-2.5 + B [+ tail sil]  ->  "A M B M" | "A M B"
  pair_hold       A(incomplete) + gap 0.3-2.5 + B + tail sil  ->  "A B M" (LS: "T M")
  long_sil        utt + sil 2-4 s                             ->  "T M"

The single_sil/complete_nosil pairing on IDENTICAL utterances is the
anti-oscillation device: opposite labels differ only in the observable
tail, so trailing silence is the only feature that separates them
(v8 gave opposite labels to identically-distributed pools — unlearnable).

Usage:
    python -m eval.build_v9_training_data \
        --ls-dir data/librispeech_raw/LibriSpeech/train-clean-100 \
        --ami-dir data/ami/ihm --out data/semantic_endpoint_v9/data.pt
"""

from __future__ import annotations
import argparse
import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from eval.run_qwen3asr_pkg import iter_librispeech_files
from eval.build_v3_training_data import meeting_in_train, decode_audio

logger = logging.getLogger(__name__)

END_TOK = "<END_SPEECH>"
EAGER_TOK = "<EAGER_END_SPEECH>"
M = f"{EAGER_TOK}{END_TOK}"
SR = 16000

# --------------------------------------------------------- audio utilities

def trim_tail_silence(audio: np.ndarray, thresh: float = 0.005,
                      frame_ms: float = 20.0, keep_ms: float = 80.0) -> np.ndarray:
    """Trim trailing low-energy frames, keeping ~keep_ms after last speech."""
    n = int(SR * frame_ms / 1000)
    if len(audio) < 2 * n:
        return audio
    frames = len(audio) // n
    rms = np.sqrt(np.mean(audio[: frames * n].reshape(frames, n) ** 2, axis=1))
    active = np.nonzero(rms > thresh)[0]
    if len(active) == 0:
        return audio
    end = min(len(audio), (active[-1] + 1) * n + int(SR * keep_ms / 1000))
    return audio[:end]


def sil(dur_s: float) -> np.ndarray:
    return np.zeros(int(dur_s * SR), dtype=np.float32)


# ------------------------------------------------- completeness classifier

INCOMPLETE_TAIL = {
    "and", "but", "so", "or", "uh", "um", "er", "the", "a", "an", "to",
    "of", "in", "with", "that", "because", "if", "then", "mean", "know",
    "like", "kind", "sort", "very", "really", "just",
}
ACK_WORDS = {
    "yeah", "okay", "ok", "right", "mm-hmm", "mmhmm", "hmm", "exactly",
    "sure", "no", "yes", "yep", "nope", "thanks", "cool", "alright",
    "fine", "good", "great",
}


def is_complete(text: str) -> bool:
    words = text.lower().replace(".", "").replace(",", "").split()
    if not words:
        return False
    if words[-1].endswith("-"):          # AMI partial-word annotation
        return False
    if words[-1] in INCOMPLETE_TAIL:
        return False
    if len(words) < 3:
        return all(w in ACK_WORDS for w in words)
    return True


# --------------------------------------------------------------- builders

def ex(audio: np.ndarray, text: str, schema: str, source: str, **extra) -> dict:
    return {"audio": audio.astype(np.float32), "text": text, "schema": schema,
            "source": source, "audio_end_s": len(audio) / SR, **extra}


def build_ls_pool(ls_dir: str, n_needed: int, rng: random.Random,
                  cache_path: Path, max_dur_s: float = 9.0) -> list[dict]:
    """LS utterances with base-model transcripts (cached, resumable)."""
    pairs = list(iter_librispeech_files(ls_dir))
    rng.shuffle(pairs)
    cands = []
    for flac, ref in pairs:
        try:
            info = sf.info(flac)
        except Exception:
            continue
        if 2.5 <= info.frames / info.samplerate <= max_dur_s:
            cands.append(flac)
        if len(cands) >= n_needed + 300:
            break

    cache: dict[str, str] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        logger.info("transcript cache: %d entries", len(cache))
    todo = [f for f in cands if f not in cache]
    if todo:
        logger.info("Transcribing %d LS utts with base model…", len(todo))
        from qwen_asr import Qwen3ASRModel
        asr = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-0.6B", cache_dir="data/qwen3-asr-0.6b-pkg",
            max_inference_batch_size=8, max_new_tokens=256)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        for i, flac in enumerate(tqdm(todo, desc="base-transcribe")):
            try:
                audio, sr = sf.read(flac, dtype="float32")
                if sr != SR:
                    cache[flac] = ""
                    continue
                if audio.ndim > 1:
                    audio = audio.mean(axis=-1)
                res = asr.transcribe(audio=(audio.astype(np.float32), sr))
                cache[flac] = (res[0].text or "") if res else ""
            except Exception as e:
                logger.warning("transcribe failed %s: %s", flac, e)
                cache[flac] = ""
            if (i + 1) % 100 == 0:
                cache_path.write_text(json.dumps(cache))
        cache_path.write_text(json.dumps(cache))
        del asr
        torch.cuda.empty_cache()

    pool = []
    for flac in cands:
        text = cache.get(flac, "").strip()
        if not text or len(text.split()) < 3:
            continue
        try:
            audio, sr = sf.read(flac, dtype="float32")
        except Exception:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        pool.append({"audio": audio.astype(np.float32), "text": text, "id": flac})
        if len(pool) >= n_needed:
            break
    logger.info("LS pool: %d utts", len(pool))
    return pool


def build_ami_pools(ami_dir: str, rng: random.Random, max_dur_s: float = 9.0):
    """Return (singles, diff_spk_pairs, same_spk_pairs) from TRAIN meetings."""
    import pyarrow.parquet as pq
    meetings: dict[str, list[dict]] = defaultdict(list)
    for pq_path in sorted(Path(ami_dir).glob("*.parquet")):
        for row in pq.read_table(pq_path).to_pylist():
            if meeting_in_train(row["meeting_id"]) and (row["text"] or "").strip():
                meetings[row["meeting_id"]].append(row)
    for mid in meetings:
        meetings[mid].sort(key=lambda r: float(r["begin_time"]))

    singles, diff_pairs, same_pairs = [], [], []
    for mid, utts in meetings.items():
        for u in utts:
            audio, sr = decode_audio(u["audio"])
            if audio is None or sr != SR or not (0.8 <= len(audio) / SR <= max_dur_s):
                continue
            singles.append({"audio": audio, "text": u["text"].strip(),
                            "meeting_id": mid})
        for a, b in zip(utts, utts[1:]):
            gap = float(b["begin_time"]) - float(a["end_time"])
            if not (0.3 <= gap <= 2.5):
                continue
            aa, sra = decode_audio(a["audio"])
            ab, srb = decode_audio(b["audio"])
            if aa is None or ab is None or sra != SR or srb != SR:
                continue
            if (len(aa) + len(ab)) / SR + gap > 14.0:
                continue
            pair = {"audio_a": aa, "audio_b": ab, "text_a": a["text"].strip(),
                    "text_b": b["text"].strip(), "gap_s": gap, "meeting_id": mid}
            (diff_pairs if a["speaker_id"] != b["speaker_id"] else same_pairs).append(pair)
    rng.shuffle(singles)
    rng.shuffle(diff_pairs)
    rng.shuffle(same_pairs)
    logger.info("AMI pools: %d singles, %d diff-spk pairs, %d same-spk pairs",
                 len(singles), len(diff_pairs), len(same_pairs))
    return singles, diff_pairs, same_pairs


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--ls-dir", default="data/librispeech_raw/LibriSpeech/train-clean-100")
    p.add_argument("--ami-dir", default="data/ami/ihm")
    p.add_argument("--out", default="data/semantic_endpoint_v9/data.pt")
    p.add_argument("--transcripts-cache", default="data/semantic_endpoint_v9/ls_transcripts.json")
    p.add_argument("--seed", type=int, default=9)
    args = p.parse_args()
    rng = random.Random(args.seed)

    N = dict(single_sil_ls=1500, single_sil_ami=1000,
             nosil_ls=900, nosil_ami=600,
             truncated=2000,
             pair_fire_diff=1000, pair_fire_same=500, pair_fire_ls=1000,
             pair_hold_ls=1000, pair_hold_ami=500,
             long_sil_ls=250, long_sil_ami=250,
             silence_only=800, lead_sil=700)

    ls_needed = (N["single_sil_ls"] + N["truncated"] + 2 * N["pair_fire_ls"]
                 + N["pair_hold_ls"] + N["long_sil_ls"] + N["lead_sil"])
    ls_pool = build_ls_pool(args.ls_dir, ls_needed, rng, Path(args.transcripts_cache))
    ami_singles, ami_diff, ami_same = build_ami_pools(args.ami_dir, rng)

    examples: list[dict] = []
    tail = lambda: sil(rng.uniform(0.3, 1.2))

    # ---- single_sil + complete_nosil (paired on the SAME utterances)
    ls_iter = iter(ls_pool)

    def take_ls(n: int) -> list[dict]:
        out = []
        for _ in range(n):
            try:
                out.append(next(ls_iter))
            except StopIteration:
                break
        return out
    ss_ls = take_ls(N["single_sil_ls"])
    ss_ami = ami_singles[: N["single_sil_ami"]]
    for i, u in enumerate(ss_ls):
        a = trim_tail_silence(u["audio"])
        examples.append(ex(np.concatenate([a, tail()]), f"{u['text']} {M}",
                            "single_sil", "ls"))
        if i < N["nosil_ls"]:
            examples.append(ex(a, u["text"], "complete_nosil", "ls"))
    for i, u in enumerate(ss_ami):
        a = trim_tail_silence(u["audio"])
        examples.append(ex(np.concatenate([a, tail()]), f"{u['text']} {M}",
                            "single_sil", "ami", meeting_id=u["meeting_id"]))
        if i < N["nosil_ami"]:
            examples.append(ex(a, u["text"], "complete_nosil", "ami",
                                meeting_id=u["meeting_id"]))

    # ---- truncated (mid-speech cut, proportional partial text, no marker)
    for u in take_ls(N["truncated"]):
        words = u["text"].split()
        if len(words) < 4:
            continue
        frac = rng.uniform(0.30, 0.80)
        a = u["audio"][: int(len(u["audio"]) * frac)]
        if len(a) < SR:
            continue
        examples.append(ex(a, " ".join(words[: max(1, round(len(words) * frac))]),
                            "truncated", "ls", trunc_frac=frac))

    # ---- pair_fire: A complete + gap + B; 2/3 get tail sil (A M B M), 1/3 none (A M B)
    def add_pair_fire(aa, ta, ab, tb, gap_s, source, **extra):
        with_tail = rng.random() < 2 / 3
        parts = [trim_tail_silence(aa), sil(gap_s), trim_tail_silence(ab)]
        text = f"{ta} {M} {tb} {M}" if with_tail else f"{ta} {M} {tb}"
        if with_tail:
            parts.append(tail())
        examples.append(ex(np.concatenate(parts), text, "pair_fire", source,
                            gap_s=gap_s, **extra))

    for pr in ami_diff[: N["pair_fire_diff"]]:
        add_pair_fire(pr["audio_a"], pr["text_a"], pr["audio_b"], pr["text_b"],
                      pr["gap_s"], "ami_diff", meeting_id=pr["meeting_id"])
    same_complete = [pr for pr in ami_same if is_complete(pr["text_a"])]
    for pr in same_complete[: N["pair_fire_same"]]:
        add_pair_fire(pr["audio_a"], pr["text_a"], pr["audio_b"], pr["text_b"],
                      pr["gap_s"], "ami_same_complete", meeting_id=pr["meeting_id"])
    pf_pool = take_ls(2 * N["pair_fire_ls"])
    for i in range(0, len(pf_pool) - 1, 2):
        ua, ub = pf_pool[i], pf_pool[i + 1]
        add_pair_fire(ua["audio"], ua["text"], ub["audio"], ub["text"],
                      rng.uniform(0.3, 2.5), "ls")

    # ---- pair_hold: A incomplete + gap + B + tail sil -> one final marker
    for u in take_ls(N["pair_hold_ls"]):        # LS split-resume synthesis
        audio, words = u["audio"], u["text"].split()
        if len(words) < 6:
            continue
        frac = rng.uniform(0.40, 0.70)
        split = int(len(audio) * frac)
        cat = np.concatenate([audio[:split], sil(rng.uniform(0.3, 2.5)),
                              audio[split:], tail()])
        examples.append(ex(cat, f"{u['text']} {M}", "pair_hold", "ls_split",
                            split_frac=frac))
    same_incomplete = [pr for pr in ami_same if not is_complete(pr["text_a"])]
    for pr in same_incomplete[: N["pair_hold_ami"]]:
        cat = np.concatenate([trim_tail_silence(pr["audio_a"]), sil(pr["gap_s"]),
                              trim_tail_silence(pr["audio_b"]), tail()])
        examples.append(ex(cat, f"{pr['text_a']} {pr['text_b']} {M}",
                            "pair_hold", "ami_same_incomplete",
                            gap_s=pr["gap_s"], meeting_id=pr["meeting_id"]))

    # ---- long_sil: fire survives long silence without marker spam
    for u in take_ls(N["long_sil_ls"]):
        examples.append(ex(np.concatenate([trim_tail_silence(u["audio"]),
                                            sil(rng.uniform(2.0, 4.0))]),
                            f"{u['text']} {M}", "long_sil", "ls"))
    for u in ami_singles[N["single_sil_ami"]: N["single_sil_ami"] + N["long_sil_ami"]]:
        examples.append(ex(np.concatenate([trim_tail_silence(u["audio"]),
                                            sil(rng.uniform(2.0, 4.0))]),
                            f"{u['text']} {M}", "long_sil", "ami",
                            meeting_id=u["meeting_id"]))

    # ---- silence_only: pure silence -> EMPTY target, no markers.
    # Discovered by the replay eval (research/61): every v1-v8 example
    # begins with speech, so on real timelines (mostly silence on a
    # single-speaker channel) the model hallucinates text and spams fires.
    for _ in range(N["silence_only"]):
        examples.append(ex(sil(rng.uniform(0.5, 4.0)), "", "silence_only", "synth"))

    # ---- lead_sil: silence BEFORE speech -> normal transcript + marker.
    # Segments in deployment start mid-silence; leading silence must not
    # change transcription or firing behavior.
    for u in take_ls(N["lead_sil"]):
        cat = np.concatenate([sil(rng.uniform(0.5, 2.0)),
                              trim_tail_silence(u["audio"]), tail()])
        examples.append(ex(cat, f"{u['text']} {M}", "lead_sil", "ls"))

    rng.shuffle(examples)
    counts = defaultdict(int)
    for e in examples:
        counts[e["schema"]] += 1
    logger.info("v9 data: %d examples, schemas=%s", len(examples), dict(counts))
    inc_frac = len(same_incomplete) / max(1, len(ami_same))
    logger.info("AMI same-spk pairs: %.0f%% classified incomplete-A", 100 * inc_frac)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(examples, args.out)
    logger.info("Saved %s", args.out)


if __name__ == "__main__":
    main()
