"""Expand the plain-ASR replay transcript cache (gap 4).

build_v15_asr_replay draws replay clips only from
data/semantic_endpoint_v9/ls_transcripts.json, which currently holds 4301
entries all <=9 s. That caps --n-asr at ~4301. This script transcribes more
train-clean-100 utterances with the base model and appends them to the same
cache (resumable, incremental writes), and raises the duration ceiling to 14 s
so the replay pool can include the longer clips the v15 recipe intended.

Cache-only: unlike build_ls_pool it does NOT load every audio into RAM.
"""
from __future__ import annotations
import argparse, json, logging, random
from pathlib import Path
import soundfile as sf, torch
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.run_qwen3asr_pkg import iter_librispeech_files  # noqa: E402

SR = 16000
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ls-dir", default="data/librispeech_raw/LibriSpeech/train-clean-100")
    p.add_argument("--cache", default="data/semantic_endpoint_v9/ls_transcripts.json")
    p.add_argument("--n-needed", type=int, default=9000)
    p.add_argument("--min-dur", type=float, default=2.5)
    p.add_argument("--max-dur", type=float, default=14.0)
    p.add_argument("--seed", type=int, default=9)
    args = p.parse_args()
    rng = random.Random(args.seed)

    pairs = list(iter_librispeech_files(args.ls_dir))
    rng.shuffle(pairs)
    cands = []
    for flac, ref in pairs:
        try:
            info = sf.info(flac)
        except Exception:
            continue
        if args.min_dur <= info.frames / info.samplerate <= args.max_dur:
            cands.append(flac)
        if len(cands) >= args.n_needed + 300:
            break
    logger.info("candidates in [%.1f,%.1f]s: %d", args.min_dur, args.max_dur, len(cands))

    cache_path = Path(args.cache)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    logger.info("cache before: %d entries", len(cache))
    todo = [f for f in cands if f not in cache]
    logger.info("to transcribe: %d", len(todo))
    if not todo:
        logger.info("nothing to do"); return

    from qwen_asr import Qwen3ASRModel
    asr = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-0.6B", cache_dir="data/qwen3-asr-0.6b-pkg",
        max_inference_batch_size=8, max_new_tokens=256,
        dtype=torch.bfloat16, device_map="cuda")  # base evals silently ran on CPU (~9x slower)
    logger.info("model device: %s", next(asr.model.parameters()).device)
    BATCH = 8
    for i in tqdm(range(0, len(todo), BATCH), desc="base-transcribe"):
        batch = todo[i:i + BATCH]
        audios, valid = [], []
        for flac in batch:
            try:
                audio, sr = sf.read(flac, dtype="float32")
                if sr != SR:
                    cache[flac] = ""; continue
                if audio.ndim > 1:
                    audio = audio.mean(axis=-1)
                audios.append((audio.astype("float32"), sr)); valid.append(flac)
            except Exception as e:
                logger.warning("read failed %s: %s", flac, e); cache[flac] = ""
        if audios:
            try:
                res = asr.transcribe(audio=audios)
                for flac, r in zip(valid, res):
                    cache[flac] = (r.text or "") if r is not None else ""
            except Exception as e:
                logger.warning("batch failed (%d): %s", len(audios), e)
                for flac in valid:
                    cache[flac] = ""
        cache_path.write_text(json.dumps(cache))
    del asr
    torch.cuda.empty_cache()
    nonempty = sum(1 for v in cache.values() if v.strip())
    logger.info("cache after: %d entries (%d nonempty)", len(cache), nonempty)


if __name__ == "__main__":
    main()
