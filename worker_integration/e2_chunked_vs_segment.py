"""E2 driver — chunked-placeholder vs single-segment WER on the vLLM path.

Decides the serving primitive (research/60 gate E2, design research/64): does feeding
0.5 s audio chunks as N SEPARATE audio placeholders (the resident fd_step_stream
shape, flat per-frame cost) degrade WER versus decoding the utterance as ONE audio
segment (the in-distribution shape the model was trained on)?

  * single-segment : prompt has ONE <|audio_start|><|audio_pad|><|audio_end|>,
                     multi_modal_data audio = [whole_utterance].
  * chunked        : prompt has N placeholders (one per 0.5 s chunk),
                     multi_modal_data audio = [chunk_0, chunk_1, …]. AuT window
                     boundaries then fall at chunk edges (out-of-distribution).

Both decode once to completion; WER computed against the LibriSpeech reference.
Decision rule (research/60): if chunked WER ≈ single-segment (< 0.5 pp), the flat-cost
resident fd_step_stream path is usable; otherwise adopt the bounded-re-feed protocol
(worker_integration/qwen3asr_stream.Qwen3ASRStreamSession).

HARD CONSTRAINT: instantiates a vLLM engine, USES THE GPU. Run only on the main
GPU-owning process (research/60 GPU-contention risk). Command in research/64.
"""

from __future__ import annotations
import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.metrics import wer

logger = logging.getLogger(__name__)
SR = 16000
APH = "<|audio_start|><|audio_pad|><|audio_end|>"


def load_ls_utterances(ls_root: Path, n: int, rng: random.Random):
    """Return n (audio float32@16k, reference_text) pairs from a LibriSpeech tree."""
    trans = sorted(ls_root.rglob("*.trans.txt"))
    refs = {}
    for tf in trans:
        for line in tf.read_text().splitlines():
            uid, _, text = line.partition(" ")
            refs[uid] = (tf.parent, text)
    uids = list(refs)
    rng.shuffle(uids)
    out = []
    for uid in uids:
        parent, text = refs[uid]
        flac = parent / f"{uid}.flac"
        if not flac.exists():
            continue
        audio, sr = sf.read(str(flac), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        if sr != SR:
            continue
        out.append((audio.astype(np.float32), text))
        if len(out) >= n:
            break
    return out


def build_prompt(n_ph: int, context: str = "") -> str:
    return (f"<|im_start|>system\n{context}<|im_end|>\n"
            f"<|im_start|>user\n{APH * n_ph}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--ls-root", default="data/librispeech_raw/LibriSpeech/train-clean-100")
    p.add_argument("--n-utts", type=int, default=30)
    p.add_argument("--chunk-s", type=float, default=0.5)
    p.add_argument("--gpu-mem", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    utts = load_ls_utterances(Path(args.ls_root), args.n_utts, random.Random(args.seed))
    logger.info("Loaded %d LS utterances", len(utts))

    # TODO(GPU): build the resident vLLM engine (uses the GPU; run on the main process).
    from vllm import LLM, SamplingParams
    engine = LLM(model=args.model_dir, trust_remote_code=True,
                 gpu_memory_utilization=args.gpu_mem, max_model_len=8192,
                 enable_prefix_caching=True, limit_mm_per_prompt={"audio": 256})
    sp = SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=False)

    def strip(t: str) -> str:
        import re
        return re.sub(r"^language\s+\S+\s*<asr_text>", "", t).replace("<|im_end|>", "").strip()

    def decode(req):
        return strip(engine.generate([req], sampling_params=sp, use_tqdm=False)[0].outputs[0].text)

    seg_wer, chk_wer = [], []
    per = []
    csz = int(args.chunk_s * SR)
    for audio, ref in utts:
        seg_hyp = decode({"prompt": build_prompt(1),
                          "multi_modal_data": {"audio": [audio]}})
        chunks = [audio[i:i + csz] for i in range(0, len(audio), csz)]
        chk_hyp = decode({"prompt": build_prompt(len(chunks)),
                          "multi_modal_data": {"audio": chunks}})
        sw, cw = wer(seg_hyp, ref), wer(chk_hyp, ref)
        seg_wer.append(sw); chk_wer.append(cw)
        per.append({"ref": ref, "seg_hyp": seg_hyp, "chk_hyp": chk_hyp,
                    "n_chunks": len(chunks), "seg_wer": sw, "chk_wer": cw})

    seg_m, chk_m = float(np.mean(seg_wer)), float(np.mean(chk_wer))
    summary = {"n": len(utts), "single_segment_wer": seg_m, "chunked_wer": chk_m,
               "delta_pp": (chk_m - seg_m) * 100.0,
               "decision": ("chunked_ok(fd_step_stream)" if (chk_m - seg_m) < 0.005
                            else "use_bounded_refeed")}
    logger.info("== E2 == single-seg WER %.4f | chunked WER %.4f | Δ %.2f pp -> %s",
                seg_m, chk_m, summary["delta_pp"], summary["decision"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "per_utt": per}, indent=2))
    logger.info("Saved %s", args.out)


if __name__ == "__main__":
    main()
