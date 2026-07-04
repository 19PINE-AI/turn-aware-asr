"""E8 data-prep: freeze a fixed biasing-preservation dataset for the in-engine sink eval.

Runs under the project .venv (python3.11) where Earnings-22 parquet reading + the LLM
entity extractor live. Selects N entity-bearing target utterances (1-8s) plus a filler
pool, and for each target builds ONE long audio buffer = [age_s of filler] ++ [target],
so the target's relevant words sit age_s*12.5 audio tokens after the pinned CTX. Dumps
everything (audio arrays, refs, entities, the CTX system prompt, and the CTX token length)
to an npz so the vLLM driver (e8_sink_biasing.py, python3.10) can load it without any
transformers/anthropic/parquet deps.

The CTX token length (max over targets, block-rounded) is written to ctx_tokens.txt so the
driver can set METRONOME_SINK_TOKENS = S >= max CTX length: the sink then covers exactly the
CTX block and the middle (aged audio) is what gets evicted under the window.

Usage:
    .venv/bin/python -m worker_integration.e8_prep_data \
        --n-utts 40 --age-s 12 --out data/e8_biasing_prep.npz
"""
from __future__ import annotations
import argparse, json, logging, random
from pathlib import Path

import numpy as np

from eval.earnings22_biasing import (
    iter_earnings22, build_system_prompt, decode_audio, DEFAULT_DATA_GLOB,
)

logger = logging.getLogger(__name__)
SR = 16000


def to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(-1)
    if sr != SR:
        n = int(round(len(audio) * SR / sr))
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                          np.arange(len(audio)), audio).astype("float32")
    return audio.astype("float32")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data-glob", default=DEFAULT_DATA_GLOB)
    p.add_argument("--n-utts", type=int, default=40)
    p.add_argument("--age-s", type=float, default=12.0)
    p.add_argument("--tokenizer", default="checkpoints/merged/qwen3-asr-0.6b-endpoint-v9")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/e8_biasing_prep.npz")
    args = p.parse_args()
    rng = random.Random(args.seed)

    from eval.llm_entities import extract_entities_llm
    cand = []
    for u in iter_earnings22(args.data_glob, args.n_utts * 8, shuffle=True, seed=args.seed):
        au = to_16k(*decode_audio(u["audio_bytes"]))
        cand.append({"audio": au, "ref": u["ref"], "dur": len(au) / SR})
        if len(cand) >= args.n_utts * 8:
            break
    ent_lists = extract_entities_llm([c["ref"] for c in cand],
                                     cache_path="data/earnings22/llm_entities.json")
    targets, fillers = [], []
    for c, ents in zip(cand, ent_lists):
        if ents and 1.0 <= c["dur"] <= 8.0 and len(targets) < args.n_utts:
            targets.append({"audio": c["audio"], "ref": c["ref"], "entities": ents})
        elif 1.0 <= c["dur"] <= 6.0 and len(fillers) < 80:
            fillers.append(c["audio"])
    logger.info("E8-prep: %d targets, %d fillers", len(targets), len(fillers))

    def prior_audio(age_s: float) -> np.ndarray:
        if age_s <= 0 or not fillers:
            return np.zeros(0, dtype="float32")
        buf, tot = [], 0.0
        while tot < age_s:
            f = fillers[rng.randrange(len(fillers))]
            buf.append(f); tot += len(f) / SR
        a = np.concatenate(buf)
        return a[-int(age_s * SR):]

    # Tokenizer to measure CTX system-block length (so the driver can size the sink S).
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    audios, refs, ents_json, ctx_prompts = [], [], [], []
    max_ctx_tok = 0
    for t in targets:
        full = np.concatenate([prior_audio(args.age_s), t["audio"]]).astype("float32")
        ctx = build_system_prompt(t["entities"])  # the hotword CTX system prompt
        # CTX block = "<|im_start|>system\n{ctx}<|im_end|>\n" — measure its token length
        sys_block = f"<|im_start|>system\n{ctx}<|im_end|>\n"
        n_ctx = len(tok(sys_block, add_special_tokens=False)["input_ids"])
        max_ctx_tok = max(max_ctx_tok, n_ctx)
        audios.append(full); refs.append(t["ref"])
        ents_json.append(json.dumps(t["entities"])); ctx_prompts.append(ctx)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out,
             audios=np.array(audios, dtype=object),
             refs=np.array(refs, dtype=object),
             entities=np.array(ents_json, dtype=object),
             ctx_prompts=np.array(ctx_prompts, dtype=object),
             age_s=args.age_s, sr=SR, max_ctx_tokens=max_ctx_tok)
    # Sink S must cover the CTX block; round up to a block (16) and add margin.
    S = ((max_ctx_tok + 32) // 16 + 1) * 16
    Path(args.out).with_name("e8_ctx_tokens.txt").write_text(str(S))
    logger.info("E8-prep saved %s: %d targets, max_ctx_tokens=%d -> sink S=%d",
                args.out, len(audios), max_ctx_tok, S)
    print("E8_PREP_DONE")


if __name__ == "__main__":
    main()
