"""Run an offline WER eval on a prepared LibriSpeech split.

Greedy decode only (no streaming simulator) — Phase 0 sanity / baseline.

Usage:
    python -m eval.run_librispeech \
        --checkpoint checkpoints/phase0-smoke/aut_frozen_step50.pt \
        --data data/librispeech/test-clean/data.pt
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.train_phase0 import Phase0Model, TrainConfig, AUDIO_START, AUDIO_END, AUDIO_PAD
from eval.metrics import wer

logger = logging.getLogger(__name__)


@torch.no_grad()
def transcribe(model: Phase0Model, mel: torch.Tensor, max_new: int = 200) -> str:
    """Greedy-decode a single utterance to text."""
    device = next(model.parameters()).device
    # Run AuT
    aut_out = model.aut(mel.unsqueeze(0).to(device=device, dtype=torch.bfloat16))[0]
    T_aud = aut_out.shape[0]

    tok = model.tokenizer
    prefix_ids = [model.audio_start_id] + [model.audio_pad_id] * T_aud + [model.audio_end_id]
    input_ids = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)

    # Build initial embeddings with audio_pads substituted
    embeds = model.llm.get_input_embeddings()(input_ids)
    pad_positions = (input_ids[0] == model.audio_pad_id).nonzero(as_tuple=True)[0]
    embeds[0, pad_positions] = aut_out.to(embeds.dtype)

    # First forward to populate KV cache
    out = model.llm(inputs_embeds=embeds, use_cache=True)
    past = out.past_key_values
    next_id = out.logits[0, -1].argmax().item()

    generated = [next_id]
    eos = tok.eos_token_id
    for _ in range(max_new):
        if next_id == eos:
            break
        next_embeds = model.llm.get_input_embeddings()(
            torch.tensor([[next_id]], device=device)
        )
        out = model.llm(inputs_embeds=next_embeds, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[0, -1].argmax().item()
        generated.append(next_id)

    if generated and generated[-1] == eos:
        generated = generated[:-1]
    return tok.decode(generated, skip_special_tokens=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None,
                    help="Path to phase0 checkpoint. If None, uses the just-loaded base model.")
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="eval/results/latest.json")
    p.add_argument("--max-utterances", type=int, default=None)
    p.add_argument("--arm", default="aut_frozen")
    args = p.parse_args()

    cfg = TrainConfig(arm=args.arm)

    logger.info("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(cfg.qwen3_path)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [AUDIO_START, AUDIO_END, AUDIO_PAD]}
    )

    logger.info("Building model (arm=%s)…", cfg.arm)
    model = Phase0Model(cfg, tokenizer).cuda().eval()

    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, weights_only=False)
        if "trainable_state" in ckpt and ckpt["trainable_state"]:
            model.load_state_dict(ckpt["trainable_state"], strict=False)
            logger.info("Loaded checkpoint %s (step %d)", args.checkpoint, ckpt["step"])

    logger.info("Loading data %s…", args.data)
    data = torch.load(args.data, weights_only=False)
    mels = data["mel"]
    texts = data["text"]
    if args.max_utterances:
        mels = mels[: args.max_utterances]
        texts = texts[: args.max_utterances]
    logger.info("Eval over %d utterances", len(mels))

    refs, hyps = [], []
    for mel, text in tqdm(list(zip(mels, texts)), desc="decode"):
        hyp = transcribe(model, mel)
        refs.append(text)
        hyps.append(hyp)

    overall_wer = wer(" \n".join(hyps), " \n".join(refs))
    logger.info("WER: %.2f %%", overall_wer * 100)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "n_utterances": len(refs),
        "wer_final": overall_wer * 100,
        "arm": args.arm,
        "checkpoint": args.checkpoint,
        "data": args.data,
    }, indent=2))
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
