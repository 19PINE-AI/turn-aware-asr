"""Compare streaming-mode (window=1s) vs offline-mode (window=8s) WER.

For Phase 2 gate per research/00-synthesis.md: streaming WER ≤ Phase 1
offline WER + 1 pp absolute (matches Qwen3-ASR-0.6B's 0.92 pp gap).

Usage:
    python -m eval.run_streaming \
        --checkpoint checkpoints/phase2/phase2_step10000.pt \
        --data data/librispeech/test-clean/data.pt
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.train_phase0 import (
    Phase0Model, TrainConfig, AUDIO_START, AUDIO_END, AUDIO_PAD,
)
from src.train_phase2 import Phase2Model, windowed_causal_mask, AUDIO_FRAME_HZ
from eval.metrics import wer

logger = logging.getLogger(__name__)


@torch.no_grad()
def transcribe_with_window(
    model: Phase2Model,
    mel: torch.Tensor,
    window_seconds: float,
    max_new: int = 200,
) -> str:
    """Greedy decode with a fixed attention window applied to the prefill pass.

    The decode steps after prefill use standard causal cross-attention against
    the prefilled KV cache, so the window only constrains the *prefill*. This
    matches Qwen3-ASR streaming behavior — the window limits how far back the
    model looks during encoding, not how it autoregresses afterward.
    """
    device = next(model.parameters()).device
    aut_out_orig = model.aut(mel.unsqueeze(0).to(device=device, dtype=torch.bfloat16))
    audio_embeds = model.adapter(aut_out_orig)[0]
    T_aud = audio_embeds.shape[0]

    tok = model.tokenizer
    prefix_ids = [model.audio_start_id] + [model.audio_pad_id] * T_aud + [model.audio_end_id]
    input_ids = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)

    embeds = model.llm.get_input_embeddings()(input_ids)
    pad_positions = (input_ids[0] == model.audio_pad_id).nonzero(as_tuple=True)[0]
    embeds[0, pad_positions] = audio_embeds.to(embeds.dtype)

    # Prefill with windowed mask
    window_frames = max(1, int(window_seconds * AUDIO_FRAME_HZ))
    L = embeds.shape[1]
    mask = windowed_causal_mask(L, window_frames, device, embeds.dtype)
    out = model.llm(inputs_embeds=embeds, attention_mask=mask, use_cache=True)
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
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--max-utterances", type=int, default=200)
    p.add_argument("--out", default="eval/results/streaming.json")
    p.add_argument("--windows", default="1,4,8",
                    help="Comma-separated list of windows in seconds")
    args = p.parse_args()

    cfg = TrainConfig(arm="aut_frozen", unfreeze_aut_proj=True)
    tok = AutoTokenizer.from_pretrained(cfg.qwen3_path)
    tok.add_special_tokens(
        {"additional_special_tokens": [AUDIO_START, AUDIO_END, AUDIO_PAD]}
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    logger.info("Building model…")
    model = Phase2Model(cfg, tok).cuda().eval()
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    model.load_state_dict(ckpt["trainable_state"], strict=False)
    logger.info("Loaded %s", args.checkpoint)

    data = torch.load(args.data, weights_only=False)
    mels = data["mel"][: args.max_utterances]
    texts = data["text"][: args.max_utterances]
    logger.info("Eval over %d utterances", len(mels))

    results = {}
    for w_str in args.windows.split(","):
        w = float(w_str)
        logger.info("Window = %.1f s", w)
        hyps = []
        for mel in tqdm(mels, desc=f"win {w}"):
            hyps.append(transcribe_with_window(model, mel, w))
        wer_val = wer(" \n".join(hyps), " \n".join(texts))
        results[f"window_{w}s"] = {"wer": wer_val * 100, "n": len(mels)}
        logger.info("  WER @ %.1f s: %.2f %%", w, wer_val * 100)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "data": args.data,
        "results": results,
    }, indent=2))
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
