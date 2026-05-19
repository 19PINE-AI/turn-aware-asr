"""Run Qwen3-ASR-0.6B natively: AuT + Qwen3-Omni thinker LM, no fine-tuning.

Goal: reproduce the paper's 2.11 % / 4.55 % WER on LibriSpeech test-clean/other.
If this works, it validates the loader and gives us the working ASR baseline
on which to layer the project's differentiators (endpoint head, context
prefix, streaming window).

Usage:
    python -m eval.run_qwen3asr_native \\
        --data data/librispeech/test-clean/data.pt --max-utterances 200
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.qwen3_asr_loader import load_full_qwen3_asr
from eval.metrics import wer

logger = logging.getLogger(__name__)


AUDIO_START_ID = 151669
AUDIO_PAD_ID = 151676
AUDIO_END_ID = 151670
IM_END_ID = 151645


def load_tokenizer():
    import json as _json
    tok = AutoTokenizer.from_pretrained("data/qwen3-asr-0.6b")
    with open("data/qwen3-asr-0.6b/chat_template.json") as f:
        tok.chat_template = _json.load(f)["chat_template"]
    return tok


def build_prompt_ids(tok, system_text: str = "") -> list[int]:
    msgs = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": [{"type": "audio"}]},
    ]
    out = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    return out["input_ids"]


@torch.no_grad()
def transcribe(
    aut,
    lm,
    tok,
    mel: torch.Tensor,
    system_text: str = "",
    max_new: int = 200,
) -> str:
    """Run AuT → audio_pad substitution → autoregressive decode."""
    device = next(lm.parameters()).device
    dtype = next(lm.parameters()).dtype

    # 1) Encode audio
    aut_out = aut(mel.unsqueeze(0).to(device=device, dtype=dtype))  # (1, T_aud, 1024)
    audio_embeds = aut_out[0]
    T_aud = audio_embeds.shape[0]

    # 2) Build prompt with the single <|audio_pad|>, then expand it to T_aud copies
    prompt_ids = build_prompt_ids(tok, system_text)
    expanded = []
    for tid in prompt_ids:
        if tid == AUDIO_PAD_ID:
            expanded.extend([AUDIO_PAD_ID] * T_aud)
        else:
            expanded.append(tid)
    input_ids = torch.tensor(expanded, dtype=torch.long, device=device).unsqueeze(0)

    # 3) Embedding lookup, then replace audio_pad positions with AuT projections
    embeds = lm.get_input_embeddings()(input_ids)
    pad_positions = (input_ids[0] == AUDIO_PAD_ID).nonzero(as_tuple=True)[0]
    assert pad_positions.shape[0] == T_aud, (pad_positions.shape, T_aud)
    embeds[0, pad_positions] = audio_embeds.to(dtype)

    # 4) First forward pass to populate KV cache
    out = lm(inputs_embeds=embeds, use_cache=True)
    past = out.past_key_values
    next_id = out.logits[0, -1].argmax().item()
    generated = []

    # Qwen3 chat assistant outputs end with <|im_end|>
    for _ in range(max_new):
        generated.append(next_id)
        if next_id == IM_END_ID or next_id == tok.eos_token_id:
            break
        next_embeds = lm.get_input_embeddings()(
            torch.tensor([[next_id]], device=device)
        )
        out = lm(inputs_embeds=next_embeds, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[0, -1].argmax().item()

    # Strip trailing im_end / eos
    if generated and generated[-1] in (IM_END_ID, tok.eos_token_id):
        generated = generated[:-1]
    return tok.decode(generated, skip_special_tokens=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--max-utterances", type=int, default=200)
    p.add_argument("--out", default="eval/results/qwen3asr_native.json")
    p.add_argument("--system", default="", help="Optional system prompt for biasing")
    args = p.parse_args()

    logger.info("Loading tokenizer + model…")
    tok = load_tokenizer()
    aut, lm = load_full_qwen3_asr()

    data = torch.load(args.data, weights_only=False)
    mels = data["mel"][: args.max_utterances]
    texts = data["text"][: args.max_utterances]
    logger.info("Eval over %d utterances", len(mels))

    refs, hyps = [], []
    for i, (mel, text) in enumerate(tqdm(list(zip(mels, texts)), desc="decode")):
        hyp = transcribe(aut, lm, tok, mel, system_text=args.system)
        refs.append(text)
        hyps.append(hyp)
        if i < 3:
            logger.info("REF[%d]: %s", i, text[:90])
            logger.info("HYP[%d]: %s", i, hyp[:90])

    overall_wer = wer(" \n".join(hyps), " \n".join(refs))
    logger.info("WER: %.2f %%", overall_wer * 100)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "n_utterances": len(refs),
        "wer_final": overall_wer * 100,
        "model": "Qwen/Qwen3-ASR-0.6B (native loader)",
        "data": args.data,
        "system": args.system,
    }, indent=2))


if __name__ == "__main__":
    main()
