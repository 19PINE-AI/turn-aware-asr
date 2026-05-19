"""Phase 0 sanity training — single-arm.

Trains Qwen3-0.6B-Base + AuT (frozen by default) on a small LibriSpeech
subset using the simple offline-mode format:

    <|audio_start|> [audio_pad × N] <|audio_end|> <transcript> <|endoftext|>

Loss is CE on the transcript tokens only (audio positions get embeddings
substituted with AuT output and contribute zero loss). This is the
Phase 0 smoke test described in research/09-phase0-engineering-plan.md.

Usage:
    python -m src.train_phase0 --arm aut_frozen --steps 50 --bsz 2
"""

from __future__ import annotations
import argparse
import logging
import math
import time
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

from .aut_encoder import load_aut_from_safetensors, freeze_aut
from .features import log_mel
from .projector import AudioProjector

logger = logging.getLogger(__name__)


AUDIO_START = "<|audio_start|>"
AUDIO_END = "<|audio_end|>"
AUDIO_PAD = "<|audio_pad|>"


@dataclass
class TrainConfig:
    arm: str = "aut_frozen"   # aut_frozen, aut_unfrozen_top6, mimi_frozen (TODO)
    qwen3_path: str = "Qwen/Qwen3-0.6B-Base"
    aut_path: str = "data/qwen3-asr-0.6b/model.safetensors"
    data_path: str = "data/librispeech/test-clean/data.pt"   # tiny smoke set
    steps: int = 50
    bsz: int = 2
    lr: float = 1e-4
    warmup_steps: int = 10
    grad_clip: float = 1.0
    seed: int = 0
    freeze_llm: bool = False  # Phase 0 minimal arm: only the adapter trains
    log_every: int = 5
    checkpoint_dir: str = "checkpoints/phase0-smoke"


class Phase0Model(nn.Module):
    """AuT + Qwen3 LLM, with `<|audio_pad|>` substitution."""

    def __init__(self, cfg: TrainConfig, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer

        logger.info("Loading AuT…")
        self.aut = load_aut_from_safetensors(cfg.aut_path, device="cpu", dtype=torch.bfloat16)
        if cfg.arm == "aut_frozen":
            freeze_aut(self.aut, unfreeze_top_n_layers=0)
        elif cfg.arm == "aut_unfrozen_top6":
            freeze_aut(self.aut, unfreeze_top_n_layers=6)

        # Domain adapter: AuT's proj2 was trained for Qwen3-Omni's LM, not
        # Qwen3-0.6B-Base. This 2-MLP+GELU projector maps the AuT output
        # distribution into Qwen3-Base's expected input distribution. Tiny
        # (~2 M params) and trainable in all arms — see synthesis 00 §3.2.
        self.adapter = AudioProjector(d_audio=1024, d_llm=1024).to(torch.bfloat16)

        logger.info("Loading Qwen3-0.6B-Base…")
        self.llm = AutoModelForCausalLM.from_pretrained(
            cfg.qwen3_path, dtype=torch.bfloat16
        )
        # Resize embeddings for the new audio control tokens. The new rows
        # (audio_start, audio_end, audio_pad) get random-init values; init
        # them as the mean of existing embeddings + σ=0.02 noise to avoid
        # the LM seeing wild OOV embeddings at step 0.
        prev_vocab_size = self.llm.get_input_embeddings().weight.shape[0]
        self.llm.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            inp = self.llm.get_input_embeddings().weight
            mean_emb = inp[:prev_vocab_size].mean(dim=0)
            for i in range(prev_vocab_size, inp.shape[0]):
                inp[i] = mean_emb + torch.randn_like(mean_emb) * 0.02
        if cfg.freeze_llm:
            # Hard-freeze everything in the LLM, including the resized
            # embedding matrix. The audio_pad embedding doesn't matter
            # (it's overwritten by the adapter output in forward); the
            # audio_start/end embeddings are now just well-init constants.
            for p in self.llm.parameters():
                p.requires_grad = False
        # Audio_pad ID after tokenizer extension
        self.audio_pad_id = tokenizer.convert_tokens_to_ids(AUDIO_PAD)
        self.audio_start_id = tokenizer.convert_tokens_to_ids(AUDIO_START)
        self.audio_end_id = tokenizer.convert_tokens_to_ids(AUDIO_END)

    def trainable_params(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def build_input(self, mel: torch.Tensor, text: str, device: torch.device):
        """Return (input_ids, labels, audio_token_indices, audio_embeds).

        - input_ids: (L,) with audio_pad placeholders for audio frames.
        - labels: (L,) with -100 except on transcript tokens (CE applied there).
        - audio_token_indices: positions of audio_pad in input_ids.
        - audio_embeds: (T_aud, d_llm) from AuT(mel).
        """
        with torch.no_grad():
            aut_out = self.aut(mel.unsqueeze(0).to(device=device, dtype=torch.bfloat16))
        # Pass AuT output through the trainable domain adapter
        audio_embeds = self.adapter(aut_out)[0]  # (T_aud, 1024)
        T_aud = audio_embeds.shape[0]

        # Build text input: <|audio_start|> [pad×T_aud] <|audio_end|> <transcript> <eos>
        prefix_ids = [self.audio_start_id] + [self.audio_pad_id] * T_aud + [self.audio_end_id]
        text_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        eos_id = self.tokenizer.eos_token_id
        input_ids = prefix_ids + text_ids + [eos_id]
        # Loss only on text + eos
        labels = ([-100] * len(prefix_ids)) + text_ids + [eos_id]
        return (
            torch.tensor(input_ids, dtype=torch.long, device=device),
            torch.tensor(labels, dtype=torch.long, device=device),
            audio_embeds,
        )

    def forward(self, batch_inputs, batch_labels, batch_audio):
        """All inputs are lists of length B; pad to max length internally."""
        device = batch_inputs[0].device
        max_len = max(x.shape[0] for x in batch_inputs)
        B = len(batch_inputs)
        pad_id = self.tokenizer.pad_token_id or 0
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
        labels = torch.full((B, max_len), -100, dtype=torch.long, device=device)
        attn_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
        for i, (ids, lab) in enumerate(zip(batch_inputs, batch_labels)):
            n = ids.shape[0]
            input_ids[i, :n] = ids
            labels[i, :n] = lab
            attn_mask[i, :n] = 1

        # Get text embeddings, then substitute audio_pad positions with audio output.
        # NOTE: batch_audio entries are already adapter-projected (see build_input).
        token_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, L, d)
        for i in range(B):
            audio_embeds = batch_audio[i].to(token_embeds.dtype)  # (T_aud, d)
            pad_positions = (input_ids[i] == self.audio_pad_id).nonzero(as_tuple=True)[0]
            n_audio = audio_embeds.shape[0]
            assert pad_positions.shape[0] == n_audio, \
                f"audio_pad count {pad_positions.shape[0]} != AuT frames {n_audio}"
            token_embeds[i, pad_positions] = audio_embeds

        out = self.llm(
            inputs_embeds=token_embeds,
            attention_mask=attn_mask,
            labels=labels,
        )
        return out.loss


def cosine_lr(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="aut_frozen")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--bsz", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--data", default="data/librispeech/test-clean/data.pt")
    p.add_argument("--checkpoint-dir", default="checkpoints/phase0-smoke")
    p.add_argument("--freeze-llm", action="store_true",
                    help="Adapter-only training — Phase 0 minimal arm")
    p.add_argument("--log-every", type=int, default=5)
    args = p.parse_args()

    cfg = TrainConfig(
        arm=args.arm, steps=args.steps, bsz=args.bsz, lr=args.lr,
        data_path=args.data, checkpoint_dir=args.checkpoint_dir,
        warmup_steps=args.warmup_steps, freeze_llm=args.freeze_llm,
        log_every=args.log_every,
    )
    torch.manual_seed(cfg.seed)

    logger.info("Loading tokenizer + adding audio special tokens…")
    tokenizer = AutoTokenizer.from_pretrained(cfg.qwen3_path)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [AUDIO_START, AUDIO_END, AUDIO_PAD]}
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Building model…")
    model = Phase0Model(cfg, tokenizer)
    model = model.cuda()
    n_trainable = sum(p.numel() for p in model.trainable_params())
    n_total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable: %.1f M / %.1f M total", n_trainable / 1e6, n_total / 1e6)

    optim = torch.optim.AdamW(model.trainable_params(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1)

    logger.info("Loading data from %s", cfg.data_path)
    data = torch.load(cfg.data_path, weights_only=False)
    mels: list[torch.Tensor] = data["mel"]
    texts: list[str] = data["text"]
    logger.info("Loaded %d utterances", len(mels))

    # Simple shuffled iterator
    indices = torch.randperm(len(mels)).tolist()
    cursor = 0

    def next_batch(bsz):
        nonlocal cursor
        batch_idx = indices[cursor : cursor + bsz]
        if len(batch_idx) < bsz:
            batch_idx = batch_idx + indices[: bsz - len(batch_idx)]
        cursor = (cursor + bsz) % len(indices)
        return [(mels[i], texts[i]) for i in batch_idx]

    # Train loop
    logger.info("Starting training: %d steps × bsz %d", cfg.steps, cfg.bsz)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    for step in range(cfg.steps):
        batch = next_batch(cfg.bsz)
        batch_inputs = []
        batch_labels = []
        batch_audio = []
        for mel, text in batch:
            ids, lab, aud = model.build_input(mel, text, device="cuda")
            batch_inputs.append(ids)
            batch_labels.append(lab)
            batch_audio.append(aud)

        lr = cosine_lr(step, cfg.lr, cfg.warmup_steps, cfg.steps)
        for g in optim.param_groups:
            g["lr"] = lr

        t0 = time.perf_counter()
        loss = model(batch_inputs, batch_labels, batch_audio)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_params(), cfg.grad_clip)
        optim.step()
        optim.zero_grad()
        dt = time.perf_counter() - t0
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            logger.info("step %4d  loss %.4f  lr %.2e  %.2f s/step  free %.0f MB",
                         step, loss.item(), lr, dt,
                         torch.cuda.mem_get_info()[0] / 1e6)

    # Save checkpoint — only the parameters that were actually trained,
    # plus the projector and endpoint head (small, useful for resume).
    ckpt_path = Path(cfg.checkpoint_dir) / f"{cfg.arm}_step{cfg.steps}.pt"
    trainable_names = {
        n for n, p in model.named_parameters() if p.requires_grad
    }
    trainable_state = {
        n: p.detach().cpu()
        for n, p in model.named_parameters()
        if n in trainable_names
    }
    torch.save(
        {
            "step": cfg.steps,
            "trainable_state": trainable_state,
            "cfg": cfg.__dict__,
            "tokenizer_vocab_size": len(tokenizer),
        },
        ckpt_path,
    )
    logger.info("Saved checkpoint to %s (%d tensors)", ckpt_path, len(trainable_state))


if __name__ == "__main__":
    main()
