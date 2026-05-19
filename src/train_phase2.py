"""Phase 2 — streaming adaptation via randomized 1–8 s attention window.

Per Qwen3-ASR §5 (and research/10-qwen3-asr-deepdive.md §5): the same model
weights serve both streaming and offline modes by switching the attention
window at inference time. Training samples the window uniformly in [1, 8] s
so the model learns to function across the whole range.

This script is a thin wrapper around train_phase0: same arch, same loss, but
each batch builds a sliding-window causal mask of width ~12.5 × window_sec
frames and passes it to the LLM via the 4-D ``attention_mask`` parameter.

Usage:
    python -m src.train_phase2 \
        --resume-from checkpoints/phase1/aut_frozen_step30000.pt \
        --steps 10000 --bsz 8 --lr 5e-5
"""

from __future__ import annotations
import argparse
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from .train_phase0 import (
    Phase0Model,
    TrainConfig,
    AUDIO_START,
    AUDIO_END,
    AUDIO_PAD,
    cosine_lr,
)

logger = logging.getLogger(__name__)


AUDIO_FRAME_HZ = 12.5


def windowed_causal_mask(
    seq_len: int,
    window_frames: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Return a 4-D (1, 1, L, L) attention mask in HF additive-bias format.

    Positions outside the window get -inf; positions inside get 0. Causal:
    no attention to future tokens.
    """
    idx = torch.arange(seq_len, device=device)
    delta = idx.unsqueeze(0) - idx.unsqueeze(1)  # (L, L), delta[q, k] = k - q
    # Allowed: -window_frames <= delta <= 0 (causal AND within-window backward)
    allowed = (delta <= 0) & (delta >= -window_frames)
    mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
    mask = mask.masked_fill(~allowed, torch.finfo(dtype).min)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)


class Phase2Model(Phase0Model):
    """Same architecture; forward accepts an attention window."""

    def forward_with_window(
        self,
        batch_inputs,
        batch_labels,
        batch_audio,
        window_seconds: float,
    ):
        device = batch_inputs[0].device
        max_len = max(x.shape[0] for x in batch_inputs)
        B = len(batch_inputs)
        pad_id = self.tokenizer.pad_token_id or 0
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
        labels = torch.full((B, max_len), -100, dtype=torch.long, device=device)
        for i, (ids, lab) in enumerate(zip(batch_inputs, batch_labels)):
            n = ids.shape[0]
            input_ids[i, :n] = ids
            labels[i, :n] = lab

        # Build embeds and substitute audio_pad positions
        token_embeds = self.llm.get_input_embeddings()(input_ids)
        for i in range(B):
            audio_embeds = batch_audio[i].to(token_embeds.dtype)
            pad_positions = (input_ids[i] == self.audio_pad_id).nonzero(as_tuple=True)[0]
            n_audio = audio_embeds.shape[0]
            assert pad_positions.shape[0] == n_audio
            token_embeds[i, pad_positions] = audio_embeds

        # Windowed causal mask
        window_frames = max(1, int(window_seconds * AUDIO_FRAME_HZ))
        mask_4d = windowed_causal_mask(max_len, window_frames, device, token_embeds.dtype)

        out = self.llm(
            inputs_embeds=token_embeds,
            attention_mask=mask_4d.expand(B, -1, -1, -1),
            labels=labels,
        )
        return out.loss


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--resume-from", required=True, help="Phase 1 checkpoint")
    p.add_argument("--data", default="data/librispeech/train-clean-100/data.pt")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--bsz", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=2500)
    p.add_argument("--checkpoint-dir", default="checkpoints/phase2")
    p.add_argument("--window-min", type=float, default=1.0)
    p.add_argument("--window-max", type=float, default=8.0)
    args = p.parse_args()

    cfg = TrainConfig(
        arm="aut_frozen", steps=args.steps, bsz=args.bsz, lr=args.lr,
        data_path=args.data, checkpoint_dir=args.checkpoint_dir,
        warmup_steps=args.warmup_steps, unfreeze_aut_proj=True,
        log_every=args.log_every, save_every=args.save_every,
    )
    torch.manual_seed(0)

    logger.info("Loading tokenizer…")
    tok = AutoTokenizer.from_pretrained(cfg.qwen3_path)
    tok.add_special_tokens(
        {"additional_special_tokens": [AUDIO_START, AUDIO_END, AUDIO_PAD]}
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    logger.info("Building model…")
    model = Phase2Model(cfg, tok).cuda()

    logger.info("Resuming from %s", args.resume_from)
    ckpt = torch.load(args.resume_from, weights_only=False, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["trainable_state"], strict=False)
    logger.info("Resumed (missing %d, unexpected %d)", len(missing), len(unexpected))

    optim = torch.optim.AdamW(
        model.trainable_params(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    logger.info("Loading data…")
    data = torch.load(cfg.data_path, weights_only=False)
    mels: list = data["mel"]
    texts: list = data["text"]
    logger.info("Loaded %d utterances", len(mels))

    indices = torch.randperm(len(mels)).tolist()
    cursor = 0
    rng = random.Random(0)

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Phase 2: %d steps × bsz %d, window ∈ [%g, %g] s",
                cfg.steps, cfg.bsz, args.window_min, args.window_max)

    for step in range(cfg.steps):
        batch_idx = indices[cursor : cursor + cfg.bsz]
        if len(batch_idx) < cfg.bsz:
            batch_idx = batch_idx + indices[: cfg.bsz - len(batch_idx)]
        cursor = (cursor + cfg.bsz) % len(indices)

        batch_inputs, batch_labels, batch_audio = [], [], []
        for i in batch_idx:
            ids, lab, aud = model.build_input(mels[i], texts[i], device="cuda")
            batch_inputs.append(ids)
            batch_labels.append(lab)
            batch_audio.append(aud)

        window_sec = rng.uniform(args.window_min, args.window_max)

        lr = cosine_lr(step, cfg.lr, cfg.warmup_steps, cfg.steps)
        for g in optim.param_groups:
            g["lr"] = lr

        t0 = time.perf_counter()
        loss = model.forward_with_window(batch_inputs, batch_labels, batch_audio, window_sec)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_params(), cfg.grad_clip)
        optim.step()
        optim.zero_grad()
        dt = time.perf_counter() - t0

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            logger.info("step %4d  loss %.4f  win %.2fs  lr %.2e  %.2f s/step",
                         step, loss.item(), window_sec, lr, dt)
            if not torch.isfinite(loss) or (loss.item() > 20 and step > cfg.warmup_steps + 100):
                logger.error("loss diverged or NaN: %s", loss.item())
                break

        if step > 0 and step % cfg.save_every == 0:
            ckpt_path = Path(cfg.checkpoint_dir) / f"phase2_step{step}.pt"
            trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
            trainable_state = {
                n: p.detach().cpu()
                for n, p in model.named_parameters()
                if n in trainable_names
            }
            torch.save(
                {"step": step, "trainable_state": trainable_state,
                 "cfg": cfg.__dict__, "tokenizer_vocab_size": len(tok)},
                ckpt_path,
            )
            logger.info("Periodic checkpoint: %s", ckpt_path)

    # Final save
    ckpt_path = Path(cfg.checkpoint_dir) / f"phase2_step{cfg.steps}.pt"
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    trainable_state = {
        n: p.detach().cpu()
        for n, p in model.named_parameters()
        if n in trainable_names
    }
    torch.save(
        {"step": cfg.steps, "trainable_state": trainable_state,
         "cfg": cfg.__dict__, "tokenizer_vocab_size": len(tok)},
        ckpt_path,
    )
    logger.info("Final checkpoint: %s", ckpt_path)


if __name__ == "__main__":
    main()
