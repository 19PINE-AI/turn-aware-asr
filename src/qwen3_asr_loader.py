"""Load the complete Qwen3-ASR-0.6B (AuT + Qwen3-Omni thinker LM) directly.

Strategy: the safetensors at `Qwen/Qwen3-ASR-0.6B/model.safetensors` contains
both `thinker.audio_tower.*` (AuT) and `thinker.model.*` + `thinker.lm_head.*`
(a Qwen3 LM, identical architecture to Qwen3-0.6B-Base except untied lm_head).

We load:
  - AuT via src.aut_encoder.load_aut_from_safetensors
  - LM via transformers.Qwen3ForCausalLM with a hand-built config matching the
    safetensors shapes, then load_state_dict from the safetensors

This is the "full Qwen3-ASR baseline" — the model that the paper reports
hits 2.11 % / 4.55 % on LibriSpeech test-clean/other. Using it directly
removes the Qwen3-Omni → Qwen3-0.6B-Base domain shift that was Phase 1's
failure mode (see research/13-phase1-failure-analysis.md).
"""

from __future__ import annotations
import logging
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open

from .aut_encoder import AuTEncoder, load_aut_from_safetensors

logger = logging.getLogger(__name__)


def load_qwen3_asr_thinker_lm(
    safetensors_path: str | Path = "data/qwen3-asr-0.6b/model.safetensors",
    dtype: torch.dtype = torch.bfloat16,
):
    """Build a Qwen3 LM and load thinker.* weights from the safetensors.

    Returns an instance of `transformers.Qwen3ForCausalLM` with weights loaded
    from the safetensors. Use it as a normal HF causal LM.
    """
    from transformers import Qwen3Config, Qwen3ForCausalLM

    # Config matches the safetensors shapes (verified by inspection):
    cfg = Qwen3Config(
        vocab_size=151_936,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=131072,
        rope_theta=1_000_000,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        hidden_act="silu",
    )

    logger.info("Building Qwen3 LM with config: %d layers, hidden=%d, vocab=%d",
                cfg.num_hidden_layers, cfg.hidden_size, cfg.vocab_size)
    lm = Qwen3ForCausalLM(cfg)

    # Build state dict from safetensors with prefix stripping.
    state = {}
    with safe_open(str(safetensors_path), framework="pt") as f:
        keys = list(f.keys())
        for key in keys:
            if key.startswith("thinker.model.") or key.startswith("thinker.lm_head."):
                short = key[len("thinker."):]
                state[short] = f.get_tensor(key)

    missing, unexpected = lm.load_state_dict(state, strict=False)
    logger.info("LM load: %d missing, %d unexpected", len(missing), len(unexpected))
    if missing:
        logger.warning("Missing (first 5): %s", missing[:5])
    if unexpected:
        logger.warning("Unexpected (first 5): %s", unexpected[:5])

    lm = lm.to(dtype=dtype).eval()
    return lm


def load_full_qwen3_asr(
    safetensors_path: str | Path = "data/qwen3-asr-0.6b/model.safetensors",
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load both AuT and the LM from the same Qwen3-ASR-0.6B safetensors."""
    logger.info("Loading AuT…")
    aut = load_aut_from_safetensors(safetensors_path, device="cpu", dtype=dtype)
    logger.info("Loading thinker LM…")
    lm = load_qwen3_asr_thinker_lm(safetensors_path, dtype=dtype)
    aut = aut.to(device=device, dtype=dtype)
    lm = lm.to(device=device, dtype=dtype)
    logger.info("Full Qwen3-ASR loaded: AuT %.1f M + LM %.1f M",
                sum(p.numel() for p in aut.parameters()) / 1e6,
                sum(p.numel() for p in lm.parameters()) / 1e6)
    return aut, lm


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    aut, lm = load_full_qwen3_asr()
    print("OK")
