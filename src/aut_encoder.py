"""AuT audio encoder — minimal standalone implementation.

Built to load `Qwen/Qwen3-ASR-0.6B`'s `thinker.audio_tower.*` weights directly,
without depending on transformers' (possibly unreleased) Qwen3-ASR class.

Architecture (confirmed by inspecting safetensors shapes):
  Conv stem:
    conv2d1: Conv2D(1→480, k=3, s=(2,2))     # mel ch 1→480, time/freq /2
    conv2d2: Conv2D(480→480, k=3, s=(2,2))   # /2
    conv2d3: Conv2D(480→480, k=3, s=(2,2))   # /2  — total 8× downsample
    conv_out: Linear(480*16 → 896)           # 128 mel / 8 = 16, project to d_model

  Sinusoidal position encoding (Whisper-style).

  18 × pre-LN transformer blocks (d=896, h=14, FFN=3584, GELU):
    x → ln1 → MHA → +x → ln2 → fc1 → GELU → fc2 → +x
    (HF naming: self_attn_layer_norm, self_attn, final_layer_norm, fc1, fc2)

  Output:
    ln_post (LayerNorm 896)
    proj1: Linear(896 → 896) with GELU
    proj2: Linear(896 → 1024)    # this is the AuT-to-LLM projector

The encoder uses **non-causal windowed attention** with n_window=50 frames
(= 4 s at 12.5 Hz post-encoder). This means during streaming we re-encode the
last 4 s of audio at every tick — the published Qwen3-ASR streaming protocol.
"""

from __future__ import annotations
from pathlib import Path
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors import safe_open

logger = logging.getLogger(__name__)


D_MODEL = 896
N_HEADS = 14
N_LAYERS = 18
FFN_DIM = 3584
CONV_HIDDEN = 480
MEL_BINS = 128
OUT_DIM = 1024            # AuT projector output dim — matches Qwen3-0.6B-Base hidden
MAX_SOURCE_POS = 1500     # pre-encoder positions (config: 100 Hz × 15 s)
N_WINDOW = 50             # post-encoder window in frames (= 4 s)


def sinusoidal_pe(max_pos: int, d_model: int) -> torch.Tensor:
    """Whisper-style sinusoidal positional embedding."""
    half = d_model // 2
    inv_freq = torch.exp(-math.log(10000.0) * torch.arange(half).float() / (half - 1))
    pos = torch.arange(max_pos).float().unsqueeze(1)
    angles = pos * inv_freq.unsqueeze(0)
    pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return pe  # (max_pos, d_model)


class AuTAttention(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class AuTBlock(nn.Module):
    """Pre-LN block matching HF Whisper / Qwen2-audio encoder naming."""
    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS, ffn_dim: int = FFN_DIM):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = AuTAttention(d_model, n_heads)
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.self_attn_layer_norm(x)
        x = x + self.self_attn(h, attn_mask=attn_mask)
        h = self.final_layer_norm(x)
        x = x + self.fc2(F.gelu(self.fc1(h)))
        return x


def _window_mask(T: int, w: int, device: torch.device) -> torch.Tensor:
    """Non-causal banded attention mask: each query attends to ±w frames.

    Returns a (T, T) bool mask where True = attend.
    """
    idx = torch.arange(T, device=device)
    return (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= w


class AuTEncoder(nn.Module):
    """Standalone AuT encoder loadable from Qwen3-ASR-0.6B safetensors.

    Forward expects (B, T_mel, 128) log-mel features at 100 Hz. Output:
    (B, T_out, 1024) continuous embeddings at 12.5 Hz.
    """
    def __init__(
        self,
        d_model: int = D_MODEL,
        n_layers: int = N_LAYERS,
        n_heads: int = N_HEADS,
        ffn_dim: int = FFN_DIM,
        conv_hidden: int = CONV_HIDDEN,
        mel_bins: int = MEL_BINS,
        out_dim: int = OUT_DIM,
        max_source_pos: int = MAX_SOURCE_POS,
        window: int = N_WINDOW,
    ):
        super().__init__()
        # Conv stem: input is treated as (B, 1, T_mel, mel_bins). Stride 2 in both
        # axes for each conv2d; total 8× downsample in time and freq.
        self.conv2d1 = nn.Conv2d(1, conv_hidden, kernel_size=3, stride=2, padding=1)
        self.conv2d2 = nn.Conv2d(conv_hidden, conv_hidden, kernel_size=3, stride=2, padding=1)
        self.conv2d3 = nn.Conv2d(conv_hidden, conv_hidden, kernel_size=3, stride=2, padding=1)
        # After 3 stride-2 convs on 128 mel, freq dim is 128 / 8 = 16. Channel * freq = 480 * 16 = 7680.
        self.conv_out = nn.Linear(conv_hidden * (mel_bins // 8), d_model, bias=False)

        self.layers = nn.ModuleList(
            [AuTBlock(d_model, n_heads, ffn_dim) for _ in range(n_layers)]
        )
        self.ln_post = nn.LayerNorm(d_model)

        # Built-in projector to LLM hidden — keep these as part of the encoder
        # because they're trained with AuT. Downstream `AudioProjector` (in
        # src/projector.py) is a separate random-init projector for the
        # ablation arm that wants to train its own.
        self.proj1 = nn.Linear(d_model, d_model, bias=True)
        self.proj2 = nn.Linear(d_model, out_dim, bias=True)

        # Sinusoidal PE buffer — generously sized. Computed at module init
        # for the worst case (~4 min of audio at 12.5 Hz = 3000 frames). PE
        # is purely a function of position so we can also recompute on the
        # fly for unusually long inputs (see forward()).
        pe = sinusoidal_pe(3000, d_model)
        self.register_buffer("positional_embedding", pe, persistent=False)

        self.window = window

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, T_mel, n_mels=128). Returns (B, T_out, OUT_DIM) at 12.5 Hz."""
        B, T_mel, M = mel.shape
        assert M == MEL_BINS, f"expected {MEL_BINS} mel bins, got {M}"
        # (B, 1, T_mel, n_mels) — convolve over (time, freq)
        x = mel.unsqueeze(1)
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))
        # x is now (B, 480, T/8, n_mels/8)
        B_, C, T_out, F_out = x.shape
        # Reorder to (B, T_out, C*F_out) then linear to d_model
        x = x.permute(0, 2, 1, 3).reshape(B_, T_out, C * F_out)
        x = self.conv_out(x)  # (B, T_out, d_model)

        # Add sinusoidal PE (recompute if longer than the precomputed buffer)
        T = x.shape[1]
        if T <= self.positional_embedding.shape[0]:
            pe = self.positional_embedding[:T]
        else:
            pe = sinusoidal_pe(T, x.shape[-1]).to(device=x.device, dtype=x.dtype)
        x = x + pe.unsqueeze(0).to(dtype=x.dtype)

        # Windowed non-causal attention
        attn_mask = _window_mask(x.shape[1], self.window, x.device)

        for blk in self.layers:
            x = blk(x, attn_mask=attn_mask)

        x = self.ln_post(x)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        return x  # (B, T_out, 1024)


def load_aut_from_safetensors(
    safetensors_path: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> AuTEncoder:
    """Build an AuTEncoder and load weights from a Qwen3-ASR safetensors file.

    Strips the ``thinker.audio_tower.`` prefix and loads everything else.
    """
    model = AuTEncoder()
    state = {}
    with safe_open(str(safetensors_path), framework="pt") as f:
        for key in f.keys():
            if key.startswith("thinker.audio_tower."):
                short = key[len("thinker.audio_tower."):]
                state[short] = f.get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Missing keys (will be random-init): %s", missing[:5])
    if unexpected:
        logger.warning("Unexpected keys (will be ignored): %s", unexpected[:5])
    model = model.to(device=device, dtype=dtype).eval()
    return model


def freeze_aut(aut: AuTEncoder, unfreeze_top_n_layers: int = 0) -> AuTEncoder:
    """Freeze AuT; optionally unfreeze the top N transformer blocks.

    proj1/proj2 are kept frozen unless explicitly unfrozen — they're trained
    for the original Qwen3-ASR LLM and serve as the bake-off baseline.
    """
    for p in aut.parameters():
        p.requires_grad = False
    if unfreeze_top_n_layers > 0:
        for blk in aut.layers[-unfreeze_top_n_layers:]:
            for p in blk.parameters():
                p.requires_grad = True
    return aut


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="data/qwen3-asr-0.6b/model.safetensors")
    args = p.parse_args()
    aut = load_aut_from_safetensors(args.source)
    print("Loaded AuT, params:", sum(p.numel() for p in aut.parameters()) / 1e6, "M")
