"""Mel filterbank features matching the Qwen3-ASR AuT input.

Uses HF's WhisperFeatureExtractor verbatim — that's what Qwen3-ASR's
`preprocessor_config.json` specifies:
  feature_extractor_type: WhisperFeatureExtractor
  feature_size: 128, hop_length: 160, n_fft: 400, chunk_length: 30 s

The standard Whisper extractor pads/truncates to a fixed 30 s window and
returns (n_mels=128, n_frames=3000). For streaming we want variable-length;
use ``log_mel`` for that path.
"""

from __future__ import annotations
from functools import lru_cache

import numpy as np
import torch


@lru_cache(maxsize=1)
def _whisper_extractor():
    """Cache the HF WhisperFeatureExtractor instance."""
    from transformers import WhisperFeatureExtractor
    return WhisperFeatureExtractor(
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
    )


def log_mel(
    audio: torch.Tensor | np.ndarray,
    sample_rate: int = 16000,
    pad_to_30s: bool = False,
) -> torch.Tensor:
    """audio: (T,) 16 kHz mono. Returns (T_frames, 128).

    If ``pad_to_30s`` is True, returns the fixed 3000-frame Whisper layout
    (use for matching Qwen3-ASR's published inference). Otherwise returns
    variable-length frames sized to the actual input.
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy().astype(np.float32)
    extractor = _whisper_extractor()
    if pad_to_30s:
        feats = extractor(audio, sampling_rate=sample_rate, return_tensors="pt").input_features
        # (1, 128, 3000) → (3000, 128)
        return feats[0].T
    # Variable length: skip the extractor's padding by computing only up to
    # the actual audio length.
    n_samples = audio.shape[-1]
    feats = extractor(audio, sampling_rate=sample_rate, return_tensors="pt").input_features
    n_frames = n_samples // 160 + 1
    return feats[0, :, :n_frames].T  # (n_frames, 128)
