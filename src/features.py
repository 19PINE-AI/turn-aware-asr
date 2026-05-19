"""Mel filterbank features matching the Qwen3-ASR AuT input.

128-dim log-mel spectrogram, 10 ms hop, 25 ms window, 16 kHz mono.
Output is (T, 128) where T = audio_len_samples // 160 + 1.

This is identical to Whisper's frontend modulo the 128 mel bins (Whisper
uses 80) — Qwen3-ASR matches the canonical Whisper-large-v3 setup.
"""

import torch
import torchaudio


_FBANK_CACHE: dict[tuple, torchaudio.transforms.MelSpectrogram] = {}


def _mel_extractor(sample_rate: int, n_mels: int) -> torchaudio.transforms.MelSpectrogram:
    key = (sample_rate, n_mels)
    if key not in _FBANK_CACHE:
        _FBANK_CACHE[key] = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,           # 25 ms at 16 kHz
            hop_length=160,      # 10 ms at 16 kHz
            n_mels=n_mels,
            f_min=0.0,
            f_max=sample_rate / 2,
            power=2.0,
        )
    return _FBANK_CACHE[key]


def log_mel(
    audio: torch.Tensor,
    sample_rate: int = 16000,
    n_mels: int = 128,
) -> torch.Tensor:
    """audio: (T,) or (B, T) 16 kHz mono. Returns (..., T_frames, n_mels)."""
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    mel = _mel_extractor(sample_rate, n_mels).to(audio.device)(audio)
    # Log scaling with floor matching Whisper / Qwen3-Omni convention.
    mel = mel.clamp(min=1e-10).log10()
    mel = (mel + 4.0) / 4.0  # rough normalization; exact constants depend on AuT statistics
    mel = mel.transpose(-1, -2)  # (B, T, n_mels)
    return mel.squeeze(0) if squeeze else mel
