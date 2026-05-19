"""Delayed-streams data loader.

Pairs an audio stream with a delayed text stream so the model learns to emit
each transcript token K frames *after* the corresponding audio is heard.
This is implemented at the *data* layout level (not the model), so the
delay can be varied at inference without retraining.

Audio frame rate: 12.5 Hz (one frame per 80 ms) — matches the AuT encoder
output.

Window strategy (Qwen3-ASR §5): each training example samples an attention
window in [1, 8] seconds. This single recipe covers both streaming
(1 s window) and offline (8 s window) modes at inference.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

import math
import random
import torch
from torch.utils.data import Dataset


AUDIO_FRAME_HZ = 12.5
AUDIO_FRAME_MS = 80.0


@dataclass
class DelayedExample:
    audio: torch.Tensor       # (T_samples,) 16 kHz float32
    text_tokens: list[int]    # length L
    word_frame_indices: list[int]  # length L; which audio frame each token aligns to
    has_endpoint: bool        # whether this example contains an utterance end
    endpoint_frame: int | None  # audio-frame index of utterance end, or None


def sample_window_seconds(rng: random.Random) -> float:
    """Sample an attention window uniformly in [1, 8] seconds.

    Distribution choice: paper says randomized 1-8 s; exact distribution is
    not reported. Uniform is the simplest defensible choice; revisit if
    streaming-mode WER ends up far from offline.
    """
    return rng.uniform(1.0, 8.0)


class StreamingASRDataset(Dataset):
    """Yields (audio_samples, text_token_ids_with_delay, endpoint_labels).

    The text stream is delayed by ``delay_frames`` so that token t in the
    text stream corresponds to audio frame t - delay_frames. Tokens beyond
    the audio's final frame are padded with the ``no_speech_id`` control.

    Endpoint labels are emitted as (eager_target, end_target) per audio
    frame, suitable for the EndpointHead BCE loss.
    """

    def __init__(
        self,
        examples: list[DelayedExample],
        no_speech_id: int,
        eager_window_frames: int = 2,   # ±2 frames around last-word boundary
        end_window_frames: int = 2,     # ±2 frames around utterance end
        delay_frames: int = 12,         # 0.96 s at 12.5 Hz; configurable
    ):
        self.examples = examples
        self.no_speech_id = no_speech_id
        self.eager_window = eager_window_frames
        self.end_window = end_window_frames
        self.delay = delay_frames

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        n_audio_frames = math.ceil(ex.audio.shape[0] / 16000 * AUDIO_FRAME_HZ)

        # Build text-stream targets, one entry per audio frame.
        text_targets = [self.no_speech_id] * n_audio_frames
        for tok_id, frame_idx in zip(ex.text_tokens, ex.word_frame_indices):
            delayed = frame_idx + self.delay
            if 0 <= delayed < n_audio_frames:
                text_targets[delayed] = tok_id
            # If delayed past the end, drop it — Phase 2 will train the model
            # to handle this with the eager-end token.

        # Build endpoint targets.
        eager = torch.zeros(n_audio_frames, dtype=torch.float32)
        end = torch.zeros(n_audio_frames, dtype=torch.float32)
        if ex.has_endpoint and ex.endpoint_frame is not None:
            # Eager target fires at the last-word boundary (we approximate as
            # the frame of the final transcript token).
            if ex.word_frame_indices:
                last_word_frame = ex.word_frame_indices[-1]
                lo = max(0, last_word_frame - self.eager_window)
                hi = min(n_audio_frames, last_word_frame + self.eager_window + 1)
                eager[lo:hi] = 1.0
            # End target fires at the explicit utterance end (post-pause).
            ef = ex.endpoint_frame
            lo = max(0, ef - self.end_window)
            hi = min(n_audio_frames, ef + self.end_window + 1)
            end[lo:hi] = 1.0

        return {
            "audio": ex.audio,
            "text_targets": torch.tensor(text_targets, dtype=torch.long),
            "eager_targets": eager,
            "end_targets": end,
            "n_frames": n_audio_frames,
        }


def collate(batch: Iterable[dict], pad_token_id: int) -> dict:
    """Right-pad to the longest sequence in the batch."""
    batch = list(batch)
    max_audio = max(b["audio"].shape[0] for b in batch)
    max_frames = max(b["n_frames"] for b in batch)
    B = len(batch)
    audio_pad = torch.zeros(B, max_audio, dtype=torch.float32)
    text_pad = torch.full((B, max_frames), pad_token_id, dtype=torch.long)
    eager_pad = torch.zeros(B, max_frames, dtype=torch.float32)
    end_pad = torch.zeros(B, max_frames, dtype=torch.float32)
    mask = torch.zeros(B, max_frames, dtype=torch.bool)
    for i, b in enumerate(batch):
        audio_pad[i, : b["audio"].shape[0]] = b["audio"]
        text_pad[i, : b["n_frames"]] = b["text_targets"][: b["n_frames"]]
        eager_pad[i, : b["n_frames"]] = b["eager_targets"][: b["n_frames"]]
        end_pad[i, : b["n_frames"]] = b["end_targets"][: b["n_frames"]]
        mask[i, : b["n_frames"]] = True
    return {
        "audio": audio_pad,
        "text_targets": text_pad,
        "eager_targets": eager_pad,
        "end_targets": end_pad,
        "frame_mask": mask,
    }
