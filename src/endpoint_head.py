"""Decoupled 0-delay acoustic endpoint head.

Reads from the audio-stream hidden states at zero delay (i.e. it sees only
audio that has actually arrived). Emits two probabilities per frame:

  - p_eager: confidence that the speaker has semantically completed
    even before silence is fully established. Maps to ``<EAGER_END_SPEECH>``.
  - p_end:   confidence that the speaker has finished and a downstream
    flush is now warranted. Maps to ``<END_SPEECH>``.

The head is intentionally small (~5 M params at default config) so it adds
negligible compute per tick and can run alongside the transcription head
without forcing the transcription head into 0-delay mode.

Why decoupled? The transcription head operates with a 0.5–8 s attention
window for accuracy. The endpoint head must fire fast — within 200 ms of
semantic completion — which is geometrically impossible if it inherits
that window. Running them separately is the only way to hit both targets.
See research/06-architecture-review.md Issue 2.
"""

import torch
import torch.nn as nn


class EndpointHead(nn.Module):
    def __init__(self, d_audio: int, hidden: int = 256, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = d_audio
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden
        self.body = nn.Sequential(*layers)
        # Two independent logits per frame: eager + final.
        self.logit_eager = nn.Linear(hidden, 1)
        self.logit_end = nn.Linear(hidden, 1)

    def forward(self, audio_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """audio_hidden: (B, T, d_audio) — frames at 12.5 Hz.

        Returns (eager_logits, end_logits), each (B, T, 1).
        """
        h = self.body(audio_hidden)
        return self.logit_eager(h), self.logit_end(h)

    @staticmethod
    def loss(
        eager_logits: torch.Tensor,
        end_logits: torch.Tensor,
        eager_targets: torch.Tensor,
        end_targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """BCE-with-logits, both heads, optionally masked.

        Targets are forced-aligned ground truth:
          eager_targets = 1 within ±2 frames of last-spoken-word boundary
          end_targets   = 1 within ±2 frames of utterance end (post-pause)
        See research/07-eval-harness-spec.md for the timing window.
        """
        bce = nn.functional.binary_cross_entropy_with_logits
        l_eager = bce(eager_logits.squeeze(-1), eager_targets.float(), reduction="none")
        l_end = bce(end_logits.squeeze(-1), end_targets.float(), reduction="none")
        loss = l_eager + l_end
        if mask is not None:
            loss = loss * mask.float()
            return loss.sum() / mask.float().sum().clamp_min(1.0)
        return loss.mean()
