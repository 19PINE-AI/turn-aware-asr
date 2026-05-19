"""Audio-to-LLM projector.

2-layer MLP with GELU between layers. Maps AuT encoder output
(d_aud = 1024 for Qwen3-ASR-0.6B AuT) to the LLM hidden size
(d_llm = 1024 for Qwen3-0.6B-Base). Output replaces
``<|audio_pad|>`` embeddings in the LLM input stream.
"""

import torch
import torch.nn as nn


class AudioProjector(nn.Module):
    def __init__(self, d_audio: int, d_llm: int, hidden: int | None = None):
        super().__init__()
        h = hidden if hidden is not None else d_llm
        self.fc1 = nn.Linear(d_audio, h, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h, d_llm, bias=True)

        # Small Gaussian init avoids the 1-2k step loss spike that random init
        # produces when the projector output collides with the existing
        # `<|audio_pad|>` embedding distribution.
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
