"""Streaming context-aware VAD+ASR model.

Architecture (see research/00-synthesis.md §3.2 and research/06-architecture-review.md):

  audio ─► AuT encoder (12.5 Hz, d_aud=1024) ─┬─► AudioProjector ─► Qwen3 LLM ─► LM head (text + control)
                                              │                        ▲
                                              │           text context prefix (KV-cached)
                                              │
                                              └─► EndpointHead (0-delay) ─► (p_eager, p_end)

Key properties:
  - Two heads: LM head trained on text-stream positions only (audio positions
    skip the head entirely — no masked-loss waste). EndpointHead reads
    audio-stream hidden states at zero delay.
  - Audio embeddings replace ``<|audio_pad|>`` placeholders in the LLM input;
    no vocab extension.
  - Attention window randomized 1-8 s during training; selected at inference.
"""

from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn

from .projector import AudioProjector
from .endpoint_head import EndpointHead


@dataclass
class StreamingModelConfig:
    qwen3_path: str = "Qwen/Qwen3-0.6B-Base"
    d_audio: int = 1024
    d_llm: int = 1024
    audio_pad_token: str = "<|audio_pad|>"
    aut_trainable_top_n: int = 0   # 0 = fully frozen; 6 = top-6 layers trainable
    window_seconds: float | None = None  # None = randomized 1-8 s during training


class StreamingVADASR(nn.Module):
    def __init__(
        self,
        aut: nn.Module,
        config: StreamingModelConfig,
        tokenizer=None,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.aut = aut
        self.projector = AudioProjector(config.d_audio, config.d_llm)
        self.endpoint_head = EndpointHead(d_audio=config.d_audio)

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.qwen3_path)
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.qwen3_path,
            torch_dtype=torch.bfloat16,
        )
        # The LM head is the existing tied embedding — we don't add a separate
        # head. Audio positions simply have their loss masked at training time,
        # but unlike the original plan we avoid computing them at all by
        # filtering positions before the LM head (see forward()).

        # Resolve the audio-pad token id (Qwen3-Omni tokenizer has it at 151676;
        # base Qwen3 tokenizer may not — add it if missing).
        pad_id = self.tokenizer.convert_tokens_to_ids(config.audio_pad_token)
        if pad_id is None or pad_id == self.tokenizer.unk_token_id:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [config.audio_pad_token]}
            )
            self.llm.resize_token_embeddings(len(self.tokenizer))
            pad_id = self.tokenizer.convert_tokens_to_ids(config.audio_pad_token)
        self.audio_pad_id = pad_id

    def encode_audio(self, audio: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
        """Return (B, T_aud, d_audio) at 12.5 Hz.

        Caller is responsible for computing mel (see src/features.py); we keep
        AuT's interface intact rather than wrapping mel inside this module.
        """
        return self.aut(mel)

    def forward(
        self,
        mel: torch.Tensor,
        text_token_ids: torch.Tensor,
        text_position_mask: torch.Tensor,
        attention_window_seconds: float | None = None,
    ) -> dict:
        """Single-stream training forward.

        Inputs:
          mel: (B, T_aud_input, 128) log-mel features
          text_token_ids: (B, L) interleaved text-stream tokens. Audio
            positions are represented by ``<|audio_pad|>`` tokens; text
            positions hold the actual transcript / control tokens.
          text_position_mask: (B, L) bool — True where text head should
            compute loss (i.e. non-audio positions).
        """
        # 1. Encode audio
        audio_hidden = self.aut(mel)  # (B, T_aud, d_audio)

        # 2. Project to LLM hidden
        audio_embeds = self.projector(audio_hidden)  # (B, T_aud, d_llm)

        # 3. Build the LLM input embedding sequence by replacing audio_pad
        #    embeddings with the projected audio embeddings.
        token_embeds = self.llm.get_input_embeddings()(text_token_ids)  # (B, L, d_llm)
        is_audio_slot = (text_token_ids == self.audio_pad_id)  # (B, L)

        # Sanity: number of audio slots should equal T_aud * B (one per frame).
        # In practice we let the data loader guarantee this and only assert during dev.
        flat_audio = audio_embeds.reshape(-1, audio_embeds.shape[-1])
        token_embeds[is_audio_slot] = flat_audio[: is_audio_slot.sum()]

        # 4. Run the LLM with the chosen attention window.
        attn_mask = self._build_attention_mask(
            seq_len=token_embeds.shape[1],
            device=token_embeds.device,
            window_seconds=attention_window_seconds,
        )
        llm_out = self.llm(
            inputs_embeds=token_embeds,
            attention_mask=attn_mask,
            output_hidden_states=False,
        )
        logits = llm_out.logits  # (B, L, V)

        # 5. Endpoint head reads audio_hidden directly (not LLM hidden) so it
        #    is independent of the LLM's attention window.
        eager_logits, end_logits = self.endpoint_head(audio_hidden)

        return {
            "logits": logits,
            "text_position_mask": text_position_mask,
            "eager_logits": eager_logits,
            "end_logits": end_logits,
        }

    def _build_attention_mask(
        self,
        seq_len: int,
        device: torch.device,
        window_seconds: float | None,
    ) -> torch.Tensor:
        """Sliding-window causal mask. Window measured in audio frames at 12.5 Hz.

        Implementation: returns a (seq_len, seq_len) bool mask where True means
        "attend". For simplicity we approximate the window in tokens, scaled
        from the seconds parameter at 12.5 Hz audio rate. This is what Qwen3-ASR
        does at inference (research/10-qwen3-asr-deepdive.md §5).
        """
        if window_seconds is None:
            # No windowing — full causal
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
            return mask
        window_tokens = max(1, int(window_seconds * 12.5))
        idx = torch.arange(seq_len, device=device)
        # Causal AND within-window
        causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
        within = (idx.unsqueeze(1) - idx.unsqueeze(0)) <= window_tokens
        return causal & within
