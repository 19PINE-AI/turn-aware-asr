"""Kyutai STT semantic-VAD head adapter — pause-prediction crossing = fire.

Model: kyutai/stt-1b-en_fr-candle (delayed-streams / moshi). Streams natively via
Mimi (24 kHz, 12.5 Hz frames). The LM carries a semantic-VAD with four pause-
prediction heads (0.5 / 1.0 / 2.0 / 3.0 s); `lm_gen.step_with_extra_heads`
returns them per frame. We read the 2.0 s head (index 2 — matches our
T_TURN = 2.0 s turn-final definition) and fire when its probability crosses
threshold. Sweep ~5 thresholds for a curve. Also records the STT transcript for
the streaming-WER row.

Fire timestamp = the frame time of the crossing, quantised up to the next 0.5 s
chunk boundary so it matches the (k+1)*chunk_s convention of every other arm.

Audio is 16 kHz in our benchmark; Mimi wants its own sample rate (24 kHz) — we
resample per stretch. License: CC-BY 4.0 (weights) / permissive code.
Deps: moshi >= 0.2.6, torchaudio (resample). Needs its own venv (moshi pins
clash with NeMo). GPU strongly preferred (1B params); CPU works but is slow.

Knobs via env:
  KYUTAI_STT_REPO   (default kyutai/stt-1b-en_fr-candle)
  KYUTAI_VAD_HEAD   (default 2 = 2.0 s pause head)
"""
from __future__ import annotations
import os
import math
import numpy as np

from eval.external.adapters.base import BaseAdapter
from eval.streaming_replay_eval import SR

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


class KyutaiVADAdapter(BaseAdapter):
    name = "kyutai_stt_vad"
    license_note = "CC-BY-4.0 weights; model kyutai/stt-1b-en_fr-candle (semantic-VAD head 2 = 2.0 s)"
    arm_keys = [f"thr{t}" for t in THRESHOLDS]

    def setup(self) -> None:
        import torch
        import moshi.models
        from moshi.models import LMGen

        repo = os.environ.get("KYUTAI_STT_REPO", "kyutai/stt-1b-en_fr-candle")
        self.head = int(os.environ.get("KYUTAI_VAD_HEAD", "2"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        info = moshi.models.loaders.CheckpointInfo.from_hf_repo(repo)
        self.mimi = info.get_mimi(device=self.device)
        self.tokenizer = info.get_text_tokenizer()
        lm = info.get_moshi(device=self.device, dtype=torch.bfloat16)
        self.lm_gen = LMGen(lm, temp=0, temp_text=0.0)
        self.mimi_sr = int(self.mimi.sample_rate)
        self.frame_size = int(self.mimi.frame_size)
        self.thresholds = THRESHOLDS

    def infer_stretch(self, audio: np.ndarray, chunk_s: float):
        import torch
        from math import gcd
        from scipy.signal import resample_poly

        a = audio.astype(np.float32)
        if self.mimi_sr != SR:
            g = gcd(self.mimi_sr, SR)
            a = resample_poly(a, self.mimi_sr // g, SR // g).astype(np.float32)
        wav = torch.from_numpy(a).to(self.device)[None, None]   # (B=1, C=1, T)

        fs = self.frame_size
        n_frames = wav.shape[-1] // fs
        vad_series = []      # (frame_time_s, prob) using ORIGINAL (16k) timeline
        text_tokens_out = []
        with torch.no_grad(), self.mimi.streaming(1), self.lm_gen.streaming(1):
            for k in range(n_frames):
                frame = wav[..., k * fs:(k + 1) * fs]
                codes = self.mimi.encode(frame)
                text_tok, vad_heads = self.lm_gen.step_with_extra_heads(codes)
                # frame end time on the mimi timeline -> seconds (sr-independent)
                t_s = (k + 1) * fs / self.mimi_sr
                if vad_heads:
                    h = min(self.head, len(vad_heads) - 1)
                    vad_series.append((t_s, float(vad_heads[h][0, 0, 0].cpu().item())))
                if text_tok is not None:
                    tid = int(text_tok[0, 0].cpu().item())
                    if tid not in (0, 3):   # skip pad/eos-ish control ids
                        text_tokens_out.append(tid)

        # fire = rising edge across each threshold (arm), quantised to 0.5 s grid
        arms = {}
        for thr in self.thresholds:
            fires, prev = [], 0.0
            for t_s, p in vad_series:
                if p >= thr and prev < thr:
                    fires.append(math.ceil(t_s / chunk_s + 1e-9) * chunk_s)
                prev = p
            arms[f"thr{thr}"] = fires

        transcript = None
        try:
            transcript = self.tokenizer.decode(text_tokens_out)
        except Exception:
            pass
        return arms, transcript
