"""NVIDIA Parakeet-Realtime-EOU-120m adapter — in-stream <EOU> token = fire.

Model: nvidia/parakeet_realtime_eou_120m-v1 (FastConformer-RNNT, cache-aware
streaming, EncDecRNNTBPEModel). It emits an `<EOU>` token in the transcript at
each end-of-utterance. Fire = <EOU> emission.

Decoding: cache-aware streaming models produce IDENTICAL predictions offline and
in-stream (NeMo cache-aware property), so we take a single decode with
timestamps=True and read each `<EOU>` token's decoded time from the char-level
timestamps. The fire is then placed at the next 0.5 s chunk boundary >= that
time — i.e. the chunk at which an in-stream decoder would surface the token —
matching the (k+1)*chunk_s timestamp convention of every other arm. (The model's
80-160 ms algorithmic latency is dominated by this 0.5 s quantisation.)

Few knobs -> a single arm ("eou"). Non-OSI license (NVIDIA Open Model /
research eval terms) — fine for evaluation; surfaced in `license_note` and to be
noted in the paper caption. Also records the transcript for the streaming-WER row.

Device: uses CUDA if visible, else CPU. For the full replay run leave the GPU
visible; the smoke test sets CUDA_VISIBLE_DEVICES="" to avoid the training GPU.
Deps: nemo_toolkit[asr] >= 2.5.3.
"""
from __future__ import annotations
import math
import tempfile
import numpy as np
import soundfile as sf

from eval.external.adapters.base import BaseAdapter
from eval.streaming_replay_eval import SR

REPO = "nvidia/parakeet_realtime_eou_120m-v1"
EOU = "<EOU>"


class ParakeetEOUAdapter(BaseAdapter):
    name = "parakeet_realtime_eou_120m"
    license_note = ("NON-OSI: NVIDIA open-model / research-eval terms — "
                    "evaluation-only; note in paper caption")
    arm_keys = ["eou"]

    def setup(self) -> None:
        import nemo.collections.asr as nemo_asr
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = nemo_asr.models.ASRModel.from_pretrained(REPO, map_location=dev)
        self.model.eval()

    def infer_stretch(self, audio: np.ndarray, chunk_s: float):
        with tempfile.NamedTemporaryFile(suffix=".wav") as tf:
            sf.write(tf.name, audio.astype(np.float32), SR)
            out = self.model.transcribe([tf.name], timestamps=True, verbose=False)[0]
        fires = []
        chars = (out.timestamp or {}).get("char", []) if hasattr(out, "timestamp") else []
        for c in chars:
            token = "".join(c.get("char", []))
            if EOU in token:
                t = float(c.get("start", c.get("end", 0.0)))
                fires.append(math.ceil(t / chunk_s + 1e-9) * chunk_s)
        # Fallback: no char timestamps but text carries <EOU> (single utterance)
        if not fires and EOU in (out.text or ""):
            segs = (out.timestamp or {}).get("segment", [])
            t = float(segs[-1]["end"]) if segs else len(audio) / SR
            fires.append(math.ceil(t / chunk_s + 1e-9) * chunk_s)
        transcript = (out.text or "").replace(EOU, " ")
        return {"eou": sorted(set(fires))}, transcript
