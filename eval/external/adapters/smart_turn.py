"""Smart Turn v3 (pipecat-ai) adapter — acoustic turn-completion classifier.

Smart Turn is NOT streaming: it judges "is this turn complete?" on a trailing
audio window when invoked. Deployment-faithful harness (its shipped design):
trigger on VAD silence — we reuse the LM arms' RMS energy detector for
apples-to-apples — then classify the trailing 8 s window and fire iff the
completion probability crosses threshold. The latency floor is therefore
VAD-silence + inference, which is the honest comparison (that is how it ships).

One model pass per stretch: we score every VAD-silence trigger once, then a
threshold sweep just re-thresholds the stored probabilities (arms = thresholds).

Model: pipecat-ai/smart-turn-v3 (Whisper-tiny encoder + linear head, ONNX, 8M
params). 16 kHz mono; window truncated/left-padded to 8 s; ONNX input tensor
"input_features"; output is a sigmoid completion probability. License: BSD-2
(OSI). Deps: onnxruntime, transformers (WhisperFeatureExtractor), huggingface_hub.
"""
from __future__ import annotations
import numpy as np

from eval.external.adapters.base import BaseAdapter, vad_silence_triggers, trailing_window

REPO = "pipecat-ai/smart-turn-v3"
ONNX_FILE = "smart-turn-v3.2-cpu.onnx"   # latest v3 CPU export on the HF repo
SR = 16000
WINDOW_S = 8.0
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]   # sweep the public decision threshold


class SmartTurnAdapter(BaseAdapter):
    name = "smart_turn_v3"
    license_note = "BSD-2 (OSI); model pipecat-ai/smart-turn-v3 (v3.2-cpu.onnx)"
    arm_keys = [f"thr{t}" for t in THRESHOLDS]

    def setup(self) -> None:
        import os
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor
        from huggingface_hub import hf_hub_download

        cache = os.environ.get("EXT_MODEL_DIR", "data/external_models")
        path = hf_hub_download(REPO, ONNX_FILE, cache_dir=cache)
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(path, sess_options=so)
        self.fe = WhisperFeatureExtractor(chunk_length=8)
        self.thresholds = THRESHOLDS

    def _prob(self, window: np.ndarray) -> float:
        # truncate to last 8 s or left-pad with zeros (audio_utils convention)
        n = int(WINDOW_S * SR)
        if len(window) > n:
            window = window[-n:]
        elif len(window) < n:
            window = np.pad(window, (n - len(window), 0))
        feats = self.fe(window, sampling_rate=SR, return_tensors="np",
                        padding="max_length", max_length=n, truncation=True,
                        do_normalize=True).input_features.squeeze(0).astype(np.float32)
        feats = np.expand_dims(feats, 0)
        out = self.session.run(None, {"input_features": feats})
        return float(out[0].reshape(-1)[0])

    def infer_stretch(self, audio: np.ndarray, chunk_s: float):
        triggers, _ = vad_silence_triggers(audio, chunk_s)
        probs = []   # (fire_time_s, probability)
        for tg in triggers:
            win = trailing_window(audio, tg["window_end_sample"], WINDOW_S)
            probs.append((tg["fire_time_s"], self._prob(win)))
        arms = {f"thr{t}": [ft for ft, p in probs if p >= t] for t in self.thresholds}
        return arms, None   # acoustic classifier: no transcript
