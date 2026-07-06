"""LiveKit turn detector adapter — text EOU classifier over a transcript.

The LiveKit detector (livekit/turn-detector, a 135M SmolLM-v2 fine-tune, INT8
ONNX, CPU) predicts end-of-utterance from the *text* of the conversation so far,
not from audio. Deployment-faithful harness (same skeleton as Smart Turn): at
each VAD-silence trigger (reusing our RMS detector) feed the transcript observed
up to that moment and fire iff the EOU probability crosses threshold. Sweep the
threshold.

Transcript source (spec lists two variants):
  (a) as-shipped over its default STT — not wired here (needs LiveKit's STT).
  (b) over a supplied streaming transcript, to isolate the detector from STT
      quality. This adapter's default `transcript_mode="oracle"` uses the
      stretch's ground-truth words with end_s <= trigger time — an *upper bound*
      on transcript quality, clearly labelled. To run variant (b) with our base
      model's real streaming transcript, set transcript_mode="external" and
      populate self.external_transcripts (see README) — that pass needs the GPU.

Text formatting mirrors the plugin: NFKC + punctuation strip, Qwen chat template
with the final <|im_end|> removed, tokenized (no special tokens); ONNX returns
the EOU probability as the last element of the flattened output.

Model: livekit/turn-detector (Apache-2.0, OSI). Deps: onnxruntime, transformers
(tokenizer), huggingface_hub — same lean venv as Smart Turn.
"""
from __future__ import annotations
import re
import unicodedata
import numpy as np

from eval.external.adapters.base import BaseAdapter, vad_silence_triggers
from eval.streaming_replay_eval import SR

REPO = "livekit/turn-detector"
ONNX_FILE = "model_quantized.onnx"
MAX_HISTORY_TOKENS = 128
THRESHOLDS = [0.3, 0.5, 0.7, 0.8, 0.9]

# Stretch events are attached by the harness caller via set_events(); the audio
# array alone can't carry ground-truth text. We stash the current stretch's
# events on the adapter right before infer_stretch (harness passes audio only,
# so we intercept through a module-level convention: see infer_stretch docstring).


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.lower())
    text = re.sub(r"[^\w\s']", "", text)
    return re.sub(r"\s+", " ", text).strip()


class LiveKitTurnAdapter(BaseAdapter):
    name = "livekit_turn_detector"
    license_note = ("Apache-2.0 (OSI); model livekit/turn-detector. "
                    "transcript_mode=oracle uses ground-truth words (upper bound); "
                    "variant-b real transcript needs a GPU STT pass.")
    arm_keys = [f"thr{t}" for t in THRESHOLDS]

    def __init__(self, transcript_mode: str = "oracle"):
        self.transcript_mode = transcript_mode
        self._events = None            # set per stretch by the harness hook
        self.external_transcripts = {}  # stretch_key -> [(end_s, word)] for mode=external

    def setup(self) -> None:
        import os
        import onnxruntime as ort
        from transformers import AutoTokenizer
        from huggingface_hub import hf_hub_download

        cache = os.environ.get("EXT_MODEL_DIR", "data/external_models")
        onnx_path = hf_hub_download(REPO, ONNX_FILE, cache_dir=cache)
        self.tokenizer = AutoTokenizer.from_pretrained(REPO, cache_dir=cache)
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"],
                                            sess_options=so)
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.thresholds = THRESHOLDS

    # The harness feeds only `audio`; it also sets adapter._events before each
    # call (see harness note). We read ground-truth words from there.
    def set_stretch(self, events, key=None):
        self._events = events
        self._key = key

    def _eou_prob(self, transcript: str) -> float:
        convo = [{"role": "user", "content": transcript or ""}]
        text = self.tokenizer.apply_chat_template(convo, add_generation_prompt=False,
                                                  tokenize=False)
        ix = text.rfind("<|im_end|>")
        if ix != -1:
            text = text[:ix]
        enc = self.tokenizer(text, add_special_tokens=False, return_tensors="np",
                             truncation=True, max_length=MAX_HISTORY_TOKENS)
        feeds = {"input_ids": enc["input_ids"].astype(np.int64)}
        if "attention_mask" in self.input_names:
            feeds["attention_mask"] = enc["attention_mask"].astype(np.int64)
        out = self.session.run(None, feeds)
        return float(np.asarray(out[0]).flatten()[-1])

    def _transcript_upto(self, t_s: float) -> str:
        if self.transcript_mode == "external":
            words = self.external_transcripts.get(getattr(self, "_key", None), [])
            return _normalize(" ".join(w for (e, w) in words if e <= t_s))
        # oracle: ground-truth words whose utterance ends before the trigger
        if not self._events:
            return ""
        parts = [e["text"] for e in self._events if e["end_s"] <= t_s + 1e-6]
        return _normalize(" ".join(parts))

    def infer_stretch(self, audio: np.ndarray, chunk_s: float):
        triggers, _ = vad_silence_triggers(audio, chunk_s)
        probs = []
        for tg in triggers:
            tr = self._transcript_upto(tg["fire_time_s"])
            probs.append((tg["fire_time_s"], self._eou_prob(tr)))
        arms = {f"thr{t}": [ft for ft, p in probs if p >= t] for t in self.thresholds}
        # transcript we (would) report for the WER row: the full oracle/external text
        full = self._transcript_upto(float("inf"))
        return arms, full
