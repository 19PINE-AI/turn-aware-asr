"""Base adapter interface + shared VAD-silence trigger for exp-4 external baselines.

Every external turn-aware system is wrapped in an Adapter that consumes ONE
benchmark stretch (the exact `build_stretches` audio, fed as 0.5 s chunks) and
returns fire timestamps in the same convention the LM arms use (a fire at chunk
k is timestamped `(k+1)*chunk_s`). The harness (eval/external/harness.py) scores
those fires with the byte-identical `score_fires` / aggregation the LM and
timeout arms use, so the boundary taxonomy, [-0.25,+1.5] s window and
false/min definition are all unchanged.

Two families of adapter:
  * streaming-native (Kyutai, Parakeet): the model emits a per-frame VAD/EOU
    signal; the adapter thresholds it. It may run the model ONCE and return a
    dict of {arm_key: fires} for a whole threshold sweep (one model pass).
  * classifier-at-trigger (Smart Turn, LiveKit): not streaming. The
    deployment-faithful harness fires the classifier only at a VAD-silence
    trigger — we reuse the SAME RMS energy detector as the LM arms'
    `--energy-gate` (gate_rms=1e-3) so the comparison isolates the policy, not
    the detector. `vad_silence_triggers` returns those trigger points; the
    latency floor is therefore trigger-silence + inference, which is how these
    systems ship.
"""
from __future__ import annotations
import numpy as np

# Reuse the LM arms' RMS speech detector verbatim (apples-to-apples).
from eval.timeout_baseline_eval import chunk_speech_flags
from eval.streaming_replay_eval import SR


def vad_silence_triggers(audio: np.ndarray, chunk_s: float,
                         gate_rms: float = 1e-3,
                         trigger_silence_chunks: int = 1):
    """Return VAD-silence trigger points on the LM arms' 0.5 s chunk grid.

    A trigger fires the classifier once continuous RMS-detected silence since
    the last speech chunk reaches `trigger_silence_chunks` chunks; it then
    re-arms on the next speech chunk (same policy skeleton as
    timeout_baseline_eval.run_timeout, but with a short fixed delay whose only
    role is to define WHEN the classifier is asked, not to decide the turn).

    Yields dicts: {"chunk_idx": k, "fire_time_s": (k+1)*chunk_s,
                   "window_end_sample": int}. `window_end_sample` marks the end
    of the trailing audio the classifier should judge (through the pause).
    """
    flags = chunk_speech_flags(audio, chunk_s, "rms", gate_rms)
    triggers = []
    seen_speech = False
    silent = 0
    for k, sp in enumerate(flags):
        if sp:
            seen_speech = True
            silent = 0
        else:
            if seen_speech:
                silent += 1
                if silent >= trigger_silence_chunks:
                    triggers.append({
                        "chunk_idx": k,
                        "fire_time_s": (k + 1) * chunk_s,
                        "window_end_sample": int((k + 1) * chunk_s * SR),
                    })
                    seen_speech = False
                    silent = 0
    return triggers, flags


def trailing_window(audio: np.ndarray, end_sample: int, window_s: float) -> np.ndarray:
    """Trailing `window_s` seconds of audio ending at `end_sample` (clamped)."""
    end = min(len(audio), max(0, end_sample))
    start = max(0, end - int(window_s * SR))
    return audio[start:end]


class BaseAdapter:
    """Subclass and implement setup() + infer_stretch().

    infer_stretch(audio, chunk_s) -> (arms, transcript)
      arms: {arm_key: [fire_time_s, ...]}   # one entry per swept threshold
      transcript: str | None                # for the streaming-WER row
    """

    name = "base"
    #: license note surfaced in the harness output (e.g. non-OSI systems)
    license_note = ""
    #: default arm keys (thresholds) this adapter sweeps; harness may echo them
    arm_keys: list[str] = ["default"]

    def setup(self) -> None:  # download / load weights
        raise NotImplementedError

    def infer_stretch(self, audio: np.ndarray, chunk_s: float):
        raise NotImplementedError
