"""CPU-only sanity adapter: RMS silence-timeout, no model, no GPU.

Reproduces eval/timeout_baseline_eval.py's rms detector as an Adapter so the
harness path (stretch loading -> adapter -> score_fires -> aggregate -> JSON)
can be validated end-to-end without downloading any external model. Sweeps a few
timeouts as arms; each arm's numbers should match timeout_baseline_eval's rms
arm at the same timeout. Not a paper baseline — a harness smoke fixture.
"""
from __future__ import annotations
import numpy as np

from eval.external.adapters.base import BaseAdapter
from eval.timeout_baseline_eval import chunk_speech_flags, run_timeout


class DummyRMSAdapter(BaseAdapter):
    name = "dummy_rms"
    license_note = "harness fixture (not a baseline)"
    arm_keys = ["0.5", "1.0", "1.5", "2.0"]

    def setup(self) -> None:
        self.timeouts = [float(x) for x in self.arm_keys]

    def infer_stretch(self, audio: np.ndarray, chunk_s: float):
        flags = chunk_speech_flags(audio, chunk_s, "rms", 1e-3)
        arms = {f"{X:.1f}": run_timeout(flags, chunk_s, X) for X in self.timeouts}
        return arms, None
