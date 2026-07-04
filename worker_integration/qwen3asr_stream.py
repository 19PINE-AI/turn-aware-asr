"""Bounded-re-feed committed-prefix streaming decoder for the endpoint Qwen3-ASR.

This is the MODEL-SIDE session logic that runs inside a Metronome worker to serve
the merged endpoint checkpoint (scripts/merge_endpoint_lora.py) as a real streaming
session. It implements the v1 protocol recommended in
research/60-metronome-integration-scope.md:

  * BOUNDED audio re-feed: per chunk, re-feed only the last ``W_audio`` seconds of
    audio as ONE segment (in-distribution: the model was trained on single audio
    segments, never on N chunked placeholders). The endpointer itself keeps the
    buffer short in conversation — on a terminal ``<END_SPEECH>`` the segment is
    flushed and its audio dropped, so utterances (not sessions) bound the buffer.
    A hard ``max_buffer_s`` cap force-flushes during continuous speech (streaming
    captions) where markers do not fire — the Metronome bound-the-resident-state
    principle applied at the application layer.

  * COMMITTED-PREFIX decoding (qwen_asr official vLLM streaming semantics, see
    qwen_asr/inference/qwen3_asr.py::streaming_transcribe): prompt = chat prefix +
    previously committed transcript with the last ``rollback`` tokens rolled back,
    so the model continues the transcript rather than re-deciding it each chunk.

  * ENERGY GATE + CONFIRM policy (research/61): never START a segment on a silent
    chunk (kills the silence-hallucination + marker-spam failure mode); optionally
    defer the system-level END by ``confirm_chunks`` silent chunks so phrase-level
    pauses (speaker resumes) do not fire (the transcript flush still stands).

  * PINNED-CTX SINK (research/60 fact 6, v2 refinement): hotwords / context live in
    the system prompt = tokens ``[0, S)`` of every request. In v1 they are simply
    re-fed inside ``HEAD`` each chunk (cheap, always attended because always
    re-prefilled). In the v2 in-engine windowed-KV build those ``[0, S)`` tokens are
    exactly what the KV-manager pin + Triton union mask keep resident while audio
    history slides out. ``pinned_prefix_len()`` exposes S for that mechanism.

Decoupling from the engine
--------------------------
This module NEVER instantiates a vLLM engine or touches the GPU. The decode step is
injected as a ``generate_fn(prompt: str, audio: np.ndarray) -> str`` callable, so the
session logic is unit-testable on CPU with a fake generator and driven by a real
engine only inside the worker. ``build_engine()`` and ``vllm_generate_fn()`` are the
GPU-side factories — both marked TODO for the main (GPU-owning) process to run.

The bounded-re-feed session emits ONE audio placeholder over the whole (bounded)
buffer — i.e. the ``fd_step`` / single-segment prompt shape, NOT the ``fd_step_stream``
one-placeholder-per-chunk shape. E2 (research/64) measures whether the chunked
placeholder shape is WER-safe; if it is, the resident ``fd_step_stream`` path can
replace this re-feed loop, but the endpoint/flush/gate/confirm control logic below
is identical either way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

SR = 16000
END_TOK = "<END_SPEECH>"
EAGER_TOK = "<EAGER_END_SPEECH>"

# Reserved-vocab ids the two markers occupy in the merged checkpoint
# (scripts/merge_endpoint_lora.py; research/60 fact 4). Kept here so the serving
# SamplingParams can assert these are NOT suppressed (no bad_words / special-token
# masking) — greedy decode must be allowed to emit them. See vllm_sampling_params().
EAGER_ID = 151705
END_ID = 151706

# GenerateFn: given the full text prompt and the (bounded) mono 16 kHz float32 audio
# buffer, return the RAW decoded string for the continuation (excluding the prompt).
# In committed mode the caller prepends the committed prefix to the prompt, so the
# return value is only the newly generated tail — matching model.generate(...) sliced
# past input_ids (transformers) and outputs[0].text (vLLM, which returns only the
# completion). See vllm_generate_fn() / transformers_generate_fn().
GenerateFn = Callable[[str, np.ndarray], str]


# --------------------------------------------------------------------------- prompt

def build_chat_prefix(context: str = "") -> str:
    """The Qwen3-ASR chat prefix up to (and including) the assistant generation tag,
    for ONE audio segment. Mirrors data/qwen3-asr-0.6b/chat_template.json exactly:

        <|im_start|>system\\n{context}<|im_end|>\\n
        <|im_start|>user\\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\\n
        <|im_start|>assistant\\n

    ``context`` is the hotword / domain context string (research/16: -37% rel. WER).
    It is the pinned-sink region (tokens [0, S); see pinned_prefix_len()).
    """
    return (
        f"<|im_start|>system\n{context}<|im_end|>\n"
        f"<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def strip_meta(text: str) -> str:
    """Drop the leading ``language X<asr_text>`` metadata Qwen3-ASR emits, and the
    turn-end tag, leaving the transcript (markers preserved). Matches
    eval/streaming_replay_eval.py::clean so streaming WER is comparable."""
    text = re.sub(r"^language\s+\S+\s*<asr_text>", "", text)
    return text.replace("<|im_end|>", "").strip()


# ------------------------------------------------------------------------- session

@dataclass
class StepResult:
    """What the session produced for one fed chunk."""
    gated: bool = False              # chunk skipped by the energy gate (no decode)
    decoded: bool = False            # a decode actually ran this chunk
    raw: str = ""                    # raw model output for the live segment (with meta)
    transcript: str = ""            # strip_meta(raw): live-segment transcript
    fired: bool = False              # a system-level END fire was EMITTED this chunk
    eager: bool = False              # the live segment currently carries an EAGER marker
    flushed_segment: Optional[str] = None   # segment text if it was flushed this chunk
    forced_flush: bool = False       # flush was the max_buffer_s cap, not a marker
    fire_time_s: Optional[float] = None      # audio-clock time of an emitted fire


@dataclass
class Qwen3ASRStreamSession:
    """One resident streaming ASR+endpoint session.

    Feed 16 kHz mono float32 chunks via ``feed(chunk)``; read markers/flushes off the
    returned :class:`StepResult`. The audio buffer is bounded (utterance flush +
    ``max_buffer_s`` cap), so per-frame decode cost is O(W_audio), independent of
    session length — the Metronome bound realized at the application layer.

    Parameters mirror research/61's operating point (v5/v8 + gate + confirm1):
      * ``window_s``        re-feed at most this many seconds each decode (bounded buffer).
      * ``max_buffer_s``    hard cap: force-flush the live segment past this length
                            (continuous-speech safety; 0 disables — utterances bound it).
      * ``overlap_s``       audio carried past a force-flush boundary so the new
                            segment's leading edge is not a hard AuT window cut
                            (research/60 risk: "AuT windowing at buffer edges").
      * ``energy_gate`` / ``gate_rms``   never start a segment on a silent chunk.
      * ``confirm_chunks``  defer the END fire by this many silent chunks; speech
                            resuming inside the window discards the END (phrase, not
                            turn). The transcript flush still stands.
      * ``rollback``        committed-prefix rollback depth K (jitter guard).
      * ``unfixed_chunks``  first N chunks of a segment decode from scratch (no prefix).
    """

    generate_fn: GenerateFn
    tokenizer: object                      # HF tokenizer (encode/decode) for rollback
    context: str = ""
    window_s: float = 16.0
    max_buffer_s: float = 30.0
    overlap_s: float = 0.75
    chunk_s: float = 0.5
    energy_gate: bool = True
    gate_rms: float = 1e-3
    confirm_chunks: int = 1
    rollback: int = 5
    unfixed_chunks: int = 2
    max_new_tokens: int = 96

    # --- live state (not user-set) ---
    prompt_prefix: str = field(init=False)
    buffer: Optional[np.ndarray] = field(default=None, init=False)
    raw: str = field(default="", init=False)          # committed raw for live segment
    seg_chunks: int = field(default=0, init=False)    # chunks decoded in live segment
    frame: int = field(default=0, init=False)         # chunks fed (for audio clock)
    prev_marker_count: int = field(default=0, init=False)
    pending_fire_frame: Optional[int] = field(default=None, init=False)
    flushed: list = field(default_factory=list, init=False)  # all flushed segment texts
    fires_s: list = field(default_factory=list, init=False)  # accepted fire times
    compute_ms: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self.prompt_prefix = build_chat_prefix(self.context)

    # -- pinned-sink hook (v2) -------------------------------------------------
    def pinned_prefix_len(self) -> int:
        """Number of leading prompt tokens (system block, incl. context/hotwords)
        that the v2 in-engine windowed-KV build should PIN as attention sinks so
        they stay attended while audio history slides out (research/60 fact 6).
        This is S in the [0, S) sink region.

        NOTE: counts the system block only (up to and including the first
        ``<|im_end|>\\n``), which is what stays constant across the session; the
        user/audio block changes every chunk under bounded re-feed."""
        sys_block = self.prompt_prefix.split("<|im_end|>\n", 1)[0] + "<|im_end|>\n"
        return len(self.tokenizer.encode(sys_block))

    # -- committed prefix ------------------------------------------------------
    def _committed_prefix(self) -> str:
        """previously-committed raw text minus the last ``rollback`` tokens (utf-8
        safe), or "" for the first ``unfixed_chunks`` chunks of a segment. Never
        rolls back across a flushed marker: ``self.raw`` is cleared on flush, so the
        prefix is always within the LIVE segment only (research/60 risk note)."""
        if self.seg_chunks < self.unfixed_chunks or not self.raw:
            return ""
        ids = self.tokenizer.encode(self.raw)
        k = self.rollback
        while True:
            end = max(0, len(ids) - k)
            prefix = self.tokenizer.decode(ids[:end]) if end else ""
            if "�" not in prefix:      # avoid splitting a multibyte char
                return prefix
            if end == 0:
                return ""
            k += 1

    # -- buffer bounding -------------------------------------------------------
    def _append_bounded(self, chunk: np.ndarray):
        self.buffer = chunk if self.buffer is None else np.concatenate([self.buffer, chunk])
        cap = int(self.window_s * SR)
        if len(self.buffer) > cap:
            self.buffer = self.buffer[-cap:]

    def _reset_segment(self, keep_overlap: bool = False):
        """Flush point: clear committed text; drop the segment's audio. On a forced
        (cap) flush keep an ``overlap_s`` audio tail so the next segment does not begin
        exactly at an AuT window boundary."""
        tail = None
        if keep_overlap and self.buffer is not None and self.overlap_s > 0:
            n = int(self.overlap_s * SR)
            tail = self.buffer[-n:] if len(self.buffer) > n else self.buffer
        self.raw = ""
        self.seg_chunks = 0
        self.prev_marker_count = 0
        self.buffer = tail

    # -- main step -------------------------------------------------------------
    def feed(self, chunk: np.ndarray) -> StepResult:
        """Consume one audio chunk; run at most one incremental decode; update
        markers/flushes. Returns a :class:`StepResult`. ``chunk`` is 16 kHz mono
        float32 in [-1, 1] (any length; nominally ``chunk_s`` seconds)."""
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        k = self.frame
        self.frame += 1
        res = StepResult()
        silent = len(chunk) == 0 or float(np.sqrt(np.mean(chunk ** 2))) < self.gate_rms

        # 1) Resolve a pending END candidate FIRST (before the gate would skip the
        #    confirming silent chunks). Its segment was already flushed at the marker.
        if self.pending_fire_frame is not None:
            if not silent:
                self.pending_fire_frame = None        # speech resumed -> phrase, not turn
            elif k - self.pending_fire_frame >= self.confirm_chunks:
                t = (k + 1) * self.chunk_s
                self.fires_s.append(t)
                res.fired = True
                res.fire_time_s = t
                self.pending_fire_frame = None

        # 2) Energy gate: never START a segment on a silent chunk.
        if self.energy_gate and self.buffer is None and silent:
            res.gated = True
            return res

        # 3) One incremental decode over the bounded buffer.
        self._append_bounded(chunk)
        prefix = self._committed_prefix()
        prompt = self.prompt_prefix + prefix
        import time as _t
        t0 = _t.perf_counter()
        gen = self.generate_fn(prompt, self.buffer)
        self.compute_ms.append((_t.perf_counter() - t0) * 1000.0)
        self.raw = (prefix + gen) if prefix else gen
        self.seg_chunks += 1
        res.decoded = True
        res.raw = self.raw
        res.transcript = strip_meta(self.raw)
        res.eager = EAGER_TOK in self.raw

        # 4) Marker / flush handling (mirrors run_stretch in streaming_replay_eval).
        n_mark = res.transcript.count(END_TOK)
        if n_mark > self.prev_marker_count:
            after = res.transcript.rsplit(END_TOK, 1)[1].replace(EAGER_TOK, "").strip()
            if not after:                              # terminal marker -> clean flush
                if self.confirm_chunks == 0:
                    t = (k + 1) * self.chunk_s
                    self.fires_s.append(t)
                    res.fired = True
                    res.fire_time_s = t
                else:
                    self.pending_fire_frame = k        # confirm on silence
                seg_text = res.transcript.replace(EAGER_TOK, "")
                self.flushed.append(seg_text)
                res.flushed_segment = seg_text
                self._reset_segment(keep_overlap=False)  # utterance done: drop its audio
                return res
        self.prev_marker_count = n_mark

        # 5) Max-buffer force-flush (continuous speech, no terminal marker): bound the
        #    committed prefix so committed-prefix decoding cannot loop (research/61 v8
        #    tail). Transcription reset, NOT an endpoint -> no fire.
        if self.max_buffer_s and self.buffer is not None and \
                len(self.buffer) >= self.max_buffer_s * SR:
            seg_text = strip_meta(self.raw).replace(EAGER_TOK, "")
            self.flushed.append(seg_text)
            res.flushed_segment = seg_text
            res.forced_flush = True
            self._reset_segment(keep_overlap=True)
        return res

    def finish(self) -> Optional[str]:
        """End of stream: flush any live segment transcript. Returns the tail text."""
        if self.raw:
            tail = strip_meta(self.raw).replace(EAGER_TOK, "")
            self.flushed.append(tail)
            self._reset_segment(keep_overlap=False)
            return tail
        return None

    def transcript(self) -> str:
        """Concatenated flushed transcript so far (markers stripped) — the streaming
        WER hypothesis (research/58 metric)."""
        return " ".join(t.replace(END_TOK, " ") for t in self.flushed).strip()


# ---------------------------------------------------------------- GPU-side factories
# Everything below TOUCHES THE GPU / loads the model. It is written but must be run by
# the main (GPU-owning) process — do NOT call from a sandboxed/parallel job (research/60
# GPU-contention risk: a second engine OOM-crashes the resident one).

def vllm_sampling_params(max_tokens: int = 96):
    """SamplingParams for greedy endpoint decoding. CRITICAL: skip_special_tokens
    MUST be False — the markers are reserved-slot special tokens (151705/151706)
    and vLLM's default (True) strips them from .text even though the model emits
    them. VERIFIED in E1b (worker_integration/e1_vllm_validate.py): with
    skip_special_tokens=False the merged v9 checkpoint fires <END_SPEECH> on
    speech+silence and stays silent on silence / speech-without-silence. Callers
    that scan .text for markers depend on this; a token-id scan is the robust
    alternative. TODO(GPU): import inside the engine process."""
    from vllm import SamplingParams  # noqa: local import; vLLM only present on GPU box
    return SamplingParams(temperature=0.0, max_tokens=max_tokens,
                          ignore_eos=False, skip_special_tokens=False)


def vllm_generate_fn(engine, sampling_params=None) -> GenerateFn:
    """Wrap a resident vLLM ``LLMEngine``/``LLM`` into a GenerateFn for the session.

    Builds the single-segment multimodal request exactly like
    metronome/backends/vllm_backend.py::_mm_prompt (ONE audio placeholder over the
    whole bounded buffer) and returns ``outputs[0].outputs[0].text`` (the completion
    only, so committed-prefix concatenation is correct).

    TODO(GPU): requires a live engine from build_engine(); do not instantiate here.
    """
    sp = sampling_params or vllm_sampling_params()

    def _gen(prompt: str, audio: np.ndarray) -> str:
        req = {"prompt": prompt, "multi_modal_data": {"audio": [audio]}}
        # LLM.generate (offline) shape; for the async worker use engine.generate /
        # AsyncLLM as in worker/stream_server.py and read out.outputs[0].text.
        outputs = engine.generate([req], sampling_params=sp, use_tqdm=False)
        return outputs[0].outputs[0].text

    return _gen


def transformers_generate_fn(model, processor, tokenizer, max_new_tokens: int = 96) -> GenerateFn:
    """CPU/GPU transformers GenerateFn — parity path against the offline eval decoder
    (eval/streaming_replay_eval.py::StreamDecoder.step). TODO(GPU): loads the model."""
    import torch

    def _gen(prompt: str, audio: np.ndarray) -> str:
        feed = audio
        min_len = int(0.3 * SR)                    # feature extractor reflection pad
        if len(feed) < min_len:
            feed = np.concatenate([feed, np.zeros(min_len - len(feed), dtype=np.float32)])
        inputs = processor(text=[prompt], audio=[feed], return_tensors="pt", padding=True)
        device = next(model.parameters()).device
        inputs = {k: (v.to(device).to(model.dtype) if torch.is_floating_point(v)
                      else v.to(device)) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        return tokenizer.decode(out.sequences[0, inputs["input_ids"].shape[1]:],
                                skip_special_tokens=False)

    return _gen


def build_engine(model_dir: str, gpu_memory_utilization: float = 0.30,
                 max_model_len: int = 8192):
    """TODO(GPU): construct the resident vLLM engine for the merged endpoint
    checkpoint. Runs ONLY in the GPU-owning process.

        from vllm import LLM
        return LLM(model=model_dir, trust_remote_code=True,
                   gpu_memory_utilization=gpu_memory_utilization,
                   max_model_len=max_model_len, enable_prefix_caching=True,
                   limit_mm_per_prompt={"audio": 1})   # ONE segment per request

    ``model_dir`` is the output of scripts/merge_endpoint_lora.py (a standard
    Qwen3-ASR HF dir; vLLM registers Qwen3ASRForConditionalGeneration). For the
    Metronome async worker use AsyncLLM as in worker/stream_server.py instead.
    Do NOT instantiate a second engine while another GPU job holds the card.
    """
    raise NotImplementedError("build_engine must be run on the GPU by the main process")
