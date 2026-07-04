"""E5 — schedulable-concurrency / throughput measurement for the streaming endpoint.

The Metronome question (paper §admission): for ONE resident vLLM engine serving the
merged v9 endpoint checkpoint with continuous batching, how many concurrent streaming
ASR sessions ``N`` can we run before the per-FRAME wall time (one batched decode step
across all due sessions) exceeds the frame budget? The largest ``N`` whose p95 per-frame
time stays under the 500 ms frame period is our estimate of ``N*`` — sessions per
Blackwell.

What this measures (vs E3)
--------------------------
E3 (research/65) measured SINGLE-stream per-chunk compute: 84 ms median on vLLM at
gpu_mem 0.15. That is the N=1 point. E5 sweeps N and drives all N sessions through ONE
engine so vLLM's continuous batching coalesces the N due requests of a frame into a
single ``engine.generate([...])`` step — exactly what the online worker does. We time
that batched step. Per-frame time grows sub-linearly with N (batching amortizes prefill /
kernel-launch overhead) until the engine saturates; that knee is N*.

Serving primitive (fixed by E2/E3, research/65)
-----------------------------------------------
BOUNDED RE-FEED, one audio segment per session per frame: each frame a session presents
the last ``window_s`` seconds of its audio as ONE ``<|audio_*|>`` placeholder (NOT one
placeholder per chunk — E2 showed the chunked shape gives 88 % WER; it is wildly OOD for
Qwen3-ASR). Committed-prefix decoding (rollback K) continues the transcript. On a terminal
``<END_SPEECH>`` the segment flushes and its audio is dropped; a ``max_buffer_s`` cap
force-flushes during continuous speech. ``skip_special_tokens=False`` so the marker
special tokens (151705/151706) survive detok. This session logic mirrors
``worker_integration/qwen3asr_stream.py::Qwen3ASRStreamSession`` — it is replicated here
in a two-phase ``prepare()`` / ``commit()`` form so the driver can collect all N frame
prompts FIRST and then issue ONE batched generate (the coupled ``feed()`` calls generate
inline, which would serialize the sessions and defeat the whole measurement).

Distinct audio per session, phase-staggered
--------------------------------------------
Each session replays a distinct AMI single-speaker stretch (``build_stretches`` from the
research/58 eval), and a per-session phase offset rotates its start point in the stream.
Distinct audio => distinct AuT embeddings => distinct KV, so prefix caching cannot
artificially collapse the batch onto a shared prefix (Metronome methodology: independent
streams). Sessions wrap their audio so N stays constant for the whole frame window.

!!! GPU / CO-TENANT CAVEAT !!!
-----------------------------
This instantiates a vLLM engine and USES THE GPU — run ONLY from the main (GPU-owning)
process. Do NOT launch it as a second heavy GPU job while another engine holds the card:
a co-resident engine OOM-crashes (research/60 GPU-contention risk). Moreover, if the GPU
has a co-tenant when E5 runs, the ABSOLUTE latencies are contaminated by contention — the
SHAPE (latency growth with N) and the RELATIVE N* remain informative, but a CALIBRATED,
citable N* needs a SOLO GPU. The default ``--gpu-mem 0.15`` matches E3 so N=1 is
comparable; raise it toward the free fraction of the card for a realistic solo N*.

The engine build and ``vllm`` import are INSIDE ``main()`` behind ``TODO(GPU)`` — importing
this module never touches the GPU. ``--dry-run`` exercises the entire scheduling / prompt /
batching path on CPU with a fake generator and a fake tokenizer (no GPU, no vLLM).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Reuse the exact bounded-re-feed prompt/marker helpers from the worker session module
# (single source of truth for the chat prefix and marker handling).
from worker_integration.qwen3asr_stream import (
    SR, END_TOK, EAGER_TOK, build_chat_prefix, strip_meta, vllm_sampling_params,
)
# Reuse the research/58 material builder so E5 drives the same real AMI streams as E3.
from eval.streaming_replay_eval import load_meetings, build_stretches

logger = logging.getLogger(__name__)

# A batched decoder: given a list of vLLM-shaped requests, return one completion string
# per request (same order). The real one wraps ``engine.generate([...])`` (continuous
# batching); the dry-run one is a CPU fake. This is the ONE call whose wall time we time.
BatchGenerateFn = Callable[[list], list]


# --------------------------------------------------------------------------- session

class E5Session:
    """One bounded-re-feed streaming session, split into ``prepare()`` (build this frame's
    request) and ``commit(gen)`` (apply the model output) so the driver can batch the
    generate across all sessions. Semantics mirror
    ``qwen3asr_stream.Qwen3ASRStreamSession.feed()`` (bounded buffer, energy gate,
    committed prefix, terminal-marker flush, max_buffer force-flush)."""

    def __init__(self, sid: int, audio: np.ndarray, tokenizer, *, context: str = "",
                 window_s: float = 16.0, chunk_s: float = 0.5, max_buffer_s: float = 30.0,
                 overlap_s: float = 0.75, energy_gate: bool = True, gate_rms: float = 1e-3,
                 rollback: int = 5, unfixed_chunks: int = 2, start_frame: int = 0):
        self.sid = sid
        self.audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        self.tokenizer = tokenizer
        self.prompt_prefix = build_chat_prefix(context)
        self.window_s = window_s
        self.chunk_s = chunk_s
        self.max_buffer_s = max_buffer_s
        self.overlap_s = overlap_s
        self.energy_gate = energy_gate
        self.gate_rms = gate_rms
        self.rollback = rollback
        self.unfixed_chunks = unfixed_chunks

        self._n_src = max(1, int(np.ceil(len(self.audio) / (chunk_s * SR))))
        self._pos = int(start_frame) % self._n_src   # read head into own audio (wraps)
        self.buffer: Optional[np.ndarray] = None
        self.raw = ""
        self.seg_chunks = 0
        # metrics
        self.n_decodes = 0
        self.n_gated = 0

    # -- audio source (wraps so the session stays active for the whole frame window) --
    def _next_chunk(self) -> np.ndarray:
        k = self._pos
        self._pos = (self._pos + 1) % self._n_src
        a, b = int(k * self.chunk_s * SR), int((k + 1) * self.chunk_s * SR)
        return self.audio[a:b]

    # -- committed prefix (utf-8-safe rollback of the last K tokens) ------------------
    def _committed_prefix(self) -> str:
        if self.seg_chunks < self.unfixed_chunks or not self.raw:
            return ""
        ids = self.tokenizer.encode(self.raw)
        k = self.rollback
        while True:
            end = max(0, len(ids) - k)
            prefix = self.tokenizer.decode(ids[:end]) if end else ""
            if "�" not in prefix:
                return prefix
            if end == 0:
                return ""
            k += 1

    def _append_bounded(self, chunk: np.ndarray):
        self.buffer = chunk if self.buffer is None else np.concatenate([self.buffer, chunk])
        cap = int(self.window_s * SR)
        if len(self.buffer) > cap:
            self.buffer = self.buffer[-cap:]

    def _reset_segment(self, keep_overlap: bool = False):
        tail = None
        if keep_overlap and self.buffer is not None and self.overlap_s > 0:
            n = int(self.overlap_s * SR)
            tail = self.buffer[-n:] if len(self.buffer) > n else self.buffer
        self.raw = ""
        self.seg_chunks = 0
        self.buffer = tail

    # -- phase A: build this frame's request, or None if the energy gate skips it -----
    def prepare(self) -> Optional[dict]:
        chunk = np.asarray(self._next_chunk(), dtype=np.float32).reshape(-1)
        silent = len(chunk) == 0 or float(np.sqrt(np.mean(chunk ** 2))) < self.gate_rms
        # Never START a segment on a silent chunk (research/61 gate): no decode this frame.
        if self.energy_gate and self.buffer is None and silent:
            self.n_gated += 1
            return None
        self._append_bounded(chunk)
        prefix = self._committed_prefix()
        prompt = self.prompt_prefix + prefix
        self._pending_prefix = prefix
        # vLLM single-segment multimodal request (bounded re-feed): ONE audio placeholder
        # over the whole bounded buffer, matching qwen3asr_stream.vllm_generate_fn.
        return {"prompt": prompt, "multi_modal_data": {"audio": [(self.buffer, SR)]}}

    # -- phase B: apply the model completion; update buffer bounding (flush) ----------
    def commit(self, gen: str):
        prefix = getattr(self, "_pending_prefix", "")
        self.raw = (prefix + gen) if prefix else gen
        self.seg_chunks += 1
        self.n_decodes += 1
        transcript = strip_meta(self.raw)
        # Terminal <END_SPEECH> -> clean flush: drop the segment's audio (utterance bound).
        if END_TOK in transcript:
            after = transcript.rsplit(END_TOK, 1)[1].replace(EAGER_TOK, "").strip()
            if not after:
                self._reset_segment(keep_overlap=False)
                return
        # Continuous-speech safety: force-flush past max_buffer_s (bounds committed prefix).
        if self.max_buffer_s and self.buffer is not None and \
                len(self.buffer) >= self.max_buffer_s * SR:
            self._reset_segment(keep_overlap=True)


# --------------------------------------------------------------------------- driver

def _pct(xs, q) -> float:
    return float(np.percentile(np.asarray(xs, dtype=float), q)) if len(xs) else 0.0


def run_sweep(stretches, tokenizer, generate_batch: BatchGenerateFn, *, ns, frame_s,
              window_s, max_frames, frame_budget_ms, energy_gate, gate_rms,
              max_buffer_s) -> dict:
    """For each N: build N phase-staggered sessions, step ``max_frames`` global frames,
    and per frame issue ONE batched generate over all due (non-gated) sessions, timing
    that step. Returns per-N summaries + the N* estimate."""
    per_n = []
    for n in ns:
        # distinct stretch per session; phase-offset the read head so reused stretches
        # (N > #stretches) and even same-length streams don't line up -> no shared prefix.
        sessions = []
        stagger = max(1, max_frames // max(1, n))
        for i in range(n):
            st = stretches[i % len(stretches)]
            sessions.append(E5Session(
                i, st["audio"], tokenizer, window_s=window_s, chunk_s=frame_s,
                max_buffer_s=max_buffer_s, energy_gate=energy_gate, gate_rms=gate_rms,
                start_frame=(i * stagger + i) % max(1, int(np.ceil(len(st["audio"]) /
                            (frame_s * SR))))))

        frame_ms: list[float] = []          # batched-step wall time per frame
        batch_sizes: list[int] = []
        session_lat: list[float] = []        # per (session,frame) participation latency
        for _ in range(max_frames):
            due = []
            for s in sessions:
                req = s.prepare()
                if req is not None:
                    due.append((s, req))
            if not due:                       # all sessions gated this frame -> no step
                continue
            reqs = [r for _, r in due]
            t0 = time.perf_counter()
            gens = generate_batch(reqs)       # THE batched continuous-batching step
            dt = (time.perf_counter() - t0) * 1000.0
            frame_ms.append(dt)
            batch_sizes.append(len(due))
            for (s, _), g in zip(due, gens):
                s.commit(g)
                session_lat.append(dt)        # each due session waited the full batch step

        p50, p95 = _pct(frame_ms, 50), _pct(frame_ms, 95)
        summary = {
            "n": n,
            "n_frames_decoded": len(frame_ms),
            "mean_batch_size": float(np.mean(batch_sizes)) if batch_sizes else 0.0,
            "frame_ms_p50": p50,
            "frame_ms_p95": p95,
            "frame_ms_mean": float(np.mean(frame_ms)) if frame_ms else 0.0,
            "frame_ms_max": float(np.max(frame_ms)) if frame_ms else 0.0,
            "per_session_lat_ms_p50": _pct(session_lat, 50),
            "per_session_lat_ms_p95": _pct(session_lat, 95),
            "frame_budget_ms": frame_budget_ms,
            "within_budget_p95": bool(p95 <= frame_budget_ms),
        }
        logger.info("N=%-3d frames=%-4d batch=%.1f  frame_ms p50=%.0f p95=%.0f (budget %.0f) -> %s",
                    n, summary["n_frames_decoded"], summary["mean_batch_size"], p50, p95,
                    frame_budget_ms, "OK" if summary["within_budget_p95"] else "OVER")
        per_n.append(summary)

    within = [s["n"] for s in per_n if s["within_budget_p95"]]
    n_star = max(within) if within else 0
    return {"per_n": per_n, "n_star_estimate": n_star}


# --------------------------------------------------------------------------- dry-run

class _FakeTokenizer:
    """Char-level tokenizer for the CPU dry-run (exercises the committed-prefix rollback
    loop without loading the real HF tokenizer / touching the GPU)."""
    def encode(self, s: str):
        return list(s)

    def decode(self, ids):
        return "".join(ids)


def _fake_generate_batch(reqs: list) -> list:
    """CPU fake for the batched engine step. Does a little audio work per request so the
    batch does something real, and NEVER emits markers (so buffers grow to the window /
    max_buffer cap — the steady-state worst case for compute). Absolute times here are
    meaningless; this validates the SCHEDULING/PROMPT/BATCHING logic only."""
    outs = []
    for r in reqs:
        audio, sr = r["multi_modal_data"]["audio"][0]
        _ = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0  # touch audio
        outs.append("language English<asr_text> hello world")
    return outs


# --------------------------------------------------------------------------- main

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default="checkpoints/merged/qwen3-asr-0.6b-endpoint-v9",
                   help="merged v9 endpoint checkpoint (scripts/merge_endpoint_lora.py out)")
    p.add_argument("--ami-dir", default="data/ami/ihm")
    p.add_argument("--split", default="data/semantic_endpoint_v3/meeting_split.json")
    p.add_argument("--ns", default="1,2,4,8,16", help="concurrency sweep, comma-separated")
    p.add_argument("--frame-s", type=float, default=0.5, help="frame period / chunk seconds")
    p.add_argument("--window-s", type=float, default=16.0, help="bounded re-feed window")
    p.add_argument("--max-buffer-s", type=float, default=30.0)
    p.add_argument("--max-frames", type=int, default=120, help="frames per N (120=60s)")
    p.add_argument("--gpu-mem", type=float, default=0.15, help="gpu_memory_utilization")
    p.add_argument("--energy-gate", action="store_true", default=True)
    p.add_argument("--no-energy-gate", dest="energy_gate", action="store_false")
    p.add_argument("--gate-rms", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="research/68-e5-concurrency.json")
    p.add_argument("--dry-run", action="store_true",
                   help="CPU-only: fake generate + fake tokenizer, no vLLM/GPU")
    args = p.parse_args()

    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    frame_budget_ms = args.frame_s * 1000.0

    # Material: distinct AMI single-speaker stretches (one per session). Request at least
    # max(ns) so every session up to the largest N gets a distinct stream.
    rng = random.Random(args.seed)
    split = json.loads(Path(args.split).read_text())
    restrict = set(split["eval_meeting_ids"])
    meetings = load_meetings(Path(args.ami_dir), restrict)
    stretches = build_stretches(meetings, rng, max(ns) if ns else 1)
    if not stretches:
        raise SystemExit("no AMI stretches built — check --ami-dir / --split")
    logger.info("Built %d stretches (%.1f min) for sweep ns=%s",
                len(stretches), sum(s["duration_s"] for s in stretches) / 60, ns)

    if args.dry_run:
        logger.info("DRY-RUN: fake batched generate + fake tokenizer (no GPU / no vLLM)")
        tokenizer = _FakeTokenizer()
        generate_batch = _fake_generate_batch
    else:
        # TODO(GPU): everything below touches the GPU. Run ONLY from the main GPU-owning
        # process (co-resident engine -> OOM; co-tenant -> contaminated absolute latency).
        from vllm import LLM  # noqa: local import; vLLM only on the GPU box
        engine = LLM(model=args.model_dir, trust_remote_code=True,
                     gpu_memory_utilization=args.gpu_mem, max_model_len=8192,
                     enable_prefix_caching=True, limit_mm_per_prompt={"audio": 1})
        tokenizer = engine.get_tokenizer()
        sp = vllm_sampling_params(max_tokens=96)   # skip_special_tokens=False (markers)

        def generate_batch(reqs):
            outs = engine.generate(reqs, sampling_params=sp, use_tqdm=False)
            return [o.outputs[0].text for o in outs]

    result = run_sweep(
        stretches, tokenizer, generate_batch, ns=ns, frame_s=args.frame_s,
        window_s=args.window_s, max_frames=args.max_frames, frame_budget_ms=frame_budget_ms,
        energy_gate=args.energy_gate, gate_rms=args.gate_rms, max_buffer_s=args.max_buffer_s)

    result.update({
        "model_dir": args.model_dir,
        "gpu_mem": args.gpu_mem,
        "frame_s": args.frame_s,
        "window_s": args.window_s,
        "max_frames": args.max_frames,
        "n_stretches": len(stretches),
        "dry_run": args.dry_run,
        "caveat": ("dry-run: absolute latencies are meaningless (fake CPU generate); only "
                   "the scheduling/batching logic is validated." if args.dry_run else
                   "If the GPU had a co-tenant, absolute latencies are contaminated by "
                   "contention; the SHAPE (growth with N) and relative N* are still "
                   "informative, a calibrated N* needs a solo GPU."),
    })
    logger.info("== E5 == N* estimate (largest N with p95 frame time <= %.0f ms): %d",
                frame_budget_ms, result["n_star_estimate"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    logger.info("Saved %s", args.out)


if __name__ == "__main__":
    main()
