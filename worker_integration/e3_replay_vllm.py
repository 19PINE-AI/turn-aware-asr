"""E3 driver — run the research/58 streaming-replay eval on the vLLM committed-prefix
path and compare per-chunk compute to the 455 ms/chunk transformers simulator.

Design: research/64-metronome-worker-integration.md. This REUSES the exact material
builder, boundary classification, and fire/WER scoring from
eval/streaming_replay_eval.py (load_meetings / build_stretches / run_stretch /
aggregate), and only swaps the per-chunk decode from transformers to a resident vLLM
engine — so the numbers are byte-identical in scoring to the re-baseline (research/61)
and the ONLY difference is the serving path. That is what E3 must isolate.

  * ``VLLMStreamDecoder`` subclasses the eval's ``StreamDecoder`` and overrides
    ``step()`` to build the single-segment prompt and call ``engine.generate`` instead
    of ``model.generate`` — same committed-prefix rollback, same buffer, so
    ``run_stretch`` drives it unchanged (identical energy-gate / confirm / max-segment
    logic and fire scoring).
  * ``compute_ms`` per chunk is collected and summarised (median / P95) against the
    455 ms/chunk simulator baseline.

HARD CONSTRAINT: this instantiates a vLLM engine and USES THE GPU. Do NOT run it in a
sandboxed/parallel job while another GPU process holds the card — it will OOM-crash the
resident engine (research/60 GPU-contention risk). Run only from the main GPU-owning
process. See the exact command in research/64.
"""

from __future__ import annotations
import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np

from eval.streaming_replay_eval import (
    SR, load_meetings, build_stretches, run_stretch, aggregate,
    build_prompt, clean, StreamDecoder,
)

logger = logging.getLogger(__name__)


class VLLMStreamDecoder(StreamDecoder):
    """StreamDecoder whose decode step runs on a resident vLLM engine (single audio
    segment per request — the bounded-re-feed shape, NOT chunked placeholders)."""

    def __init__(self, engine, sampling_params, tokenizer, committed: bool,
                 rollback: int = 5, max_new: int = 96):
        # Skip StreamDecoder.__init__ (which wants an HF model/processor); set only
        # what step() and run_stretch() touch.
        self.engine = engine
        self.sampling_params = sampling_params
        self.tokenizer = tokenizer
        self.committed = committed
        self.rollback = rollback
        self.max_new = max_new
        self.prompt = build_prompt(tokenizer)   # <|im_start|>system…assistant\n
        self.raw = ""
        self.buffer = None
        self.compute_ms = []

    def step(self, chunk: np.ndarray) -> str:
        self.buffer = chunk if self.buffer is None else np.concatenate([self.buffer, chunk])
        prefix = ""
        if self.committed and self.raw:
            ids = self.tokenizer.encode(self.raw)
            k = self.rollback
            while True:
                end = max(0, len(ids) - k)
                prefix = self.tokenizer.decode(ids[:end]) if end else ""
                if "�" not in prefix:
                    break
                if end == 0:
                    prefix = ""
                    break
                k += 1
        prompt = self.prompt + prefix
        req = {"prompt": prompt, "multi_modal_data": {"audio": [self.buffer]}}
        t0 = time.perf_counter()
        outputs = self.engine.generate([req], sampling_params=self.sampling_params,
                                       use_tqdm=False)
        self.compute_ms.append((time.perf_counter() - t0) * 1000.0)
        gen = outputs[0].outputs[0].text
        self.raw = (prefix + gen) if self.committed else gen
        return clean(self.raw)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True,
                   help="merged endpoint checkpoint (scripts/merge_endpoint_lora.py out-dir)")
    p.add_argument("--ami-dir", default="data/ami/ihm")
    p.add_argument("--split", default="data/semantic_endpoint_v3/meeting_split.json")
    p.add_argument("--n-stretches", type=int, default=25)
    p.add_argument("--chunk-s", type=float, default=0.5)
    p.add_argument("--from-scratch", action="store_true")
    p.add_argument("--energy-gate", action="store_true")
    p.add_argument("--gate-rms", type=float, default=1e-3)
    p.add_argument("--confirm-silent-chunks", type=int, default=0)
    p.add_argument("--max-segment-chunks", type=int, default=0)
    p.add_argument("--gpu-mem", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rng = random.Random(args.seed)
    split = json.loads(Path(args.split).read_text())
    restrict = set(split["eval_meeting_ids"])
    meetings = load_meetings(Path(args.ami_dir), restrict)
    stretches = build_stretches(meetings, rng, args.n_stretches)
    logger.info("Built %d stretches (%.1f min)", len(stretches),
                sum(s["duration_s"] for s in stretches) / 60)

    # TODO(GPU): build the resident vLLM engine (uses the GPU; run on the main process)
    from vllm import LLM, SamplingParams
    engine = LLM(model=args.model_dir, trust_remote_code=True,
                 gpu_memory_utilization=args.gpu_mem, max_model_len=8192,
                 enable_prefix_caching=True, limit_mm_per_prompt={"audio": 1})
    tokenizer = engine.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=96, ignore_eos=False)

    per, all_ms = [], []
    for st in stretches:
        dec = VLLMStreamDecoder(engine, sp, tokenizer, committed=not args.from_scratch)
        per.append(run_stretch(dec, st, args.chunk_s,
                               energy_gate=args.energy_gate, gate_rms=args.gate_rms,
                               confirm_chunks=args.confirm_silent_chunks,
                               max_segment_chunks=args.max_segment_chunks))
        all_ms.extend(dec.compute_ms)

    summary = aggregate(per, args.chunk_s, "vllm_committed")
    ms = np.array(all_ms) if all_ms else np.array([0.0])
    summary["chunk_compute_ms_median"] = float(np.median(ms))
    summary["chunk_compute_ms_p95"] = float(np.percentile(ms, 95))
    summary["chunk_compute_ms_vs_sim_455"] = float(np.median(ms) / 455.0)
    logger.info("== E3 vLLM REPLAY == median %.0f ms / chunk (sim baseline 455 ms), P95 %.0f ms",
                summary["chunk_compute_ms_median"], summary["chunk_compute_ms_p95"])
    for k, v in summary.items():
        logger.info("  %-32s %s", k, f"{v:.3f}" if isinstance(v, float) else v)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"model_dir": args.model_dir, "summary": summary, "per_stretch": per}, indent=2))
    logger.info("Saved %s", args.out)


if __name__ == "__main__":
    main()
