"""E11: VAD / silence-timeout baselines on the streaming-replay eval.

The deployed alternative to semantic endpointing is a cascade: a speech/silence
detector + a fixed timeout — fire END after X seconds of continuous silence
following speech. This driver sweeps the timeout family through the SAME
stretches, boundary classification, and scoring as the LM arms
(eval/streaming_replay_eval.py::score_fires), producing the latency vs
recall/false-fire tradeoff curve the semantic model has to beat.

Two detectors:
  * rms    — energy threshold (the same RMS rule as the LM arms' --energy-gate,
             so the comparison isolates the POLICY, not the detector)
  * silero — Silero VAD v5 (the standard production VAD), per-chunk speech prob

Policy per timeout X: within a stream, once speech has been observed, when
continuous detected-silence since the last speech chunk reaches X seconds ->
fire once at that chunk boundary, then arm again on next speech. Decisions on
the same 0.5 s chunk grid as the LM arms.

No WER column: the cascade's ASR is a separate component; this measures the
endpoint policy only.

CPU-only. Usage:
    .venv/bin/python -m eval.timeout_baseline_eval \
        --detector rms --timeouts 0.5,1.0,1.5,2.0 \
        --n-stretches 25 --seed 0 --out research/71-timeout-rms-seed0.json
"""
from __future__ import annotations
import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np

from eval.streaming_replay_eval import (
    load_meetings, build_stretches, score_fires, SR,
)

logger = logging.getLogger(__name__)


def chunk_speech_flags(audio: np.ndarray, chunk_s: float, detector: str,
                       gate_rms: float, silero=None) -> list[bool]:
    """Per-chunk speech decision on the same grid the LM arms use."""
    n_chunks = int(np.ceil(len(audio) / (chunk_s * SR)))
    flags = []
    for k in range(n_chunks):
        chunk = audio[int(k * chunk_s * SR): int((k + 1) * chunk_s * SR)]
        if len(chunk) == 0:
            flags.append(False)
            continue
        if detector == "rms":
            flags.append(float(np.sqrt(np.mean(chunk ** 2))) >= gate_rms)
        else:  # silero: max window prob over the chunk
            import torch
            model, get_speech_ts = silero
            with torch.no_grad():
                # silero expects 512-sample windows at 16k; run get_speech_timestamps
                ts = get_speech_ts(torch.from_numpy(chunk), model,
                                   sampling_rate=SR, min_speech_duration_ms=60)
            flags.append(len(ts) > 0)
            model.reset_states()
    return flags


def run_timeout(flags: list[bool], chunk_s: float, timeout_s: float) -> list[float]:
    """Silence-timeout endpoint policy over per-chunk speech flags."""
    fires = []
    seen_speech = False
    silent_chunks = 0
    need = max(1, int(round(timeout_s / chunk_s)))
    for k, sp in enumerate(flags):
        if sp:
            seen_speech = True
            silent_chunks = 0
        else:
            if seen_speech:
                silent_chunks += 1
                if silent_chunks == need:
                    fires.append((k + 1) * chunk_s)   # same timestamp rule as LM arms
                    seen_speech = False               # re-arm on next speech
    return fires


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--detector", default="rms", choices=["rms", "silero"])
    p.add_argument("--timeouts", default="0.5,1.0,1.5,2.0")
    p.add_argument("--ami-dir", default="data/ami/ihm")
    p.add_argument("--split", default="data/semantic_endpoint_v3/meeting_split.json")
    p.add_argument("--n-stretches", type=int, default=25)
    p.add_argument("--chunk-s", type=float, default=0.5)
    p.add_argument("--gate-rms", type=float, default=1e-3)   # same as LM arms
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    timeouts = [float(x) for x in args.timeouts.split(",")]

    rng = random.Random(args.seed)
    split = json.loads(Path(args.split).read_text())
    meetings = load_meetings(Path(args.ami_dir), set(split["eval_meeting_ids"]))
    stretches = build_stretches(meetings, rng, args.n_stretches)
    logger.info("Built %d stretches (seed=%d): %.1f min, %d turn / %d cont bounds",
                len(stretches), args.seed,
                sum(s["duration_s"] for s in stretches) / 60,
                sum(1 for s in stretches for e in s["events"] if e["boundary"] == "turn_final"),
                sum(1 for s in stretches for e in s["events"] if e["boundary"] == "continuation"))

    silero = None
    if args.detector == "silero":
        from silero_vad import load_silero_vad, get_speech_timestamps
        silero = (load_silero_vad(), get_speech_timestamps)

    # Detector pass once per stretch; policy sweep reuses the flags.
    all_flags = [chunk_speech_flags(s["audio"], args.chunk_s, args.detector,
                                    args.gate_rms, silero) for s in stretches]

    arms = {}
    for X in timeouts:
        per = []
        for s, flags in zip(stretches, all_flags):
            fires = run_timeout(flags, args.chunk_s, X)
            sc = score_fires(fires, s["events"])
            per.append({
                "meeting_id": s["meeting_id"], "speaker_id": s["speaker_id"],
                "n_turn_bounds": len(sc["turn_bounds"]),
                "n_cont_bounds": len(sc["cont_bounds"]),
                "n_fires": len(fires), "n_hits": len(sc["hits"]),
                "n_resume_fires": len(sc["resume_fires"]),
                "n_false_fires": len(sc["false_fires"]),
                "n_dup_or_silence_fires": len(sc["dup_fires"]),
                "latencies_s": sc["recalled"], "speech_s": sc["speech_s"],
            })
        lat = sorted(l for r in per for l in r["latencies_s"])
        n_turn = sum(r["n_turn_bounds"] for r in per)
        n_cont = sum(r["n_cont_bounds"] for r in per)
        speech_min = sum(r["speech_s"] for r in per) / 60.0
        summ = {
            "detector": args.detector, "timeout_s": X, "chunk_s": args.chunk_s,
            "n_stretches": len(per), "n_turn_bounds": n_turn, "n_cont_bounds": n_cont,
            "boundary_recall": sum(r["n_hits"] for r in per) / max(1, n_turn),
            "false_fires_per_speech_min":
                sum(r["n_false_fires"] for r in per) / max(1e-9, speech_min),
            "resume_after_fire_rate":
                sum(r["n_resume_fires"] for r in per) / max(1, n_cont),
        }
        if lat:
            summ["latency_s_p50"] = lat[len(lat) // 2]
            summ["latency_s_p95"] = lat[int(0.95 * (len(lat) - 1))]
        arms[str(X)] = {"summary": summ, "per_stretch": per}
        logger.info("X=%.2fs: recall %.3f  P50 %s  false/min %.1f  resume %.2f",
                    X, summ["boundary_recall"],
                    f"{summ.get('latency_s_p50', float('nan')):.2f}s",
                    summ["false_fires_per_speech_min"], summ["resume_after_fire_rate"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "detector": args.detector, "seed": args.seed,
        "n_stretches": args.n_stretches, "arms": arms,
    }, indent=2))
    logger.info("Saved %s", args.out)


if __name__ == "__main__":
    main()
