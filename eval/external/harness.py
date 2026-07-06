"""Exp-4 external-baseline driver: run any turn-aware system on our replay set.

Loads the EXACT benchmark stretches (eval.streaming_replay_eval.build_stretches,
same seed/split/n-stretches as the LM and timeout arms), feeds each stretch to a
named adapter, collects fire timestamps, and scores them with the byte-identical
`score_fires` + aggregation used everywhere else. Output JSON matches
eval/timeout_baseline_eval.py's shape:

    {"system", "seed", "n_stretches", "license_note",
     "arms": {arm_key: {"summary": {...}, "per_stretch": [...]}}}

so figure code and the main table ingest it unchanged. One adapter may emit
several arms (a threshold sweep) from a single model pass.

This driver does model inference (GPU where the adapter uses it). Run it inside
the adapter's dedicated venv — see eval/external/README.md. Stretch building is
CPU/IO only.

Usage:
    <venv>/bin/python -m eval.external.harness \
        --adapter smart_turn --n-stretches 25 --seed 0 \
        --out research/exp4-smart_turn-seed0.json
"""
from __future__ import annotations
import argparse
import importlib
import json
import logging
import random
import time
from pathlib import Path

import numpy as np

from eval.streaming_replay_eval import load_meetings, build_stretches, score_fires

logger = logging.getLogger(__name__)

ADAPTERS = {
    "dummy_rms": "eval.external.adapters.dummy_rms:DummyRMSAdapter",
    "smart_turn": "eval.external.adapters.smart_turn:SmartTurnAdapter",
    "parakeet_eou": "eval.external.adapters.parakeet_eou:ParakeetEOUAdapter",
    "kyutai_vad": "eval.external.adapters.kyutai_vad:KyutaiVADAdapter",
    "livekit_turn": "eval.external.adapters.livekit_turn:LiveKitTurnAdapter",
}


def load_adapter(name: str):
    spec = ADAPTERS.get(name, name)
    mod_name, cls_name = spec.split(":")
    return getattr(importlib.import_module(mod_name), cls_name)()


def _try_wer(hyp: str, ref: str):
    try:
        from eval.metrics import wer
        return wer(hyp, ref)
    except Exception:  # jiwer missing in a lean venv, or empty transcript
        return None


def aggregate(per: list[dict], chunk_s: float, arm_key: str, system: str) -> dict:
    lat = sorted(l for r in per for l in r["latencies_s"])
    n_turn = sum(r["n_turn_bounds"] for r in per)
    n_cont = sum(r["n_cont_bounds"] for r in per)
    speech_min = sum(r["speech_s"] for r in per) / 60.0
    summ = {
        "system": system, "arm": arm_key, "chunk_s": chunk_s,
        "n_stretches": len(per), "n_turn_bounds": n_turn, "n_cont_bounds": n_cont,
        "boundary_recall": sum(r["n_hits"] for r in per) / max(1, n_turn),
        "false_fires_per_speech_min":
            sum(r["n_false_fires"] for r in per) / max(1e-9, speech_min),
        "resume_after_fire_rate":
            sum(r["n_resume_fires"] for r in per) / max(1, n_cont),
        "dup_or_silence_fires_total": sum(r["n_dup_or_silence_fires"] for r in per),
    }
    if lat:
        summ["latency_s_p50"] = lat[len(lat) // 2]
        summ["latency_s_p95"] = lat[int(0.95 * (len(lat) - 1))]
        summ["latency_s_mean"] = float(np.mean(lat))
    wers = [r["wer"] for r in per if r.get("wer") is not None]
    if wers:
        summ["wer_mean"] = float(np.mean(wers))
    return summ


def score_arm(stretches, fires_per_stretch, transcripts, chunk_s, arm_key, system):
    per = []
    for st, fires, tr in zip(stretches, fires_per_stretch, transcripts):
        sc = score_fires(fires, st["events"])
        rec = {
            "meeting_id": st["meeting_id"], "speaker_id": st["speaker_id"],
            "n_turn_bounds": len(sc["turn_bounds"]),
            "n_cont_bounds": len(sc["cont_bounds"]),
            "n_fires": len(fires), "n_hits": len(sc["hits"]),
            "n_resume_fires": len(sc["resume_fires"]),
            "n_false_fires": len(sc["false_fires"]),
            "n_dup_or_silence_fires": len(sc["dup_fires"]),
            "latencies_s": sc["recalled"], "speech_s": sc["speech_s"],
            "fires_s": sorted(fires),
        }
        if tr is not None:
            ref = " ".join(e["text"] for e in st["events"])
            rec["wer"] = _try_wer(tr, ref)
        per.append(rec)
    return {"summary": aggregate(per, chunk_s, arm_key, system), "per_stretch": per}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, help=f"one of {list(ADAPTERS)} or module:Class")
    p.add_argument("--ami-dir", default="data/ami/ihm")
    p.add_argument("--split", default="data/semantic_endpoint_v3/meeting_split.json")
    p.add_argument("--n-stretches", type=int, default=25)
    p.add_argument("--chunk-s", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-train-meetings", action="store_true",
                   help="dev/smoke: draw stretches from TRAIN meetings")
    p.add_argument("--limit", type=int, default=0,
                   help="smoke test: only run the first N stretches (0 = all)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rng = random.Random(args.seed)
    split = json.loads(Path(args.split).read_text())
    key = "train_meeting_ids" if args.use_train_meetings else "eval_meeting_ids"
    logger.info("Loading %s meetings from %s …", key, args.ami_dir)
    meetings = load_meetings(Path(args.ami_dir), set(split[key]))
    stretches = build_stretches(meetings, rng, args.n_stretches)
    if args.limit:
        stretches = stretches[:args.limit]
    logger.info("Built %d stretches: %.1f min, %d turn / %d cont bounds",
                len(stretches), sum(s["duration_s"] for s in stretches) / 60,
                sum(1 for s in stretches for e in s["events"] if e["boundary"] == "turn_final"),
                sum(1 for s in stretches for e in s["events"] if e["boundary"] == "continuation"))

    adapter = load_adapter(args.adapter)
    logger.info("Setting up adapter '%s' …", adapter.name)
    t0 = time.time()
    adapter.setup()
    logger.info("Adapter ready in %.1fs", time.time() - t0)

    # infer_stretch returns (arms, transcript); collect per-arm fire lists.
    arm_fires: dict[str, list] = {}
    transcripts: list = []
    for i, st in enumerate(stretches):
        # Optional hook: adapters that need ground-truth text / a transcript
        # source (e.g. LiveKit's text EOU classifier) receive the whole stretch
        # here. Audio-only adapters ignore it.
        if hasattr(adapter, "set_stretch"):
            adapter.set_stretch(st["events"],
                                key=f"{st['meeting_id']}:{st['speaker_id']}")
        arms, tr = adapter.infer_stretch(st["audio"], args.chunk_s)
        transcripts.append(tr)
        for arm_key, fires in arms.items():
            arm_fires.setdefault(arm_key, [[] for _ in stretches])
            arm_fires[arm_key][i] = list(fires)
        logger.info("stretch %d/%d (%s) fires: %s", i + 1, len(stretches),
                    st["meeting_id"], {k: len(v[i]) for k, v in arm_fires.items()})

    out = {
        "system": adapter.name, "seed": args.seed,
        "n_stretches": len(stretches), "chunk_s": args.chunk_s,
        "license_note": getattr(adapter, "license_note", ""),
        "arms": {},
    }
    for arm_key, fires_per in arm_fires.items():
        res = score_arm(stretches, fires_per, transcripts, args.chunk_s, arm_key, adapter.name)
        out["arms"][arm_key] = res
        s = res["summary"]
        logger.info("arm %-10s recall %.3f  P50 %s  false/min %.1f  resume %.2f  wer %s",
                    arm_key, s["boundary_recall"],
                    f"{s.get('latency_s_p50', float('nan')):.2f}s",
                    s["false_fires_per_speech_min"], s["resume_after_fire_rate"],
                    f"{s.get('wer_mean', float('nan')):.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    logger.info("Saved %s", args.out)


if __name__ == "__main__":
    main()
