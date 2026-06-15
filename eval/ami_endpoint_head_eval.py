"""Eval EndpointHead on real AMI conversational audio.

Matches eval/ami_conversational_eval.py protocol but evaluates the
acoustic head instead of the LM end tokens. Walks AMI parquet from the
HELD-OUT meeting split, builds 50 single / 50 double-turn / 50
disfluency examples (same joining as the LM eval), and counts fire
events in each.

Targets to compare against v3 / v5:
  - SINGLE: head fires ≥ 1 time (= v3's 100% / v5's 82%)
  - DOUBLE: head fires ≥ 2 distinct times (= v3's 90% / v5's 76%)
  - DISFLUENCY: head fires exactly 1 time (= v3's 82% correct)

Usage:
    python -m eval.ami_endpoint_head_eval \\
        --ckpt checkpoints/endpoint_head_v2.pt \\
        --restrict-to-meetings data/semantic_endpoint_v5/meeting_split.json \\
        --max-per-schema 50 \\
        --tau 0.5 --k 2
"""

from __future__ import annotations
import argparse
import io
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

from src.aut_encoder import load_aut_from_safetensors
from src.endpoint_head import EndpointHead
from src.features import log_mel
from src.endpoint_eval_v4 import k_consec_runs

logger = logging.getLogger(__name__)
FRAME_S = 0.08
SR = 16000


def decode_audio(blob: dict):
    if "array" in blob and blob["array"] is not None:
        return np.asarray(blob["array"], dtype=np.float32), int(blob.get("sampling_rate", SR))
    if "bytes" in blob:
        audio, sr = sf.read(io.BytesIO(blob["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        return audio.astype(np.float32), int(sr)
    return None, None


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/endpoint_head_v2.pt")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--ami-dir", default="data/ami/ihm")
    p.add_argument("--restrict-to-meetings", required=True)
    p.add_argument("--max-per-schema", type=int, default=50)
    p.add_argument("--max-dur-s", type=float, default=12.0)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--refractory-s", type=float, default=1.5)
    p.add_argument("--out", default="research/55-ami-endpoint-head-eval.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    with open(args.restrict_to_meetings) as f:
        msplit = json.load(f)
    eval_meetings = set(msplit["eval_meeting_ids"])
    logger.info("Eval meetings: %d", len(eval_meetings))

    logger.info("Loading AuT…")
    aut = load_aut_from_safetensors(
        "data/qwen3-asr-0.6b/model.safetensors",
        device="cuda", dtype=torch.bfloat16,
    )

    logger.info("Loading head from %s", args.ckpt)
    state = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    head = EndpointHead(d_audio=1024, hidden=args.hidden).cuda().bfloat16()
    head.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    head.eval()

    # Walk AMI parquet, collect utts grouped by meeting
    meetings: dict[str, list[dict]] = defaultdict(list)
    for pq_path in sorted(Path(args.ami_dir).glob("*.parquet")):
        t = pq.read_table(pq_path)
        for row in t.to_pylist():
            if row["meeting_id"] in eval_meetings:
                meetings[row["meeting_id"]].append(row)
    for mid in meetings:
        meetings[mid].sort(key=lambda r: r["begin_time"])

    refr = int(args.refractory_s / FRAME_S)

    def encode_and_fire(audio: np.ndarray) -> tuple[list[int], int]:
        mel = log_mel(torch.from_numpy(audio))
        with torch.no_grad():
            aut_out = aut(mel.unsqueeze(0).cuda().bfloat16())
            _, end_log = head(aut_out)
            p = torch.sigmoid(end_log).squeeze(0).squeeze(-1)
        events = k_consec_runs(p, args.tau, args.k, refractory=refr)
        return events, p.shape[0]

    singles = []
    doubles = []
    disfluencies = []
    per_example = []

    # Build singles and pair-derived examples
    for mid, utts in meetings.items():
        if len(utts) < 2:
            continue
        # Singles
        for utt in utts:
            if len(singles) >= args.max_per_schema:
                break
            audio, sr = decode_audio(utt["audio"])
            if audio is None or sr != SR or not utt["text"].strip():
                continue
            if len(audio) > int(args.max_dur_s * SR) or len(audio) == 0:
                continue
            events, T = encode_and_fire(audio)
            singles.append({"n_fires": len(events), "events": events,
                            "audio_frames": T, "meeting_id": mid,
                            "text": utt["text"].strip()})
            per_example.append({"schema": "single", "n_fires": len(events),
                                  "meeting_id": mid, "events": events,
                                  "T": T, "text": utt["text"][:80]})
        # Pairs
        for i in range(len(utts) - 1):
            if len(doubles) >= args.max_per_schema and len(disfluencies) >= args.max_per_schema:
                break
            a, b = utts[i], utts[i + 1]
            if not a["text"].strip() or not b["text"].strip():
                continue
            gap = float(b["begin_time"]) - float(a["end_time"])
            if not (0.05 < gap < 3.0):
                continue
            aa, sra = decode_audio(a["audio"])
            ab, srb = decode_audio(b["audio"])
            if aa is None or ab is None or sra != SR or srb != SR:
                continue
            total_dur = (len(aa) + len(ab)) / SR + gap
            if total_dur > args.max_dur_s or len(aa) == 0 or len(ab) == 0:
                continue
            gap_samples = int(gap * SR)
            cat = np.concatenate([aa, np.zeros(gap_samples, dtype=np.float32), ab])
            events, T = encode_and_fire(cat)
            is_double = a["speaker_id"] != b["speaker_id"]
            kind = "double" if is_double else "disfluency"
            tgt_list = doubles if is_double else disfluencies
            if len(tgt_list) >= args.max_per_schema:
                continue
            tgt_list.append({"n_fires": len(events), "events": events,
                              "audio_frames": T, "meeting_id": mid,
                              "gap_s": gap})
            per_example.append({"schema": kind, "n_fires": len(events),
                                  "meeting_id": mid, "events": events,
                                  "T": T, "gap_s": gap,
                                  "text_a": a["text"][:60], "text_b": b["text"][:60]})

    # Summaries
    def summarize(lst, target_min: int, target_max: int | None = None):
        if not lst:
            return {}
        n = len(lst)
        n_hit = sum(1 for x in lst if x["n_fires"] >= target_min
                     and (target_max is None or x["n_fires"] <= target_max))
        return {"n": n, "n_hit": n_hit, "hit_rate": n_hit / n,
                 "avg_n_fires": sum(x["n_fires"] for x in lst) / n}

    single_summary = summarize(singles, target_min=1)
    # Disfluency = exactly 1 fire is correct (single speaker continues then ends)
    disfl_summary = summarize(disfluencies, target_min=1, target_max=1)
    disfl_overfire = sum(1 for x in disfluencies if x["n_fires"] >= 2) / max(1, len(disfluencies))
    double_summary = summarize(doubles, target_min=2)

    summary = {
        "tau": args.tau, "k": args.k, "refractory_s": args.refractory_s,
        "n_meetings_used": len(meetings),
        "single": single_summary,
        "double_turn_detection": double_summary,
        "disfl_correct_1_fire": disfl_summary,
        "disfl_overfire_rate": disfl_overfire,
    }

    logger.info("Summary:")
    logger.info("  single  hit=%.2f (n=%d, avg fires=%.2f)",
                 single_summary.get("hit_rate", 0), single_summary.get("n", 0),
                 single_summary.get("avg_n_fires", 0))
    logger.info("  double  hit=%.2f (n=%d, avg fires=%.2f)",
                 double_summary.get("hit_rate", 0), double_summary.get("n", 0),
                 double_summary.get("avg_n_fires", 0))
    logger.info("  disfl   correct=%.2f overfire=%.2f (n=%d, avg fires=%.2f)",
                 disfl_summary.get("hit_rate", 0), disfl_overfire,
                 disfl_summary.get("n", 0), disfl_summary.get("avg_n_fires", 0))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "summary": summary,
        "per_example": per_example,
    }, indent=2))
    logger.info("Saved %s", args.out)


if __name__ == "__main__":
    main()
