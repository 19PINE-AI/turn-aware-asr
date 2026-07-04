"""Unified streaming-replay endpoint eval (design: research/58-unified-eval-design.md).

Replaces ami_conversational_eval.py (offline marker counting, silence-free
clips) and streaming_latency.py (synthetic LS, no real conversation) with
one deployment-matched protocol:

  * Material: continuous single-channel stretches from held-out AMI
    meetings — one target speaker's utterances placed at their true
    timeline offsets, silence in between. Audio never ends at a speech
    boundary; silence keeps arriving, as in deployment.
  * Ground truth: per utterance-end boundary, classified causally:
      - turn_final    (gap to same-speaker resume >= T_turn, or another
                       speaker interleaves): model SHOULD fire.
      - continuation  (0.3 <= gap < T_turn, same speaker, no interleave):
                       fire is a measured cost (resume_after_fire), NOT an
                       error — the correct label depends on future audio.
      - merged        (gap < 0.3 s): no boundary event; speech effectively
                       continues.
  * Protocol: feed CHUNK_S chunks; per chunk one incremental decode.
    Default committed-prefix decoding (qwen-asr official streaming
    semantics: emitted text is committed, last K tokens rolled back);
    --from-scratch preserves the v3-v8 re-decode protocol.
  * Fires are timestamped at the chunk boundary where <END_SPEECH> first
    appears; scored against boundaries with tolerance [-0.25, +1.5] s.

Usage:
    python -m eval.streaming_replay_eval \
        --checkpoint checkpoints/semantic_endpoint_v5_es/best.pt \
        --split data/semantic_endpoint_v3/meeting_split.json \
        --n-stretches 25 --out research/61-replay-v5.json
"""

from __future__ import annotations
import argparse
import io
import json
import logging
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from tqdm import tqdm

from eval.metrics import wer

logger = logging.getLogger(__name__)

END_TOK = "<END_SPEECH>"
EAGER_TOK = "<EAGER_END_SPEECH>"
SR = 16000

# Boundary classification (research/58)
T_TURN = 2.0          # same-speaker gap >= this => turn_final
T_MERGE = 0.3         # gap < this => no boundary event (merged)
TOL_EARLY = 0.25      # fire up to this much before the boundary still a hit
TOL_LATE = 1.5        # fire later than boundary + this => late (missed)


# ---------------------------------------------------------------- material

def decode_audio(blob: dict):
    if "array" in blob and blob["array"] is not None:
        return np.asarray(blob["array"], dtype=np.float32), int(blob.get("sampling_rate", SR))
    if "bytes" in blob:
        audio, sr = sf.read(io.BytesIO(blob["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        return audio.astype(np.float32), int(sr)
    return None, None


def load_meetings(parquet_dir: Path, restrict_to: set[str]) -> dict[str, list[dict]]:
    meetings: dict[str, list[dict]] = defaultdict(list)
    for pq_path in sorted(parquet_dir.glob("*.parquet")):
        t = pq.read_table(pq_path)
        for row in t.to_pylist():
            if row["meeting_id"] in restrict_to and (row["text"] or "").strip():
                meetings[row["meeting_id"]].append(row)
    for mid in meetings:
        meetings[mid].sort(key=lambda r: float(r["begin_time"]))
    return meetings


def build_stretches(meetings: dict[str, list[dict]], rng: random.Random,
                    n_stretches: int, min_dur: float = 30.0, max_dur: float = 60.0,
                    tail_s: float = 2.5) -> list[dict]:
    """One stretch = one target speaker's channel over a timeline window.

    Returns dicts with: audio (float32), events (utterance intervals with
    text + boundary class), meeting_id, speaker_id, duration_s.
    """
    candidates = []
    for mid, utts in meetings.items():
        by_spk: dict[str, list[dict]] = defaultdict(list)
        for u in utts:
            by_spk[u["speaker_id"]].append(u)
        for spk, sutts in by_spk.items():
            if len(sutts) < 3:
                continue
            # Slide over this speaker's utterances looking for >=3 utts
            # inside a max_dur window.
            for i in range(len(sutts) - 2):
                t0 = float(sutts[i]["begin_time"])
                group = [u for u in sutts
                         if t0 <= float(u["begin_time"])
                         and float(u["end_time"]) <= t0 + max_dur]
                if len(group) >= 3 and float(group[-1]["end_time"]) - t0 >= min_dur * 0.5:
                    candidates.append((mid, spk, group, utts))
                    break   # one candidate window per speaker

    rng.shuffle(candidates)
    stretches = []
    seen_meetings: dict[str, int] = defaultdict(int)
    for mid, spk, group, all_utts in candidates:
        if len(stretches) >= n_stretches:
            break
        if seen_meetings[mid] >= 3:      # diversity across meetings
            continue
        t0 = float(group[0]["begin_time"])
        t1 = float(group[-1]["end_time"]) + tail_s
        dur = t1 - t0
        audio = np.zeros(int(dur * SR), dtype=np.float32)
        events = []
        ok = True
        for j, u in enumerate(group):
            clip, sr = decode_audio(u["audio"])
            if clip is None or sr != SR:
                ok = False
                break
            ub, ue = float(u["begin_time"]), float(u["end_time"])
            off = int((ub - t0) * SR)
            n = min(len(clip), len(audio) - off)
            if n <= 0:
                ok = False
                break
            audio[off:off + n] = clip[:n]
            # classify the boundary at this utterance's end
            if j + 1 < len(group):
                gap = float(group[j + 1]["begin_time"]) - ue
            else:
                gap = float("inf")      # stream ends in silence
            interleave = any(
                u2["speaker_id"] != spk
                and ue < float(u2["begin_time"]) < ue + max(gap, 0.0)
                for u2 in all_utts
            )
            if gap < T_MERGE and not interleave:
                btype = "merged"
            elif gap >= T_TURN or interleave:
                btype = "turn_final"
            else:
                btype = "continuation"
            events.append({
                "begin_s": ub - t0, "end_s": ue - t0,
                "text": u["text"].strip(), "boundary": btype, "gap_s": gap,
            })
        if not ok:
            continue
        stretches.append({
            "meeting_id": mid, "speaker_id": spk, "duration_s": dur,
            "audio": audio, "events": events,
        })
        seen_meetings[mid] += 1
    return stretches


# ---------------------------------------------------------------- decoding

def build_prompt(tokenizer) -> str:
    msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio"}]},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def clean(text: str) -> str:
    text = re.sub(r"^language\s+\S+\s*<asr_text>", "", text)
    return text.replace("<|im_end|>", "").strip()


class StreamDecoder:
    """Segment-local incremental decoder over a growing audio buffer.

    committed mode: prompt-forces previously decoded text (minus rollback)
    so the model continues rather than re-deciding (qwen-asr protocol).
    """

    def __init__(self, model, processor, tokenizer, committed: bool,
                 rollback: int = 5, max_new: int = 96):
        self.model, self.processor, self.tokenizer = model, processor, tokenizer
        self.committed = committed
        self.rollback = rollback
        self.max_new = max_new
        self.prompt = build_prompt(tokenizer)
        self.raw = ""            # raw decoded text incl. "language ...<asr_text>"
        self.buffer: np.ndarray | None = None
        self.compute_ms: list[float] = []

    def reset_segment(self):
        self.raw = ""
        self.buffer = None

    def step(self, chunk: np.ndarray) -> str:
        self.buffer = chunk if self.buffer is None else np.concatenate([self.buffer, chunk])
        device = next(self.model.parameters()).device
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
        text_in = self.prompt + prefix
        # The feature extractor's reflection padding needs a minimum input
        # length; a fresh segment whose first chunk is a tiny stream tail
        # can be shorter. Zero-pad the FED copy only (silence-equivalent).
        feed = self.buffer
        min_len = int(0.3 * SR)
        if len(feed) < min_len:
            feed = np.concatenate([feed, np.zeros(min_len - len(feed), dtype=np.float32)])
        inputs = self.processor(text=[text_in], audio=[feed],
                                return_tensors="pt", padding=True)
        inputs = {k: v.to(device).bfloat16() if torch.is_floating_point(v) else v.to(device)
                  for k, v in inputs.items()}
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new)
        self.compute_ms.append((time.perf_counter() - t0) * 1000)
        gen = self.tokenizer.decode(
            out.sequences[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        self.raw = (prefix + gen) if self.committed else gen
        return clean(self.raw)


# ---------------------------------------------------------------- scoring

def run_stretch(dec: StreamDecoder, stretch: dict, chunk_s: float,
                energy_gate: bool = False, gate_rms: float = 1e-3,
                confirm_chunks: int = 0) -> dict:
    """energy_gate models the production composition policy (research/60):
    never START a segment on a silent chunk — the LM only decodes once
    speech energy has been observed. Motivated by the v5 replay finding
    that the LM hallucinates text + spams fires on silence-only segments
    (all v1-v8 training examples begin with speech).

    confirm_chunks=h treats a marker as a CANDIDATE: the system-level fire
    is accepted only after h further silent chunks (phrase-vs-turn policy
    dial found in the first gated run — spontaneous speech is full of
    complete phrases + 0.3-0.7 s pauses where the speaker continues).
    Latency cost is +h*chunk_s on accepted fires; resumed speech within
    the window cancels the candidate (no flush, segment continues)."""
    audio = stretch["audio"]
    n_chunks = int(np.ceil(len(audio) / (chunk_s * SR)))
    fires: list[float] = []
    flushed: list[str] = []
    prev_marker_count = 0
    dec.reset_segment()
    n_gated = 0
    n_cancelled = 0
    pending_fire: int | None = None      # chunk idx of candidate marker

    def chunk_silent(idx: int) -> bool:
        c = audio[int(idx * chunk_s * SR): int((idx + 1) * chunk_s * SR)]
        return len(c) == 0 or float(np.sqrt(np.mean(c ** 2))) < gate_rms

    for k in range(n_chunks):
        chunk = audio[int(k * chunk_s * SR): int((k + 1) * chunk_s * SR)]
        if (energy_gate and dec.buffer is None and
                (len(chunk) == 0 or float(np.sqrt(np.mean(chunk ** 2))) < gate_rms)):
            n_gated += 1
            continue

        # resolve a pending candidate before decoding this chunk
        if pending_fire is not None:
            if not chunk_silent(k):
                pending_fire = None      # speech resumed — cancel candidate
                n_cancelled += 1
            elif k - pending_fire >= confirm_chunks:
                fires.append((k + 1) * chunk_s)
                seg_text = clean(dec.raw).replace(EAGER_TOK, "")
                flushed.append(seg_text)
                dec.reset_segment()
                prev_marker_count = 0
                pending_fire = None
                n_gated += int(energy_gate)   # this silent chunk starts no segment
                continue
            else:
                continue                  # still confirming; skip decode

        text = dec.step(chunk)
        n_mark = text.count(END_TOK)
        if n_mark > prev_marker_count:
            after = text.rsplit(END_TOK, 1)[1].replace(EAGER_TOK, "").strip()
            if confirm_chunks > 0:
                if not after:
                    pending_fire = k      # candidate; confirm on silence
                # markers with trailing text stay textual (mid-stream A M B)
            else:
                t_fire = (k + 1) * chunk_s
                fires.extend([t_fire] * (n_mark - prev_marker_count))
                # flush rule (research/58): clean flush only when the marker
                # is terminal — otherwise the boundary is textual and the
                # buffer keeps growing until a clean flush.
                if not after:
                    seg_text = text.replace(EAGER_TOK, "")
                    flushed.append(seg_text)
                    dec.reset_segment()
                    prev_marker_count = 0
                    continue
        prev_marker_count = n_mark

    if dec.raw:
        flushed.append(clean(dec.raw).replace(EAGER_TOK, ""))

    # --- score fires against boundaries
    events = stretch["events"]
    bounds = []
    for i, e in enumerate(events):
        if e["boundary"] == "merged":
            continue
        # hit window must not reach into the next utterance's speech
        nxt = events[i + 1]["begin_s"] if i + 1 < len(events) else float("inf")
        bounds.append((e["end_s"], e["boundary"], min(e["end_s"] + TOL_LATE, nxt + TOL_EARLY)))
    speech = [(e["begin_s"], e["end_s"]) for e in events]

    def in_speech(t: float) -> bool:
        return any(b + 0.1 < t < e + TOL_EARLY for b, e in speech)

    hits, resume_fires, false_fires, dup_fires, late_info = [], [], [], [], []
    claimed: set[int] = set()
    for f in sorted(fires):
        matched = False
        for bi, (bt, btype, wend) in enumerate(bounds):
            if bt - TOL_EARLY <= f <= wend:
                if bi in claimed:
                    dup_fires.append(f)
                else:
                    claimed.add(bi)
                    (hits if btype == "turn_final" else resume_fires).append((f, bt))
                matched = True
                break
        if not matched:
            (false_fires if in_speech(f) else dup_fires).append(f)

    turn_bounds = [b for b in bounds if b[1] == "turn_final"]
    cont_bounds = [b for b in bounds if b[1] == "continuation"]
    recalled = [f - bt for f, bt in hits]
    speech_s = sum(e - b for b, e in speech)

    hyp_words = " ".join(t.replace(END_TOK, " ") for t in flushed)
    ref_words = " ".join(e["text"] for e in events)
    return {
        "meeting_id": stretch["meeting_id"], "speaker_id": stretch["speaker_id"],
        "duration_s": stretch["duration_s"], "speech_s": speech_s,
        "n_turn_bounds": len(turn_bounds), "n_cont_bounds": len(cont_bounds),
        "n_fires": len(fires), "n_hits": len(hits),
        "n_resume_fires": len(resume_fires), "n_false_fires": len(false_fires),
        "n_dup_or_silence_fires": len(dup_fires),
        "latencies_s": recalled,
        "wer": wer(hyp_words, ref_words),
        "median_chunk_ms": float(np.median(dec.compute_ms)) if dec.compute_ms else None,
        "n_chunks_gated": n_gated,
        "n_chunks_total": n_chunks,
        "n_cancelled_candidates": n_cancelled,
    }


def aggregate(per: list[dict], chunk_s: float, mode: str) -> dict:
    lat = sorted(l for r in per for l in r["latencies_s"])
    n_turn = sum(r["n_turn_bounds"] for r in per)
    n_cont = sum(r["n_cont_bounds"] for r in per)
    speech_min = sum(r["speech_s"] for r in per) / 60.0
    s = {
        "mode": mode, "chunk_s": chunk_s,
        "n_stretches": len(per), "n_turn_bounds": n_turn, "n_cont_bounds": n_cont,
        "boundary_recall": sum(r["n_hits"] for r in per) / max(1, n_turn),
        "false_fires_per_speech_min": sum(r["n_false_fires"] for r in per) / max(1e-9, speech_min),
        "resume_after_fire_rate": sum(r["n_resume_fires"] for r in per) / max(1, n_cont),
        "dup_or_silence_fires_total": sum(r["n_dup_or_silence_fires"] for r in per),
        "wer_mean": float(np.mean([r["wer"] for r in per])),
        "median_chunk_ms": float(np.median([r["median_chunk_ms"] for r in per
                                            if r["median_chunk_ms"]])),
    }
    if lat:
        s["latency_s_p50"] = lat[len(lat) // 2]
        s["latency_s_p95"] = lat[int(0.95 * (len(lat) - 1))]
        s["latency_s_mean"] = float(np.mean(lat))
    return s


# ---------------------------------------------------------------- main

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ami-dir", default="data/ami/ihm")
    p.add_argument("--split", default="data/semantic_endpoint_v3/meeting_split.json")
    p.add_argument("--n-stretches", type=int, default=25)
    p.add_argument("--chunk-s", type=float, default=0.5)
    p.add_argument("--from-scratch", action="store_true",
                    help="v3-v8 protocol: re-decode from scratch each chunk (default: committed-prefix)")
    p.add_argument("--energy-gate", action="store_true",
                    help="production composition policy: never start a segment on a silent chunk")
    p.add_argument("--gate-rms", type=float, default=1e-3)
    p.add_argument("--confirm-silent-chunks", type=int, default=0,
                    help="accept a marker only after this many further silent chunks "
                         "(phrase-vs-turn dial; adds h*chunk_s latency)")
    p.add_argument("--use-train-meetings", action="store_true",
                    help="dev mode: draw stretches from TRAIN meetings (for early stopping)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rng = random.Random(args.seed)
    split = json.loads(Path(args.split).read_text())
    key = "train_meeting_ids" if args.use_train_meetings else "eval_meeting_ids"
    restrict = set(split[key])
    logger.info("Loading %d %s meetings…", len(restrict), key)
    meetings = load_meetings(Path(args.ami_dir), restrict)
    stretches = build_stretches(meetings, rng, args.n_stretches)
    logger.info("Built %d stretches (%.1f min audio, %d turn bounds, %d cont bounds)",
                 len(stretches), sum(s["duration_s"] for s in stretches) / 60,
                 sum(1 for s in stretches for e in s["events"] if e["boundary"] == "turn_final"),
                 sum(1 for s in stretches for e in s["events"] if e["boundary"] == "continuation"))

    from src.train_semantic_endpoint import (
        load_base_model, extend_tokenizer_and_model,
        apply_lora, freeze_except_lora_and_new_rows,
    )
    model, tokenizer, processor = load_base_model()
    model = model.cuda().bfloat16()
    new_ids, _, _ = extend_tokenizer_and_model(model, tokenizer, processor)
    apply_lora(model.thinker, rank=16, alpha=32)
    freeze_except_lora_and_new_rows(model, new_ids)
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    model.load_state_dict(ckpt["trainable"], strict=False)
    model.eval()
    logger.info("Loaded %s (step %s)", args.checkpoint, ckpt.get("step"))

    mode = "from_scratch" if args.from_scratch else "committed_prefix"
    if args.energy_gate:
        mode += "+gate"
    if args.confirm_silent_chunks:
        mode += f"+confirm{args.confirm_silent_chunks}"
    per = []
    for st in tqdm(stretches, desc=f"replay[{mode}]"):
        dec = StreamDecoder(model, processor, tokenizer, committed=not args.from_scratch)
        per.append(run_stretch(dec, st, args.chunk_s,
                                energy_gate=args.energy_gate, gate_rms=args.gate_rms,
                                confirm_chunks=args.confirm_silent_chunks))

    summary = aggregate(per, args.chunk_s, mode)
    logger.info("== REPLAY SUMMARY ==")
    for k, v in summary.items():
        logger.info("  %-32s %s", k, f"{v:.3f}" if isinstance(v, float) else v)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "checkpoint": args.checkpoint, "summary": summary, "per_stretch": per,
    }, indent=2))
    logger.info("Saved %s", args.out)


if __name__ == "__main__":
    main()
