"""Export website data: benchmark audio + original trajectories + chart data.

Reconstructs the exact evaluation material (deterministic seeds / recorded
events) and pairs it with the recorded trajectories from research/*.json.
Nothing is re-run through a model — every fire, hypothesis, and metric comes
from the original result files.

Run from repo root:  .venv/bin/python website/scripts/export_data.py
"""
from __future__ import annotations
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.streaming_replay_eval import (  # noqa: E402
    load_meetings, build_stretches, score_fires, SR,
)

R = ROOT / "research"
OUT = ROOT / "website" / "public"
AUDIO = OUT / "audio"
DATA = OUT / "data"


# ---------------------------------------------------------------- helpers

def encode_mp3(audio: np.ndarray, out_path: Path, kbps: int = 40):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    raw = audio.astype(np.float32).tobytes()
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "f32le", "-ar", str(SR),
         "-ac", "1", "-i", "-", "-codec:a", "libmp3lame", "-b:a", f"{kbps}k",
         str(out_path)],
        input=raw, check=True)


def peaks(audio: np.ndarray, bin_s: float = 0.08) -> list[float]:
    n = int(bin_s * SR)
    m = [float(np.abs(audio[i:i + n]).max()) if len(audio[i:i + n]) else 0.0
         for i in range(0, len(audio), n)]
    top = max(m) or 1.0
    return [round(v / top, 3) for v in m]


def classify_fires(fires: list[float], events: list[dict]) -> list[dict]:
    s = score_fires(fires, events)
    out = []
    for f, bt in s["hits"]:
        out.append({"t": round(f, 2), "cls": "hit", "lat": round(f - bt, 2)})
    for f, bt in s["resume_fires"]:
        out.append({"t": round(f, 2), "cls": "resume", "lat": round(f - bt, 2)})
    for f in s["false_fires"]:
        out.append({"t": round(f, 2), "cls": "false"})
    for f in s["dup_fires"]:
        out.append({"t": round(f, 2), "cls": "dup"})
    return sorted(out, key=lambda d: d["t"])


def arm_entry(label, group, summary, fires_by_idx, extra=None):
    e = {"label": label, "group": group,
         "summary": {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in summary.items()
                     if k in ("boundary_recall", "false_fires_per_speech_min",
                              "resume_after_fire_rate", "latency_s_p50",
                              "latency_s_p95", "wer_mean", "n_stretches")},
         "fires": fires_by_idx}
    if extra:
        e.update(extra)
    return e


# ---------------------------------------------------------------- AMI replay

def export_replay():
    print("== loading AMI meetings…")
    split = json.loads((ROOT / "data/semantic_endpoint_v3/meeting_split.json").read_text())
    meetings = load_meetings(ROOT / "data/ami/ihm", set(split["eval_meeting_ids"]))

    import random
    sets = {}
    print("== building stretches (seed 0, n=100) …")
    sets["ami100"] = build_stretches(meetings, random.Random(0), 100)
    print(f"   {len(sets['ami100'])} stretches")
    print("== building stretches (seed 1, n=50) …")
    sets["fresh50"] = build_stretches(meetings, random.Random(1), 50)
    print(f"   {len(sets['fresh50'])} stretches")

    # ---- validate against recorded events
    def validate(stretches, rec_file, n=None):
        rec = json.loads((R / rec_file).read_text())["per_stretch"]
        n = n or len(rec)
        assert len(stretches) >= n, f"{rec_file}: built {len(stretches)} < {n}"
        for i in range(n):
            a, b = stretches[i], rec[i]
            assert a["meeting_id"] == b["meeting_id"] and a["speaker_id"] == b["speaker_id"], \
                f"{rec_file}[{i}]: {a['meeting_id']}/{a['speaker_id']} != {b['meeting_id']}/{b['speaker_id']}"
            ta = [e["text"] for e in a["events"]]
            tb = [e["text"] for e in b["events"]]
            assert ta == tb, f"{rec_file}[{i}]: event texts differ"
        print(f"   OK {rec_file}: {n} stretches match")

    validate(sets["ami100"], "124-replay-v18-dev25.json")   # first 25
    validate(sets["ami100"], "125-replay-v18-big.json")     # all 100
    validate(sets["fresh50"], "73-replay-v9gate-fresh50.json")

    # ---- timeout policy (identical code to eval/timeout_baseline_eval.py, rms)
    def rms_flags(audio, chunk_s=0.5, gate_rms=1e-3):
        n_chunks = int(np.ceil(len(audio) / (chunk_s * SR)))
        return [len(audio[int(k * chunk_s * SR):int((k + 1) * chunk_s * SR)]) > 0
                and float(np.sqrt(np.mean(
                    audio[int(k * chunk_s * SR):int((k + 1) * chunk_s * SR)] ** 2))) >= gate_rms
                for k in range(n_chunks)]

    def run_timeout(flags, chunk_s, timeout_s):
        fires, seen_speech, silent = [], False, 0
        need = max(1, int(round(timeout_s / chunk_s)))
        for k, sp in enumerate(flags):
            if sp:
                seen_speech, silent = True, 0
            else:
                if seen_speech:
                    silent += 1
                    if silent == need:
                        fires.append((k + 1) * chunk_s)
        return fires

    # ---- assemble per set
    out_sets = {}
    for set_id, stretches, prefix in [("ami100", sets["ami100"], "ami"),
                                      ("fresh50", sets["fresh50"], "fresh")]:
        entries = []
        for i, st in enumerate(stretches):
            mp3 = AUDIO / prefix / f"s{i:03d}.mp3"
            encode_mp3(st["audio"], mp3)
            entries.append({
                "id": f"{prefix}{i:03d}",
                "meeting": st["meeting_id"], "speaker": st["speaker_id"],
                "dur": round(st["duration_s"], 2),
                "audio": f"audio/{prefix}/s{i:03d}.mp3",
                "peaks": peaks(st["audio"]),
                "events": [{"b": round(e["begin_s"], 2), "e": round(e["end_s"], 2),
                            "text": e["text"], "cls": e["boundary"],
                            "gap": round(e["gap_s"], 2) if e["gap_s"] is not None else None}
                           for e in st["events"]],
            })
        out_sets[set_id] = {"stretches": entries, "arms": {}}
        print(f"   encoded {len(entries)} {set_id} mp3s")

    def add_lm_arm(set_id, arm_id, label, group, rec_file):
        d = json.loads((R / rec_file).read_text())
        stretches = sets[set_id]
        fires_by_idx = {}
        for i, r in enumerate(d["per_stretch"]):
            fires_by_idx[str(i)] = {
                "f": classify_fires(r["fires_s"], stretches[i]["events"]),
                "wer": round(r["wer"], 3) if r.get("wer") is not None else None,
            }
        out_sets[set_id]["arms"][arm_id] = arm_entry(label, group, d["summary"], fires_by_idx)

    print("== LM arms…")
    add_lm_arm("ami100", "causal", "Causal endpointer (ours, v9 + gate)", "ours", "62-replay-v9gate.json")
    add_lm_arm("ami100", "causal_h1", "Causal + confirm h=1", "ours", "62-replay-v9gate-confirm1.json")
    add_lm_arm("ami100", "unified", "Released unified model (rank-32)", "ours", "124-replay-v18-dev25.json")
    add_lm_arm("ami100", "unified_h1", "Released unified + confirm h=1", "ours", "124-replay-v18-dev25-h1.json")
    add_lm_arm("ami100", "unified_100", "Released unified (100-stretch held-out)", "ours", "125-replay-v18-big.json")
    add_lm_arm("ami100", "mixed", "Mixed-pools (row 3) + confirm h=1", "prior", "61-replay-v5gate-confirm1.json")
    add_lm_arm("ami100", "opposed", "Opposed-pools (row 6) + confirm h=1", "prior", "61-replay-v8gate-confirm1.json")
    add_lm_arm("fresh50", "causal", "Causal endpointer (ours, fresh 50)", "ours", "73-replay-v9gate-fresh50.json")

    # external systems (recorded fires, dev-25)
    print("== external arms…")
    for fname, sys_label in [("95-exp4-kyutai-dev25.json", "Kyutai STT semantic-VAD"),
                             ("95-exp4-parakeet-dev25.json", "Parakeet-EOU"),
                             ("95-exp4-livekit_turn-dev25.json", "LiveKit turn detector (oracle transcript)"),
                             ("95-exp4-smart_turn-dev25.json", "Smart Turn v3")]:
        d = json.loads((R / fname).read_text())
        for arm_name, arm in d["arms"].items():
            fires_by_idx = {}
            for i, r in enumerate(arm["per_stretch"]):
                if "fires_s" not in r:
                    continue
                fires_by_idx[str(i)] = {
                    "f": classify_fires(r["fires_s"], sets["ami100"][i]["events"]),
                    "wer": round(r["wer"], 3) if r.get("wer") is not None else None,
                }
            suffix = "" if len(d["arms"]) == 1 else f" ({arm_name.replace('thr', 'thr ')})"
            out_sets["ami100"]["arms"][f"{d['system']}_{arm_name}"] = arm_entry(
                sys_label + suffix, "external", arm["summary"], fires_by_idx)

    # timeout baselines: deterministic recompute on reconstructed audio,
    # validated against the recorded per-stretch counts (71-timeout-rms-seed0)
    print("== timeout baselines (recomputed, validated)…")
    rec = json.loads((R / "71-timeout-rms-seed0.json").read_text())["arms"]
    for X in [0.5, 1.0, 1.5, 2.0]:
        fires_by_idx = {}
        mismatch = 0
        for i, st in enumerate(sets["ami100"][:25]):
            fires = run_timeout(rms_flags(st["audio"]), 0.5, X)
            rp = rec[str(X)]["per_stretch"][i]
            if len(fires) != rp["n_fires"]:
                mismatch += 1
            fires_by_idx[str(i)] = {"f": classify_fires(fires, st["events"]), "wer": None}
        assert mismatch == 0, f"timeout X={X}: {mismatch} stretches disagree with recorded counts"
        out_sets["ami100"]["arms"][f"timeout_rms_{X}"] = arm_entry(
            f"RMS + {X:g}s silence timeout", "timeout", rec[str(X)]["summary"], fires_by_idx)
        print(f"   X={X:g}: matches recorded counts on all 25 stretches")

    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "replay.json").write_text(json.dumps({"sets": out_sets}))
    print(f"   wrote replay.json ({(DATA/'replay.json').stat().st_size/1e6:.1f} MB)")


# ---------------------------------------------------------------- probes

def export_dictation():
    print("== dictation probes…")
    out = {"sets": {}}
    for set_id, probe_dir, arms in [
        ("big", ROOT / "data/probes_big",
         [("released", "Released unified model (rank-32)", "119-probe-v18-9000.json")]),
        ("small", ROOT / "data/probes",
         [("conversational", "Conversational-only (causal v9)", "74-dictation-probe-v9.json"),
          ("dictation_matched", "+ dictation schemas (match-ctx, v11)", "75-dictation-probe-v11.json")]),
    ]:
        probe = json.loads((probe_dir / "digit_probe.json").read_text())
        items = []
        for it in probe:
            wav_path = ROOT / it["wav"]
            audio, sr = sf.read(wav_path, dtype="float32")
            assert sr == SR
            mp3 = AUDIO / f"digit_{set_id}" / f"d{it['id']:03d}.mp3"
            encode_mp3(audio, mp3)
            items.append({
                "id": it["id"],
                "text": it["text"], "digits": it["digits"], "groups": it["groups"],
                "pauses": [[round(a, 2), round(b, 2)] for a, b in it["pauses"]],
                "last_speech_end": round(it["last_speech_end_s"], 2),
                "dur": round(it["duration_s"], 2),
                "audio": f"audio/digit_{set_id}/d{it['id']:03d}.mp3",
                "peaks": peaks(audio),
            })
        set_arms = {}
        for arm_id, label, rec_file in arms:
            d = json.loads((R / rec_file).read_text())["digit"]
            per = {str(r["id"]): {
                "fires": [round(f, 2) for f in r["fires_s"]],
                "premature": r["n_premature"], "final_recall": r["final_recall"],
                "final_latency": round(r["final_latency_s"], 2) if r.get("final_latency_s") is not None else None,
                "digit_acc": r["digit_acc"], "hyp": r["hyp"].strip(),
            } for r in d["per_item"]}
            set_arms[arm_id] = {"label": label, "summary": d["summary"], "items": per}
        out["sets"][set_id] = {"items": items, "arms": set_arms}
        print(f"   digit[{set_id}]: {len(items)} items")
    (DATA / "dictation.json").write_text(json.dumps(out))


def export_spelled():
    print("== spelled-entity probe…")
    probe = json.loads((ROOT / "data/probes_big/spelled_probe.json").read_text())
    res = json.loads((R / "119-probe-v18-9000.json").read_text())["spelled"]
    by_id = {r["id"]: r for r in res["per_item"]}
    items = []
    for i, it in enumerate(probe):
        r = by_id.get(i)
        if r is None:
            continue
        wav = ROOT / "data/probes_big/spelled_wavs" / f"sp_{i:04d}.wav"
        assert wav.exists(), f"no wav for spelled item {i}"
        audio, sr = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        mp3 = AUDIO / "spelled" / f"p{i:03d}.mp3"
        encode_mp3(audio.astype(np.float32), mp3)
        items.append({
            "id": i, "kind": it["kind"], "style": it.get("style", ""),
            "utt": it["utt"], "target": it["target"],
            "ctx": it["ctx"], "ctx_distractor": it["ctx_distractor"],
            "audio": f"audio/spelled/p{i:03d}.mp3",
            "hyp_none": r["hyp_none"].strip(), "hit_none": r["hit_none"],
            "hyp_profile": r["hyp_profile"].strip(), "hit_profile": r["hit_profile"],
            "hyp_distractor": r["hyp_distractor"].strip(), "hit_distractor": r["hit_distractor"],
            "intrusion": r["intrusion_distractor"],
        })
    (DATA / "spelled.json").write_text(json.dumps({
        "summary": res["summary"], "model": "Released unified model (rank-32)",
        "items": items}))
    print(f"   spelled: {len(items)} items")


def export_earnings():
    print("== earnings-22 biasing…")
    import pyarrow.parquet as pq
    base = json.loads((R / "63-earnings22-biasing.json").read_text())
    rel = json.loads((R / "120-biasing-v18-9000.json").read_text())
    wanted = {u["utt_id"] for u in base["per_utterance"]} | \
             {u["utt_id"] for u in rel["per_utterance"]}
    audio_by_id = {}
    for p in sorted((ROOT / "data/earnings22/chunked").glob("*.parquet")):
        t = pq.read_table(p, columns=["file_id", "segment_id", "audio"])
        for row in t.to_pylist():
            uid = f"{row['file_id']}-{row['segment_id']}"
            if uid in wanted and uid not in audio_by_id:
                audio_by_id[uid] = row["audio"]["bytes"]
        if len(audio_by_id) == len(wanted):
            break
    print(f"   found audio for {len(audio_by_id)}/{len(wanted)} utterances")

    def item(u, arm):
        return {k: u[k] for k in
                ("utt_id", "ref", "ref_entities", "distractor_entities",
                 "hyp_no_ctx", "hyp_relevant", "hyp_distractor",
                 "recall_no_ctx", "recall_relevant", "hallucination_distractor_count")
                if k in u}

    out = {"arms": {}}
    for arm_id, d, label in [("base", base, "Base model (Qwen3-ASR-0.6B)"),
                             ("released", rel, "Released unified model (rank-32)")]:
        out["arms"][arm_id] = {"label": label, "summary": d["summary"],
                               "items": [item(u, arm_id) for u in d["per_utterance"]]}
    audio_map = {}
    for uid, blob in audio_by_id.items():
        audio, sr = sf.read(io.BytesIO(blob), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        if sr != SR:  # resample cheaply if needed
            import math
            idx = np.linspace(0, len(audio) - 1, int(len(audio) * SR / sr))
            audio = audio[idx.astype(int)]
        mp3 = AUDIO / "earnings" / f"{uid}.mp3"
        encode_mp3(audio.astype(np.float32), mp3)
        audio_map[uid] = f"audio/earnings/{uid}.mp3"
    out["audio"] = audio_map
    (DATA / "earnings.json").write_text(json.dumps(out))
    print(f"   earnings: {len(audio_map)} clips")


# ---------------------------------------------------------------- chart data

def export_results():
    print("== chart data (results.json)…")
    res = {}

    # trade-off frontier (fig3)
    def fam(fname):
        arms = json.loads((R / fname).read_text())["arms"]
        pts = []
        for x, a in sorted(arms.items(), key=lambda kv: float(kv[0])):
            s = a["summary"]
            if "latency_s_p50" not in s:
                continue
            pts.append({"X": float(x), "lat": round(s["latency_s_p50"], 3),
                        "ff": round(s["false_fires_per_speech_min"], 3),
                        "recall": round(s["boundary_recall"], 3)})
        return pts
    res["tradeoff"] = {
        "rms": fam("71-timeout-rms-seed0.json"),
        "silero": fam("71-timeout-silero-seed0.json"),
        "priors": [{"name": "mixed-pools + confirm h=1", "lat": 0.68, "ff": 3.2, "recall": 0.927},
                   {"name": "opposed-pools + confirm h=1", "lat": 0.77, "ff": 0.0, "recall": 0.938}],
        "ours": {"name": "causal endpointer", "lat": 0.39, "ff": 0.32, "recall": 0.969},
        "unified": {"name": "released unified", "lat": 0.39, "ff": 0.97, "recall": 0.948},
        "external": [{"name": "Parakeet-EOU", "lat": 0.47, "ff": 0.22, "recall": 0.17},
                     {"name": "Kyutai VAD", "lat": 0.53, "ff": 14.6, "recall": 0.88},
                     {"name": "Smart Turn v3", "lat": 0.64, "ff": 3.4, "recall": 0.71},
                     {"name": "LiveKit (oracle)", "lat": 0.65, "ff": 6.25, "recall": 0.96}],
    }

    # oscillation curves (fig1)
    v8 = json.loads((ROOT / "checkpoints/semantic_endpoint_v8_es/eval_log.json").read_text())
    v8b = json.loads((ROOT / "checkpoints/semantic_endpoint_v8_seed1_es/eval_log.json").read_text())
    v9 = json.loads((ROOT / "checkpoints/semantic_endpoint_v9_es/eval_log.json").read_text())
    res["oscillation"] = {
        "opposed_seed0": [{"step": e["step"], "fire": e["single"], "hold": e["no_fire_correct"]} for e in v8],
        "opposed_seed1": [{"step": e["step"], "fire": e["single"], "hold": e["no_fire_correct"]} for e in v8b],
        "causal": [{"step": e["step"], "fire": e["single_sil_exact"],
                    "hold": round(float(np.mean([e["complete_nosil_exact"],
                                                 e["silence_only_exact"],
                                                 e["truncated_exact"]])), 4),
                    "composite": e["score"]} for e in v9],
    }

    # biasing (fig4)
    e4 = json.loads((R / "69-e4-pinned-ctx.json").read_text())["summary"]["by_age"]
    ages = sorted(float(a) for a in e4)
    key = lambda a: f"{a:g}.0" if f"{a:g}.0" in e4 else str(a)
    res["biasing"] = {
        "bars": {"no_ctx": 66.7, "relevant": 95.6, "halluc_base": 3.7, "halluc_released": 2.0},
        "by_age": [{"age": a, "ctx": round(100 * e4[key(a)]["pin_recall"], 1),
                    "no_ctx": round(100 * e4[key(a)]["nopin_recall"], 1)} for a in ages],
    }

    # silence incompetence scatter (fig6)
    d = json.loads((R / "61-replay-v5-specv1.json").read_text())["per_stretch"]
    res["silence"] = [{"sil": round(r["duration_s"] - r["speech_s"], 1),
                       "spam": r["n_dup_or_silence_fires"]} for r in d]

    # ranking inversion (fig5), silence-append (tab:silappend) — paper values
    res["inversion"] = [
        {"name": "offline-labeled", "offline_recall": 100, "stream_ff": 88.8},
        {"name": "mixed-pools", "offline_recall": 82, "stream_ff": 27.3},
        {"name": "opposed-pools", "offline_recall": 10, "stream_ff": 3.6}]
    res["silappend"] = [
        {"name": "offline-labeled", "orig": 1.00, "plus1s": 1.00},
        {"name": "mixed-pools", "orig": 0.82, "plus1s": 0.98},
        {"name": "opposed-pools", "orig": 0.10, "plus1s": 1.00}]

    # main endpointing table (tab:main)
    res["main_table"] = [
        {"policy": "mixed-pools + gate + confirm h=1", "recall": 0.927, "p50": 0.68, "p95": 1.27, "ff": 3.2},
        {"policy": "opposed-pools + gate + confirm h=1", "recall": 0.938, "p50": 0.77, "p95": 1.06, "ff": 0.0},
        {"policy": "causal + gate (ours)", "recall": 0.969, "p50": 0.39, "p95": 0.70, "ff": 0.3, "hero": True},
        {"policy": "causal + gate + confirm h=1", "recall": 0.969, "p50": 0.89, "p95": 1.20, "ff": 0.0},
        {"policy": "causal + gate, fresh 50-stretch set", "recall": 0.924, "p50": 0.42, "p95": 0.70, "ff": 0.2}]

    # dictation / spelled table (tab:dictation)
    res["dictation_table"] = [
        {"model": "conversational only", "premat": 4.96, "final": 0.82, "digit": 0.889,
         "name": [0.25, 0.82], "email": [0.04, 0.58], "intrusion": 1.7},
        {"model": "timeout X=0.5 s", "premat": 2.00, "final": 1.00},
        {"model": "timeout X=1.0 s", "premat": 0.74, "final": 1.00},
        {"model": "+ dictation (match-ctx)", "premat": 0.36, "final": 0.78, "digit": 0.996,
         "name": [1.00, 1.00], "email": [0.03, 0.93], "intrusion": 11.3},
        {"model": "released (r32, unified)", "premat": 0.16, "final": 0.88, "digit": 0.998,
         "name": [0.98, 1.00], "email": [0.11, 0.93], "intrusion": 0.8, "hero": True}]

    # supervision-composition record (tab:composition, compressed)
    res["composition"] = [
        {"row": 1, "change": "offline endpoint pool only: fire at every clip end",
         "behavior": "100% fire recall offline; 89 false fires/speech-min in streaming replay"},
        {"row": 2, "change": "+ truncated-speech hold pool (11%→38%)",
         "behavior": "median fire timing improves −13.1 → −9.6 → −7.6 s — still seconds early"},
        {"row": 3, "change": "+ trailing-silence fire pool",
         "behavior": "latency crosses zero (P50 +0.32 s) but offline fire recall 100 → 82%; “trade-off” declared"},
        {"row": 4, "change": "+ complete utterances with and without silence tails, both labeled fire",
         "behavior": "silence becomes optional again; frontier declared fundamental"},
        {"row": 5, "change": "+ hold pool on the same audio as existing fire examples",
         "behavior": "collapse: opposite labels on identical inputs; fire recall 0%"},
        {"row": 6, "change": "hold pool rebuilt on disjoint but identically distributed audio",
         "behavior": "bimodal oscillation between fire-mode and no-fire-mode for the whole run"},
        {"row": 7, "change": "causal relabel: fire iff complete AND ≥0.3 s observed silence",
         "behavior": "monotone training, both classes at ceiling; single checkpoint dominates all", "causal": True}]

    # toy study (fig:toy) — clairvoyant fraction sweep
    toy = json.loads((R / "94-exp5-toy-task.json").read_text())
    if "by_f" in toy:
        res["toy"] = toy["by_f"]

    # held-out summaries
    res["heldout"] = {
        "unified_100": {"recall": 0.982, "ff": 1.30, "p50": 0.38},
        "fresh50": {"recall": 0.924, "ff": 0.21, "p50": 0.42},
    }

    (DATA / "results.json").write_text(json.dumps(res))
    print("   wrote results.json")


if __name__ == "__main__":
    AUDIO.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    export_results()
    export_dictation()
    export_spelled()
    export_earnings()
    export_replay()
    print("done.")
