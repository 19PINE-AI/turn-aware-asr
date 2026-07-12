"""Capture a verbatim per-chunk streaming-decode trace for the paper's
'how a turn-aware recognizer decodes' figure (fig17).

Runs the real unified (v18, r32) checkpoint on one real AMI window in 0.5 s
chunks and logs, per chunk: the exact prompt fed (system CTX + committed
transcript prefix), the exact tokens generated, wall-clock compute, and any
marker. Also emits the cascade lane: RMS silence timeline + a silence-timeout
(X) decision computed on the identical audio. Nothing here is synthetic.

Usage:
  python -m eval.capture_decode_trace --out research/130-decode-trace.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from eval.streaming_replay_eval import (
    build_prompt, clean, SR, END_TOK, EAGER_TOK, load_meetings, build_stretches,
)

ROOT = Path(__file__).resolve().parents[1]

# --- conversational scenario: a remote-control design-review meeting ---
HOTWORDS = ["prototype", "remote control", "presentation", "share folder", "conceptual design"]
CONV_CTX = ("The user will provide an audio recording. Transcribe it verbatim. "
            "The recording may contain these proper nouns or named entities: "
            + ", ".join(HOTWORDS) + ".")
WIN_T0, WIN_T1 = 18.0, 24.5    # AMI IS1007d/MIO049: "...see the what" (pause) "did you prepare" (turn end)

# --- dictation scenario: an agent confirming the caller's number on file.
# The session profile (pinned in the system slot) carries the number the
# audio reads back — realistic "read me your callback number" grounding.
DICT_CTX = "User profile — name: Priya Raman; email: priya.raman@gmail.com; plan: business tier."
DICT_WAV = "data/probes/digit_wavs/digit_000.wav"

CHUNK_S = 0.5
GATE_RMS = 1e-3
TIMEOUT_X = 1.0                # cascade silence-timeout threshold (s)


def load_model(checkpoint: str):
    import torch
    from src.train_semantic_endpoint import (
        load_base_model, extend_tokenizer_and_model, apply_lora,
        freeze_except_lora_and_new_rows,
    )
    model, tokenizer, processor = load_base_model()
    model = model.cuda().bfloat16()
    new_ids, _, _ = extend_tokenizer_and_model(model, tokenizer, processor)
    ckpt = torch.load(checkpoint, weights_only=False, map_location="cpu")
    rank = next(v.shape[0] for k, v in ckpt["trainable"].items() if "lora_A" in k)
    apply_lora(model.thinker, rank=rank, alpha=2 * rank)
    freeze_except_lora_and_new_rows(model, new_ids)
    model.load_state_dict(ckpt["trainable"], strict=False)
    model.eval()
    return model, tokenizer, processor, rank


def stream_trace(model, tokenizer, processor, audio, ctx):
    """Committed-prefix incremental decode, logging every chunk verbatim."""
    import torch
    device = next(model.parameters()).device
    base_prompt = build_prompt(tokenizer)
    # inject the CTX into the system slot of the chat template
    prompt = base_prompt.replace("<|im_start|>system\n<|im_end|>",
                                 f"<|im_start|>system\n{ctx}<|im_end|>")
    assert ctx in prompt, "CTX injection failed — check chat template"

    n_chunks = int(np.ceil(len(audio) / (CHUNK_S * SR)))
    raw = ""
    buffer = None
    rows = []
    fires = []
    prev_marks = 0
    for k in range(n_chunks):
        chunk = audio[int(k * CHUNK_S * SR):int((k + 1) * CHUNK_S * SR)]
        rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0
        silent = rms < GATE_RMS
        t_end = round((k + 1) * CHUNK_S, 2)

        # energy gate: never START a segment on silence
        if buffer is None and silent:
            rows.append({"k": k, "t_end": t_end, "rms": rms, "gated": True,
                         "prompt_prefix": "", "gen": "", "text": "", "compute_ms": 0.0,
                         "fired": False})
            continue

        buffer = chunk if buffer is None else np.concatenate([buffer, chunk])
        # committed prefix = decoded-so-far minus a 5-token rollback
        prefix = ""
        if raw:
            ids = tokenizer.encode(raw)
            kk = 5
            while True:
                end = max(0, len(ids) - kk)
                prefix = tokenizer.decode(ids[:end]) if end else ""
                if "�" not in prefix or end == 0:
                    break
                kk += 1
        text_in = prompt + prefix
        feed = buffer
        min_len = int(0.3 * SR)
        if len(feed) < min_len:
            feed = np.concatenate([feed, np.zeros(min_len - len(feed), dtype=np.float32)])
        inputs = processor(text=[text_in], audio=[feed], return_tensors="pt", padding=True)
        inputs = {k2: v.to(device).bfloat16() if torch.is_floating_point(v) else v.to(device)
                  for k2, v in inputs.items()}
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=96)
        dt = (time.perf_counter() - t0) * 1000
        gen = tokenizer.decode(out.sequences[0, inputs["input_ids"].shape[1]:],
                               skip_special_tokens=False)
        raw = prefix + gen
        text = clean(raw)

        n_marks = text.count(END_TOK)
        fired = n_marks > prev_marks
        after = text.rsplit(END_TOK, 1)[1].replace(EAGER_TOK, "").strip() if END_TOK in text else "x"
        if fired:
            fires.append(t_end)
        rows.append({"k": k, "t_end": t_end, "rms": rms, "gated": False,
                     "prompt_prefix": clean(prefix), "gen": gen.replace("<|im_end|>", "").strip(),
                     "text": text, "compute_ms": round(dt, 1), "fired": fired})
        # terminal marker flushes the segment
        if fired and not after:
            raw = ""
            buffer = None
            prev_marks = 0
        else:
            prev_marks = n_marks
    return rows, fires


def cascade_timeout(audio, x=TIMEOUT_X):
    """VAD + silence-timeout: fire when trailing silence reaches x seconds
    (measured per chunk on the identical audio). Returns fire times + the
    per-chunk silence-run length for the timer lane."""
    n_chunks = int(np.ceil(len(audio) / (CHUNK_S * SR)))
    fires, silence_run = [], []
    run = 0.0
    spoke = False
    fired_this_run = False
    for k in range(n_chunks):
        chunk = audio[int(k * CHUNK_S * SR):int((k + 1) * CHUNK_S * SR)]
        rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0
        if rms >= GATE_RMS:
            run = 0.0
            spoke = True
            fired_this_run = False
        else:
            run += CHUNK_S
        silence_run.append(round(run, 2))
        if spoke and not fired_this_run and run >= x:
            fires.append(round((k + 1) * CHUNK_S, 2))
            fired_this_run = True
    return fires, silence_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="checkpoints/semantic_endpoint_v18_es/snap_step9000.pt")
    ap.add_argument("--out", default="research/130-decode-trace.json")
    ap.add_argument("--scenario", choices=["conv", "dictation", "spelled"], default="dictation")
    ap.add_argument("--ctx", choices=["profile", "none"], default="profile")
    ap.add_argument("--item", type=int, default=0)
    args = ap.parse_args()

    if args.scenario == "spelled":
        import soundfile as sf
        items = json.load(open(ROOT / "data/probes_big/spelled_probe.json"))
        it = items[args.item]
        audio, sr = sf.read(ROOT / it["wav"], dtype="float32")
        assert sr == SR
        ctx = it["ctx"] if args.ctx == "profile" else ""
        source = f"spelled probe #{it['id']} ({it['kind']}: {it['target']})"
    elif args.scenario == "conv":
        import random
        split = json.load(open(ROOT / "data/semantic_endpoint_v3/meeting_split.json"))
        meetings = load_meetings(ROOT / "data/ami/ihm", set(split["eval_meeting_ids"]))
        st0 = build_stretches(meetings, random.Random(0), 1)[0]
        assert st0["meeting_id"] == "IS1007d"
        audio = st0["audio"][int(WIN_T0 * SR):int(WIN_T1 * SR)]
        ctx, source = CONV_CTX, f"AMI IS1007d/MIO049 window [{WIN_T0},{WIN_T1}]s"
    else:
        import soundfile as sf
        audio, sr = sf.read(ROOT / DICT_WAV, dtype="float32")
        assert sr == SR
        ctx = DICT_CTX if args.ctx == "profile" else ""
        source = f"dictation probe #0 ({DICT_WAV})"

    model, tokenizer, processor, rank = load_model(args.checkpoint)
    rows, fires = stream_trace(model, tokenizer, processor, audio, ctx)
    to_fires, silence_run = cascade_timeout(audio)

    out = {
        "source": f"{source}; unified r{rank} {args.checkpoint}",
        "scenario": args.scenario, "ctx": ctx, "hotwords": HOTWORDS,
        "win_t0": WIN_T0, "win_t1": WIN_T1, "chunk_s": CHUNK_S,
        "ours_fires": fires, "ours_rows": rows,
        "timeout_x": TIMEOUT_X, "timeout_fires": to_fires, "silence_run": silence_run,
    }
    Path(ROOT / args.out).write_text(json.dumps(out, indent=2))
    print(f"saved {args.out}")
    print("ours fires:", fires, "| timeout fires:", to_fires)
    for r in rows:
        tag = "FIRE" if r["fired"] else ("gate" if r["gated"] else "    ")
        print(f"  t={r['t_end']:5.2f} {tag} rms={r['rms']:.4f} {r['compute_ms']:6.1f}ms  gen={r['gen'][:60]!r}")


if __name__ == "__main__":
    main()
