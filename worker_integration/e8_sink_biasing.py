"""E8: does the in-engine pinned CTX-token sink preserve context biasing across a long
session, IN the real vLLM serving path? (the capability payoff of Part B)

E4 (research/69) proved biasing survives session length at the APPLICATION layer (bounded
re-feed re-pins CTX every frame). E8 proves the same capability with the IN-ENGINE sink:
the CTX system block sits in KV positions [0,S); age_s seconds of filler audio scroll past;
the target's relevant words are decoded far behind the CTX. Three engine configs (each a
separate process — attention backend + window are fixed at engine init):

  FULL  : SWA=0  SINK=0  FLASH     -> CTX always attended (upper bound on recall)
  WIN   : SWA=W  SINK=0  TRITON    -> window only; CTX evicted once audio > W (the decay)
  SINK  : SWA=W  SINK=S  TRITON    -> CTX block pinned + window (the fix)

If recall(SINK) >= recall(WIN) and tracks recall(FULL), the in-engine sink preserves biasing
that a bare window drops. Loads the frozen dataset from e8_prep_data.py (no transformers/LLM
deps here). Runs under system python3.10 (vLLM).

Usage (one config per invocation, env-selected exactly like E7):
    METRONOME_SWA_TOKENS=128 METRONOME_SINK_TOKENS=160 METRONOME_ATTN_BACKEND=TRITON_ATTN \
      python3 worker_integration/e8_sink_biasing.py \
        --data data/e8_biasing_prep.npz --model checkpoints/merged/qwen3-asr-0.6b-endpoint-v9 \
        --tag SINK --out research/e8_SINK.json
"""
import argparse, json, os
import numpy as np

SR = 16000


def entity_recall(hyp, ref_entities):
    hyp_lo = hyp.lower()
    hits = sum(1 for e in ref_entities if e.lower() in hyp_lo)
    return hits, len(ref_entities)


def clean(t):
    import re
    t = re.sub(r"^language\s+\S+\s*<asr_text>", "", t)
    return (t.replace("<|im_end|>", "").replace("<EAGER_END_SPEECH>", "")
             .replace("<END_SPEECH>", "").strip())


def prompt(ctx):
    sysblk = f"<|im_start|>system\n{ctx}<|im_end|>\n" if ctx else "<|im_start|>system\n<|im_end|>\n"
    return (sysblk +
            "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
            "<|im_start|>assistant\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)          # FULL | WIN | SINK
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.30)
    ap.add_argument("--ctx-mode", default="pin", choices=["pin", "nopin"])
    args = ap.parse_args()

    W = os.environ.get("METRONOME_SWA_TOKENS", "0")
    S = os.environ.get("METRONOME_SINK_TOKENS", "0")
    backend = os.environ.get("METRONOME_ATTN_BACKEND", "").strip() or None
    print(f"### E8 tag={args.tag} SWA={W} SINK={S} BACKEND={backend or '(default)'} "
          f"ctx_mode={args.ctx_mode}")

    d = np.load(args.data, allow_pickle=True)
    audios = list(d["audios"]); refs = list(d["refs"])
    entities = [json.loads(e) for e in d["entities"]]
    ctx_prompts = list(d["ctx_prompts"])
    age_s = float(d["age_s"])
    print(f"### {len(audios)} targets, age_s={age_s}, max_ctx_tokens={int(d['max_ctx_tokens'])}")

    from vllm import LLM, SamplingParams
    kw = dict(model=args.model, trust_remote_code=True, gpu_memory_utilization=args.gpu_mem,
              max_model_len=8192, enable_prefix_caching=False, limit_mm_per_prompt={"audio": 1},
              enforce_eager=True)  # lighter init under co-tenant contention; eager kernel path
    if backend:
        kw["attention_backend"] = backend
    llm = LLM(**kw)
    sp = SamplingParams(temperature=0.0, max_tokens=128, skip_special_tokens=False)

    per, hits_tot, ent_tot = [], 0, 0
    for i, (audio, ref, ents, ctxp) in enumerate(zip(audios, refs, entities, ctx_prompts)):
        ctx = ctxp if args.ctx_mode == "pin" else ""
        o = llm.generate([{"prompt": prompt(ctx),
                           "multi_modal_data": {"audio": [(audio.astype("float32"), SR)]}}],
                         sampling_params=sp, use_tqdm=False)[0].outputs[0]
        hyp = clean(o.text)
        h, tot = entity_recall(hyp, ents)
        hits_tot += h; ent_tot += tot
        per.append({"ref": ref, "entities": ents, "hyp": hyp, "hits": h, "total": tot})

    recall = hits_tot / max(1, ent_tot)
    summary = {"tag": args.tag, "swa": int(W), "sink": int(S), "backend": backend,
               "ctx_mode": args.ctx_mode, "n_targets": len(per), "age_s": age_s,
               "entity_hits": hits_tot, "entity_total": ent_tot, "recall": recall}
    print(f"### E8 {args.tag}: entity recall = {recall:.3f} ({hits_tot}/{ent_tot}) "
          f"over {len(per)} targets, age={age_s}s")
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "per_target": per}, indent=2))
    print("E8_DONE")


if __name__ == "__main__":
    main()
