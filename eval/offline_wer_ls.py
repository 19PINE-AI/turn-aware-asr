"""E12: offline ASR regression check — did the endpoint LoRA damage base ASR?

Decodes LibriSpeech test-clean and test-other (full official splits, offline
single-shot) with a given checkpoint on vLLM and reports WER (jiwer,
whisper-norm). Run once with the stock Qwen/Qwen3-ASR-0.6B and once with the
merged endpoint checkpoint; the delta is the regression cost of the endpoint
capability. Markers are stripped before scoring (skip_special_tokens=False so
their emission is observable; also reports marker-emission rate — offline
clips end at speech, so the causal v9 model should rarely fire).

Runs under system python3.10 (vLLM).
    python3 -m eval.offline_wer_ls --model Qwen/Qwen3-ASR-0.6B \
        --out research/72-wer-base.json
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
END = "<END_SPEECH>"
EAGER = "<EAGER_END_SPEECH>"


def iter_split(root: Path):
    for trans in sorted(root.rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            utt_id, ref = line.split(" ", 1)
            yield trans.parent / f"{utt_id}.flac", ref.strip()


def clean(t: str) -> str:
    import re
    t = re.sub(r"^language\s+\S+\s*<asr_text>", "", t)
    return (t.replace("<|im_end|>", "").replace(EAGER, "").replace(END, "").strip())


def prompt() -> str:
    return ("<|im_start|>system\n<|im_end|>\n"
            "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
            "<|im_start|>assistant\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ls-root", default=str(Path.home() / "data/LibriSpeech"))
    ap.add_argument("--splits", default="test-clean,test-other")
    ap.add_argument("--gpu-mem", type=float, default=0.28)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="0 = full split")
    ap.add_argument("--suppress-markers", action="store_true",
                    help="ban the endpoint marker tokens at decode time (offline mode): "
                         "isolates how much of any WER delta is marker emission ending "
                         "decodes early vs. genuine capability loss")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from eval.metrics import wer as wer_fn
    llm = LLM(model=args.model, trust_remote_code=True,
              gpu_memory_utilization=args.gpu_mem, max_model_len=8192,
              enable_prefix_caching=False, limit_mm_per_prompt={"audio": 1},
              enforce_eager=True)
    kw = {}
    if args.suppress_markers:
        kw["bad_words"] = [EAGER, END]
    sp = SamplingParams(temperature=0.0, max_tokens=440, skip_special_tokens=False, **kw)

    results = {}
    for split in args.splits.split(","):
        pairs = list(iter_split(Path(args.ls_root) / split))
        if args.limit:
            pairs = pairs[: args.limit]
        print(f"### {split}: {len(pairs)} utts")
        hyps, refs, fired = [], [], 0
        t0 = time.time()
        for i in range(0, len(pairs), args.batch):
            batch = pairs[i: i + args.batch]
            reqs = []
            for flac, _ in batch:
                a, sr = sf.read(flac, dtype="float32")
                reqs.append({"prompt": prompt(),
                             "multi_modal_data": {"audio": [(a, sr)]}})
            outs = llm.generate(reqs, sampling_params=sp, use_tqdm=False)
            for (flac, ref), o in zip(batch, outs):
                text = o.outputs[0].text
                if END in text:
                    fired += 1
                hyps.append(clean(text))
                refs.append(ref)
            if (i // args.batch) % 10 == 0:
                print(f"  {i + len(batch)}/{len(pairs)} ({time.time()-t0:.0f}s)")
        # corpus WER = wer over concatenation is wrong; use jiwer over lists via join
        w = wer_fn(" ".join(hyps), " ".join(refs))
        results[split] = {"n": len(pairs), "wer": w,
                          "marker_fire_rate": fired / max(1, len(pairs))}
        print(f"### {split}: WER={w:.4f}  marker_fire_rate={fired/max(1,len(pairs)):.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"model": args.model, "results": results}, indent=2))
    print("E12_DONE")


if __name__ == "__main__":
    main()
