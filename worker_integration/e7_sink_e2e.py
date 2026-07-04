"""E7: end-to-end validation of the Metronome in-engine CTX-token sink on Qwen3-ASR.

Emits transcription + fire status + per-decode latency for a set of probe clips, under
whatever windowing/sink the env selects (METRONOME_SWA_TOKENS, METRONOME_SINK_TOKENS,
VLLM_ATTENTION_BACKEND). The runner (run_e7.sh) invokes it in three configs:

  A) baseline     — no window, no sink, FLASH_ATTN  -> reference transcription
  B) sink-covers-all — window=128 + sink=8192, TRITON_ATTN. The union [0,S)∪[t-W,t]
     with S >= context covers ALL keys == full attention, so B's transcription MUST
     equal A's. This is the CORRECTNESS GATE: if the sink kernel + block-pin corrupt
     anything, B diverges from A.
  C) real sink    — window=128 + sink=64, TRITON_ATTN, long clip (context > window > sink):
     the middle is freed, the sink pinned; transcription must stay sane (no garbage).

Runs under system python3.10 (vLLM).
"""
import argparse, os, time
import numpy as np
import soundfile as sf

SR = 16000
END = "<END_SPEECH>"


def load_16k(path):
    a, sr = sf.read(os.path.expanduser(path), dtype="float32")
    if a.ndim > 1:
        a = a.mean(-1)
    if sr != SR:
        n = int(round(len(a) * SR / sr))
        a = np.interp(np.linspace(0, len(a), n, endpoint=False), np.arange(len(a)), a).astype("float32")
    return a


def clean(t):
    import re
    t = re.sub(r"^language\s+\S+\s*<asr_text>", "", t)
    return t.replace("<|im_end|>", "").replace("<EAGER_END_SPEECH>", "").replace(END, "").strip()


def prompt():
    return ("<|im_start|>system\n<|im_end|>\n"
            "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
            "<|im_start|>assistant\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--flac", default="~/data/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac")
    ap.add_argument("--gpu-mem", type=float, default=0.30)
    args = ap.parse_args()

    print(f"### SWA={os.environ.get('METRONOME_SWA_TOKENS','0')} "
          f"SINK={os.environ.get('METRONOME_SINK_TOKENS','0')} "
          f"BACKEND={os.environ.get('METRONOME_ATTN_BACKEND','(default)')}")

    from vllm import LLM, SamplingParams
    # vLLM 0.19 dropped the VLLM_ATTENTION_BACKEND env var; the backend is now an
    # engine arg. The sink kernel lives only in TRITON_ATTN, so force it when a sink
    # is active (FLASH_ATTN can't express the [0,S)∪[t-W,t] union).
    backend = os.environ.get("METRONOME_ATTN_BACKEND", "").strip() or None
    llm_kw = dict(model=args.model, trust_remote_code=True,
                  gpu_memory_utilization=args.gpu_mem, max_model_len=8192,
                  enable_prefix_caching=False, limit_mm_per_prompt={"audio": 1},
                  enforce_eager=True)  # skip inductor compile + cudagraph capture:
    # robust init under GPU co-tenant contention, and runs the patched Triton sink
    # kernel in eager mode (correctness path we're validating).
    if backend:
        llm_kw["attention_backend"] = backend
    llm = LLM(**llm_kw)
    tok = llm.get_tokenizer()
    END_ID = tok.convert_tokens_to_ids(END)
    sp = SamplingParams(temperature=0.0, max_tokens=96, skip_special_tokens=False)

    speech = load_16k(args.flac)
    sil = np.zeros(int(0.5 * SR), dtype="float32")
    long_ctx = np.concatenate([np.tile(speech, 8)[: 40 * SR], speech, sil])  # ~44s, target at end

    probes = {
        "short_speech": speech,
        "speech+sil(FIRE)": np.concatenate([speech, sil]),
        "silence(no fire)": np.zeros(int(2.0 * SR), dtype="float32"),
        "long44s+target": long_ctx,
    }
    for name, audio in probes.items():
        t0 = time.perf_counter()
        o = llm.generate([{"prompt": prompt(), "multi_modal_data": {"audio": [(audio, SR)]}}],
                         sampling_params=sp, use_tqdm=False)[0].outputs[0]
        ms = (time.perf_counter() - t0) * 1000
        fired = list(o.token_ids).count(END_ID) > 0
        print(f"TEXT[{name}] fired={fired} {ms:.0f}ms :: {clean(o.text)[:140]!r}")

    print("E7_DONE")


if __name__ == "__main__":
    main()
