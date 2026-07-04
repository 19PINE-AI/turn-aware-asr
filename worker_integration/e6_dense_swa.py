"""E6: validate Metronome's dense-Qwen3 windowed-KV (SWA) on the Qwen3-ASR endpoint model.

Metronome's sliding-window patch (metronome_vllm_plugin / patches/qwen3_swa.py) originally
covered only the Qwen3-Omni-MoE text backbone. This test exercises the extension to the DENSE
Qwen3ForCausalLM thinker used by the merged v9 endpoint checkpoint.

With METRONOME_SWA_TOKENS=W set, vLLM builds the thinker's decoder attention with a
per-layer sliding window -> a SlidingWindowSpec KV manager that bounds BOTH per-frame
attention compute AND resident KV. This probes three things:
  1. the dense patch fires inside EngineCore (grep the captured log),
  2. endpoint behavior is preserved (fires <END_SPEECH> on speech+silence, silent otherwise),
  3. per-decode latency stays FLAT as audio context grows past W (the windowed-KV signature;
     unwindowed full attention grows with context).

Runs under system python3.10 (vLLM). Set METRONOME_SWA_TOKENS before launching.

Usage:
    METRONOME_SWA_TOKENS=256 python3 worker_integration/e6_dense_swa.py \
        --model checkpoints/merged/qwen3-asr-0.6b-endpoint-v9
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


def prompt():
    return ("<|im_start|>system\n<|im_end|>\n"
            "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
            "<|im_start|>assistant\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--flac", default="~/data/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac")
    ap.add_argument("--gpu-mem", type=float, default=0.15)
    ap.add_argument("--durations", default="5,15,30,60")
    args = ap.parse_args()

    W = int(os.environ.get("METRONOME_SWA_TOKENS", "0") or "0")
    print(f"### METRONOME_SWA_TOKENS = {W}")

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, trust_remote_code=True,
              gpu_memory_utilization=args.gpu_mem, max_model_len=8192,
              enable_prefix_caching=False, limit_mm_per_prompt={"audio": 1})
    tok = llm.get_tokenizer()
    END_ID = tok.convert_tokens_to_ids(END)
    sp = SamplingParams(temperature=0.0, max_tokens=96, skip_special_tokens=False)

    speech = load_16k(args.flac)

    # (2) endpoint behavior preserved under the window
    print("\n## behavior probes (under window):")
    sil = np.zeros(int(0.5 * SR), dtype="float32")
    for name, audio in [("speech+0.5s_sil (expect FIRE)", np.concatenate([speech, sil])),
                        ("pure_silence (expect no fire)", np.zeros(int(2.0 * SR), dtype="float32"))]:
        o = llm.generate([{"prompt": prompt(), "multi_modal_data": {"audio": [(audio, SR)]}}],
                         sampling_params=sp, use_tqdm=False)[0].outputs[0]
        fired = list(o.token_ids).count(END_ID) > 0
        print(f"    {name}: fired={fired}")

    # (3) latency vs growing audio context — flat => window active
    print("\n## per-decode latency vs audio context (flat => windowed KV active):")
    durs = [float(d) for d in args.durations.split(",")]
    for d in durs:
        reps = int(np.ceil(d * SR / len(speech)))
        buf = np.tile(speech, reps)[: int(d * SR)]
        # warm + timed
        req = {"prompt": prompt(), "multi_modal_data": {"audio": [(buf, SR)]}}
        llm.generate([req], sampling_params=sp, use_tqdm=False)
        t0 = time.perf_counter()
        llm.generate([req], sampling_params=sp, use_tqdm=False)
        ms = (time.perf_counter() - t0) * 1000
        print(f"    ctx={d:>4.0f}s (~{int(d*12.5):>4d} audio pos)   decode {ms:7.1f} ms")

    print("\nE6_DONE")


if __name__ == "__main__":
    main()
