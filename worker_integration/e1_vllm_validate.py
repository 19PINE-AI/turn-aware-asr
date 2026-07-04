"""E1b: validate that vLLM can serve the merged v9 endpoint checkpoint and
that the endpoint markers decode as ordinary tokens.

Runs under the system python3.10 (vLLM 0.19, native Qwen3ASRForConditionalGeneration).
Two probes:
  * speech + 0.5 s trailing silence  -> expect <END_SPEECH> (fire)
  * pure silence                     -> expect NO marker, ~empty text (no-fire)

Usage (GPU):
    python3 worker_integration/e1_vllm_validate.py \
        --model checkpoints/merged/qwen3-asr-0.6b-endpoint-v9 \
        --flac ~/data/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac
"""
import argparse, glob, os
import numpy as np
import soundfile as sf

SR = 16000
END = "<END_SPEECH>"
EAGER = "<EAGER_END_SPEECH>"


def load_16k(path):
    a, sr = sf.read(os.path.expanduser(path), dtype="float32")
    if a.ndim > 1:
        a = a.mean(-1)
    if sr != SR:
        import math
        # cheap linear resample to 16k
        n = int(round(len(a) * SR / sr))
        a = np.interp(np.linspace(0, len(a), n, endpoint=False), np.arange(len(a)), a).astype("float32")
    return a


def prompt():
    # MUST match the v9 training chat template EXACTLY, incl. the empty system
    # block — the model learned marker emission conditioned on this prompt.
    # (apply_chat_template([{system,''},{user,[audio]}], add_generation_prompt=True))
    return ("<|im_start|>system\n<|im_end|>\n"
            "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
            "<|im_start|>assistant\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--flac", default="~/data/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac")
    ap.add_argument("--gpu-mem-util", type=float, default=0.15)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, trust_remote_code=True,
              gpu_memory_utilization=args.gpu_mem_util, max_model_len=4096,
              enable_prefix_caching=True, limit_mm_per_prompt={"audio": 1})
    # CRITICAL: the markers are reserved-slot special tokens; without
    # skip_special_tokens=False vLLM strips them from .text even though the
    # model emits them. We also scan raw token_ids to be certain.
    sp = SamplingParams(temperature=0.0, max_tokens=96, skip_special_tokens=False)

    tok = llm.get_tokenizer()
    EAGER_ID = tok.convert_tokens_to_ids(EAGER)
    END_ID = tok.convert_tokens_to_ids(END)
    print(f"marker ids: EAGER={EAGER_ID} END={END_ID}")

    speech = load_16k(args.flac)
    sil = np.zeros(int(0.5 * SR), dtype="float32")
    probes = {
        "speech+0.5s_sil (expect FIRE)": np.concatenate([speech, sil]),
        "pure_silence (expect NO fire)": np.zeros(int(2.0 * SR), dtype="float32"),
        "speech_no_sil (may not fire)": speech,
    }
    for name, audio in probes.items():
        req = {"prompt": prompt(), "multi_modal_data": {"audio": [(audio, SR)]}}
        out = llm.generate([req], sampling_params=sp, use_tqdm=False)
        o = out[0].outputs[0]
        text = o.text
        ids = list(o.token_ids)
        n_end_ids = ids.count(END_ID)
        n_eager_ids = ids.count(EAGER_ID)
        print(f"\n### {name}")
        print(f"    text n_END={text.count(END)} n_EAGER={text.count(EAGER)} | "
              f"token_ids n_END={n_end_ids} n_EAGER={n_eager_ids}  fired={n_end_ids>0}")
        print(f"    text: {text[:200]!r}")

    print("\nE1B_DONE")


if __name__ == "__main__":
    main()
