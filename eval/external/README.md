# Evaluate external turn detectors

[Project README](../../README.md) · [Training and evaluation](../../REPRODUCING.md) · [Recorded comparison](../../research/102-exp4-external-baselines.md)

This harness runs other turn detectors on the same continuous AMI audio used to
evaluate our model. Each adapter produces turn-ending timestamps; shared scoring
code computes recall, latency, and false fires. A configuration in a threshold
sweep is called an **arm**.

## Before you run

Prepare the AMI IHM parquet files and meeting split described in the
[training guide](../../REPRODUCING.md#prepare-the-data). Run commands below from
the repository root. Install `uv` for the environment-creation commands, or create
equivalent Python 3.11 virtual environments yourself. Separate environments avoid
conflicts between the different model runtimes. Weights download on first use.

| System | Adapter name | Environment | Runtime |
| --- | --- | --- | --- |
| Smart Turn v3 | `smart_turn` | `.venv_ext_smart_turn` | ONNX, CPU |
| LiveKit turn detector | `livekit_turn` | `.venv_ext_smart_turn` | ONNX, CPU; oracle text by default |
| Parakeet-Realtime-EOU-120m | `parakeet_eou` | `.venv_ext_nemo` | NeMo; GPU recommended |
| Kyutai STT semantic-VAD | `kyutai_vad` | `.venv_ext_moshi` | Moshi; GPU recommended |

The completed development-set comparison is in
[research/102-exp4-external-baselines.md](../../research/102-exp4-external-baselines.md);
its recorded results are `research/95-exp4-*-dev25.json`. The commands below write
to `eval/results/` to keep those records intact. External weights retain their
upstream licenses; see each model's published terms.

## Shared command and output

After installing an adapter's environment:

```bash
<venv>/bin/python -m eval.external.harness \
  --adapter <name> --n-stretches 25 --seed 0 \
  --out eval/results/exp4-<name>-seed0.json
```

Replace `<venv>` and `<name>` using the table. Use `--limit 1` for a short smoke
run. `--ami-dir` and `--split` override data paths; `--chunk-s` defaults to 0.5.
`--use-train-meetings` explicitly changes the meeting selection to training
meetings, so do not use it for a held-out comparison. `--n-stretches 50 --seed 1`
selects the fresh-set construction used in the experiments.

Output uses `arms[key] = {summary, per_stretch}`. A fire during chunk `k` is
reported at `(k + 1) * chunk_s`; boundary matching uses the shared
`[-0.25, +1.5]` second window. Smart Turn and LiveKit use the common RMS energy
threshold (`1e-3`) to trigger classification after silence. Keep the audio,
split, seed, chunk size, and scoring settings matched across systems.

`EXT_MODEL_DIR=data/external_models` controls the ONNX/HF cache used by these
adapters; NeMo and Moshi also use their own Hugging Face caches.

## 1. Smart Turn v3

Acoustic turn-completion classifier (Whisper-tiny encoder + linear head, 8M,
ONNX). Fires the classifier at each RMS VAD-silence trigger on the trailing 8 s;
fire iff completion prob ≥ threshold. Arms = thresholds {0.3,0.4,0.5,0.6,0.7}.

Setup:

```bash
uv venv .venv_ext_smart_turn --python 3.11
uv pip install --python .venv_ext_smart_turn/bin/python \
    numpy pyarrow soundfile tqdm jiwer onnxruntime "transformers>=4.40" \
    huggingface_hub jinja2
```
Model `pipecat-ai/smart-turn-v3` (`smart-turn-v3.2-cpu.onnx`) auto-downloads.

Smoke:

```bash
EXT_MODEL_DIR=data/external_models .venv_ext_smart_turn/bin/python -m eval.external.harness \
  --adapter smart_turn --n-stretches 2 --limit 2 --out eval/results/exp4-smart_turn-smoke.json
```

**Full scoring (dev 25):**

```bash
EXT_MODEL_DIR=data/external_models .venv_ext_smart_turn/bin/python -m eval.external.harness \
    --adapter smart_turn --n-stretches 25 --seed 0 --out eval/results/exp4-smart_turn-dev25.json
```
Fresh 50: `--n-stretches 50 --seed 1` → `...-fresh50.json`. CPU-only.

## 2. LiveKit turn detector

Text EOU classifier (135M SmolLM-v2, INT8 ONNX, CPU) over the transcript so far,
invoked at each RMS VAD-silence trigger; fire iff EOU prob ≥ threshold. Arms =
thresholds {0.3,0.5,0.7,0.8,0.9}. Shares `.venv_ext_smart_turn`.

Model `livekit/turn-detector` (`model_quantized.onnx` + tokenizer) auto-downloads.

**Transcript source:** the default is `transcript_mode="oracle"`: ground-truth
words available up to the trigger. This measures the detector with an upper
bound on transcript quality; it is not a complete speech-recognition pipeline.
Do not interpret its near-zero WER as a recognizer result. The paper's recorded
LiveKit comparison uses this oracle mode.

To supply real transcripts, populate `adapter.external_transcripts` with
`(word_end_s, word)` pairs keyed by `"{meeting_id}:{speaker_id}"` and set
`adapter.transcript_mode="external"`. Generating those transcripts is a separate
step; the harness does not wire up LiveKit's default speech recognizer.

**Full scoring (dev 25, oracle transcript):**

```bash
EXT_MODEL_DIR=data/external_models .venv_ext_smart_turn/bin/python -m eval.external.harness \
    --adapter livekit_turn --n-stretches 25 --seed 0 --out eval/results/exp4-livekit-dev25.json
```

## 3. Parakeet-Realtime-EOU-120m (evaluation-only license)

FastConformer-RNNT (`EncDecRNNTBPEModel`), cache-aware streaming, emits `<EOU>`
at each end-of-utterance. Fire = each `<EOU>` token's decoded timestamp
(read from char-level timestamps of a single decode; cache-aware models produce
identical offline/streaming predictions) quantised to the 0.5 s grid. Single arm
`eou`. Records the transcript for the WER row.

Setup:

```bash
uv venv .venv_ext_nemo --python 3.11
uv pip install --python .venv_ext_nemo/bin/python \
    numpy pyarrow soundfile tqdm jiwer "nemo_toolkit[asr]>=2.5.3"
```
Model `nvidia/parakeet_realtime_eou_120m-v1` auto-downloads via NeMo.

Smoke (CPU):

```bash
EXT_MODEL_DIR=data/external_models CUDA_VISIBLE_DEVICES="" .venv_ext_nemo/bin/python -m \
    eval.external.harness --adapter parakeet_eou --n-stretches 1 --limit 1 --out eval/results/exp4-parakeet-smoke.json
```

**Full scoring (dev 25, GPU):**

```bash
EXT_MODEL_DIR=data/external_models .venv_ext_nemo/bin/python -m eval.external.harness \
    --adapter parakeet_eou --n-stretches 25 --seed 0 --out eval/results/exp4-parakeet-dev25.json
```
The model has its own NVIDIA license; the repository Apache license does not cover these weights.
Fresh 50: `--n-stretches 50 --seed 1`.

## 4. Kyutai STT semantic-VAD head (GPU strongly preferred)

1B delayed-streams model. Fire = 2.0 s pause head (index 2, matches T_TURN)
crossing threshold on a rising edge; arms = {0.3,0.4,0.5,0.6,0.7}; records the
STT transcript for the WER row. Smoke on 1 stretch: emitted 13 fires + a real
transcript (WER 0.38) — correctly shaped and scored.

Setup:

```bash
uv venv .venv_ext_moshi --python 3.11
uv pip install --python .venv_ext_moshi/bin/python \
    numpy pyarrow soundfile tqdm jiwer "moshi>=0.2.6" scipy setuptools
```
NOTE: do **not** install `torchaudio` (its prebuilt binary is ABI-incompatible
with the torch moshi pulls, and crashes on import) — we resample 16k→24k with
`scipy.signal.resample_poly` instead. `setuptools` is required by moshi's
`torch.compile`. Model `kyutai/stt-1b-en_fr-candle` auto-downloads. Env knobs:
`KYUTAI_STT_REPO`, `KYUTAI_VAD_HEAD` (default 2).

Smoke on CPU:

```bash
EXT_MODEL_DIR=data/external_models CUDA_VISIBLE_DEVICES="" .venv_ext_moshi/bin/python -m eval.external.harness \
  --adapter kyutai_vad --n-stretches 1 --limit 1 --out eval/results/exp4-kyutai-smoke.json
```

GPU execution is recommended for full scoring. `KYUTAI_VAD_HEAD` selects the
pause head; changing it changes the evaluated configuration. Keep its value and
the thresholds fixed when comparing to the recorded paper results.

**Full scoring (dev 25, GPU):**

```bash
EXT_MODEL_DIR=data/external_models .venv_ext_moshi/bin/python -m eval.external.harness \
    --adapter kyutai_vad --n-stretches 25 --seed 0 --out eval/results/exp4-kyutai-dev25.json
```

---

## Adding an adapter

Subclass `eval/external/adapters/base.py::BaseAdapter`, implement `setup()` and
`infer_stretch(audio, chunk_s) -> ({arm_key: [fire_time_s]}, transcript_or_None)`,
register it in `harness.py::ADAPTERS`. Reuse `vad_silence_triggers` /
`trailing_window` for classifier-at-trigger systems. If it needs ground-truth
text or an external transcript, add a `set_stretch(events, key)` method — the
harness calls it before each `infer_stretch`.

`dummy_rms` is a CPU-only harness fixture (RMS silence-timeout, no model) to
validate the pipeline end-to-end; it should reproduce `timeout_baseline_eval`'s
rms arm.
