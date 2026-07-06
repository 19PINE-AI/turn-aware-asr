# Exp-4: external turn-aware baselines on the replay benchmark

First cross-system measurement of the open turn-aware class on our
deployment-matched protocol. Each external system is wrapped in an **adapter**
that consumes the *exact* benchmark stretches
(`eval.streaming_replay_eval.build_stretches`, same seed / split / n-stretches as
the LM and timeout arms) and emits fire timestamps, which the shared **harness**
scores with the byte-identical `score_fires` + aggregation used everywhere else.
So the boundary taxonomy, the `[-0.25, +1.5] s` window, and the false/min
definition are unchanged; output JSON matches `eval/timeout_baseline_eval.py`'s
shape (`arms[key] = {summary, per_stretch}`), ready for `fig3_tradeoff` / `tab:main`.

**Convention:** a fire "at chunk k" is timestamped `(k+1)*chunk_s` (chunk_s=0.5),
identical to the LM arms. Classifier-at-trigger systems (Smart Turn, LiveKit)
reuse the LM arms' **RMS energy detector** (`gate_rms=1e-3`) as the VAD-silence
trigger, so the comparison isolates the *policy*, not the detector.

## Status (2026-07-06)

| System | Adapter | Venv | Weights | Smoke | License |
|---|---|---|---|---|---|
| Smart Turn v3 | `smart_turn` | `.venv_ext_smart_turn` | auto (HF) | ✅ CPU | BSD-2 (OSI) |
| LiveKit turn detector | `livekit_turn` | `.venv_ext_smart_turn` | auto (HF) | ✅ CPU | Apache-2.0 (OSI) |
| Parakeet-Realtime-EOU-120m | `parakeet_eou` | `.venv_ext_nemo` | auto (HF/NeMo) | ✅ CPU | **non-OSI, eval-only** |
| Kyutai STT semantic-VAD | `kyutai_vad` | `.venv_ext_moshi` | auto (HF) | ✅ CPU (slow) | CC-BY-4.0 |

`✅ CPU` = loaded + emitted a correctly-shaped fire on a benchmark stretch
through the harness (no GPU touched — a training job holds the GPU). None of the
**full 25/50-stretch scoring runs** have been done; commands below.

## Harness

```
<venv>/bin/python -m eval.external.harness \
    --adapter <name> --n-stretches 25 --seed 0 \
    --out research/exp4-<name>-seed0.json
```
Flags: `--limit N` (only first N stretches, for smoke), `--use-train-meetings`
(dev set), `--chunk-s 0.5`, `--split`, `--ami-dir`. One adapter can emit several
arms (a threshold sweep) from one model pass; each arm is scored + summarised.
Stretch building reads the AMI parquet (~10 s, CPU) each run — deterministic
given `--seed`.

Set `EXT_MODEL_DIR=data/external_models` to cache ONNX/HF weights there (NeMo and
moshi use their own `~/.cache/huggingface` cache).

---

## 1. Smart Turn v3  ✅ ready

Acoustic turn-completion classifier (Whisper-tiny encoder + linear head, 8M,
ONNX). Fires the classifier at each RMS VAD-silence trigger on the trailing 8 s;
fire iff completion prob ≥ threshold. Arms = thresholds {0.3,0.4,0.5,0.6,0.7}.

Setup (done):
```
uv venv .venv_ext_smart_turn --python 3.11
uv pip install --python .venv_ext_smart_turn/bin/python \
    numpy pyarrow soundfile tqdm jiwer onnxruntime "transformers>=4.40" \
    huggingface_hub jinja2
```
Model `pipecat-ai/smart-turn-v3` (`smart-turn-v3.2-cpu.onnx`) auto-downloads.

Smoke: `EXT_MODEL_DIR=data/external_models .venv_ext_smart_turn/bin/python -m \
eval.external.harness --adapter smart_turn --n-stretches 2 --limit 2 --out research/exp4-smart_turn-smoke.json`

**Full scoring (dev 25 — run me):**
```
EXT_MODEL_DIR=data/external_models .venv_ext_smart_turn/bin/python -m eval.external.harness \
    --adapter smart_turn --n-stretches 25 --seed 0 --out research/exp4-smart_turn-dev25.json
```
Fresh 50 (top-2 only): `--n-stretches 50 --seed 1` → `...-fresh50.json`. CPU-only.

## 2. LiveKit turn detector  ✅ ready

Text EOU classifier (135M SmolLM-v2, INT8 ONNX, CPU) over the transcript so far,
invoked at each RMS VAD-silence trigger; fire iff EOU prob ≥ threshold. Arms =
thresholds {0.3,0.5,0.7,0.8,0.9}. Shares `.venv_ext_smart_turn`.

Model `livekit/turn-detector` (`model_quantized.onnx` + tokenizer) auto-downloads.

**Transcript source (spec's two variants):**
- **default `transcript_mode="oracle"`** — ground-truth words with `end_s ≤`
  trigger. An *upper bound* on transcript quality (clearly labelled in the
  `license_note`); good enough to size the detector's discrimination and to
  smoke-test. WER row will read ~0 (it *is* the reference) — ignore it.
- **variant (b) real transcript** — run our base model's streaming decode,
  collect `(word_end_s, word)` per stretch keyed `"{meeting_id}:{speaker_id}"`,
  set `adapter.transcript_mode="external"` and `adapter.external_transcripts`.
  This needs the GPU (base-model decode) — wire it in the scoring run.

**Full scoring (dev 25, oracle transcript — run me):**
```
EXT_MODEL_DIR=data/external_models .venv_ext_smart_turn/bin/python -m eval.external.harness \
    --adapter livekit_turn --n-stretches 25 --seed 0 --out research/exp4-livekit-dev25.json
```
CPU-only. (a) as-shipped over LiveKit's default STT is **not wired** — would need
LiveKit's STT pairing; report the oracle/base-model variants instead.

## 3. Parakeet-Realtime-EOU-120m  ✅ ready (⚠ non-OSI, eval-only)

FastConformer-RNNT (`EncDecRNNTBPEModel`), cache-aware streaming, emits `<EOU>`
at each end-of-utterance. Fire = each `<EOU>` token's decoded timestamp
(read from char-level timestamps of a single decode; cache-aware models produce
identical offline/streaming predictions) quantised to the 0.5 s grid. Single arm
`eou`. Records the transcript for the WER row.

Setup (done):
```
uv venv .venv_ext_nemo --python 3.11
uv pip install --python .venv_ext_nemo/bin/python \
    numpy pyarrow soundfile tqdm jiwer "nemo_toolkit[asr]>=2.5.3"
```
Model `nvidia/parakeet_realtime_eou_120m-v1` auto-downloads via NeMo.

Smoke (CPU, GPU untouched):
```
EXT_MODEL_DIR=data/external_models CUDA_VISIBLE_DEVICES="" .venv_ext_nemo/bin/python -m \
    eval.external.harness --adapter parakeet_eou --n-stretches 1 --limit 1 --out research/exp4-parakeet-smoke.json
```

**Full scoring (dev 25 — run me, GPU; leave CUDA visible so it uses the GPU):**
```
EXT_MODEL_DIR=data/external_models .venv_ext_nemo/bin/python -m eval.external.harness \
    --adapter parakeet_eou --n-stretches 25 --seed 0 --out research/exp4-parakeet-dev25.json
```
Non-OSI license (NVIDIA open-model / research-eval) — note in the paper caption.
Fresh 50: `--n-stretches 50 --seed 1`.

## 4. Kyutai STT semantic-VAD head  ✅ ready (GPU strongly preferred)

1B delayed-streams model. Fire = 2.0 s pause head (index 2, matches T_TURN)
crossing threshold on a rising edge; arms = {0.3,0.4,0.5,0.6,0.7}; records the
STT transcript for the WER row. Smoke on 1 stretch: emitted 13 fires + a real
transcript (WER 0.38) — correctly shaped and scored.

Setup (done):
```
uv venv .venv_ext_moshi --python 3.11
uv pip install --python .venv_ext_moshi/bin/python \
    numpy pyarrow soundfile tqdm jiwer "moshi>=0.2.6" scipy setuptools
```
NOTE: do **not** install `torchaudio` (its prebuilt binary is ABI-incompatible
with the torch moshi pulls, and crashes on import) — we resample 16k→24k with
`scipy.signal.resample_poly` instead. `setuptools` is required by moshi's
`torch.compile`. Model `kyutai/stt-1b-en_fr-candle` auto-downloads. Env knobs:
`KYUTAI_STT_REPO`, `KYUTAI_VAD_HEAD` (default 2).

Smoke (CPU is ~30 s per 60 s stretch): `EXT_MODEL_DIR=data/external_models \
CUDA_VISIBLE_DEVICES="" .venv_ext_moshi/bin/python -m eval.external.harness \
--adapter kyutai_vad --n-stretches 1 --limit 1 --out research/exp4-kyutai-smoke.json`

**Caveat for the full run:** on CPU the 1B model is slow (~30 s / stretch) — run
the dev-25/fresh-50 scoring on the **GPU** (leave CUDA visible). The 13 fires on
one stretch suggest the 2.0 s head fires on many internal pauses; the threshold
sweep (arms) produces the trade-off curve, and `KYUTAI_VAD_HEAD` can be raised
to the 3.0 s head if the 2.0 s head is too eager. This is a tuning dial for the
scoring run, not a blocker.

**Full scoring (dev 25 — run me, GPU):**
```
EXT_MODEL_DIR=data/external_models .venv_ext_moshi/bin/python -m eval.external.harness \
    --adapter kyutai_vad --n-stretches 25 --seed 0 --out research/exp4-kyutai-dev25.json
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
