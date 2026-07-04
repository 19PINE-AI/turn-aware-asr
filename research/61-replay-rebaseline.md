# Replay-eval re-baseline: v3/v5/v8 under deployment conditions (2026-07-04)

First run of the unified streaming-replay eval (design research/58,
driver `eval/streaming_replay_eval.py`): 25 single-speaker stretches from
held-out AMI meetings (22.2 min audio, 108 turn-final + 15 continuation
boundaries), 0.5 s chunks, committed-prefix decoding, marker fires
timestamped and scored causally.

## Ungated results (LM alone on the raw channel)

| Model | Recall | P50 | P95 | False fires /speech-min | Silence-spam fires | WER |
|---|---|---|---|---|---|---|
| v3 | 0.991 | 0.04 s | 0.25 s | **97.8** | 1519 | 3.59 |
| v5 | 0.991 | 0.15 s | 0.88 s | 31.4 | 1475 | 3.03 |
| v8-12k | 0.954 | 0.25 s | 0.56 s | **3.9** | 1449 | 7.56 |

Two headline findings:

**1. All three models are silence-incompetent — a universal training-data
gap the offline evals could never see.** Every v1–v8 training example
begins with speech; a real single-speaker channel is mostly silence
(13.0 of 22.2 min here). On silence-only segments the LM hallucinates
text and fires markers at ~1.9/s (corr(silence duration, spam fires) =
0.997 across stretches). Consequences: (a) the flush→fresh-segment→
silence→hallucinate loop wrecks streaming WER (median stretch ≈ 2.0);
(b) **ungated recall is uninterpretable** — a random 1.9/s spammer
would score 0.96 recall by chance against the 1.75 s hit window. The
gated arms below are the real metric.

**2. The historical ranking inverts on speech behavior.** v3 — the
research/50 "meeting champion" — fires ~98×/speech-min mid-utterance:
its offline strength (fire at semantic completion) is a premature-fire
pathology under causal streaming, the same behavior that showed as
P50 −9.57 s in the old synthetic eval. v8-12k — declared broken at 10 %
AMI single — has the cleanest speech behavior (3.9 false/min) at 0.954
recall / 0.25 s P50: its streaming no-fire-mid-utterance training
works. The dual-checkpoint ship recommendation of research/50 would
have failed in production in BOTH deployments.

## Gated results (energy gate: never start a segment on a silent chunk)

`--energy-gate` models the production composition policy from
research/60: a trivial RMS check in front of the LM. Within-segment
behavior (including post-speech silence for the fire decision) is
untouched — the gate only stops the LM from decoding pure silence.

*(results pending — arms running; table filled in below when complete)*

| Model | Recall | P50 | P95 | False /speech-min | Resume-rate | WER | Chunks gated |
|---|---|---|---|---|---|---|---|
| v3 + gate | | | | | | | |
| v5 + gate | | | | | | | |
| v8 + gate | | | | | | | |

## Consequences

1. **The endpoint answer is system-shaped**: LM semantics + an energy
   gate. The gate is one RMS comparison per chunk — no acoustic head
   required for the silence problem (the 0.33 M head of research/56
   remains an option for *refining* fire timing, not a necessity).
   This is also the Metronome-stack policy (research/60): skipping
   silent channels saves the decode entirely, which multiplies
   sessions-per-GPU in realistic (mostly-silent) conversations.
2. **v9's silence schemas are mandatory, not optional**: the model must
   be safe even without the gate (gate misses low-energy speech onset;
   defense in depth). `silence_only` (empty target) + `lead_sil` added
   to the builder before the v9 run.
3. **resume_after_fire ≈ 0.9 for all models**: same-speaker quick
   resumes (0.3–2 s gap) essentially always get a fire. Under the causal
   analysis (research/58) this is expected behavior with a cost knob;
   v9's pair_hold schema (incomplete-A ⇒ hold) is the lever that can
   selectively reduce it without clairvoyance.
4. Streaming WER must be read post-gate; ungated WER mixes real
   transcription with silence hallucination.

## Files

- `research/61-replay-{v3,v5,v8}.json` — ungated arms
- `research/61-replay-{v3,v5,v8}gate.json` — gated arms
- `eval/streaming_replay_eval.py` — driver (committed-prefix,
  `--energy-gate`, `--from-scratch`)
