# 74 — Dictation probes: baseline (causal/v9 checkpoint)

**Date:** 2026-07-05. **Probe builder:** `eval/build_dictation_probes.py`;
**eval:** `eval/dictation_probe_eval.py`; **raw:** `74-dictation-probe-v9.json`.

Motivation (paper §2): the two turns no timeout can get right — hold through
dictation pauses, fire instantly on semantic completion. The v9 completeness
heuristic reads digit strings as complete, so we predicted premature fires in
inter-group pauses. Also: spelled names/emails should improve with a profile
CTX in the system slot.

## Probe A — digit dictation (50 ten-digit numbers, 3-3-4 groups from FSDD
held-out speakers george+lucas; pauses 0.6–1.2 s; committed-prefix streaming,
gate on, no confirm)

| metric | v9 | timeout X=0.5 | timeout X=1.0 |
|---|---|---|---|
| premature fires / sequence | **4.80** | 2.00 | 0.74 |
| sequences with ≥1 premature | **100%** | 100% | ~52% |
| final-boundary recall | 0.76 | 1.00 | 1.00 |
| final latency P50 | 0.50 s | 0.50 s | 1.00 s |
| digit word accuracy | 0.91 | n/a | n/a |

**Finding A1 (prediction confirmed, worse than predicted):** the causal model
is *worse than a plain 0.5 s silence timeout* on dictation — it fires not only
in the two inter-group pauses but after individual digits mid-group (fires at
~1 s intervals; see per_item hyps: segments of 1–3 digits). Digit-group
prosody + ≥0.3 s silence satisfies the v9 fire rule everywhere. This is the
continuation-expectation gap named in the paper's Limitations, now measured.

**Finding A2:** streaming digit WER is decent (0.91 word acc) but transcripts
are shredded into 5–7 segments per number by the premature fires — a
downstream agent would answer mid-number 4.8 times per dictation.

## Probe B — spelled names/emails (40+40 edge-tts items, voices Jenny/Guy;
offline single-shot decode; conditions: no ctx / profile ctx / distractor ctx)

| entity | none | profile | distractor |
|---|---|---|---|
| name (spelled) exact | 0.300 | **0.825** | 0.300 |
| email exact | 0.000 | **0.400** | 0.000 |
| email (spelled half) | 0.000 | 0.300 | — |
| email (natural half) | 0.000 | 0.500 | — |

Distractor intrusion rate: 1.3% (profiles for the wrong user do not poison).

**Finding B1:** profile CTX lifts spelled-name exact match +52.5 pp (0.30 →
0.825) with zero distractor cost — Bo's user-profile thesis holds on the base
capability alone (v9 never trained with ctx).

**Finding B2:** emails are unusable without context (0.000: the model
transcribes "ana dot kowalski … at meridianhealth dot org" with spelling
drift and no normalization) and only 0.40 with it. The model often ignores
the profile string when the audio spells letter-by-letter. Headroom for v10's
`spell_email` schema (target = normalized written form; 50% with profile ctx).

## v10 response (data built, training launched)

`eval/build_v10_training_data.py`: v9's 11,950 examples + 1,660 new across 5
schemas — digit_hold 450 (3/6-digit prefixes + pause → NO marker), digit_fire
450 (full 10 digits w/ internal pauses + tail → marker), digit_nosil 200
(full digits, no tail → NO marker), spell_name 280, spell_email 280 (targets
in normalized written form; 50% carry profile ctx; ~15% no-tail minimal
pairs). FSDD train speakers jackson/nicolas/theo/yweweler (probe speakers
held out); TTS train voices exclude probe's Jenny/Guy; name lists disjoint.
Trainer patched to pass per-example `ctx` into the system slot (empty default
— v9 examples unchanged). Run: `scripts/train_v10_resilient.sh` → 15k steps,
eval every 1500, holdout 130 (10/schema), score-spec v9, early-stop 4.

**Success criteria for v10:** premature/seq ≤ 0.5 with final recall ≥ 0.9 on
Probe A (beat both timeout rows simultaneously); email profile ≥ 0.7 on Probe
B; no regression on the 25-stretch replay benchmark (recall ≥ 0.95, false/min
≤ 0.5) or on the v9 holdout schemas.
