# Silence-appended smoke test: the v3/v5 frontier was an eval artifact (2026-07-04)

Hypothesis (from re-reading the v1–v8 record): the AMI conversational eval
cuts clips at the forced-alignment end of speech, so "fire at end-of-audio
without trailing silence" — a condition that never occurs in deployment,
where silence keeps arriving — was being scored as the definition of
endpoint competence. Models trained toward streaming behavior (fire on
silence-after-speech) were scored as broken on AMI for correctly waiting
for silence the eval had cut off.

Test: identical eval (`ami_conversational_eval.py`, 50/schema, seed 0,
held-out meetings), with `--append-silence-s 1.0` adding one second of
silence to every clip. 1.0 s is in-distribution for v5's trailing-silence
training (0.3–1.0 s tails).

## Setup notes

`data/` had been cleaned from disk; AMI ihm parquet re-downloaded from
`edinburghcstr/ami` (refs/convert/parquet, all 51 shards) and the
train/eval meeting split reconstructed from the deterministic
md5(meeting_id) % 3 rule (115 train / 56 eval meetings). Reproduction
fidelity of the 0 s arms: **single recall reproduces exactly** for all
three models (1.00 / 0.82 / 0.10); double within 6 pp; the disfluency
pool drew differently (correct rates shifted up ~10–16 pp vs the
historical JSONs), so silence-effect comparisons below are within-run
(0 s vs 1 s arms on identical examples), not vs the historical numbers.

## Results (all: 50 single / 50 double / 50 disfluency, held-out meetings)

| Model | Arm | Single ≥1 | Double ≥2 | Disfl =1 (correct) | Disfl ≥2 (over-fire) |
|---|---|---|---|---|---|
| v3  | 0 s   | 1.00 | 0.88 | 0.88 | 0.10 |
| v3  | 1.0 s | 1.00 | 0.88 | 0.90 | 0.10 |
| v5  | 0 s   | 0.82 | 0.76 | 0.82 | 0.10 |
| v5  | **1.0 s** | **0.98** | 0.80 | **0.88** | **0.10** |
| v8-12k | 0 s | 0.10 | 0.84 | 0.84 | 0.16 |
| v8-12k | **1.0 s** | **1.00** | **0.98** | 0.54 | 0.46 |

## Findings

1. **The frontier was the eval.** One second of deployment-realistic
   silence takes v8-12k's single recall from 0.10 to 1.00 and v5's from
   0.82 to 0.98. v3 (control) is unchanged — silence hurts nothing.
   Under fair conditions there is no v3-vs-v5 trade on AMI singles:
   all three models are at 0.98–1.00.
2. **v8-12k is the strongest turn detector**: double 0.98 with silence.
   Its "10 % single / no-fire-mode attractor" verdict in research/50 was
   ~entirely the artifact. Eight experiments (v4–v8) chased a data recipe
   for a contradiction the eval itself manufactured.
3. **Gap stratification (1.0 s arms).** Double recall by internal gap:
   v3 0.92/0.87/0.87, v5 1.00/0.73/0.74, v8 1.00/1.00/0.96 for
   gap [0,0.3)/[0.3,1)/[1,3) s. Two surprises: (a) sub-300 ms speaker
   changes are detected near-perfectly by ALL models — in offline decode
   the internal marker is placed from speaker-change/semantic cues, not
   silence; tight turn-taking is not the weak spot offline (streaming
   behavior may differ — replay eval will tell). (b) v5's weakness is
   *mid-length* gaps, where it appears to treat a 0.5–3 s pause as
   possible continuation.
4. **v8's disfluency "over-fire" (0.46) is a policy, not a defect.**
   Disfluency examples have median internal gap 1.39 s (88 % ≥ 1 s —
   research/58 contradiction 2). v8 fires at long same-speaker pauses
   after complete-sounding speech — the causally correct behavior when
   the future is unobservable; the metric demands clairvoyance. v3/v5
   suppress those fires using same-speaker voice continuity (observable
   offline because the resume IS in the clip). In streaming the resume
   hasn't happened yet at decision time; the honest metric is
   `resume_after_fire` (research/58), not "over-fire".

## Implications

- Retire the 0 s offline protocol; it scored a deployment-impossible
  condition and produced the false Pareto conclusion of research/50.
- v8-12k and v5 are both live candidates again; the deciding axis moves
  to streaming premature-fire behavior + latency on the unified replay
  eval (research/58, runs in progress: `61-replay-{v3,v5,v8}.json`).
- v9 (research/59) remains motivated by contradiction 2 (causal labels
  for pause policy), not by the single-utterance frontier, which is gone.

## Files

- `research/57-smoke-{v3,v5,v8}-sil{0.0,1.0}.json` — six arms
- `eval/ami_conversational_eval.py` — new `--append-silence-s` flag
