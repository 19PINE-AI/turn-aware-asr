# v9 training recipe — consistent causal label spec (2026-07-04)

v6–v8 failed to unify because the training data (inherited from the two
evals) labeled causally identical inputs with opposite outputs — see
`58-unified-eval-design.md` for both contradictions. v9 rebuilds the data
under one rule and changes nothing else (same LoRA recipe, same base
model), so the experiment isolates the label spec.

## The label rule

Emit `<EAGER_END_SPEECH><END_SPEECH>` (the historical adjacent-pair format)
in the target transcript at a point iff, at that point in the audio:

1. the speech so far forms a **semantically complete** utterance, AND
2. **≥ 0.3 s of silence** has elapsed since the last speech.

Both conditions are observable from the audio prefix — the rule is causal.
Two corollaries that fix the historical contradictions:

- A complete utterance whose audio ends without trailing silence gets **no
  marker** (transcribe, don't fire *yet*) — this one class was labeled
  "fire" by v3-style AMI examples and "no fire" by v7/v8 `no_fire`
  examples. It is now always no-marker, and the *same* utterance appears
  again with a silence tail and a marker, so trailing silence is the only
  feature separating the two labels.
- A same-speaker pause ≥ 0.3 s after a **complete** utterance gets a
  marker even though the speaker later resumes (quick resumes become two
  segments — the `resume_after_fire` cost knob in eval 58, not an error).
  Holding fire through a pause is trained **only** where the speech so far
  is incomplete — information that IS in the prefix. This removes the
  double-vs-disfluency contradiction (overlapping gap distributions,
  opposite labels keyed on future audio).

Abandonment (incomplete speech, speaker never resumes) is handled by an
inference-policy timeout (~2 s of silence → force fire), not by labels.

## Completeness classifier (for AMI pair examples)

Utterance A is **incomplete** iff its transcript ends in a
filler/connective ("and, but, so, or, uh, um, er, the, a, an, to, of, in,
with, i mean, you know, kind of, sort of, like"), ends mid-word (AMI
partial-word annotation), or has < 3 words and is not in a
closed acknowledgment list ("yeah, okay, right, mm-hmm, exactly, sure,
no, yes, thanks, cool"). Everything else is complete. Borderline
misclassifications land in `resume_after_fire` bias, which is a measured
knob — they never create fire/no-fire contradictions on identical
observables.

## Schemas (~10.5 k examples)

| # | Schema | Construction | Target text | n |
|---|---|---|---|---|
| 1 | `single_sil` | LS or AMI utterance + 0.3–1.2 s silence | `text M` | 2500 |
| 2 | `complete_nosil` | **same utterances as #1**, tail ≤ 0.1 s | `text` | 1500 |
| 3 | `truncated` | cut mid-speech at 30–80 % | partial text | 2000 |
| 4 | `pair_fire` | A *complete* + gap 0.3–2.5 s + B (+ tail sil on 2/3) | `A M B M` (or `A M B`) | 2500 |
| 5 | `pair_hold` | A *incomplete* (same spk) + gap 0.3–2.5 s + B + tail sil | `A B M` | 1500 |
| 6 | `long_sil` | utterance + 2–4 s silence | `text M` | 500 |
| 7 | `silence_only` | pure silence 0.5–4 s | *(empty)* | 800 |
| 8 | `lead_sil` | silence 0.5–2 s + utterance + tail sil | `text M` | 700 |

`M` = `<EAGER_END_SPEECH><END_SPEECH>`. Sources: LibriSpeech
(~/data/LibriSpeech) and AMI train meetings (md5 % 3 ∈ {0,1}) roughly
50/50; pair examples use both same-speaker and different-speaker AMI
pairs — the fire decision keys on completeness + silence, never on
speaker identity (single-channel deployment can't observe it).

Design notes:
- #1/#2 pairing on identical utterances is the anti-oscillation device:
  v8 gave opposite labels to disjoint-but-identically-distributed pools
  (unlearnable, hence the two attractors); v9 gives opposite labels to the
  same content differing only in the tail (learnable, and the *intended*
  feature).
- #4 includes `A M B` (no tail): mid-stream state where one fire already
  happened and speech is ongoing — the steady state of a long session.
- #6 guards the timeout region: silence far beyond 1.2 s must still hold
  the already-emitted marker (no marker spam), and keeps long-silence
  audio in-distribution for the replay eval.
- #7/#8 added 2026-07-04 after the first replay-eval run (research/61):
  v5 on real timelines hallucinates text and spams fires on silence-only
  segments — corr(silence duration, spurious fires) = 0.997 — because
  every v1–v8 example begins with speech. #7 teaches silence → empty
  output; #8 teaches that segments starting mid-silence (the normal case
  after a flush) transcribe and fire normally. The production system
  additionally front-gates the LM with an energy check (`--energy-gate`
  in the replay eval), but the model must also be safe without it.
- Fire markers ride ~0.3–0.5 s into the pause by construction (the pause
  exists in the audio); single_sil tail of 0.3–1.2 s trains the same
  anchor v5 had.

## What is deliberately NOT in v9

- No `no_fire`-style complete-utterance-with-natural-tail examples labeled
  no-marker (that class is now split by observable tail into #1/#2).
- No offline-eval-shaped "fire at end-of-clip without silence" labels.
- No context-prefix examples (endpoint isolation; context biasing is
  already strong in the base model per research/16).

## Training

Identical to v5–v8 (isolate the data change): LoRA r=16 α=32 on
q/k/v/o_proj of the thinker LM + trainable rows for the two marker tokens,
bsz 8, up to 18 k steps. Early stopping on a holdout composite matching
the new spec (equal-weight: #1 fire-rate, #2 no-fire-rate, #3
no-fire-rate, #4 internal-fire-rate, #5 internal-hold-rate, #6
single-fire-rate), then the top-3 checkpoints re-ranked by the
streaming-replay eval (research/58) on a 10-stretch dev subset from
TRAIN meetings (eval meetings stay untouched until the final report).

## Success criteria (on the unified replay eval, held-out meetings)

- Boundary recall ≥ 90 %, latency P50 ≤ 0.5 s / P95 ≤ 1.2 s
- False-fire ≤ 1/min of speech; streaming WER ≤ offline + 1 pp
- No fire-mode/no-fire-mode oscillation in the holdout trajectory
  (the v8 signature of contradictory labels — its absence is the direct
  test of the label-spec hypothesis)

## Priority note (2026-07-04)

The silence-appended smoke test (research/57) shows v5 *already* reaches
~v3-level AMI behavior under deployment-realistic silence tails. If the
full smoke results + a v5 replay-eval baseline confirm this, v9's marginal
value is the pair_fire/pair_hold causal cleanup (turn-taking + disfluency
under one policy) rather than frontier collapse per se — run it, but the
bar it must beat is v5-under-fair-eval, not v8.
