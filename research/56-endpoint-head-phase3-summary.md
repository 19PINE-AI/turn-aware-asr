# Endpoint head v2: Phase 1-3 results (2026-06-15)

After 8 LM-LoRA experiments confirmed the Pareto frontier between
AMI conversation and streaming endpoint accuracy, we pivoted to
architectural unification: separate the **acoustic VAD** from the
**LM transcription** and compose them at inference.

Phase 1 (data, 1.5 h)
   - 2500 forced-aligned examples: 1500 solo LS + 500 solo AMI
     + 500 LS concat pairs
   - Per-frame labels at 12.5 Hz: eager at last-word, end after
     last-word + 200 ms post-pause, end=0 during internal silences

Phase 2 (train, 1 min on Blackwell)
   - 0.33 M-param 2-layer MLP head on AuT 12.5 Hz hidden states
   - BCE + pos_weight=1.5 for sparse end labels
   - 3000 steps, bsz=8, AdamW lr=1e-3

Phase 3 (eval)
   - Solo LS / AMI: best policy τ=0.5, k=2 hits LS P50=240 ms,
     AMI P50=80 ms, both ≥97 % detection
   - Concat (LS+LS, 1 s gap): 100 % fire, **37 % find BOTH turn
     boundaries**, 53 % find only post-utterance B
   - **Real AMI conv. eval** (the actual production test):

| Metric | Head v2 best (τ=0.3, k=1) | v3 LM | v5 LM | v8 LM |
|---|---|---|---|---|
| AMI single ≥1 fire | **92 %** | 100 % | 82 % | 90 % |
| AMI double ≥2 fires | 64 % | **90 %** | 76 % | 88 % |
| AMI disfluency =1 fire (correct) | 12 % | **82 %** | 66 % | 62 % |
| AMI disfluency ≥2 fires (over-fire) | **88 %** | 18 % | 22 % | 36 % |

## Why the head over-fires on disfluency

Disfluency = same speaker stops, pauses, restarts ("um… and then…").
The head correctly detects the pause and fires. But there's no
semantic signal saying "the speaker isn't done." Pure acoustic VAD
fires at every silent gap ≥ 200 ms — that includes mid-thought pauses.

The LM gets this right when fine-tuned (v3: 82 % disfl correct)
because it learns the linguistic structure: "and then" isn't a
sentence-final pattern.

## Architectural conclusion

Pure acoustic VAD on AuT features:
  - Beats v5 on AMI single (92 % vs 82 %)
  - Beats v6/v8 on AMI single
  - Loses to v3 on double detection (64 % vs 90 %) and disfluency
    (12 % vs 82 % correct)
  - Adds another point to the same frontier — doesn't dominate

The unification hypothesis ("frame-level VAD breaks the LM frontier")
is **wrong in this form**. A frame VAD fires correctly at silence,
but firing at silence isn't sufficient — the model needs semantic
context to distinguish "end of utterance silence" from "disfluency
silence."

## What WOULD unify

The honest unification requires a composition policy that combines:

1. **LM semantic signal**: when the next-token distribution favors
   end-of-utterance vs continuation (Qwen3-ASR doesn't expose this
   directly, but v3 LoRA learned to produce explicit end markers)
2. **Acoustic VAD signal**: when speech has paused for ≥ K frames

The cleanest decision rule is:
  - Emit `<END_SPEECH>` IFF (LM produced end marker OR LM has been
    silent for K tokens) AND (acoustic VAD fired)

That requires both signals to be available and properly time-aligned.
This is a system design change — and it's exactly the dual-checkpoint
v3+v5 ship we already concluded with.

The frame head COULD become the streaming signal that v5 currently
gets from learned LM behavior — but the head is currently slower
(P50 ~240 ms vs v5's +320 ms; comparable) and noisier on AMI conv.
audio (no acoustic gain over v5).

## Two paths from here

### Path A: accept that frame-level VAD does not unify
Ship v3 + v5 LM-based dual-checkpoint. The head experiment confirms
the frontier isn't a data-or-training-recipe artifact; it's structural.

### Path B: build the composition wrapper anyway
A v3 LoRA emit-marker LM plus the head's confirmation signal, fused
at inference. Concretely:
  - Run v3 LM on audio chunks → emit `<END_SPEECH>` when LM thinks
    semantic completion
  - Concurrently run head on AuT hidden states → emit fire event
    when silence held K frames
  - Final decision: emit END iff both signals agree within ±400 ms
This costs ~half a day of engineering and gives an honest test of
whether the AND composition narrows the frontier further.

Note: Path B might HURT some metrics (AND is more restrictive than
either alone) and is unlikely to be a clear win. But it's the only
remaining experiment that could combine both signals without
architectural change.

## Files

- `data/endpoint_v2/data.pt` — 2500 forced-aligned examples
- `checkpoints/endpoint_head_v2.pt` — 0.33 M-param trained head
- `research/51-endpoint-head-v2-results.json` — training + first sweep
- `research/52-endpoint-policy-sweep.json` — K-consecutive sweep
- `research/53-endpoint-eval-v3.json` — spurious-speech metric sweep
- `research/54-endpoint-multi-fire.json` — multi-event detection
- `research/55-ami-endpoint-head-eval.json` — real AMI conv. eval
- `eval/ami_endpoint_head_eval.py` — AMI eval driver
- `src/train_endpoint_v2.py` — scaled training
- `src/endpoint_eval_v3.py`, `src/endpoint_eval_v4.py` — eval scripts
