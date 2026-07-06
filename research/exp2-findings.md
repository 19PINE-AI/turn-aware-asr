# exp-2 — Marker-token PROBABILITY across the opposed-pools oscillation

**Question (research/86, exp-2).** The opposed-pools (v8) recipe, retrained under
seed-1, reproduces the oscillation fingerprint: holdout argmax accuracy on the
fire-class bounces between ~1.0 and ~0.3 across training steps
(`checkpoints/semantic_endpoint_v8_seed1_es/eval_log.json`). Argmax bouncing has
two possible explanations, and this experiment separates them by reading the
marker **probability** (not argmax):

- **Mode circulation** — the probability mass on firing genuinely swings
  high↔low across steps; the weights move between two attractors.
- **Threshold flapping** — the fire probability hovers near the 0.5 decision
  boundary the whole time and tiny changes flip the argmax.

**Verdict: MODE CIRCULATION.** Threshold flapping is ruled out.

---

## Method

`eval/exp2_marker_prob.py`. GPU-inference only, one snapshot loaded at a time,
bfloat16, `torch.no_grad`. Probe = the exact balanced holdout the training
reserved (`random.Random(12345)`, 10/schema, never trained on), so P_fire is
directly comparable to the recorded argmax oscillation:

- **FIRE examples** = schemas `single` (10) + `trailing_silence` (10) — complete
  utterance, should fire.
- **HOLD examples** = schemas `truncated` (10) + `no_fire` (10) — incomplete /
  no trailing silence, should hold.

**P_fire.** The trained marker pathway is a rigid two-token sequence: after the
transcript the model emits *space → `<EAGER_END_SPEECH>` → `<END_SPEECH>`*, or it
emits `<|im_end|>` and stops. The `EAGER→END` bigram is **deterministic**
(`P(<END_SPEECH> | <EAGER_END_SPEECH>) ≈ 1.0` at every snapshot), so reading
`P(<END_SPEECH>)` as a bare next-token probability at the end of the transcript
is ≈0 and uninformative — the graded fire decision lives one position earlier
(stop vs enter the marker path). We therefore define **P_fire = the marginal
probability that the continuation emits `<END_SPEECH>` before `<|im_end|>`**,
evaluated at the end-of-audio decision position (transcript teacher-forced),
computed by marginalizing over the first continuation token (top-k covering
≥0.98 mass) and greedy-rolling each branch to termination. Because the marker
bigram is deterministic once entered, this equals the probability the model
commits to firing the marker, and lies in [0, 1].

**Validation.** Mean P_fire tracks the recorded argmax accuracy almost exactly:
`corr(single-schema P_fire, eval_log single argmax) = +0.98`;
`corr(HOLD P_fire, eval_log no_fire_correct) = −0.95` (a hold example that fires
is a `no_fire` miss). The probability metric is measuring the same phenomenon the
argmax metric reports.

---

## Results

Per-step mean P_fire (full table in `research/exp2-marker-prob.json`,
figure `paper/figures/exp2_marker_prob.pdf`):

| step | single P_fire | eval single argmax | HOLD P_fire | eval no_fire_correct |
|-----:|:-------------:|:------------------:|:-----------:|:--------------------:|
| 1500 | 0.36 | 0.3 | 0.08 | 1.0 |
| 3000 | **0.77** | **0.9** | **0.34** | **0.2** |
| 4500 | 0.40 | 0.3 | 0.04 | 1.0 |
| 6000 | 0.52 | 0.5 | 0.06 | 1.0 |
| 7500 | 0.66 | 0.7 | 0.18 | 0.7 |
| 9000 | 0.73 | 0.8 | 0.18 | 0.5 |
|10500 | 0.42 | 0.3 | 0.02 | 1.0 |
|12000 | 0.66 | 0.7 | 0.10 | 0.8 |
|13500 | 0.46 | 0.5 | 0.12 | 0.7 |
|15000 | 0.41 | 0.4 | 0.04 | 0.9 |
|16500 | 0.60 | 0.6 | 0.12 | 0.8 |

**The probabilities swing.**
- `single`-schema fire class (the class whose argmax bounces): mean P_fire
  swings **0.36 ↔ 0.77**, amplitude **0.40**, straddling the 0.5 boundary in a
  sawtooth locked to the argmax oscillation.
- HOLD class (`truncated`+`no_fire`): mean P_fire swings **0.02 ↔ 0.34**,
  amplitude **0.32** — the hold examples periodically gain firing mass (the
  `no_fire` misfire episodes at steps 3000 / 9000).
- `trailing_silence` fire class: pinned at **≈1.0** at every step — a stable,
  never-oscillating attractor (its argmax `trail_correct` is 1.0 throughout).
  The oscillation is confined to the *contested* schemas, not global noise.

**It is not the boundary that these examples sit on — it is two confident modes
they flip between.** A mean near 0.5 could arise from every example hovering at
0.5 (flapping) or from confident examples flipping between ~1 and ~0
(circulation). The per-example distribution settles it:

- **0 / 20 FIRE and 0 / 20 HOLD examples remain inside the ambiguous [0.2, 0.8]
  band across all 11 snapshots** — the persistent-boundary signature of
  threshold flapping is entirely absent.
- At any single snapshot only **~13 % (FIRE) / ~15 % (HOLD)** of examples fall in
  the [0.2, 0.8] band; the rest are confidently committed to fire (>0.8) or
  hold (<0.2).
- **Full attractor reversals:** 2 / 20 FIRE and **5 / 20 HOLD** examples visit
  *both* a confident-fire (>0.8) *and* a confident-hold (<0.2) state over
  training — the same input swaps modes as the weights move.
- Per-example swing amplitude (max−min over steps) averages **0.30 (FIRE)** /
  **0.43 (HOLD)**, i.e. individual examples move most of the way across the
  probability range, not jitter around 0.5.

---

## Interpretation and honest caveats

The marker probability genuinely circulates: individual examples are, at most
steps, confidently committed to one mode, and the training trajectory carries
them between the fire and hold attractors — visible both as the sawtooth in the
means and as full >0.8↔<0.2 reversals of specific examples. This is **mode
circulation**, and it strengthens the §4 "loss surface reporting a contradiction"
mechanism: the weights are not parked on a knife-edge threshold, they are
orbiting between two basins.

Caveats, stated quantitatively:
- The swing is **partial-amplitude**, not an idealized hard limit cycle. The
  contested `single` schema's per-step mean straddles 0.5 (0.36–0.77) and its
  mean per-example amplitude is ~0.30–0.40, so a minority of examples are
  genuinely uncertain at some steps rather than snapping cleanly 1↔0. The
  decisive evidence against flapping is the **absence of any persistently-boundary
  example** plus the confident full reversals — not a claim that every example
  executes a perfect 0→1→0 oscillation.
- `trailing_silence` never participates; the circulation is a property of the
  schemas the two opposed pools actually contest, consistent with the
  contradiction being *local to the disputed examples*.

**Bottom line:** the opposed-pools argmax oscillation is driven by the marker
probability mass genuinely swinging between two confident modes
(single-schema P_fire 0.36↔0.77, HOLD 0.02↔0.34, with 7/40 examples fully
reversing >0.8↔<0.2), not by probabilities hovering at the 0.5 decision
boundary. Mode circulation, confirmed.
