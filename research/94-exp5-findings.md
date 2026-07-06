# exp-5 — synthetic clairvoyant-fraction toy task: findings

**Date:** 2026-07-06. **Status:** run complete (CPU, ~18 min, 27 runs).
**Artifacts:** `eval/toy_clairvoyant.py` (task + model + sweep),
`research/94-exp5-toy-task.json` (per-f, per-step, per-seed trajectories),
`paper/figures/exp5_toy.pdf` / `.png`, `paper/figures/make_exp5_figure.py`.
**Reproduce:** `.venv/bin/python eval/toy_clairvoyant.py --sweep --steps 1200 --eval-every 20 --lr 2e-3`
then `.venv/bin/python paper/figures/make_exp5_figure.py`. Everything is seeded (torch/numpy/python);
CPU-only, does not touch the GPU.

## What the toy is

A discrete streaming stream (no speech anywhere): phrases of content tokens separated by
silence runs ("gaps"). A phrase ends "complete" (terminal token) or "incomplete" (non-terminal)
— always readable from the prefix. Each gap's *full* length is short (speaker resumes) or long
(turn ends), 50/50, and is only knowable **after** the decision point. At the g_min-th silence
token of each gap a tiny causal transformer (2 layers, d=32, ~30k params) must emit FIRE vs HOLD.

- **Causal rule** (prefix-computable): FIRE iff the phrase was complete.
- **Clairvoyant rule** (target leak in time): FIRE iff the gap *turns out* long — a genuine
  future-token dependence, the exact analog of "offline clips cut at the boundary / transcripts
  record who spoke next".

completeness ⟂ gap-length, so the two rules are maximally opposed while each is internally
consistent (the paper's "pools with incompatible optima"). A fraction **f** of training
*sequences* are clairvoyant-annotated (coherent pools); the holdout is **always** causal.
Metrics logged every 20 steps: `fire_acc = P(pred FIRE | causal-FIRE)` (recall) and
`hold_acc = P(pred HOLD | causal-HOLD)` (precision proxy).

## Results (mean over 3 seeds; full trajectories in the JSON)

| f | fire-acc std (tail) | bistable mode-flips | corr(fire,hold) | worst holdout acc | final fire / hold |
|---|---|---|---|---|---|
| 0.00 | 0.002 | 0.0 | +0.18 | 0.99 | 1.00 / 1.00 |
| 0.10 | 0.002 | 0.0 | −0.29 | 0.99 | 1.00 / 1.00 |
| 0.20 | 0.003 | 0.0 | −0.48 | 0.99 | 1.00 / 0.99 |
| 0.30 | 0.005 | 0.0 | −0.45 | 0.98 | 1.00 / 0.99 |
| 0.40 | 0.007 | 0.0 | −0.49 | 0.98 | 0.99 / 0.98 |
| 0.50 | 0.011 | 0.0 | −0.51 | 0.96 | 0.98 / 0.97 |
| 0.70 | 0.036 | 0.0 | −0.63 | 0.87 | 0.92 / 0.91 |
| 0.85 | 0.104 | 8.3 | −0.83 | 0.63 | 0.81 / 0.76 |
| 1.00 | 0.241 | 16.3 | −1.00 | 0.47 | 0.50 / 0.49 |

("mode-flips" counts sign changes of (fire−hold) whose amplitude exceeds 0.15, so it fires only
on genuine mode circulation, not float noise near 1.0. "worst holdout acc" = min overall causal
accuracy over the post-warmup trajectory.)

## Does oscillation onset appear?

Yes, and it is **monotone in f** with a clear threshold:

- **f = 0 (causal):** textbook monotone convergence. Both classes climb to 1.00, oscillation
  amplitude 0.002, zero mode-flips, worst-case accuracy 0.99. A single dominating checkpoint.
- **f = 0.1 – 0.5:** no bistable swinging yet (0 flips), but the fingerprint is already present in
  embryo: fire/hold accuracy become **anti-correlated** (−0.3 → −0.5) and swing amplitude and
  precision erosion grow smoothly. This is the "phantom frontier" forming as a whisper.
- **f = 0.7:** amplitude 18× the f=0 baseline (0.036), anti-corr −0.63, worst-case accuracy down
  to 0.87 — visibly noisy, on the cusp.
- **f = 0.85:** **onset of true bistability** — 8 mode-flips, amplitude 0.10, anti-corr −0.83,
  worst-case accuracy collapses to 0.63. The model ping-pongs between fire-mode and hold-mode.
- **f = 1.0:** full mode circulation — 16 flips, amplitude 0.24, anti-corr −1.00, overall holdout
  accuracy oscillating around chance (0.47). This is the direct analog of the v8 opposed-pools
  oscillation in `fig1_oscillation` (left panel).

## Does the phantom frontier appear/dissolve?

Yes — this is the cleanest part (Fig. `exp5_toy` panel b). Plotting every checkpoint's
(fire-acc, hold-acc):

- **f = 0:** all checkpoints collapse into a tight dot in the **top-right corner (1, 1)** — no
  frontier, no trade-off, one operating point.
- **high f:** the checkpoints **spread into an anti-diagonal cloud** along `recall + precision ≈ 1`
  — an apparent recall-vs-precision Pareto front. At f=1 the cloud sits *exactly* on that line:
  it is not a set of models with different capabilities, it is **one model swinging**, sampled at
  different steps. The "frontier" is entirely an artifact of the clairvoyant labels and dissolves
  to a point the instant they are removed.

## Verdict

**The mechanism reproduces outside speech.** A future-dependent ("clairvoyant") streaming label,
introduced as a tunable fraction f of an otherwise-causal synthetic stream, manufactures both
symptoms the paper attributes to it in the ASR model — training-time oscillation between two mode
attractors and an apparent recall-vs-precision frontier across checkpoints — and both vanish at
f=0 under one causal relabel, with nothing else changed. The effect is a clean, monotone
dose-response in f, replicated across 3 seeds, on a task with no acoustics, no pretraining, and a
30k-parameter model. This elevates the principle from "two instances in one system" to a
demonstrated mechanism.

**One honest nuance worth stating in the paper.** Visible *bistable* oscillation (mode-flips)
needs a **high** clairvoyant fraction (≈0.85+); in the requested 0–0.5 range you see the milder
precursor — growing fire/hold anti-correlation and precision erosion (the phantom frontier
forming), but not yet flips. The reason is mechanical and itself supports the thesis: because
clairvoyant corruption on any given prefix is ~50/50, the causal label stays the per-prefix
*majority* for every f < 1, so a calm single-mode optimum still exists and the optimizer can find
it until the contradictory pool becomes large enough to destabilize it. The speech model lived in
the high-f regime by construction — its offline and streaming pools were near-balanced and
directly opposed, i.e. equivalent to f near 1 here — which is exactly why it oscillated hard. So
the toy both reproduces the phenomenon and explains *when* it becomes severe: severity scales with
how balanced-and-opposed the clairvoyant pool is, and near-balanced opposed pools are the common
case whenever offline supervision is mixed with streaming reality.
