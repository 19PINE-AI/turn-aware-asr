# Exp-3 (schema ablation) and Exp-4 (external baselines): specs

**Date:** 2026-07-06. **Status:** agreed as pre-submission work; not started.
**Paper landing spots:** `%% FIXME(exp-3)` and `%% FIXME(exp-4)` blocks in `paper/main.tex`
(consolidated block after Fig. 1; inline markers in §5.1/§6).
These are *additional to* the four morning-notes gaps (natural distractor, WER, dictation,
oscillation) being closed in the concurrent session — do not conflate the runs.

---

## Exp-3: schema ablation — which component of the causal recipe does the work?

### Why
§5.1 of the paper credits the endpointing win to the causal rule plus two constructions
(minimal pair, pause pair) plus the silence schemas. Reviewers will ask: is that attribution
measured or asserted? Three ablation arms answer it. This also hardens the science claim:
if removing the minimal pair reproduces a *specific* pathology from the Table-composition
record (row 4: silence-optional firing), the recipe→pathology mapping is causal, not narrative.

### Arms
Ablate on the **pure endpointing recipe (v9 pool)**, not the unified v15/v16 pool — keeps the
attribution clean of dictation/context confounds. Baseline arm = the released causal recipe
unchanged.

| Arm | Drop | Schemas remaining |
|---|---|---|
| A. −minimal-pair | schema 2 (the no-tail twins of schema 1) | 1, 3–8 |
| B. −silence | schemas 7–8 (pure silence; leading silence) | 1–6 |
| C. −pause-pair | schema 5 (incomplete-A + gap + B) | 1–4, 6–8 |

### How to run
1. **Builder flag (small code change):** add `--drop-schemas 2` (comma list) to
   `eval/build_v9_training_data.py` (CLI at ~line 225). Two pool variants per arm:
   - *raw-drop* (pool shrinks) — the honest "recipe minus component";
   - *iso-count* (redistribute the dropped quota proportionally across remaining schemas) —
     controls for data volume. **Report iso-count as primary**, raw-drop in a footnote;
     if the two disagree, volume matters and that is itself a finding.
2. **Train:** copy `scripts/train_v16.sh` per arm (single-process, no resilient wrapper —
   see its header comment), pointing `--data` at the ablated pool and `--checkpoint-dir` at
   `checkpoints/semantic_endpoint_abl_{A,B,C}`. Same hyperparameters, same seed as release
   (add seed 1 for any arm whose result is surprising). ~hours/arm on the shared GPU;
   3 arms ≈ 1 day serial.
3. **Select:** same protocol as the paper (`eval/select_checkpoint.py` on the holdout
   composite — *not* on the replay benchmark, to avoid selection leakage into the eval).
4. **Score:** per arm, on the 25-stretch dev replay set:
   - `eval/streaming_replay_eval.py` gated → recall / P50 / P95 / false-per-min / resume;
   - **ungated silence-fire rate** (fires per second of silence, the Fig. `fig6_silence`
     metric) — this is the primary metric for arm B; gated numbers can mask it;
   - confirm-horizon dependence: candidates cancelled at h=1 (the "crutch retirement" metric,
     Table appconfirm) — primary secondary metric for arms A and C.

### How to interpret
Each arm has a *specific predicted failure*; the table is the point of the experiment:

| Arm | Prediction if component is load-bearing | If instead ≈ baseline |
|---|---|---|
| A. −minimal-pair | Silence becomes optional again (Table-composition row-4 pathology): false/min ↑ several×, fires land at Δ<0.3 s silence, confirm h=1 starts cancelling many candidates again | The minimal pair is redundant given schemas 4/6; soften the §5.1 "this construction is what removes the pathology" sentence to "the rule, jointly enforced by schemas 1–2/4/6" |
| B. −silence | Ungated silence-fire rate returns toward ~1.9 fires/s; gated false/min mildly ↑ | Silence competence is learned incidentally from tails/gaps; the energy gate remains a cheap belt-and-suspenders, and Fig. 6's "consequence" framing weakens slightly |
| C. −pause-pair | False fires concentrate at *internal/continuation* boundaries (phrase-level pauses fire); turn-final recall roughly unchanged | Pause discrimination comes free from schema 4's complete-A gaps; merge schemas 4–5 in the paper's description |

Write-up: one paragraph in §5.1 + one table in App. C. Report per-boundary-class false-fire
breakdown, not just aggregate false/min — the arms are *predicted to fail in different
places*, and the breakdown is what shows it.

**Caution:** do not iterate checkpoints against the replay set (that is how the legacy
benchmark corrupted recipes 1–4). One selection, one scoring run, report what lands.

---

## Exp-4: external baselines on the replay benchmark

### Why
The main table currently compares only our own predecessors and the timeout family. The
open turn-aware systems named in the landscape table have never been run on a common
deployment-matched protocol — by anyone. This is both the biggest reviewer ask and a
contribution in itself (first cross-system measurement of the open turn-aware class).

### Systems and adapters
Write one adapter per system in `eval/external/`, each consuming the exact benchmark
stretches (0.5 s chunks, same audio) and emitting fires as timestamped events in the same
JSON shape `streaming_replay_eval.py` outputs, so `eval/metrics.py` scoring applies
unchanged (same boundary taxonomy, same [−0.25,+1.5] s window, same false/min definition).

1. **Kyutai STT semantic-VAD head** (`kyutai/stt-*` delayed-streams models). Streams natively;
   fire = the VAD head's pause-prediction crossing its threshold. Sweep the threshold
   (~5 values) to get a curve, not a point. Also record its transcript → streaming WER row.
2. **NVIDIA Parakeet-Realtime-EOU-120m** (NeMo). Fire = `<EOU>` token emission in-stream.
   Few knobs; one point. Non-OSI license: fine for evaluation, note it in the caption.
3. **Smart Turn v3** (pipecat). Acoustic *classifier*, not streaming: it judges "turn done?"
   on an audio window when invoked. Deployment-faithful harness = its intended one: trigger
   on VAD silence (use our RMS detector for apples-to-apples), classify the trailing window,
   fire on positive. Latency floor is therefore VAD-silence + inference — that is the honest
   comparison, since it is how the system ships. Sweep its decision threshold.
4. **LiveKit turn detector** (text EOU classifier over a separate STT). Two variants worth
   one run each: (a) as-shipped over its default STT pairing if practical; (b) over *our*
   base model's streaming transcript — isolates the detector from STT quality. Fire =
   classifier positive at a VAD-silence trigger, same harness shape as Smart Turn.

Practicalities: separate venv per system (NeMo/moshi dependency clash risk);
GPU inference only, no training; expect the adapters (not the compute) to be the cost —
budget 1–2 days of engineering, then hours of replay.

### How to interpret
Score all on the dev 25-stretch set; run the top-2 also on the fresh 50-stretch set.
Add: rows to `tab:main`, points/curves to `fig3_tradeoff`, and delete the "no external
baselines" sentence in Limitations. Reading the outcomes:

- **They land on or near the timeout curve** (plausible for Smart Turn/LiveKit, whose fires
  are gated on VAD silence by construction): strongest outcome — "escaping the curve"
  becomes a measured property distinguishing in-model semantics from bolt-on detectors.
- **Kyutai/Parakeet land inside the curve but behind us:** system claim intact; report
  honestly, including where they win (e.g., Parakeet latency).
- **Someone matches or beats our operating point:** the *system* superiority claim narrows,
  but the paper's spine survives unchanged — the science claim (phantom frontier, causal
  supervision) and the openness claim (first *recipe*; none of these publish one) do not
  depend on winning the benchmark. Adjust §6 wording, keep the row.
- **Interpretive caveat to state up front:** these systems were not tuned for AMI close-talk
  replay; we sweep their public thresholds but do not retrain them. The comparison is
  "as-shipped on a common protocol", not "best achievable".

---

## Do we need further experiments? (recommendation, priority order)

**Paper tracking (2026-07-06):** exp-3 and exp-4 have FIXME landing spots in `main.tex`
(consolidated block after Fig. 1 + inline in §5.1/§6). The three additional experiments below
(exp-2 logit-level, dictation-generalization, exp-5 toy task) are now *also* tracked in the
paper: consolidated in the FIXME block, with inline landing spots at §diagnosis "Why
oscillation, specifically" (exp-2), Limitations (dictation-generalization), and §discussion
"Who else is likely affected" (exp-5). Fill the inline sentence/paragraph when each run lands.


1. **Gap-4 seeded oscillation** *(in flight, other session)* — the single most attackable
   point of the science-first framing; nothing else matters as much for reviewers of §3–4.
2. **Exp-4** — biggest external-validity upgrade per unit effort. Engineering, no training.
3. **Exp-3** — turns §5.1's attribution from narrative into measurement. Three short runs.
4. **Exp-2, logit-level oscillation analysis** *(cheap, recommended, no training)*: plot
   marker-token *probability* (not argmax) across saved opposed-pools checkpoints.
   Distinguishes true mode circulation (probabilities swing) from threshold-flapping
   (probabilities hover at 0.5). Strengthens the §4 "loss surface reporting a contradiction"
   mechanism with one afternoon of eval compute. Depends on which v8 snapshots were retained
   (`rebuild_v8_opposed.sh` exists if the pool must be rebuilt, but historical *checkpoints*
   are the thing needed — check `checkpoints/` first; if only the final v8 survives, scope
   this down or drop it).
5. **Dictation generalization probe** *(cheap, recommended)*: zero-shot street addresses /
   16-digit card numbers through the existing dictation harness
   (`eval/build_dictation_probes.py` + `dictation_probe_eval.py`). Limitations currently
   *asserts* other enumerations are untrained; measuring turns an assertion into a number,
   and if partial transfer shows up it upgrades the claim.
6. **Exp-5, synthetic contradiction toy task** *(optional flagship)*: tiny transformer on a
   synthetic stream where a fraction of labels is made clairvoyant by construction; show
   oscillation onset and phantom frontier as a function of the clairvoyant fraction, outside
   speech entirely. This is the one experiment that would elevate the principle from
   "two instances in one system" to "demonstrated mechanism". Days of work, trivial compute.
   Worth it if targeting ICLR; skip for a fast arXiv release.

**Not needed:** more release-recipe seeds (3 done), larger backbones (out of scope, stated),
more benchmark stretches (fresh 50-stretch + 100-stretch sets already confirm), human evals.
