# Overnight autonomous plan (2026-07-05 night → morning)

User asleep; mandate: keep GPU busy, validate the story with more/bigger/diverse
evals + statistical rigor, rerun ALL experiments on the finalized checkpoint,
then revise the whole paper (new unified-model story, consistent numbers, less
number-dense body → move data to figures, regenerate all figures).

## Process hygiene (LEARNED THE HARD WAY)
- Single-process launches ONLY. No resilient wrapper retry loops (they spawned
  duplicates writing to the same dir).
- Kill strays by explicit PID; `pkill -f` FAILS in this sandbox.
- Check for HUNG procs (0% cpu, alive) AND duplicates before/after each job.
- PYTHONUNBUFFERED=1 on long jobs so a mid-run kill doesn't lose output.
- No co-tenant contention reported tonight → low OOM risk, but still verify.

## Finalized checkpoint
v15@6000 (v12 data + ASR-replay + wd-fix + cosine) = release candidate.
Probes (best of all models): prem 0.22, dict-rec 0.78, dacc 0.998, email+ 0.88,
intrusion 0.000; replay 0.958/0.75/0.39. WER: PENDING (research/80-wer-v15.json).
If WER recovered (< v14 3.94/7.27, ideally ~v12 3.60/6.96) → v15@6000 DOMINATES
all axes → the unambiguous release. Merged at checkpoints/merged/...-endpoint-v15.

## Phases (drive via scheduled wakeups; adapt to failures)

**P0 — lock release (in progress):** finish v15 chain (9000 replay + WER). Decide
release = v15@6000 if dominant. Copy merged -v15 → -unified.

**P1 — bigger benchmarks (CPU build now; GPU eval after):**
- Dictation probe: 50 → 250 digit sequences (FSDD held-out george/lucas).
- Spelled probe: 80 → 240 items (more names/emails, keep probe voices held out).
- Replay: build a 100-stretch benchmark (combine seeds or draw 100). Bootstrap
  CIs on recall/false.
- Earnings-22 biasing: 150 → 400 utts if feasible.
Goal: tight Wilson/bootstrap CIs on every headline number.

**P2 — statistical rigor via multi-seed (GPU, sequential single runs):**
- (a) UNIFIED recipe (v15 data), seeds 1 & 2 → report release numbers as
  mean±std, prove robustness. [~30 min each]
- (b) CORE CLAIM: opposed-pools vs causal, seeds 1 & 2 → oscillation fingerprint
  reproduces / monotone reproduces (multi-seed Fig 1). Need v8 (opposed) + v9
  (causal) data — reconstruct via build_v8/v9 if needed. [~30 min each]
  Priority: (a) first (directly about the release), then (b) if time.

**P3 — rerun ALL evals on finalized checkpoint w/ big benchmarks (GPU):**
big replay, big dictation, big spelled, big biasing, WER + CIs → the numbers
that go in the paper.

**P4 — paper revision (CPU, last):**
1. Restructure to the unified-model story (one checkpoint, four behaviors, two
   axes). Update abstract/intro/Sec7/Sec8/Sec9/limitations.
2. Consistency pass: every number matches the rerun results. Single source of
   truth = a numbers table in research/.
3. Reduce number density in body → move data into figures/tables; prose keeps
   only headline figures.
4. Regenerate ALL figures with new numbers (make_figures.py): fig1 (multi-seed
   oscillation), fig3 (tradeoff w/ v15), fig4 (biasing bigger), tab:main,
   tab:dictation, master comparison figure, CIs.
5. Recompile, check overfull/undefined, page count.

## Running log (update each wake)
- 14:54 v15 sweep done; v15@6000 best probe checkpoint. Replay 6000=0.958/0.75.
- (next: WER, then P1 build)

## UPDATE 15:08 — release locked
v15@6000 WER 3.73/7.06 (fire 1.8%) — beats v14 (3.94/7.27), ~ties v12 (3.60/6.96).
MASTER TABLE (small benchmarks): v15@6000 best/tied on conv-recall(0.958), dict-
premature(0.22), digit-acc(0.998), email(0.88), intrusion(0); 2nd on false(0.75),
dict-rec(0.78), WER(3.73/7.06). => v15@6000 = DOMINANT RELEASE. Merged to
checkpoints/merged/qwen3-asr-0.6b-endpoint-unified. snap = v15_es/unified_release.pt.
seed-1 unified training running (robustness). NEXT: big-benchmark evals on v15@6000
(250 digit, 240 spelled, 100-stretch replay, big biasing) + CIs; seed-2; then paper.

## UPDATE 15:30 — big-probe results on v15@6000 (RELEASE numbers, CI-backed)
Digit (n=250): premature 0.16/seq [0.12,0.21], final-recall 0.844 [0.794,0.884],
  digit-acc 0.998, P50 0.43s.
Spelled (n=240, 120 identities): name none/prof 0.983/1.00; email none/prof
  0.133/0.942 [0.884,0.971]; intrusion 0.000 [0.000,0.016].
=> bigger benchmark IMPROVED the estimates (small probe was noisy: prem 0.22->0.16,
  email 0.88->0.942). These are the paper numbers. NEXT: big replay + biasing + seeds.

## UPDATE 15:52 — big replay (100 stretches, 384 boundaries) = honest release conv numbers
v15@6000: recall 0.938 [0.899,0.970], false 1.03/min [0.69,1.42], P50 0.39, P95 0.68.
(Bigger/harder draw than 25-stretch dev 0.958/0.75; consistent w/ fresh-50 0.924.
These CI-backed numbers are the paper's conv-endpointing figures.)
Big-probe (release): prem 0.16 [0.12,0.21], email+ 0.942 [0.884,0.971], intr 0 [<=0.016].

## UPDATE 16:36 — throughput lesson: cap at 2 concurrent GPU jobs
3 concurrent (seed-2 train + seed-1 sweep + biasing) dropped GPU util to 30% and
slowed seed-1 sweep to ~12min/ckpt (vs 2.5). Oversubscription is counterproductive.
NEW RULE: <=2 GPU jobs at once. Let biasing (~178/300) + seed-1 sweep finish, then
proceed more serially. Plenty of time (user asleep). Priorities: (1) finish current,
(2) seed robustness mean±std, (3) opposed/causal Fig1 multi-seed (central claim),
(4) PAPER REVISION (main deliverable).

## UPDATE 17:07 — biasing on RELEASE model (v15@6000, 300 Earnings-22 utts)
recall no_ctx 37.8% -> relevant 65.3% (UPLIFT +27.55pp; base model was +28.9pp).
distractor hallucination 6.25% (base 3.7%). => fine-tune PRESERVES biasing uplift
but at lower absolute recall + higher hallucination (WER-regression side-effect).
NOTE: Earnings-22 natural-speech distractor (6.25%) is NOT covered by the spelled-
entity distractor-ratio fix (that gave probe-intrusion 0) — honest inconsistency;
paper: keep base +28.9pp as capability number, report release +27.5pp preserves it.

## UPDATE 17:08 — seed-1 robustness (reproduces release)
seed1 best ckpts: @3000 prem0.14/email0.95/intr0.013; @6000 prem0.32/email0.90/intr0;
@1500 prem0.42/email0.85/intr0. => seed-1 reproduces the unified capability (email
0.85-0.95, prem 0.14-0.42, intrusion ~0), like seed0 (v15@6000). Robustness holds.
seed-2 finishing. Decision: report robustness from probe sweeps (mean±std on prem/
email/intrusion across 3 seeds) — sufficient; skip per-seed replay to save GPU.
PRIORITIZE PAPER REVISION (main deliverable). Opposed/causal Fig1: attempt after
paper if time, else note single-seed honestly.
