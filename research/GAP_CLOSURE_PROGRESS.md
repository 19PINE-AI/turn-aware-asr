# Closing the four morning-report gaps (2026-07-06)

Mandate: close the four "honest gaps" from research/MORNING_SUMMARY.md.
Release baseline: v15@6000 = checkpoints/merged/qwen3-asr-0.6b-endpoint-unified.

## Infra fix found along the way
The whole qwen_asr eval/transcribe suite silently ran on **CPU** (~9x slower):
`Qwen3ASRModel.from_pretrained` forwards kwargs to AutoModel with no device_map.
Fix = pass `dtype=torch.bfloat16, device_map="cuda"`. Applied to my new scripts +
`eval/semantic_endpoint_data_v2.py` + `eval/earnings22_biasing.py` (pkg backend).
Cut the transcript-cache build from ~2h to ~13min. Re-baseline release biasing on
GPU too (fair comparison; slight bf16-vs-fp32 shift expected but small).

## Gap 2 — dictation "~2s late": CLOSED (analysis) ✅
research/85-gap2-dictation-reanalysis.md. Re-scored release big probe (250 seqs)
by delta = fire - last_speech_end:
- genuine late fires (>+1.75s): **0/250**
- genuine mid-seq prematures (<-0.30s): **2/250 = 0.008**
- on-time recall [-0.30,+1.75]: **0.992** (vs shipped 0.844), p50 latency 0.40s
The claim was a scoring-window artifact (shipped premature cutoff +0.25 clips eager
fires quantized to the 0.5s chunk grid). Paper caption (main.tex:363) says "~5 fires
landing ~2s after speech end" — factually inverted; misses are EARLY, not late.
TODO: paper edit (batch with other gaps).

## Gap 4 — offline WER (+0.9/+1.9pp): IN PROGRESS
Base 2.79/5.14 vs release 3.73/7.06. Lever = larger ASR-replay dose.
- Expanded ls_transcripts cache 4301 -> 10774 (eval/expand_ls_cache.py).
- v16 pool built (21530 ex): asr_plain 5000 = 23% replay (was 14.5%).
- Training: checkpoints/semantic_endpoint_v16_es (12k steps, ~45min). research/88.
- After: select_checkpoint sweep -> merge winner -> offline_wer_ls. scripts/eval_v16_winner.sh
- Watch: digit final_recall/premature, email+, intrusion, conv-recall must hold.

## Gap 1 — natural-speech distractor hallucination (6.3%): IN PROGRESS
Fold into v16. Counterfactuals sourced from LibriSpeech (NOT Earnings-22 -> no
contamination), entities via Haiku LLM (eval/llm_entities, matches the eval's
extractor), eval's exact build_system_prompt comma-list format.
eval/build_natural_biasing.py -> bias_nat_distractor 1200 + bias_nat_relevant 600.
Eval: earnings22_biasing on v16 + re-baselined release. Target: distractor_halluc
<=3% while uplift holds ~+27pp. scripts/eval_v16_winner.sh runs it.

## Gap 3 — opposed-pools oscillation multi-seed: IN PROGRESS
Fully regenerable (raw LibriSpeech+AMI present). scripts/rebuild_v8_opposed.sh
rebuilds v2->v3->v4->v5->v6->v8 chain (research/86). Then scripts/train_v8_seed1.sh
= seed-1, --score-spec legacy, 18k steps (matches historical trajectory length),
same historical recipe (constant LR, no wd/cosine fix) so ONLY the seed differs.
Compare per-step single/no_fire_correct vs retained
checkpoints/semantic_endpoint_v8_es/eval_log.json (do NOT overwrite it).

## Launch order (1 GPU, cap 2 jobs)
1. cache expand (done) 2. v16 build ∥ gap-3 data (done/running) 3. v16 train (running)
4. gap-3 seed train (after v16 train frees GPU) 5. evals 6. paper edits.

## Additional documented experiments (paper FIXMEs + research/86-...md) — 2026-07-06
Running in parallel with the GPU chain (engineering on CPU while GPU trains).
1. dictation-generalization (task5): card(16-digit 4-4-4-4)/phone13/address probes
   built (data/probes_{card,phone13,addr}); release model eval running -> research/93-gen-*.json.
   eval/build_generalization_probes.py + scripts/eval_generalization.sh.
2. exp-5 toy task (task6): subagent building eval/toy_clairvoyant.py + sweep +
   paper/figures/exp5_toy.pdf + research/94-exp5-findings.md. Self-contained, CPU.
3. exp-2 logit-level (task7): rides on gap-3 seed-1 (now --snapshot-evals). After
   gap-3 trains, script loads each snap, computes P(<END_SPEECH>) at decision
   positions across steps -> mode-circulation vs threshold-flap.
4. exp-3 schema ablation (task8): --drop-schemas + --iso-count added to
   build_v9_training_data.py. Arms (confirm schema# vs paper Table first):
   A -minimal-pair=drop complete_nosil; B -silence=drop silence_only,lead_sil;
   C -pause-pair=drop pair_hold. Build pools (CPU, cache reuse) -> train each
   (queue after v16+gap-3) -> select -> replay + ungated silence-fire + confirm h=1
   + per-boundary-class false-fire breakdown.
5. exp-4 external baselines: subagent scaffolding eval/external/ adapters
   (Smart Turn v3, Parakeet-EOU, Kyutai STT, LiveKit) + per-system venvs + downloads
   + smoke tests. I run the GPU replay scoring later (slot between trainings).

## GPU queue (serial, I control): v16 train (running) -> v16 evals -> gap-3 seed-1
## train -> exp-3 arms A/B/C -> exp-4 replay scoring. exp-2/exp-5/gen = light/CPU.
