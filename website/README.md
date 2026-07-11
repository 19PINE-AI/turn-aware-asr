# Paper website — *The Trade-off Was in the Labels*

A React (Vite) site for the paper, in three sections:

1. **How it works** — interactive teaser (timeout-dilemma slider), system diagram,
   the clairvoyant-label failure class, the causal recipe (minimal pair /
   counterfactual twin / eight schemas), and the supervision checklist.
2. **Key results** — every headline figure redrawn as interactive SVG from the
   original result files (`research/*.json`, checkpoint eval logs): trade-off
   frontier, main table, composition record, oscillation curves, silence-append
   intervention, ranking inversion, biasing, dictation, silence scatter, and the
   synthetic clairvoyant-fraction study.
3. **Trajectory explorer** — audio playback + timeline visualization of **all
   original recorded trajectories**: the streaming replay benchmark (dev-25 with
   7 model arms, 15 external-system arms, 4 timeout arms; the 100-stretch
   held-out set; the fresh 50-stretch confirmation set), the dictation probes
   (250 + 50 items, three models), the spelled-entity probe (240 items × 3
   context conditions), and Earnings-22 biasing (base + released models, 300
   utterances × 3 context conditions).

## Data provenance

`scripts/export_data.py` (run from the **repo root** with the project venv)
regenerates everything under `public/audio/` and `public/data/`:

```bash
.venv/bin/python website/scripts/export_data.py
```

- Benchmark stretch audio is rebuilt with the *same deterministic builder and
  seeds* the evaluations used (`eval.streaming_replay_eval.build_stretches`,
  seed 0 for dev-25/held-out-100, seed 1 for fresh-50) and validated
  event-by-event against the recorded replay JSONs; the recomputed RMS-timeout
  fires match the recorded per-stretch counts exactly, confirming sample-exact
  audio.
- All fires, hypotheses, latencies, and summaries come verbatim from
  `research/*.json` — nothing is re-run through a model.
- Probe/earnings audio is transcoded from `data/probes*`, `data/earnings22`.

Total generated payload ≈ 78 MB (lazy-loaded MP3s + ~1.7 MB JSON). These
generated directories are gitignored; run the export before building/deploying.

## Develop / build / deploy

```bash
npm install
npm run dev        # local dev server
npm run build      # static site in dist/ (self-contained, relative paths)
```

`vite.config.js` uses `base: './'`, so `dist/` can be served from any path
(e.g. `01.me/research/turn-aware-asr`). No server-side code is needed.

`scripts/smoke.mjs` is a headless render test (requires the devDependency
`playwright` and a system Chromium): it loads the built site, walks the
explorer tabs, and fails on any console error.
