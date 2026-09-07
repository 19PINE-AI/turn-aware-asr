# Paper website

[Project README](../README.md) · [Live site](https://01.me/research/turn-aware-asr/) · [Paper](https://arxiv.org/abs/2609.04225)

A static React/Vite site explaining the method and showing recorded results.
The trajectory explorer plays audio alongside saved transcripts and turn
boundaries; it does not run a model in the browser.

## Run locally

Requirements: Node.js 18 or later and npm. From the repository root:

```bash
cd website
npm ci
npm run dev
```

A fresh clone can build the site code, but **result charts and audio exploration
require generated data**. The `public/data/` and `public/audio/` directories are
excluded from Git. Use the live site to explore the complete results immediately,
or prepare the data below for local development.

## Prepare result data and audio

Requirements: the project Python environment, `ffmpeg`, the recorded files in
`research/`, and the source benchmark/probe audio under `data/`.
See the [training guide](../REPRODUCING.md#prepare-the-data) for corpus prerequisites
and inspect [`scripts/export_data.py`](scripts/export_data.py) for exact inputs.

Run from the **repository root**, not `website/`:

```bash
.venv/bin/python website/scripts/export_data.py
```

This generates approximately 78 MB of MP3 and JSON assets under `website/public/`.
It reconstructs benchmark audio with the same stretch builder and seeds used by
the evaluations: seed 0 for development/100-stretch sets and seed 1 for the fresh
50-stretch set. It validates reconstructed events against recorded replay JSONs.
Probe and Earnings-22 audio are transcoded from the local source files.

Transcripts, fires, latencies, and summaries come from the recorded results.
Exporting assets does not rerun model inference. Missing source audio or data
must be prepared before the export can produce a complete site.

## Build and check

From `website/`, after generating the data:

```bash
npm run build
npm run preview
node scripts/smoke.mjs
```

The smoke script uses Playwright and `/usr/bin/chromium-browser`. It visits the
explorer tabs, saves screenshots under `shots/`, and reports browser console
errors. Inspect that report; console errors are collected but do not currently
set a failing exit status.

The output is `dist/`. Vite uses `base: './'`, so the complete directory can be
served from a subpath such as `/research/turn-aware-asr/`. Deploy assets before
replacing `index.html` so the page always refers to available bundles.

## Where to edit

| Content | Source |
| --- | --- |
| Title, summary, paper links, headline numbers | [`src/components/Hero.jsx`](src/components/Hero.jsx) |
| Published abstract and citation | [`src/components/Publication.jsx`](src/components/Publication.jsx) |
| Method explanation and interactive examples | [`src/components/method/`](src/components/method/) |
| Result charts and tables | [`src/components/results/`](src/components/results/) |
| Audio and trajectory explorer | [`src/components/explorer/`](src/components/explorer/) |
| Navigation and page structure | [`src/App.jsx`](src/App.jsx) |
| Search metadata | [`index.html`](index.html) |
| Visual styling | [`src/styles.css`](src/styles.css) |
| Result/audio export | [`scripts/export_data.py`](scripts/export_data.py) |

Keep the pure endpointing model's headline numbers distinct from the unified
model's results. Update paper details against [arXiv:2609.04225](https://arxiv.org/abs/2609.04225).
