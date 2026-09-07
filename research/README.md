# Research records and results

[Project overview](../README.md) · [Training guide](../REPRODUCING.md) · [Code map](../docs/code-map.md)

This directory is the experiment archive. Numbered files record what was tried,
measured, and concluded at that point in development. They are not a sequential
setup tutorial, and early plans can contradict the final method. Start with the
published [paper](https://arxiv.org/abs/2609.04225) for the current claims.

## Suggested reading

| Question | Record |
| --- | --- |
| How is continuous streaming evaluated? | [58: replay protocol](58-unified-eval-design.md) |
| How are causal training labels constructed? | [59: causal recipe](59-v9-recipe-design.md) |
| What did the pure endpointing model achieve? | [62: causal endpointing results](62-v9-results.md) |
| Which unified configuration was selected? | [123: v18 selection](123-v18-release-decision.md) |
| How were external systems compared? | [102: external baselines](102-exp4-external-baselines.md) |
| What are the dictation generalization limits? | [93: generalization findings](93-gen-findings.md) |
| Where are serving measurements discussed? | [134: v18 serving confirmation](134-v18-serving-confirm.md) |

`v9` denotes the pure causal endpointing model; `v18` denotes the selected unified
configuration with dictation and context grounding. A historical “release” label
selects a checkpoint within the experiments; it does not imply a public weight
download. The repository currently hosts code and results, not trained weights.

## Recorded outputs

| File | Configuration |
| --- | --- |
| [124-replay-v18-dev25.json](124-replay-v18-dev25.json) | Unified model, 25-stretch development replay |
| [124-replay-v18-dev25-h1.json](124-replay-v18-dev25-h1.json) | Same model with one-chunk silence confirmation |
| [125-replay-v18-big.json](125-replay-v18-big.json) | Unified model, 100-stretch replay |

Replay JSONs contain aggregate metrics in `summary` and individual trajectories
in `per_stretch`. Other experiment families have their own JSON structures.
The website's [data exporter](../website/scripts/export_data.py) maps recorded
results into the figures and audio explorer. Use the
[live explorer](https://01.me/research/turn-aware-asr/#explorer) to inspect them
without rebuilding audio locally.

Write new experiment outputs to `eval/results/` or a new named location. Keep the
recorded paper results intact so comparisons retain their provenance.
