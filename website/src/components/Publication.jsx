import React from 'react'

const abstract = 'A voice agent must decide, moment to moment, whether the user has finished; silence rarely settles it: a caller reading a phone number pauses mid-digits, a one-word "Stop!" ends a turn, a long question carries pauses longer than real turn-gaps. A voice-activity detector plus a silence timeout (the deployed default) cannot separate these, because within-turn pauses routinely exceed between-turn gaps; what distinguishes them is whether the words so far form a complete thought: what a recognizer computes to produce a transcript. We present the first open training recipe and benchmark for turn-aware streaming ASR: a small LoRA adapter on Qwen3-ASR-0.6B, trained in hours on one GPU, that transcribes, detects end-of-turn from meaning and silence, handles dictation, and grounds transcription in context. On a deployment-matched benchmark it reaches 0.97 boundary recall at 0.39 s median latency with 0.3 false fires per speech-minute, replicated on a fresh test set; no silence timeout reaches this point. The recipe rests on one principle: every streaming-decision label must be computable from input up to the decision point. Offline corpora violate it, encoding the future; such clairvoyant labels manufactured oscillation and a phantom recall-versus-precision trade-off, exposed when one appended second of silence raised a "broken" model\'s end-of-turn recall from 0.10 to 1.00. The same leak recurred with context: an always-matching biasing prefix became a copied shortcut (40% intrusion), and counterfactuals disagreeing with the audio cut this to 0.8% while keeping most of a +28.9 pp entity-recall benefit.'
const citation = `@misc{li2026tradeofflabels,
  title = {The Trade-off Was in the Labels: Causal Supervision for Turn-Aware Streaming ASR},
  author = {Bojie Li and Noah Shi},
  year = {2026},
  eprint = {2609.04225},
  archivePrefix = {arXiv},
  primaryClass = {eess.AS},
  doi = {10.48550/arXiv.2609.04225},
  url = {https://arxiv.org/abs/2609.04225}
}`

export default function Publication() {
  return (
    <section className="block" id="publication">
      <div className="wrap">
        <h2>Abstract</h2>
        <p className="lede">{abstract}</p>
        <h3 style={{ marginTop: 28 }}>Citation</h3>
        <p><a href="https://arxiv.org/abs/2609.04225" target="_blank" rel="noreferrer">arXiv:2609.04225</a> · <a href="https://doi.org/10.48550/arXiv.2609.04225" target="_blank" rel="noreferrer">DOI: 10.48550/arXiv.2609.04225</a></p>
        <pre className="hyp" style={{ whiteSpace: 'pre-wrap' }}><code>{citation}</code></pre>
      </div>
    </section>
  )
}
