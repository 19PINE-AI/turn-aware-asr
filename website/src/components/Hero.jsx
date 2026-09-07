import React from 'react'

const stats = [
  { v: '0.97', k: 'boundary recall on the deployment-matched streaming benchmark' },
  { v: <>0.39<small> s</small></>, k: 'median end-of-turn latency — no silence timeout reaches this point' },
  { v: '0.3', k: 'false fires per speech-minute, raw (no confirm horizon)' },
  { v: <>0.10 → 1.00</>, k: 'fire recall of a “broken” model after appending 1 s of silence to the eval clips' },
]

export default function Hero() {
  return (
    <header className="hero">
      <div className="wrap">
        <div className="kicker">arXiv:2609.04225 · Open training recipe · benchmark</div>
        <h1>The Trade-off Was in the Labels</h1>
        <div className="subtitle">Causal Supervision for Turn-Aware Streaming ASR</div>
        <div className="authors">
          <b>Bojie Li</b> (Pine AI) &nbsp;·&nbsp; <b>Noah Shi</b> (University of Washington)
        </div>
        <p className="lede" style={{ marginTop: 18 }}>
          A small LoRA adapter on Qwen3-ASR-0.6B, trained in hours on one GPU,
          transcribes, detects end-of-turn from meaning and silence, handles dictation,
          and grounds transcription in context. We present the first <b>open training
          recipe and benchmark</b> for turn-aware streaming ASR. Its core rule:
          every streaming-decision label must be computable from input up to the
          decision point. Violating this rule manufactured a phantom recall-versus-precision
          trade-off through <b>clairvoyant labels</b>.
        </p>
        <div className="links">
          <a className="btn primary" href="https://arxiv.org/abs/2609.04225" target="_blank" rel="noreferrer">Paper · arXiv:2609.04225</a>
          <a className="btn" href="https://arxiv.org/pdf/2609.04225" target="_blank" rel="noreferrer">PDF</a>
          <a className="btn" href="https://github.com/19PINE-AI/turn-aware-asr" target="_blank" rel="noreferrer">Code &amp; training recipe</a>
          <a className="btn" href="#explorer">Explore the raw trajectories ↓</a>
        </div>
        <div className="statstrip">
          {stats.map((s, i) => (
            <div className="stat" key={i}>
              <div className="v">{s.v}</div>
              <div className="k">{s.k}</div>
            </div>
          ))}
        </div>
      </div>
    </header>
  )
}
