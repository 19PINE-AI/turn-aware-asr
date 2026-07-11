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
        <div className="kicker">Open weights · training recipe · benchmark</div>
        <h1>The Trade-off Was in the Labels</h1>
        <div className="subtitle">Causal Supervision for Turn-Aware Streaming ASR</div>
        <div className="authors">
          <b>Bojie Li</b> (Pine AI) &nbsp;·&nbsp; <b>Noah Shi</b> (University of Washington)
        </div>
        <p className="lede" style={{ marginTop: 18 }}>
          Every voice agent must decide, in real time, whether you have finished talking.
          We present the first <b>open</b> turn-aware streaming ASR system with a published training
          recipe — a small LoRA adapter on Qwen3-ASR-0.6B, trained in hours on one GPU — and show
          that the recall-versus-precision “frontier” that haunted eight successive supervision
          compositions was manufactured by <b>clairvoyant labels</b>: supervision that peeks at
          audio after the decision point.
        </p>
        <div className="links">
          <a className="btn primary" href="https://github.com/19PINE-AI/turn-aware-asr" target="_blank" rel="noreferrer">Code &amp; weights</a>
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
