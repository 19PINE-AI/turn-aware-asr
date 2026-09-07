import React from 'react'
import Publication from './components/Publication.jsx'
import Hero from './components/Hero.jsx'
import Method from './components/method/Method.jsx'
import Results from './components/results/Results.jsx'
import Explorer from './components/explorer/Explorer.jsx'

export default function App() {
  return (
    <>
      <nav className="nav">
        <div className="nav-inner">
          <span className="nav-title">The Trade-off Was in the Labels</span>
          <a href="#method">How it works</a>
          <a href="#results">Results</a>
          <a href="#explorer">Trajectory explorer</a>
          <a className="hide-sm" href="https://arxiv.org/abs/2609.04225" target="_blank" rel="noreferrer">Paper ↗</a>
          <a className="hide-sm" href="https://github.com/19PINE-AI/turn-aware-asr" target="_blank" rel="noreferrer">Code ↗</a>
        </div>
      </nav>
      <Hero />
      <Method />
      <Results />
      <Explorer />
      <Publication />
      <footer className="footer">
        <div className="wrap">
          <div>
            <b>The Trade-off Was in the Labels: Causal Supervision for Turn-Aware Streaming ASR</b>
            <div>Bojie Li (Pine AI) · Noah Shi (University of Washington)</div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div><a href="https://github.com/19PINE-AI/turn-aware-asr" target="_blank" rel="noreferrer">github.com/19PINE-AI/turn-aware-asr</a></div>
            <div>Every trajectory on this page is the original recorded evaluation output — nothing is re-simulated.</div>
          </div>
        </div>
      </footer>
    </>
  )
}
