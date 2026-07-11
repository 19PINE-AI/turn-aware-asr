import React, { useState } from 'react'
import ReplayExplorer from './ReplayExplorer.jsx'
import DictationExplorer from './DictationExplorer.jsx'
import SpelledExplorer from './SpelledExplorer.jsx'
import EarningsExplorer from './EarningsExplorer.jsx'

const TABS = [
  { id: 'replay', label: 'Streaming replay benchmark', comp: ReplayExplorer },
  { id: 'dictation', label: 'Dictation probe', comp: DictationExplorer },
  { id: 'spelled', label: 'Spelled entities & context', comp: SpelledExplorer },
  { id: 'earnings', label: 'Earnings-22 biasing', comp: EarningsExplorer },
]

export default function Explorer() {
  const [tab, setTab] = useState('replay')
  const Active = TABS.find(t => t.id === tab).comp
  return (
    <section className="block" id="explorer">
      <div className="wrap">
        <div className="kicker">Section 3 · Trajectory explorer</div>
        <h2>Listen to the benchmark. Inspect every trajectory.</h2>
        <p className="lede">
          Everything below is the <b>original recorded evaluation output</b> — the exact audio the
          systems heard (reconstructed sample-exactly from the deterministic benchmark builder and
          validated against the recorded event logs), and the exact fires, transcripts, and
          hypotheses they produced. Nothing is re-simulated or hand-picked.
        </p>
        <div className="explorer-tabs">
          {TABS.map(t => (
            <button key={t.id} className={tab === t.id ? 'on' : ''} onClick={() => setTab(t.id)}>{t.label}</button>
          ))}
        </div>
        <div style={{ marginTop: 6 }}>
          <Active />
        </div>
      </div>
    </section>
  )
}
