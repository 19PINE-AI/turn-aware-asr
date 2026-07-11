import React from 'react'
import InteractiveTeaser from './InteractiveTeaser.jsx'
import SystemDiagram from './SystemDiagram.jsx'
import Clairvoyant from './Clairvoyant.jsx'
import Recipe from './Recipe.jsx'
import Checklist from './Checklist.jsx'

export default function Method() {
  return (
    <section className="block" id="method">
      <div className="wrap">
        <div className="kicker">Section 1 · How it works</div>
        <h2>Turn-aware ASR, and the labels that make it possible</h2>
        <p className="lede">
          The deployed default decides end-of-turn from acoustics alone — a voice-activity
          detector plus a silence timeout. Frontier commercial recognizers moved the decision
          into the ASR model, where the words are; every member of that class is closed. This
          work opens it, and finds that the entire turn capability comes from <b>what the labels
          must satisfy</b>, not from architecture.
        </p>

        <InteractiveTeaser />
        <SystemDiagram />

        <h2 style={{ marginTop: 56, fontSize: 27 }}>The failure class: clairvoyant labels</h2>
        <p className="lede">
          Supervision inherited from offline corpora encodes the future in two places — and a
          streaming model can never see the future. Eight successive supervision compositions
          with identical architecture, adapter, and data volume traced what looked like a
          fundamental recall-versus-precision frontier. It wasn’t.
        </p>
        <Clairvoyant />

        <h2 style={{ marginTop: 56, fontSize: 27 }}>The causal recipe</h2>
        <p className="lede">
          The cure is a construction, not a loss term: make the untrustworthy signal
          uninformative by design, so the only policy consistent with the training data is the
          one that keys on observables.
        </p>
        <Recipe />
        <Checklist />
      </div>
    </section>
  )
}
