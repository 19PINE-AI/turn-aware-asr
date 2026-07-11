import React from 'react'

export function MainTable({ rows, heldout }) {
  return (
    <div className="card">
      <h3>Main endpointing result (deployment-matched replay, energy gate on)</h3>
      <p className="sub">
        Rows 1–4: the 25-stretch development benchmark (96 turn-final boundaries). Last row: a
        fresh twice-larger confirmation set drawn after all development ended. The causal model
        with the gate alone is better on every axis at once — its <i>raw</i> false-fire rate beats
        the mixed-pools model’s <i>confirmed</i> rate at half the latency and higher recall.
      </p>
      <div className="tbl-scroll">
        <table className="tbl">
          <thead>
            <tr><th>Policy</th><th className="num">Recall ↑</th><th className="num">P50 ↓</th>
              <th className="num">P95 ↓</th><th className="num">False/min ↓</th></tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className={r.hero ? 'hero-row' : ''}>
                <td>{r.policy}</td>
                <td className="num">{r.recall.toFixed(3)}</td>
                <td className="num">{r.p50.toFixed(2)} s</td>
                <td className="num">{r.p95.toFixed(2)} s</td>
                <td className="num">{r.ff.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="note">
        The released unified model scores 0.982 boundary recall at {heldout.unified_100.ff} false
        fires/min on the 100-stretch held-out set — still strictly inside the timeout family. The
        confirm horizon the earlier compositions needed effectively retires: it cancels 3 fire
        candidates for the causal model vs 241 for mixed-pools; the phrase-vs-turn discrimination
        moved from inference policy into the weights.
      </p>
    </div>
  )
}

export function DictationTable({ rows }) {
  const f = (v, d = 2) => (v === undefined ? '—' : typeof v === 'number' ? v.toFixed(d) : v)
  return (
    <div className="card">
      <h3>Dictation and spelled-entity probes</h3>
      <p className="sub">
        Premature fires per dictated 10-digit number, final-boundary recall, digit accuracy;
        exact-match on spelled names/emails without/with the profile prefix; wrong-profile
        intrusion. The conversational-only model reads a digit group as a finished thought,
        firing five times per number; matched-context training then produces 11.3% → 40%
        context copying; the counterfactual twin drives it to 0.8% with everything else intact.
      </p>
      <div className="tbl-scroll">
        <table className="tbl">
          <thead>
            <tr><th>Model</th><th className="num">Premature/seq ↓</th><th className="num">Final recall ↑</th>
              <th className="num">Digit acc ↑</th><th className="num">Name −/+ctx</th>
              <th className="num">Email −/+ctx</th><th className="num">Intrusion ↓</th></tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className={r.hero ? 'hero-row' : ''}>
                <td>{r.model}</td>
                <td className="num">{f(r.premat)}</td>
                <td className="num">{f(r.final)}</td>
                <td className="num">{f(r.digit, 3)}</td>
                <td className="num">{r.name ? `${r.name[0].toFixed(2)} / ${r.name[1].toFixed(2)}` : '—'}</td>
                <td className="num">{r.email ? `${r.email[0].toFixed(2)} / ${r.email[1].toFixed(2)}` : '—'}</td>
                <td className="num">{r.intrusion !== undefined ? `${r.intrusion}%` : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export function CompositionTable({ rows }) {
  return (
    <div className="card">
      <h3>The supervision-composition record: eight compositions, one architecture</h3>
      <p className="sub">
        Identical architecture, adapter, data volume, and source corpora — only the labels differ.
        Every addition before the causal relabel moved the model <i>along</i> an apparent
        frontier, never off it; by row 6 the recorded recommendation was to ship two checkpoints,
        one eager and one conservative.
      </p>
      <div className="tbl-scroll">
        <table className="tbl">
          <thead><tr><th>#</th><th>Supervision change (cumulative)</th><th>Observed behavior</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.row} className={r.causal ? 'hero-row' : ''}>
                <td className="num">{r.row}</td>
                <td>{r.change}</td>
                <td>{r.behavior}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
