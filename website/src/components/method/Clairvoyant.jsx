import React from 'react'

const Chip = ({ txt, color }) => (
  <span style={{
    background: color, color: '#fff', fontWeight: 700, fontSize: 11.5,
    padding: '3px 12px', borderRadius: 999, letterSpacing: '0.04em', whiteSpace: 'nowrap', flexShrink: 0,
  }}>{txt}</span>
)

function Lane({ label, tail, tailLabel, future, chip, chipColor }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '9px 0', borderBottom: '1px dashed var(--line)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flex: 1, minWidth: 0 }}>
        <div style={{
          background: '#dbe9f6', border: '1px solid #3b7db5', borderRadius: 8,
          padding: '6px 12px', fontSize: 12.5, whiteSpace: 'nowrap', overflow: 'hidden',
          textOverflow: 'ellipsis', minWidth: 0,
        }}>{label}</div>
        {tail && (
          <div style={{
            background: '#f2e8c8', border: '1px solid #c9a227', borderRadius: 5,
            padding: '2px 8px', fontSize: 10.5, color: '#7a5c10', whiteSpace: 'nowrap', flexShrink: 0,
          }}>{tailLabel || '+ silence'}</div>
        )}
        {future && (
          <div style={{
            background: 'repeating-linear-gradient(45deg,#e8e6e0,#e8e6e0 4px,#f5f4f0 4px,#f5f4f0 8px)',
            border: '1px solid #b9b6ad', borderRadius: 6, padding: '4px 10px',
            fontSize: 10.5, color: '#6b7078', whiteSpace: 'nowrap', flexShrink: 0,
          }}>future ▒</div>
        )}
      </div>
      <Chip txt={chip} color={chipColor} />
    </div>
  )
}

export default function Clairvoyant() {
  return (
    <>
      <div className="callout" style={{ marginTop: 32 }}>
        <b className="title">Definition — clairvoyant labels</b>
        <p>
          A label for a streaming decision at time <i>t</i> is <b>clairvoyant</b> if its value
          depends on input after <i>t</i>. Such labels don’t merely add noise: because the
          dependence is systematic, they partition training data into internally consistent pools
          with <b>incompatible optima</b>. The symptoms: (a) training-time oscillation between mode
          attractors, and (b) apparent capability trade-offs across checkpoints — a phantom Pareto
          frontier.
        </p>
      </div>
      <div className="callout green">
        <b className="title">The principle</b>
        <p>
          Every training label and every evaluation target for a streaming decision must be
          computable from the input up to the decision point.
        </p>
      </div>

      <div className="two-col" style={{ marginTop: 8 }}>
        <div className="card">
          <h3 style={{ fontSize: 17 }}>Leak 1 — the trailing-silence clip cut</h3>
          <p className="sub" style={{ fontSize: 13.5 }}>
            Offline corpora are segmented by forced alignment, so clips end exactly at end of
            speech. An endpoint pool built on them says <b>fire</b> on a prefix that a streaming
            pool — correctly — labels <b>hold</b> (no silence observed yet). Opposite labels,
            causally identical inputs.
          </p>
          <Lane label="“let’s move on to the next slide” ⏐ clip ends" chip="OFFLINE: FIRE" chipColor="#c23b22" />
          <Lane label="“let’s move on to the next slide”" tail tailLabel="stream continues…" chip="STREAMING: HOLD" chipColor="#0d5c96" />
          <p className="note">
            Deployment never presents audio that ends at end-of-speech — on a live channel the
            next silence always arrives.
          </p>
        </div>
        <div className="card">
          <h3 style={{ fontSize: 17 }}>Leak 2 — pause labels keyed on the future</h3>
          <p className="sub" style={{ fontSize: 13.5 }}>
            Historical pools separated “disfluent pause → hold” from “turn boundary → fire” using
            who spoke <i>next</i> in the transcript. Measured at the decision point the two are
            statistically indistinguishable — the within-turn pauses are even <i>longer</i>{' '}
            (medians 1.39 s vs 0.83 s).
          </p>
          <Lane label="“so what I think is…” (pause 1.2 s)" future chip="same speaker → HOLD" chipColor="#0d5c96" />
          <Lane label="“so what I think is…” (pause 1.2 s)" future chip="next speaker → FIRE" chipColor="#c23b22" />
          <p className="note">
            The label is a function of audio after the decision point — of the future.
          </p>
        </div>
      </div>
    </>
  )
}
