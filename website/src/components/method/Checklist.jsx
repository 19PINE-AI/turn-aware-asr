import React from 'react'

const items = [
  {
    b: '1 · Prefix test',
    p: 'For each label at decision time t: is it computable from input up to t?',
    eg: 'Our pause labels were keyed on who spoke next.',
  },
  {
    b: '2 · Twin test',
    p: 'Can two causally identical prefixes receive different labels anywhere in the data? Minimal pairs differing in an observable are the fix, not the bug.',
    eg: 'The same complete-utterance prefix carried fire in one pool, hold in the other.',
  },
  {
    b: '3 · Phantom-condition test',
    p: 'Does the evaluation demand behavior on conditions deployment never produces?',
    eg: 'Clips cut at end-of-speech; a live channel always delivers the next silence.',
  },
  {
    b: '4 · Oscillation test',
    p: 'Does training bounce between mode attractors that each satisfy part of the data? That is the loss surface reporting a contradiction no individual example states.',
    eg: 'Two attractors: fire-mode and no-fire-mode, for the entire run, across seeds.',
  },
]

export default function Checklist() {
  return (
    <div className="card">
      <h3>Is your supervision clairvoyant? A four-test checklist</h3>
      <p className="sub">
        The failure mode generalizes to any model that must act mid-stream on offline-labeled
        data: turn-taking policies for full-duplex agents, barge-in, simultaneous translation with
        reference-aligned read/write decisions, tool-call timing for agents cloned from completed
        trajectories.
      </p>
      <div className="checklist">
        {items.map((it, i) => (
          <div className="check-item" key={i}>
            <b>{it.b}</b>
            <p>{it.p}</p>
            <div className="eg">{it.eg}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
