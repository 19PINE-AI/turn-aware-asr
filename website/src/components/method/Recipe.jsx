import React from 'react'

const M = () => <code style={{ background: '#f2efe6', borderRadius: 4, padding: '0 4px', color: '#9a6b00' }}>⟨M⟩</code>

function PairRow({ audio, tail, ctx, ctxWrong, target, chip, chipColor }) {
  return (
    <div className="schema-row">
      <div style={{ flex: 1, minWidth: 0 }}>
        {ctx && (
          <div style={{
            display: 'inline-block', fontSize: 11.5, padding: '2px 10px', borderRadius: 6, marginBottom: 5,
            background: ctxWrong ? 'var(--red-soft)' : 'var(--accent-soft)',
            color: ctxWrong ? 'var(--red)' : 'var(--accent)', fontWeight: 600,
          }}>{ctx}</div>
        )}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{
            background: '#dbe9f6', border: '1px solid #3b7db5', borderRadius: 8,
            padding: '6px 12px', fontSize: 13, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
          }}>{audio}</div>
          {tail && (
            <div style={{
              background: '#f2e8c8', border: '1px solid #c9a227', borderRadius: 5,
              padding: '2px 8px', fontSize: 11, color: '#7a5c10', whiteSpace: 'nowrap',
            }}>+ {tail} silence</div>
          )}
        </div>
      </div>
      <div style={{ fontFamily: 'var(--mono)', fontSize: 12.5, minWidth: 170 }}>{target}</div>
      <span style={{
        background: chipColor, color: '#fff', fontWeight: 700, fontSize: 11,
        padding: '3px 11px', borderRadius: 999, whiteSpace: 'nowrap', flexShrink: 0,
      }}>{chip}</span>
    </div>
  )
}

const schemas = [
  ['1', 'complete utterance + 0.3–1.2 s silence tail', <>text <M /></>],
  ['2', 'same utterances as #1, tail ≤ 0.1 s', 'text'],
  ['3', 'speech truncated mid-utterance at 30–80%', 'partial text'],
  ['4', 'complete A + gap 0.3–2.5 s + B (+ tail)', <>A <M /> B <M /> / A <M /> B</>],
  ['5', 'incomplete A + gap + B + tail', <>A B <M /></>],
  ['6', 'complete utterance + long silence (2–4 s)', <>text <M /></>],
  ['7', 'pure silence 0.5–4 s', '(empty)'],
  ['8', 'leading silence 0.5–2 s + utterance + tail', <>text <M /></>],
]

export default function Recipe() {
  return (
    <>
      <div className="callout green" style={{ marginTop: 36 }}>
        <b className="title">One labeling rule generates every training example</b>
        <p style={{ fontSize: 16 }}>
          Emit the end-of-turn marker at a point <b>iff</b> the speech so far is{' '}
          <b>semantically complete</b> <u>and</u> <b>≥ 0.3 s of silence has been observed</b> since
          the last speech. Both conditions are functions of the audio prefix — nothing else changed
          between the oscillating compositions and the monotone one.
        </p>
      </div>

      <div className="two-col">
        <div className="card">
          <h3 style={{ fontSize: 17 }}>The minimal pair (temporal axis)</h3>
          <p className="sub" style={{ fontSize: 13.5 }}>
            The same complete utterance appears twice with opposite targets. The only difference
            is an <i>observable</i> silence tail — so completeness alone must not fire, and the
            model is forced to key on the one signal it can trust.
          </p>
          <PairRow audio="“turn the lights off in the kitchen”" target="…kitchen" chip="HOLD" chipColor="#0d5c96" />
          <PairRow audio="“turn the lights off in the kitchen”" tail="0.3–1.2 s"
            target={<>…kitchen <M /></>} chip="FIRE" chipColor="#c23b22" />
        </div>
        <div className="card">
          <h3 style={{ fontSize: 17 }}>The counterfactual twin (context axis)</h3>
          <p className="sub" style={{ fontSize: 13.5 }}>
            Trained only on matching profiles, the model learns to <i>copy</i> the context instead
            of listening — a wrong-user profile wrote the wrong entity into the transcript 40% of
            the time. A third of examples carry a deliberately <i>conflicting</i> profile whose
            target follows the audio; intrusion drops to 0.8%.
          </p>
          <PairRow audio="audio spells “K-O-W-A-L-S-K-I”" ctx="ctx: profile = Kowalski"
            target="kowalski" chip="GROUND" chipColor="#00694f" />
          <PairRow audio="audio spells “K-O-W-A-L-S-K-I”" ctx="ctx: profile = Nguyen (wrong)" ctxWrong
            target="kowalski" chip="IGNORE CTX" chipColor="#00694f" />
        </div>
      </div>

      <div className="card">
        <h3 style={{ fontSize: 17 }}>The eight training schemas (~12k examples, LibriSpeech + AMI ≈ 50/50)</h3>
        <div className="tbl-scroll">
          <table className="tbl">
            <thead><tr><th>#</th><th>Construction</th><th>Target</th></tr></thead>
            <tbody>
              {schemas.map(([n, c, t]) => (
                <tr key={n}>
                  <td className="num">{n}</td>
                  <td>{c}</td>
                  <td className="mono">{t}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="note">
          Schemas 1–2 are the minimal pair; 4–5 the pause pair (bridge or fire a gap depending only
          on whether the words before it are complete); 7–8 cover the silence-heavy channel shape
          pure-utterance corpora never show. Pair examples mix same- and different-speaker sources,
          so the fire decision cannot key on speaker identity. Dictation and spelled-entity schemas
          extend the pool to ~20k for the released unified model.
        </p>
      </div>
    </>
  )
}
