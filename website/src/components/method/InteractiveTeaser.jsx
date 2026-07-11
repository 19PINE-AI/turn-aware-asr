import React, { useState } from 'react'

const C = { red: '#c23b22', green: '#00694f', blue: '#0d5c96', grey: '#8a8f98' }

// schematic seconds → svg x
const W = 940, PAD = 24
const x = (s) => PAD + (s / 10.2) * (W - 2 * PAD)

function Speech({ b, e, y, label }) {
  return (
    <g>
      <rect x={x(b)} y={y - 17} width={x(e) - x(b)} height={34} rx={8}
        fill="#dbe9f6" stroke="#3b7db5" strokeWidth={1.2} />
      <text x={(x(b) + x(e)) / 2} y={y + 4} textAnchor="middle" fontSize={12.5} fill="#1c1e21">{label}</text>
    </g>
  )
}

function FireX({ s, y, label, above }) {
  return (
    <g>
      <path d={`M${x(s) - 6},${y - 6} L${x(s) + 6},${y + 6} M${x(s) - 6},${y + 6} L${x(s) + 6},${y - 6}`}
        stroke={C.red} strokeWidth={3.4} strokeLinecap="round" />
      {label && <text x={x(s)} y={above ? y - 14 : y + 24} textAnchor="middle" fontSize={11.5} fill={C.red} fontWeight={600}>{label}</text>}
    </g>
  )
}

function Star({ s, y, label }) {
  const pts = []
  for (let i = 0; i < 10; i++) {
    const r = i % 2 === 0 ? 9.5 : 4.2
    const a = -Math.PI / 2 + (i * Math.PI) / 5
    pts.push(`${x(s) + r * Math.cos(a)},${y + r * Math.sin(a)}`)
  }
  return (
    <g>
      <polygon points={pts.join(' ')} fill={C.green} stroke="#083f30" strokeWidth={0.8} />
      {label && <text x={x(s)} y={y + 26} textAnchor="middle" fontSize={11.5} fill={C.green} fontWeight={700}>{label}</text>}
    </g>
  )
}

export default function InteractiveTeaser() {
  const [X, setX] = useState(0.7)

  // Turn 1: dictated number, gaps 0.9 s and 1.1 s inside the turn.
  const t1 = [
    { b: 0.3, e: 2.1, label: '“four one five …”' },
    { b: 3.0, e: 4.9, label: '“five five five …”' },
    { b: 6.0, e: 8.3, label: '“zero one nine two.”' },
  ]
  const gaps = [{ after: 2.1, len: 0.9 }, { after: 4.9, len: 1.1 }]
  const midFires = gaps.filter(g => X <= g.len).map(g => g.after + X)
  // Turn 2: complete thought at 3.1 s.
  const t2End = 3.1

  return (
    <div className="card">
      <h3>Try it: no timeout gets both turns right</h3>
      <p className="sub">
        A silence timeout fires after <b>X</b> seconds of quiet. Drag X and watch the dilemma:
        short timeouts interrupt the dictated number mid-turn; long ones leave the user hanging
        after a completed thought. The turn-aware model reads <i>completeness</i>, so it holds
        through the pauses and fires 0.39&thinsp;s after the thought ends.
      </p>
      <div className="strip" style={{ marginBottom: 4 }}>
        <label>silence timeout X = {X.toFixed(1)} s</label>
        <input type="range" min={0.3} max={2.0} step={0.1} value={X}
          onChange={e => setX(parseFloat(e.target.value))} style={{ width: 260 }} />
        <span style={{ fontSize: 13, color: 'var(--ink-soft)' }}>
          {midFires.length > 0
            ? <>❌ interrupts the number <b>{midFires.length}×</b>, </>
            : <>✓ survives the pauses, </>}
          {X > 0.41
            ? <>then makes the user wait <b>{(X - 0.39).toFixed(2)} s longer</b> than ours after “I’m still here.”</>
            : <>and answers promptly — but no deployed timeout runs this low (it would fire inside ordinary pauses).</>}
        </span>
      </div>
      <svg className="timeline-svg" viewBox={`0 0 ${W} 245`}>
        {/* Turn 1 */}
        <text x={PAD} y={18} fontSize={13.5} fontStyle="italic" fill="#1c1e21">Turn 1 — the pause that means “wait” (dictating a phone number)</text>
        <line x1={x(0.15)} x2={x(10.1)} y1={52} y2={52} stroke="#555" strokeWidth={1} />
        {t1.map((s, i) => <Speech key={i} {...s} y={52} />)}
        {gaps.map((g, i) => (
          <text key={i} x={(x(g.after) + x(g.after + g.len)) / 2} y={30} textAnchor="middle" fontSize={11} fill={C.grey}>{g.len.toFixed(1)} s pause</text>
        ))}
        {midFires.map((s, i) => <FireX key={i} s={s} y={52} label={i === 0 ? `X=${X.toFixed(1)}s fires mid-number` : null} above={false} />)}
        <Star s={8.3 + 0.39} y={52} label="ours: +0.39 s" />
        {midFires.length === 0 && X <= 2.0 &&
          <FireX s={8.3 + X} y={52} label={`timeout: +${X.toFixed(1)} s`} above />}
        <text x={x(4.55)} y={96} textAnchor="middle" fontSize={11.5} fill={C.green}>
          ours: holds — six digits of a ten-digit pattern predict four more
        </text>

        {/* Turn 2 */}
        <text x={PAD} y={133} fontSize={13.5} fontStyle="italic" fill="#1c1e21">Turn 2 — the completion that means “go”</text>
        <line x1={x(0.15)} x2={x(10.1)} y1={170} y2={170} stroke="#555" strokeWidth={1} />
        <Speech b={0.3} e={t2End} y={170} label='“Hello? I’m still here.”' />
        <Star s={t2End + 0.39} y={170} label="ours: +0.39 s — thought complete" />
        <FireX s={t2End + X} y={170} label={`timeout: still waiting ${X.toFixed(1)} s`} above />
        <text x={x(0.3)} y={228} fontSize={11.5} fill={C.grey}>
          Humans respond within ≈0.2 s of a completed turn; deployed timeouts sit at 0.5–1.0 s.
        </text>
      </svg>
      <p className="note">
        Interactive version of Figure 1 in the paper. The pauses <i>inside</i> turns are longer
        than the gaps <i>between</i> turns on real meeting audio — no threshold separates them.
        The resolving signal is in the words, and the words live inside the recognizer.
      </p>
    </div>
  )
}
