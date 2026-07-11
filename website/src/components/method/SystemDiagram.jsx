import React from 'react'

function Box({ x, y, w, h, title, sub, fill, stroke, dashed }) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={10} fill={fill} stroke={stroke}
        strokeWidth={1.4} strokeDasharray={dashed ? '5 4' : 'none'} />
      <text x={x + w / 2} y={y + h / 2 + (sub ? -6 : 4)} textAnchor="middle" fontSize={13} fontWeight={600} fill="#1c1e21">{title}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 12} textAnchor="middle" fontSize={10.5} fill="#4a4f57">{sub}</text>}
    </g>
  )
}

function Arrow({ x1, y1, x2, y2, color = '#4a4f57', label, labelDy = -6 }) {
  const id = `${x1}${y1}${x2}${y2}`.replace(/[.-]/g, '')
  return (
    <g>
      <defs>
        <marker id={`ah${id}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
        </marker>
      </defs>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={1.6} markerEnd={`url(#ah${id})`} />
      {label && <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 + labelDy} textAnchor="middle" fontSize={10.5} fill={color}>{label}</text>}
    </g>
  )
}

export default function SystemDiagram() {
  return (
    <div className="card">
      <h3>The system: one small model, three interpretable knobs</h3>
      <p className="sub">
        A LoRA adapter (rank 16–32) on Qwen3-ASR-0.6B claims two reserved vocabulary slots as
        marker tokens and learns to emit them <i>inside</i> the transcript at endpoint moments.
        Training: ~20k synthesized examples, hours on one GPU. Everything else sits outside the
        model as auditable policy.
      </p>
      <svg className="timeline-svg" viewBox="0 0 960 250">
        {/* audio chunks */}
        {[0, 1, 2, 3].map(i => (
          <rect key={i} x={20 + i * 26} y={95} width={20} height={52} rx={5} fill="#cfe8f7" stroke="#3b7db5" strokeWidth={1.2} />
        ))}
        <text x={72} y={172} textAnchor="middle" fontSize={11} fill="#4a4f57">0.5 s audio chunks</text>
        <Arrow x1={132} y1={121} x2={168} y2={121} />

        <Box x={166} y={92} w={112} h={58} title="energy gate" sub="skips silent chunks" fill="#f7edd4" stroke="#c9a227" />
        <Arrow x1={274} y1={121} x2={312} y2={121} />

        {/* model */}
        <rect x={314} y={40} width={330} height={186} rx={12} fill="#f4f4f2" stroke="#9aa0a8" strokeWidth={1.3} />
        <text x={479} y={64} textAnchor="middle" fontSize={13} fontWeight={700} fill="#1c1e21">Qwen3-ASR-0.6B + LoRA (merged)</text>
        <Box x={334} y={92} w={116} h={54} title="audio encoder" fill="#dbe9f6" stroke="#3b7db5" />
        <Arrow x1={450} y1={119} x2={470} y2={119} />
        <Box x={472} y={92} w={150} h={54} title="LM decoder" sub="+ 2 marker token rows" fill="#dbe9f6" stroke="#3b7db5" />
        <Box x={334} y={162} w={288} h={46} title="committed transcript prefix" sub="bounded re-feed of the last audio window" fill="#ececea" stroke="#9aa0a8" />
        {/* ctx */}
        <Box x={380} y={6} w={200} h={30} title="⟨CTX⟩ session context prefix" fill="#e0efe9" stroke="#00694f" />
        <Arrow x1={480} y1={36} x2={480} y2={88} color="#00694f" />

        <Arrow x1={644} y1={121} x2={684} y2={121} />
        <Box x={686} y={92} w={120} h={58} title="policy knobs" sub="confirm h · force-flush" fill="#fbe3d2" stroke="#c26a2a" />
        <Arrow x1={806} y1={104} x2={846} y2={74} />
        <Arrow x1={806} y1={138} x2={846} y2={172} />
        <Box x={848} y={48} w={100} h={50} title="transcript" sub="segments" fill="#e8e8f8" stroke="#5555aa" />
        <Box x={848} y={150} w={100} h={50} title="END events" sub="end-of-turn fires" fill="#f9e7e2" stroke="#c23b22" />
      </svg>
      <p className="note">
        The knobs: an RMS <b>energy gate</b> never <i>starts</i> a segment on a silent chunk; a{' '}
        <b>confirm horizon h</b> holds a fire candidate for h further silent chunks before
        signalling END (the causal model barely needs it: 3 cancelled candidates vs 241 for the
        mixed-pools model); a <b>max-segment force-flush</b> bounds the committed transcript.
      </p>
    </div>
  )
}
