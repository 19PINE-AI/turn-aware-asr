import React from 'react'
import { COL, linScale, XAxis, polyline } from './chart.jsx'

const W = 460, H = 300, ML = 46, MR = 12, MT = 16, MB = 46

function Panel({ title, series, xmax }) {
  const x = linScale(0, xmax, ML, W - MR)
  const y = linScale(0, 1.05, H - MB, MT)
  return (
    <svg className="timeline-svg" viewBox={`0 0 ${W} ${H}`}>
      {[0, 0.25, 0.5, 0.75, 1].map((t, i) => (
        <g key={i}>
          <line x1={ML} x2={W - MR} y1={y(t)} y2={y(t)} stroke="#ecebe4" />
          <text x={ML - 6} y={y(t) + 4} textAnchor="end" fontSize={10.5} fill="#6b7078">{t}</text>
        </g>
      ))}
      <XAxis x={x} y={H - MB} x0={ML} x1={W - MR}
        ticks={Array.from({ length: Math.floor(xmax / 3) + 1 }, (_, i) => i * 3)}
        label="training step (k)" />
      {series.map((s, i) => (
        <g key={i}>
          <polyline points={polyline(s.pts.map(p => [x(p[0]), y(p[1])]))} fill="none"
            stroke={s.color} strokeWidth={s.width || 2} strokeDasharray={s.dash || 'none'} opacity={s.op || 1} />
          {(s.dots !== false) && s.pts.map((p, j) => (
            <circle key={j} cx={x(p[0])} cy={y(p[1])} r={2.4} fill={s.color} opacity={s.op || 1} />
          ))}
        </g>
      ))}
      <text x={(ML + W - MR) / 2} y={13} textAnchor="middle" fontSize={12.5} fontWeight={600} fill="#1c1e21">{title}</text>
    </svg>
  )
}

export default function OscillationChart({ data }) {
  const s0 = data.opposed_seed0, s1 = data.opposed_seed1, c = data.causal
  const xmax = Math.max(...s0.map(e => e.step), ...c.map(e => e.step)) / 1000
  return (
    <div className="card">
      <h3>The fingerprint of contradictory supervision — and its cure</h3>
      <p className="sub">
        Holdout accuracy over training, from the original run logs. Left: the opposed-pools
        composition oscillates for the <i>entire run</i> between a fire-mode and a no-fire-mode
        attractor — reproducibly across two seeds, so the oscillation is a property of the labels.
        Right: the causal recipe with the same architecture, adapter, and data volume, only the
        labels changed.
      </p>
      <div className="two-col">
        <div>
          <Panel title="opposed-pools: clairvoyant labels (2 seeds)" xmax={xmax} series={[
            { pts: s0.map(e => [e.step / 1000, e.fire]), color: COL.red },
            { pts: s0.map(e => [e.step / 1000, e.hold]), color: COL.blue },
            { pts: s1.map(e => [e.step / 1000, e.fire]), color: COL.red, dash: '5 4', op: 0.45, width: 1.4, dots: false },
            { pts: s1.map(e => [e.step / 1000, e.hold]), color: COL.blue, dash: '5 4', op: 0.45, width: 1.4, dots: false },
          ]} />
        </div>
        <div>
          <Panel title="causal labels (only the supervision changed)" xmax={xmax} series={[
            { pts: c.map(e => [e.step / 1000, e.fire]), color: COL.red },
            { pts: c.map(e => [e.step / 1000, e.hold]), color: COL.blue },
            { pts: c.map(e => [e.step / 1000, e.composite]), color: COL.grey, dash: '4 4', width: 1.4, dots: false },
          ]} />
        </div>
      </div>
      <div className="legend">
        <span><i style={{ background: COL.red }} /> fire class</span>
        <span><i style={{ background: COL.blue }} /> hold classes</span>
        <span><i style={{ background: COL.grey }} /> composite (8 schemas, right panel)</span>
        <span style={{ color: 'var(--ink-faint)' }}>dashed = second seed</span>
      </div>
    </div>
  )
}
