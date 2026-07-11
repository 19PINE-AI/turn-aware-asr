import React from 'react'
import { COL, linScale, polyline } from './chart.jsx'

export default function BiasingChart({ data }) {
  const bars = data.bars
  const W1 = 380, H = 280, MB = 58, MT = 30
  const yb = linScale(0, 105, H - MB, MT)
  const bar = (i, v, w, color, op = 1) => (
    <g>
      <rect x={i} y={yb(v)} width={w} height={yb(0) - yb(v)} rx={4} fill={color} opacity={op} />
    </g>
  )

  const W2 = 460
  const ages = data.by_age
  const x2 = linScale(0, Math.max(...ages.map(a => a.age)), 56, W2 - 20)
  const y2 = linScale(40, 100, H - MB, MT)

  return (
    <div className="card">
      <h3>Context biasing works — and survives session length</h3>
      <p className="sub">
        On Earnings-22 (spontaneous speech dense in tickers, executives, products), a relevant
        hotword prefix lifts entity recall by <b>+28.9 pp</b> while nudging WER slightly{' '}
        <i>down</i>. The counterfactual twin keeps the prefix a prior, not a substitute for
        listening: the released model hallucinates a wrong-profile entity <i>less often than the
        base model</i>. Right: the advantage does not decay as speech interposes between prefix
        and target — the property a voice agent needs at minute five, not just utterance one.
      </p>
      <div className="two-col">
        <svg className="timeline-svg" viewBox={`0 0 ${W1} ${H}`}>
          {[0, 25, 50, 75, 100].map((t, i) => (
            <g key={i}>
              <line x1={48} x2={W1 - 12} y1={yb(t)} y2={yb(t)} stroke="#ecebe4" />
              <text x={42} y={yb(t) + 4} textAnchor="end" fontSize={10.5} fill="#6b7078">{t}</text>
            </g>
          ))}
          {bar(70, bars.no_ctx, 62, COL.grey)}
          <text x={101} y={yb(bars.no_ctx) - 7} textAnchor="middle" fontSize={13} fontWeight={700}>{bars.no_ctx}%</text>
          <text x={101} y={H - MB + 18} textAnchor="middle" fontSize={11} fill="#4a4f57">no context</text>
          {bar(160, bars.relevant, 62, COL.green)}
          <text x={191} y={yb(bars.relevant) - 7} textAnchor="middle" fontSize={13} fontWeight={700} fill={COL.green}>{bars.relevant}%</text>
          <text x={191} y={H - MB + 18} textAnchor="middle" fontSize={11} fill="#4a4f57">relevant prefix</text>
          <text x={191} y={H - MB + 32} textAnchor="middle" fontSize={10} fill="#8a8f98">entity recall</text>
          {bar(262, bars.halluc_base, 30, COL.red, 0.45)}
          {bar(298, bars.halluc_released, 30, COL.red)}
          <text x={277} y={yb(bars.halluc_base) - 6} textAnchor="middle" fontSize={11} fill={COL.red}>{bars.halluc_base}%</text>
          <text x={313} y={yb(bars.halluc_released) - 6} textAnchor="middle" fontSize={11} fontWeight={700} fill={COL.red}>{bars.halluc_released}%</text>
          <text x={295} y={H - MB + 18} textAnchor="middle" fontSize={11} fill="#4a4f57">distractor halluc.</text>
          <text x={295} y={H - MB + 32} textAnchor="middle" fontSize={10} fill="#8a8f98">base / released</text>
          <text x={W1 / 2} y={16} textAnchor="middle" fontSize={12.5} fontWeight={600}>Earnings-22 hotword biasing</text>
          <g>
            <line x1={131} y1={yb(bars.relevant) + 4} x2={131} y2={yb(bars.no_ctx) - 4} stroke="#1c1e21" strokeWidth={1.4} markerEnd="none" />
            <text x={138} y={(yb(bars.relevant) + yb(bars.no_ctx)) / 2} fontSize={11.5} fontWeight={700}>+28.9 pp</text>
          </g>
        </svg>
        <svg className="timeline-svg" viewBox={`0 0 ${W2} ${H}`}>
          {[40, 60, 80, 100].map((t, i) => (
            <g key={i}>
              <line x1={56} x2={W2 - 20} y1={y2(t)} y2={y2(t)} stroke="#ecebe4" />
              <text x={50} y={y2(t) + 4} textAnchor="end" fontSize={10.5} fill="#6b7078">{t}</text>
            </g>
          ))}
          {ages.map((a, i) => (
            <text key={i} x={x2(a.age)} y={H - MB + 16} textAnchor="middle" fontSize={10.5} fill="#6b7078">{a.age}</text>
          ))}
          <text x={(56 + W2 - 20) / 2} y={H - 8} textAnchor="middle" fontSize={11.5} fontWeight={600} fill="#4a4f57">
            intervening audio between prefix and target (s)
          </text>
          <polygon points={[...ages.map(a => `${x2(a.age)},${y2(a.ctx)}`), ...[...ages].reverse().map(a => `${x2(a.age)},${y2(a.no_ctx)}`)].join(' ')}
            fill={COL.green} opacity={0.1} />
          <polyline points={polyline(ages.map(a => [x2(a.age), y2(a.ctx)]))} fill="none" stroke={COL.green} strokeWidth={2.2} />
          {ages.map((a, i) => <circle key={i} cx={x2(a.age)} cy={y2(a.ctx)} r={3.6} fill={COL.green} />)}
          <polyline points={polyline(ages.map(a => [x2(a.age), y2(a.no_ctx)]))} fill="none" stroke={COL.grey} strokeWidth={2.2} />
          {ages.map((a, i) => <rect key={i} x={x2(a.age) - 3.4} y={y2(a.no_ctx) - 3.4} width={6.8} height={6.8} fill={COL.grey} />)}
          <text x={W2 / 2} y={16} textAnchor="middle" fontSize={12.5} fontWeight={600}>biasing survives session length</text>
          <text x={x2(6)} y={y2((ages[2].ctx + ages[2].no_ctx) / 2) + 4} textAnchor="middle" fontSize={11.5}
            fontStyle="italic" fill={COL.green}>advantage ≈ +31 pp, flat</text>
          <g transform={`translate(64, ${H - MB - 14})`} fontSize={11}>
            <rect x={0} y={-8} width={11} height={11} rx={3} fill={COL.green} />
            <text x={16} y={2} fill="#4a4f57">CTX prefix present</text>
            <rect x={150} y={-8} width={11} height={11} rx={3} fill={COL.grey} />
            <text x={166} y={2} fill="#4a4f57">no prefix</text>
          </g>
        </svg>
      </div>
    </div>
  )
}
