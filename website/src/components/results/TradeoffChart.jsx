import React, { useState } from 'react'
import { COL, linScale, logScale, XAxis, Star, polyline } from './chart.jsx'

const W = 720, H = 420, ML = 64, MR = 20, MT = 16, MB = 52
const INK = '#3d4148'

export default function TradeoffChart({ data }) {
  const [hover, setHover] = useState(null)
  const x = linScale(0.25, 1.6, ML, W - MR)
  const y = logScale(0.04, 35, H - MB, MT)
  const yTicks = [0.05, 0.3, 1, 3, 10, 30]
  const ffv = (ff) => Math.max(ff, 0.05)

  const Dot = ({ px, py, color, shape = 'o', size = 6, info }) => {
    const common = {
      onMouseEnter: () => setHover(info ? { px, py, ...info } : null),
      onMouseLeave: () => setHover(null),
      style: { cursor: info ? 'pointer' : 'default' },
    }
    if (shape === 'D') return <rect x={px - size} y={py - size} width={size * 2} height={size * 2}
      transform={`rotate(45 ${px} ${py})`} fill={color} stroke="#fff" strokeWidth={1} {...common} />
    if (shape === 'X') return (
      <g {...common}>
        <circle cx={px} cy={py} r={size + 4} fill="transparent" />
        <path d={`M${px - size},${py - size} L${px + size},${py + size} M${px - size},${py + size} L${px + size},${py - size}`}
          stroke={color} strokeWidth={3} strokeLinecap="round" />
      </g>
    )
    if (shape === '^') return <polygon points={`${px},${py - size - 1} ${px + size + 1},${py + size} ${px - size - 1},${py + size}`}
      fill={color} stroke="#fff" strokeWidth={1} {...common} />
    return <circle cx={px} cy={py} r={size} fill={color} stroke="#fff" strokeWidth={1} {...common} />
  }

  const fam = (pts, color, shape, name) => {
    const usable = pts.filter(p => p.recall >= 0.5)
    return (
      <g key={name}>
        <polyline points={polyline(usable.map(p => [x(p.lat), y(ffv(p.ff))]))}
          fill="none" stroke={color} strokeWidth={2} opacity={0.85} />
        {pts.map((p, i) => (
          <Dot key={i} px={x(p.lat)} py={y(ffv(p.ff))} color={color} shape={shape}
            info={{ name: `${name}, X=${p.X}s`, lat: p.lat, ff: p.ff, recall: p.recall }} />
        ))}
      </g>
    )
  }

  // direct labels, each positioned to clear its known neighbours
  const label = (px, py, txt, { anchor = 'start', dx = 0, dy = 0, bold = false } = {}) => (
    <text x={px + dx} y={py + dy} textAnchor={anchor} fontSize={bold ? 12.5 : 11}
      fontWeight={bold ? 700 : 500} fill={INK}
      style={{ paintOrder: 'stroke', stroke: '#fff', strokeWidth: 3.5 }}>{txt}</text>
  )
  const ext = Object.fromEntries(data.external.map(p => [p.name, p]))
  const pri = Object.fromEntries(data.priors.map(p => [p.name.split(' ')[0], p]))
  const P = (p) => [x(p.lat), y(ffv(p.ff))]

  const legend = [
    [COL.green, 'causal endpointer (ours)'],
    [COL.sky, 'released unified model'],
    [COL.grey, 'prior supervision compositions'],
    [COL.amber, 'RMS + timeout family'],
    [COL.purple, 'Silero VAD + timeout family'],
    [COL.brown, 'external turn-aware (as-shipped)'],
  ]

  return (
    <div className="card">
      <h3>The causal model escapes the timeout curve</h3>
      <p className="sub">
        Median end-of-turn latency vs. false fires per speech-minute (log scale), on identical
        audio, detector, and scoring. A silence timeout can only slide <i>along</i> its curve; the
        causally supervised model sits strictly inside the whole family because it reads
        completeness. Hover any point for details.
      </p>
      <div style={{ position: 'relative' }}>
        <svg className="timeline-svg" viewBox={`0 0 ${W} ${H}`}>
          {yTicks.map((t, i) => (
            <g key={i}>
              <line x1={ML} x2={W - MR} y1={y(t)} y2={y(t)} stroke="#ecebe4" />
              <text x={ML - 7} y={y(t) + 4} textAnchor="end" fontSize={11} fill="#6b7078">
                {t === 0.05 ? '0*' : t}
              </text>
            </g>
          ))}
          <text x={20} y={(y(0.05) + y(30)) / 2} textAnchor="middle" fontSize={12} fontWeight={600} fill="#4a4f57"
            transform={`rotate(-90 20 ${(y(0.05) + y(30)) / 2})`}>false fires / speech-minute</text>
          <XAxis x={x} y={H - MB} x0={ML} x1={W - MR} ticks={[0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]}
            fmt={v => v.toFixed(1)} label="median end-of-turn latency P50 (s)" />

          {fam(data.rms, COL.amber, 'o', 'RMS + timeout')}
          {fam(data.silero, COL.purple, '^', 'Silero VAD + timeout')}

          {data.priors.map((p, i) => (
            <Dot key={i} px={x(p.lat)} py={y(ffv(p.ff))} color={COL.grey} shape="D"
              info={{ name: p.name, lat: p.lat, ff: p.ff, recall: p.recall }} />
          ))}
          {data.external.map((p, i) => (
            <Dot key={i} px={x(p.lat)} py={y(ffv(p.ff))} color={COL.brown} shape="X" size={7}
              info={{ name: `${p.name} (as-shipped)`, lat: p.lat, ff: p.ff, recall: p.recall }} />
          ))}

          <Dot px={x(data.unified.lat)} py={y(ffv(data.unified.ff))} color={COL.sky} shape="D" size={8}
            info={{ name: data.unified.name, lat: data.unified.lat, ff: data.unified.ff, recall: data.unified.recall }} />
          <g onMouseEnter={() => setHover({ px: x(data.ours.lat), py: y(ffv(data.ours.ff)), name: data.ours.name, ...data.ours })}
            onMouseLeave={() => setHover(null)} style={{ cursor: 'pointer' }}>
            <Star cx={x(data.ours.lat)} cy={y(ffv(data.ours.ff))} r={14} />
          </g>

          {/* direct labels — offsets chosen to clear coincident neighbours */}
          {label(...P(data.ours), 'causal endpointer · R=0.97', { anchor: 'start', dx: -28, dy: -22, bold: true })}
          {label(...P(data.unified), 'unified release · R=0.95', { anchor: 'start', dx: 14, dy: 4, bold: true })}
          {/* Kyutai sits on the Silero X=0.5 point → label to the left */}
          {label(...P(ext['Kyutai VAD']), 'Kyutai VAD', { anchor: 'end', dx: -13, dy: 4 })}
          {/* LiveKit coincides with RMS X=0.5 → label up-right, clear of both */}
          {label(...P(ext['LiveKit (oracle)']), 'LiveKit (oracle)', { anchor: 'start', dx: 12, dy: -10 })}
          {/* Smart Turn abuts the mixed-pools diamond → label left, prior label right */}
          {label(...P(ext['Smart Turn v3']), 'Smart Turn v3', { anchor: 'end', dx: -13, dy: 4 })}
          {label(...P(pri['mixed-pools']), 'mixed-pools + confirm h=1', { anchor: 'start', dx: 12, dy: 4 })}
          {/* Parakeet shares its y with RMS X=2.0 and sits under the causal star → label below */}
          {label(...P(ext['Parakeet-EOU']), 'Parakeet-EOU · R=0.17', { anchor: 'middle', dy: 24 })}
          {label(...P(pri['opposed-pools']), 'opposed-pools + confirm h=1', { anchor: 'middle', dy: -14 })}
        </svg>
        {hover && (
          <div style={{
            position: 'absolute', left: `${(hover.px / W) * 100}%`, top: `${(hover.py / H) * 100}%`,
            transform: 'translate(-50%, -130%)', background: '#1c1e21', color: '#fff',
            padding: '7px 11px', borderRadius: 8, fontSize: 12, pointerEvents: 'none',
            whiteSpace: 'nowrap', boxShadow: 'var(--shadow)', zIndex: 5,
          }}>
            <b>{hover.name}</b><br />
            recall {hover.recall.toFixed(2)} · P50 {hover.lat.toFixed(2)} s · {hover.ff.toFixed(2)} false/min
          </div>
        )}
      </div>
      <div className="legend">
        {legend.map(([c, t], i) => (
          <span key={i}><i style={{ background: c }} /> {t}</span>
        ))}
      </div>
      <p className="note">
        Timeout points with recall &lt; 0.5 are unusable operating points (X = 1.5–2 s misses most
        turns). “0*” = zero false fires plotted at 0.05. External systems were run as-shipped
        through the identical benchmark, sweeping their public thresholds — the first measurement
        of the open turn-aware class on a common protocol.
      </p>
    </div>
  )
}
