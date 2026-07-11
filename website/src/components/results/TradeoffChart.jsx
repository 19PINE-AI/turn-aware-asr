import React, { useState } from 'react'
import { COL, linScale, logScale, XAxis, Star, polyline } from './chart.jsx'

const W = 720, H = 440, ML = 64, MR = 20, MT = 18, MB = 52

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

          <g onMouseEnter={() => setHover({ px: x(data.unified.lat), py: y(ffv(data.unified.ff)), name: data.unified.name, ...data.unified })}
            onMouseLeave={() => setHover(null)}>
            <Dot px={x(data.unified.lat)} py={y(ffv(data.unified.ff))} color={COL.sky} shape="D" size={8}
              info={{ name: data.unified.name, lat: data.unified.lat, ff: data.unified.ff, recall: data.unified.recall }} />
          </g>
          <g onMouseEnter={() => setHover({ px: x(data.ours.lat), py: y(ffv(data.ours.ff)), name: data.ours.name, ...data.ours })}
            onMouseLeave={() => setHover(null)} style={{ cursor: 'pointer' }}>
            <Star cx={x(data.ours.lat)} cy={y(ffv(data.ours.ff))} r={14} />
          </g>
          <text x={x(0.39)} y={y(0.32) - 18} textAnchor="middle" fontSize={12.5} fontWeight={700} fill={COL.green}>
            causal endpointer · R=0.97
          </text>
          <text x={x(0.39) + 14} y={y(0.97) + 4} fontSize={11.5} fontWeight={600} fill={COL.sky}>
            unified release · R=0.95
          </text>
          <text x={x(data.external[1].lat) + 12} y={y(data.external[1].ff) + 4} fontSize={11} fill={COL.brown}>Kyutai VAD</text>
          <text x={x(data.external[3].lat) + 12} y={y(data.external[3].ff) + 4} fontSize={11} fill={COL.brown}>LiveKit (oracle)</text>
          <text x={x(data.external[2].lat) + 12} y={y(data.external[2].ff) + 4} fontSize={11} fill={COL.brown}>Smart Turn v3</text>
          <text x={x(data.external[0].lat) + 12} y={y(Math.max(data.external[0].ff, 0.05)) + 4} fontSize={11} fill={COL.brown}>Parakeet-EOU (R=0.17)</text>

          {/* legend */}
          <g transform={`translate(${W - 250}, ${MT + 4})`} fontSize={11.5}>
            {[[COL.amber, 'RMS + timeout family'], [COL.purple, 'Silero VAD + timeout family'],
              [COL.grey, 'prior supervision compositions'], [COL.brown, 'external turn-aware (as-shipped)'],
              [COL.green, 'causal endpointer (ours)'], [COL.sky, 'released unified model']].map(([c, t], i) => (
              <g key={i} transform={`translate(0, ${i * 18})`}>
                <rect x={0} y={-8} width={11} height={11} rx={3} fill={c} />
                <text x={17} y={2} fill="#4a4f57">{t}</text>
              </g>
            ))}
          </g>
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
      <p className="note">
        Points with open interiors in the paper (recall &lt; 0.5) are unusable operating points.
        “0*” = zero false fires plotted at 0.05. External systems were run as-shipped through the
        identical benchmark, sweeping their public thresholds — the first measurement of the open
        turn-aware class on a common protocol.
      </p>
    </div>
  )
}
