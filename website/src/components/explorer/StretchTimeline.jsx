import React, { useRef, useState } from 'react'

export const CLS_COLOR = {
  hit: '#00694f', resume: '#b07c00', false: '#c23b22', dup: '#8a8f98',
}
export const EVT_COLOR = {
  turn_final: 'rgba(0,105,79,0.16)', continuation: 'rgba(176,124,0,0.15)', merged: 'rgba(120,125,135,0.12)',
}
export const EVT_EDGE = {
  turn_final: '#00694f', continuation: '#b07c00', merged: '#9aa0a8',
}

/**
 * Waveform + reference events + one fire-lane per arm + playhead.
 * props: stretch {dur, peaks, events}, arms: [{id,label,color,fires:[{t,cls,lat}]}],
 *        t (playhead s), onSeek(s)
 */
export default function StretchTimeline({ stretch, arms, t, onSeek }) {
  const [hover, setHover] = useState(null)
  const svgRef = useRef(null)
  const W = 1000, PADL = 8, PADR = 8
  const waveH = 84, waveY = 26
  const laneH = 30
  const lanesY = waveY + waveH + 26
  const H = lanesY + arms.length * laneH + 14
  const dur = stretch.dur
  const x = (s) => PADL + (s / dur) * (W - PADL - PADR)

  const seekFromClient = (clientX) => {
    const rect = svgRef.current.getBoundingClientRect()
    const frac = (clientX - rect.left) / rect.width
    onSeek(Math.max(0, Math.min(dur, ((frac * W) - PADL) / (W - PADL - PADR) * dur)))
  }

  const peaks = stretch.peaks
  const binW = (W - PADL - PADR) / peaks.length

  return (
    <div className="timeline-wrap" style={{ position: 'relative' }}>
      <svg ref={svgRef} className="timeline-svg" viewBox={`0 0 ${W} ${H}`}
        onClick={(e) => seekFromClient(e.clientX)} style={{ cursor: 'crosshair' }}>
        {/* time grid */}
        {Array.from({ length: Math.floor(dur / 5) + 1 }, (_, i) => i * 5).map(s => (
          <g key={s}>
            <line x1={x(s)} x2={x(s)} y1={waveY - 6} y2={H - 8} stroke="#eeede6" strokeWidth={1} />
            <text x={x(s)} y={waveY - 10} textAnchor="middle" fontSize={10} fill="#9aa0a8">{s}s</text>
          </g>
        ))}

        {/* reference events behind waveform */}
        {stretch.events.map((e, i) => (
          <g key={i}
            onMouseEnter={() => setHover({ x: x((e.b + e.e) / 2), y: waveY - 2, html: `<b>${e.cls.replace('_', '-')}</b> · ${e.b.toFixed(1)}–${e.e.toFixed(1)}s${e.gap != null ? ` · gap ${e.gap.toFixed(1)}s` : ''}<br/>${e.text.length > 90 ? e.text.slice(0, 90) + '…' : e.text}` })}
            onMouseLeave={() => setHover(null)}>
            <rect x={x(e.b)} y={waveY} width={Math.max(x(e.e) - x(e.b), 2)} height={waveH}
              fill={EVT_COLOR[e.cls]} stroke="none" />
            <rect x={x(e.b)} y={waveY + waveH + 3} width={Math.max(x(e.e) - x(e.b), 2)} height={5} rx={2}
              fill={EVT_EDGE[e.cls]} />
            {e.cls !== 'merged' && (
              <line x1={x(e.e)} x2={x(e.e)} y1={waveY} y2={waveY + waveH}
                stroke={EVT_EDGE[e.cls]} strokeWidth={1.6} strokeDasharray={e.cls === 'continuation' ? '4 3' : 'none'} />
            )}
          </g>
        ))}

        {/* waveform */}
        {peaks.map((p, i) => {
          const h = Math.max(p * (waveH - 8), 1.2)
          return <rect key={i} x={PADL + i * binW + 0.3} y={waveY + (waveH - h) / 2}
            width={Math.max(binW - 0.6, 0.7)} height={h} fill="#5b6470" opacity={0.55} rx={0.6} />
        })}

        {/* arm lanes */}
        {arms.map((arm, ai) => {
          const y = lanesY + ai * laneH
          return (
            <g key={arm.id}>
              <line x1={PADL} x2={W - PADR} y1={y + laneH / 2} y2={y + laneH / 2} stroke="#e7e5de" strokeWidth={1} />
              <rect x={PADL} y={y + 4} width={10} height={laneH - 12} rx={3} fill={arm.color} />
              <text x={PADL + 16} y={y + laneH / 2 + 4} fontSize={11} fill="#4a4f57" fontWeight={600}
                style={{ paintOrder: 'stroke', stroke: '#fff', strokeWidth: 3 }}>{arm.label}</text>
              {arm.fires.map((f, fi) => (
                <g key={fi}
                  onMouseEnter={() => setHover({
                    x: x(f.t), y: y - 4,
                    html: `<b>${arm.label}</b><br/>${f.cls === 'hit' ? 'hit (turn-final)' : f.cls === 'resume' ? 'fire at continuation boundary' : f.cls === 'false' ? 'false fire (mid-speech)' : 'duplicate / silence fire'} · t=${f.t.toFixed(1)}s${f.lat != null ? ` · latency ${f.lat >= 0 ? '+' : ''}${f.lat.toFixed(2)}s` : ''}`,
                  })}
                  onMouseLeave={() => setHover(null)}
                  onClick={(ev) => { ev.stopPropagation(); onSeek(Math.max(0, f.t - 2)) }}
                  style={{ cursor: 'pointer' }}>
                  <line x1={x(f.t)} x2={x(f.t)} y1={y + 5} y2={y + laneH - 5} stroke={CLS_COLOR[f.cls]} strokeWidth={2.4} />
                  {f.cls === 'hit'
                    ? <circle cx={x(f.t)} cy={y + laneH / 2} r={5.2} fill={CLS_COLOR.hit} stroke="#fff" strokeWidth={1.2} />
                    : f.cls === 'false'
                      ? <path d={`M${x(f.t) - 4.5},${y + laneH / 2 - 4.5} l9,9 M${x(f.t) - 4.5},${y + laneH / 2 + 4.5} l9,-9`}
                        stroke={CLS_COLOR.false} strokeWidth={2.6} strokeLinecap="round" />
                      : <rect x={x(f.t) - 4} y={y + laneH / 2 - 4} width={8} height={8}
                        transform={`rotate(45 ${x(f.t)} ${y + laneH / 2})`} fill={CLS_COLOR[f.cls]} stroke="#fff" strokeWidth={1} />}
                </g>
              ))}
            </g>
          )
        })}

        {/* playhead */}
        <line x1={x(Math.min(t, dur))} x2={x(Math.min(t, dur))} y1={waveY - 4} y2={H - 8}
          stroke="#1c1e21" strokeWidth={1.6} />
        <polygon points={`${x(Math.min(t, dur)) - 5},${waveY - 12} ${x(Math.min(t, dur)) + 5},${waveY - 12} ${x(Math.min(t, dur))},${waveY - 3}`}
          fill="#1c1e21" />
      </svg>
      {hover && (
        <div style={{
          position: 'absolute', left: `${(hover.x / W) * 100}%`, top: `${(hover.y / H) * 100}%`,
          transform: 'translate(-50%, -110%)', background: '#1c1e21', color: '#fff',
          padding: '7px 11px', borderRadius: 8, fontSize: 12, pointerEvents: 'none',
          maxWidth: 340, zIndex: 6, lineHeight: 1.45,
        }} dangerouslySetInnerHTML={{ __html: hover.html }} />
      )}
    </div>
  )
}
