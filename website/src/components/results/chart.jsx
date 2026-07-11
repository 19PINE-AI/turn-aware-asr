import React from 'react'

export const COL = {
  green: '#00694f', red: '#c23b22', blue: '#0d5c96', amber: '#b07c00',
  purple: '#8a4f76', grey: '#8a8f98', sky: '#3f8fc4', brown: '#8a5a2b',
  ink: '#1c1e21', faint: '#8a8f98',
}

export function linScale(d0, d1, r0, r1) {
  const f = (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0)
  f.ticks = (n = 5) => {
    const step = (d1 - d0) / n
    return Array.from({ length: n + 1 }, (_, i) => d0 + i * step)
  }
  return f
}

export function logScale(d0, d1, r0, r1) {
  const l0 = Math.log10(d0), l1 = Math.log10(d1)
  const f = (v) => r0 + ((Math.log10(v) - l0) / (l1 - l0)) * (r1 - r0)
  return f
}

export function XAxis({ x, y, ticks, fmt = (v) => v, label, x0, x1 }) {
  return (
    <g>
      <line x1={x0} x2={x1} y1={y} y2={y} stroke="#c9c6bd" strokeWidth={1} />
      {ticks.map((t, i) => (
        <g key={i}>
          <line x1={x(t)} x2={x(t)} y1={y} y2={y + 4} stroke="#c9c6bd" />
          <text x={x(t)} y={y + 17} textAnchor="middle" fontSize={11} fill="#6b7078">{fmt(t)}</text>
        </g>
      ))}
      {label && <text x={(x0 + x1) / 2} y={y + 36} textAnchor="middle" fontSize={12} fill="#4a4f57" fontWeight={600}>{label}</text>}
    </g>
  )
}

export function YAxis({ y, x, ticks, fmt = (v) => v, label, y0, y1, grid, gx1 }) {
  return (
    <g>
      {ticks.map((t, i) => (
        <g key={i}>
          {grid && <line x1={x} x2={gx1} y1={y(t)} y2={y(t)} stroke="#ecebe4" strokeWidth={1} />}
          <text x={x - 7} y={y(t) + 4} textAnchor="end" fontSize={11} fill="#6b7078">{fmt(t)}</text>
        </g>
      ))}
      {label && (
        <text x={x - 40} y={(y(ticks[0]) + y(ticks[ticks.length - 1])) / 2} textAnchor="middle"
          fontSize={12} fill="#4a4f57" fontWeight={600}
          transform={`rotate(-90 ${x - 40} ${(y(ticks[0]) + y(ticks[ticks.length - 1])) / 2})`}>{label}</text>
      )}
    </g>
  )
}

export function Star({ cx, cy, r = 11, fill = COL.green }) {
  const pts = []
  for (let i = 0; i < 10; i++) {
    const rr = i % 2 === 0 ? r : r * 0.44
    const a = -Math.PI / 2 + (i * Math.PI) / 5
    pts.push(`${cx + rr * Math.cos(a)},${cy + rr * Math.sin(a)}`)
  }
  return <polygon points={pts.join(' ')} fill={fill} stroke="#083f30" strokeWidth={0.9} />
}

export function polyline(pts) {
  return pts.map(p => p.join(',')).join(' ')
}
