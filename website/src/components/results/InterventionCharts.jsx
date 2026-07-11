import React from 'react'
import { COL, linScale } from './chart.jsx'

/* ---------- silence-append dumbbell ---------- */
export function SilAppend({ data }) {
  const W = 460, H = 200, ML = 150, MR = 70
  const x = linScale(0, 1.0, ML, W - MR)
  return (
    <div className="card">
      <h3>Diagnosis by intervention: append 1 s of silence</h3>
      <p className="sub">
        One variable changed on the <i>evaluation</i>: 1.0 s of silence appended to each clip of
        the legacy offline benchmark. The “broken” opposed-pools model’s fire recall jumps from
        0.10 to 1.00 — the capability axis four compositions of data engineering had been fighting
        was an artifact of where the clips ended.
      </p>
      <svg className="timeline-svg" viewBox={`0 0 ${W} ${H}`}>
        {[0, 0.25, 0.5, 0.75, 1].map((t, i) => (
          <g key={i}>
            <line x1={x(t)} x2={x(t)} y1={28} y2={H - 40} stroke="#ecebe4" />
            <text x={x(t)} y={H - 24} textAnchor="middle" fontSize={10.5} fill="#6b7078">{t}</text>
          </g>
        ))}
        <text x={(ML + W - MR) / 2} y={H - 6} textAnchor="middle" fontSize={11.5} fontWeight={600} fill="#4a4f57">
          fire recall on complete utterances
        </text>
        {data.map((d, i) => {
          const y = 46 + i * 42
          return (
            <g key={i}>
              <text x={ML - 10} y={y + 4} textAnchor="end" fontSize={12} fill="#1c1e21">{d.name}</text>
              <line x1={x(d.orig)} x2={x(d.plus1s)} y1={y} y2={y} stroke="#c9c6bd" strokeWidth={2.5} />
              <circle cx={x(d.orig)} cy={y} r={7} fill={COL.grey} stroke="#fff" strokeWidth={1.5} />
              <circle cx={x(d.plus1s)} cy={y} r={7} fill={COL.green} stroke="#fff" strokeWidth={1.5} />
              <text x={x(1) + 12} y={y + 4} fontSize={11.5} fontWeight={700}
                fill={d.plus1s - d.orig > 0.5 ? COL.green : '#6b7078'}>
                {d.orig.toFixed(2)} → {d.plus1s.toFixed(2)}
              </text>
            </g>
          )
        })}
        <g transform={`translate(${ML}, 16)`} fontSize={11}>
          <circle cx={5} cy={0} r={6} fill={COL.grey} /><text x={16} y={4} fill="#4a4f57">original clips</text>
          <circle cx={125} cy={0} r={6} fill={COL.green} /><text x={136} y={4} fill="#4a4f57">+ 1.0 s silence</text>
        </g>
      </svg>
    </div>
  )
}

/* ---------- ranking inversion slopegraph ---------- */
export function Inversion({ data }) {
  const W = 460, H = 250
  const colors = [COL.red, COL.amber, COL.blue]
  const offRank = [...data].sort((a, b) => b.offline_recall - a.offline_recall)
  const stRank = [...data].sort((a, b) => a.stream_ff - b.stream_ff)
  const yFor = (rank) => 70 + rank * 55
  return (
    <div className="card">
      <h3>Evaluation is the port of entry: the ranking inverts</h3>
      <p className="sub">
        The legacy offline benchmark didn’t just mis-score the models — it steered four
        compositions of data engineering toward a frontier that does not exist. Under the
        deployment-matched replay, the offline champion is the worst live model (its offline
        strength <i>is</i> a premature-fire pathology) and the model written off at 10% offline
        recall is the best prior endpointer.
      </p>
      <svg className="timeline-svg" viewBox={`0 0 ${W} ${H}`}>
        <text x={115} y={30} textAnchor="middle" fontSize={11.5} fontWeight={600} fill="#4a4f57">offline benchmark</text>
        <text x={115} y={44} textAnchor="middle" fontSize={10} fill="#8a8f98">(fire rate on clipped audio)</text>
        <text x={345} y={30} textAnchor="middle" fontSize={11.5} fontWeight={600} fill="#4a4f57">deployment-matched replay</text>
        <text x={345} y={44} textAnchor="middle" fontSize={10} fill="#8a8f98">(false fires / min, gate on)</text>
        {data.map((d, i) => {
          const y0 = yFor(offRank.indexOf(d)), y1 = yFor(stRank.indexOf(d))
          const c = colors[i]
          return (
            <g key={i}>
              <line x1={170} x2={290} y1={y0} y2={y1} stroke={c} strokeWidth={2.5} opacity={0.85} />
              <circle cx={170} cy={y0} r={5.5} fill={c} />
              <circle cx={290} cy={y1} r={5.5} fill={c} />
              <text x={160} y={y0 + 4} textAnchor="end" fontSize={11.5} fontWeight={600} fill={c}>
                {d.name} · {d.offline_recall}%
              </text>
              <text x={300} y={y1 + 4} fontSize={11.5} fontWeight={600} fill={c}>
                {d.stream_ff} false/min
              </text>
            </g>
          )
        })}
        <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10.5} fontStyle="italic" fill="#6b7078">
          any team selecting a turn-aware checkpoint on clip-cut offline evals should expect the same inversion
        </text>
      </svg>
    </div>
  )
}

/* ---------- silence incompetence scatter ---------- */
export function SilenceScatter({ data }) {
  const W = 460, H = 260, ML = 50, MR = 16, MT = 14, MB = 46
  const xmax = Math.max(...data.map(d => d.sil)) * 1.05
  const ymax = Math.max(...data.map(d => d.spam)) * 1.08
  const x = linScale(0, xmax, ML, W - MR)
  const y = linScale(0, ymax, H - MB, MT)
  // least squares
  const n = data.length
  const mx = data.reduce((s, d) => s + d.sil, 0) / n
  const my = data.reduce((s, d) => s + d.spam, 0) / n
  const b = data.reduce((s, d) => s + (d.sil - mx) * (d.spam - my), 0) /
            data.reduce((s, d) => s + (d.sil - mx) ** 2, 0)
  const a = my - b * mx
  const r = data.reduce((s, d) => s + (d.sil - mx) * (d.spam - my), 0) /
    Math.sqrt(data.reduce((s, d) => s + (d.sil - mx) ** 2, 0) * data.reduce((s, d) => s + (d.spam - my) ** 2, 0))
  return (
    <div className="card">
      <h3>Why every comparison runs behind an energy gate</h3>
      <p className="sub">
        Models trained only on speech-initial examples hallucinate markers on silence at ~1.9
        fires per second: the ungated mixed-pools model’s spurious fires track each stretch’s
        silence content almost perfectly. Ungated recall is uninterpretable — a random 1.9 Hz
        spammer scores 0.96 against the tolerance window.
      </p>
      <svg className="timeline-svg" viewBox={`0 0 ${W} ${H}`}>
        {[0, 20, 40, 60].filter(t => t <= ymax).map((t, i) => (
          <g key={i}>
            <line x1={ML} x2={W - MR} y1={y(t)} y2={y(t)} stroke="#ecebe4" />
            <text x={ML - 6} y={y(t) + 4} textAnchor="end" fontSize={10.5} fill="#6b7078">{t}</text>
          </g>
        ))}
        {[0, 10, 20, 30].filter(t => t <= xmax).map((t, i) => (
          <text key={i} x={x(t)} y={H - MB + 16} textAnchor="middle" fontSize={10.5} fill="#6b7078">{t}</text>
        ))}
        <line x1={ML} x2={W - MR} y1={H - MB} y2={H - MB} stroke="#c9c6bd" />
        <text x={(ML + W - MR) / 2} y={H - 8} textAnchor="middle" fontSize={11.5} fontWeight={600} fill="#4a4f57">
          silence in stretch (s)
        </text>
        <text x={16} y={(y(0) + y(ymax)) / 2} textAnchor="middle" fontSize={11.5} fontWeight={600} fill="#4a4f57"
          transform={`rotate(-90 16 ${(y(0) + y(ymax)) / 2})`}>spurious fires (ungated)</text>
        <line x1={x(0)} x2={x(xmax)} y1={y(a)} y2={y(a + b * xmax)} stroke={COL.red} strokeWidth={1.6} strokeDasharray="6 4" />
        {data.map((d, i) => <circle key={i} cx={x(d.sil)} cy={y(d.spam)} r={4.5} fill={COL.blue} opacity={0.75} />)}
        <text x={W - MR - 6} y={MT + 12} textAnchor="end" fontSize={12} fontWeight={700} fill={COL.red}>r = {r.toFixed(3)}</text>
      </svg>
    </div>
  )
}

/* ---------- synthetic toy study ---------- */
export function ToyChart({ data }) {
  const fs = Object.keys(data).map(parseFloat).sort((a, b) => a - b)
  const W = 460, H = 250, ML = 52, MR = 56, MT = 16, MB = 46
  const x = linScale(0, 1, ML, W - MR)
  const yA = linScale(-0.2, 1, H - MB, MT)     // anticorrelation
  const flipsMax = Math.max(...fs.map(f => data[f.toFixed(2)].mean_mode_flips)) || 1
  const yF = linScale(0, flipsMax * 1.1, H - MB, MT)
  const key = (f) => f.toFixed(2)
  return (
    <div className="card">
      <h3>The mechanism, isolated outside speech</h3>
      <p className="sub">
        A ~30k-parameter transformer on a synthetic stream where a tunable fraction <i>f</i> of
        labels is made clairvoyant by construction. As <i>f</i> grows, fire- and hold-class
        accuracy become anti-correlated (a phantom frontier — one model swinging, sampled at
        different steps), and true bistable mode-flipping onsets only at high <i>f</i>. Nothing
        about the failure is specific to speech, acoustics, or pretraining.
      </p>
      <svg className="timeline-svg" viewBox={`0 0 ${W} ${H}`}>
        {[0, 0.5, 1].map((t, i) => (
          <line key={i} x1={ML} x2={W - MR} y1={yA(t)} y2={yA(t)} stroke="#ecebe4" />
        ))}
        <line x1={ML} x2={W - MR} y1={H - MB} y2={H - MB} stroke="#c9c6bd" />
        {fs.map((f, i) => (
          <text key={i} x={x(f)} y={H - MB + 16} textAnchor="middle" fontSize={10.5} fill="#6b7078">{f}</text>
        ))}
        <text x={(ML + W - MR) / 2} y={H - 8} textAnchor="middle" fontSize={11.5} fontWeight={600} fill="#4a4f57">
          clairvoyant label fraction f
        </text>
        {[-0, 0.5, 1].map((t, i) => (
          <text key={i} x={ML - 6} y={yA(t) + 4} textAnchor="end" fontSize={10.5} fill={COL.purple}>{t}</text>
        ))}
        <text x={W - MR + 8} y={yF(0) + 4} fontSize={10.5} fill={COL.red}>0</text>
        <text x={W - MR + 8} y={yF(flipsMax) + 4} fontSize={10.5} fill={COL.red}>{flipsMax.toFixed(0)}</text>
        <polyline points={fs.map(f => `${x(f)},${yA(data[key(f)].mean_fire_hold_anticorr)}`).join(' ')}
          fill="none" stroke={COL.purple} strokeWidth={2.2} />
        {fs.map((f, i) => <circle key={i} cx={x(f)} cy={yA(data[key(f)].mean_fire_hold_anticorr)} r={3.4} fill={COL.purple} />)}
        <polyline points={fs.map(f => `${x(f)},${yF(data[key(f)].mean_mode_flips)}`).join(' ')}
          fill="none" stroke={COL.red} strokeWidth={2.2} strokeDasharray="6 4" />
        {fs.map((f, i) => <circle key={i} cx={x(f)} cy={yF(data[key(f)].mean_mode_flips)} r={3.4} fill={COL.red} />)}
        <g transform={`translate(${ML + 8}, ${MT + 6})`} fontSize={11}>
          <rect x={0} y={-7} width={11} height={11} rx={3} fill={COL.purple} />
          <text x={16} y={2} fill="#4a4f57">fire↔hold anti-correlation (3 seeds)</text>
          <rect x={0} y={12} width={11} height={11} rx={3} fill={COL.red} />
          <text x={16} y={21} fill="#4a4f57">bistable mode flips per run</text>
        </g>
      </svg>
    </div>
  )
}
