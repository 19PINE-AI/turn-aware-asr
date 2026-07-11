import React, { useMemo, useState } from 'react'
import { useJson } from '../../lib/useJson.js'
import { useAudio, fmtTime } from '../../lib/useAudio.js'

const ARMS = [
  { id: 'released', set: 'big', color: '#00694f' },
  { id: 'dictation_matched', set: 'small', color: '#0d5c96' },
  { id: 'conversational', set: 'small', color: '#c23b22' },
]

export default function DictationExplorer() {
  const { data, error } = useJson('data/dictation.json')
  const [armId, setArmId] = useState('released')
  const [onlyBad, setOnlyBad] = useState(false)
  const [idx, setIdx] = useState(0)

  const armDef = ARMS.find(a => a.id === armId)
  const set = data?.sets?.[armDef.set]
  const arm = set?.arms?.[armId]

  const filtered = useMemo(() => {
    if (!set) return []
    let ids = set.items.map(it => it.id)
    if (onlyBad && arm) ids = ids.filter(id => (arm.items[String(id)]?.premature || 0) > 0 || arm.items[String(id)]?.final_recall === 0)
    return ids
  }, [set, arm, onlyBad])

  const itemId = filtered[Math.min(idx, Math.max(filtered.length - 1, 0))]
  const item = set?.items?.find(it => it.id === itemId)
  const res = arm?.items?.[String(itemId)]
  const audio = useAudio(item?.audio)

  if (error) return <div className="loading">Failed to load dictation data: {String(error)}</div>
  if (!data) return <div className="loading">Loading dictation probes…</div>
  if (!item || !res) return <div className="loading">No items match this filter.</div>

  const W = 1000, PADL = 8, PADR = 8, waveY = 24, waveH = 70
  const H = waveY + waveH + 58
  const x = (s) => PADL + (s / item.dur) * (W - PADL - PADR)
  const binW = (W - PADL - PADR) / item.peaks.length
  const isPremature = (t) => t <= item.last_speech_end + 0.01

  const groupsLabel = item.groups.join('-')
  const s = arm.summary

  return (
    <div>
      <div className="strip">
        <div className="seg">
          {ARMS.map(a => (
            <button key={a.id} className={armId === a.id ? 'on' : ''}
              onClick={() => { setArmId(a.id); setIdx(0); audio.stop() }}>
              {data.sets[a.set].arms[a.id].label}
            </button>
          ))}
        </div>
        <label style={{ display: 'flex', gap: 6, alignItems: 'center', fontWeight: 500 }}>
          <input type="checkbox" checked={onlyBad} onChange={e => { setOnlyBad(e.target.checked); setIdx(0) }} />
          only failures (premature fire or missed final)
        </label>
      </div>
      <p className="note" style={{ marginTop: 8 }}>
        Ten-digit phone numbers ({groupsLabel} groups from held-out Free Spoken Digit Dataset
        speakers) with inter-group pauses, replayed through the exact streaming stack.
        A fire before the last digit is <b>premature</b> — the agent just interrupted the caller
        mid-number. Set-level: <b>{s.premature_per_seq}</b> premature fires/sequence ·
        final recall <b>{s.final_recall}</b> · digit accuracy <b>{(s.digit_acc_mean * 100).toFixed(1)}%</b>
        {armDef.set === 'small' ? ' (original 50-item probe)' : ' (enlarged 250-item post-development probe)'}.
      </p>

      <div className="strip">
        <label>Item</label>
        <select value={idx} onChange={e => { setIdx(+e.target.value); audio.stop() }}>
          {filtered.map((id, i) => {
            const r = arm.items[String(id)]
            return <option key={id} value={i}>
              #{id} · {set.items.find(it => it.id === id).digits.join('')}{r.premature ? ` · ${r.premature} premature` : ''}{r.final_recall === 0 ? ' · missed final' : ''}
            </option>
          })}
        </select>
        <div className="pager" style={{ margin: 0 }}>
          <button disabled={idx === 0} onClick={() => { setIdx(idx - 1); audio.stop() }}>← prev</button>
          <button disabled={idx >= filtered.length - 1} onClick={() => { setIdx(idx + 1); audio.stop() }}>next →</button>
        </div>
        <span style={{ fontSize: 12.5, color: 'var(--ink-faint)' }}>{filtered.length} items{onlyBad ? ' (filtered)' : ''}</span>
      </div>

      <div className="player">
        <button className="playbtn" onClick={audio.toggle}>{audio.playing ? '❚❚' : '▶'}</button>
        <span className="time">{fmtTime(audio.t)} / {fmtTime(item.dur)}</span>
      </div>

      <div className="timeline-wrap">
        <svg className="timeline-svg" viewBox={`0 0 ${W} ${H}`}
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect()
            audio.seek(((e.clientX - rect.left) / rect.width * W - PADL) / (W - PADL - PADR) * item.dur)
          }} style={{ cursor: 'crosshair' }}>
          {/* pauses */}
          {item.pauses.map(([a, b], i) => (
            <g key={i}>
              <rect x={x(a)} y={waveY} width={x(b) - x(a)} height={waveH} fill="rgba(176,124,0,0.13)" />
              <text x={(x(a) + x(b)) / 2} y={waveY + 14} textAnchor="middle" fontSize={10.5} fill="#8a6b10">
                pause {(b - a).toFixed(1)}s
              </text>
            </g>
          ))}
          {/* waveform */}
          {item.peaks.map((p, i) => {
            const h = Math.max(p * (waveH - 6), 1.2)
            return <rect key={i} x={PADL + i * binW + 0.3} y={waveY + (waveH - h) / 2}
              width={Math.max(binW - 0.6, 0.7)} height={h} fill="#5b6470" opacity={0.55} rx={0.6} />
          })}
          {/* end of speech */}
          <line x1={x(item.last_speech_end)} x2={x(item.last_speech_end)} y1={waveY - 4} y2={waveY + waveH + 20}
            stroke="#00694f" strokeWidth={1.6} strokeDasharray="5 4" />
          <text x={x(item.last_speech_end) + 5} y={waveY + waveH + 16} fontSize={10.5} fill="#00694f">last digit ends</text>
          {/* fires */}
          {res.fires.map((t, i) => {
            const bad = isPremature(t)
            const prev = i > 0 ? res.fires[i - 1] : -Infinity
            const showLabel = x(t) - x(prev) > 80
            return (
              <g key={i} onClick={(ev) => { ev.stopPropagation(); audio.seek(Math.max(0, t - 1.5)) }} style={{ cursor: 'pointer' }}>
                <line x1={x(t)} x2={x(t)} y1={waveY} y2={waveY + waveH} stroke={bad ? '#c23b22' : '#00694f'} strokeWidth={2.6} />
                <circle cx={x(t)} cy={waveY - 8} r={7} fill={bad ? '#c23b22' : '#00694f'} />
                <text x={x(t)} y={waveY - 4.5} textAnchor="middle" fontSize={9} fill="#fff" fontWeight={700}>{bad ? '✗' : '✓'}</text>
                {showLabel && (
                  <text x={x(t)} y={waveY + waveH + 32} textAnchor="middle" fontSize={10.5}
                    fill={bad ? '#c23b22' : '#00694f'} fontWeight={600}>
                    {bad ? 'premature fire' : `END +${(t - item.last_speech_end).toFixed(2)}s`}
                  </text>
                )}
              </g>
            )
          })}
          {res.fires.length === 0 && (
            <text x={W - 12} y={waveY + waveH / 2} textAnchor="end" fontSize={12} fill="#c23b22" fontWeight={600}>
              never fired — user left waiting
            </text>
          )}
          {/* playhead */}
          <line x1={x(Math.min(audio.t, item.dur))} x2={x(Math.min(audio.t, item.dur))} y1={waveY - 4} y2={waveY + waveH + 4}
            stroke="#1c1e21" strokeWidth={1.5} />
        </svg>
      </div>

      <div className="two-col" style={{ marginTop: 14 }}>
        <div className="hyp"><span className="lab">reference ({groupsLabel})</span>{item.text}</div>
        <div className="hyp">
          <span className="lab">model transcript · digit accuracy {(res.digit_acc * 100).toFixed(0)}%{res.final_latency != null ? ` · final latency +${res.final_latency.toFixed(2)}s` : ''}</span>
          {res.hyp || <i>(empty)</i>}
        </div>
      </div>
    </div>
  )
}
