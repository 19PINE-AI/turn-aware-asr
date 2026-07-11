import React, { useMemo, useState } from 'react'
import { useJson } from '../../lib/useJson.js'
import { useAudio, fmtTime } from '../../lib/useAudio.js'
import StretchTimeline, { CLS_COLOR } from './StretchTimeline.jsx'

const ARM_PALETTE = ['#00694f', '#0d5c96', '#b07c00', '#8a4f76', '#c23b22', '#3f8fc4', '#8a5a2b', '#546a3a']

const VIEWS = {
  dev25: { set: 'ami100', n: 25, label: 'Development set (25 stretches)', exclude: ['unified_100'] },
  heldout100: { set: 'ami100', n: 100, label: 'Held-out set (100 stretches)', only: ['unified_100'] },
  fresh50: { set: 'fresh50', n: 50, label: 'Fresh confirmation set (50 stretches)' },
}
const GROUP_LABEL = { ours: 'This work', prior: 'Prior compositions', timeout: 'Silence timeouts', external: 'External systems' }

export default function ReplayExplorer() {
  const { data, error } = useJson('data/replay.json')
  const [viewId, setViewId] = useState('dev25')
  const [idx, setIdx] = useState(0)
  const [sel, setSel] = useState(['causal', 'timeout_rms_1.0'])

  const view = VIEWS[viewId]
  const set = data?.sets?.[view.set]

  const armIds = useMemo(() => {
    if (!set) return []
    let ids = Object.keys(set.arms)
    if (view.only) ids = ids.filter(a => view.only.includes(a))
    if (view.exclude) ids = ids.filter(a => !view.exclude.includes(a))
    return ids
  }, [set, viewId])

  const selected = sel.filter(a => armIds.includes(a))
  const effSel = selected.length ? selected : armIds.slice(0, 1)

  const stretch = set?.stretches?.[idx]
  const audio = useAudio(stretch?.audio)

  if (error) return <div className="loading">Failed to load replay data: {String(error)}</div>
  if (!data) return <div className="loading">Loading the streaming replay benchmark (≈1 MB)…</div>

  const groups = {}
  for (const id of armIds) {
    const g = set.arms[id].group
    ;(groups[g] = groups[g] || []).push(id)
  }

  const armsForTimeline = effSel.map((id, i) => ({
    id,
    label: set.arms[id].label,
    color: ARM_PALETTE[i % ARM_PALETTE.length],
    fires: set.arms[id].fires[String(idx)]?.f || [],
  }))

  const switchView = (v) => { setViewId(v); setIdx(0); audio.stop() }

  const counts = (id) => {
    const f = set.arms[id].fires[String(idx)]?.f || []
    const c = { hit: 0, resume: 0, false: 0, dup: 0 }
    f.forEach(x => c[x.cls]++)
    return c
  }
  const nTurn = stretch.events.filter(e => e.cls === 'turn_final').length
  const nCont = stretch.events.filter(e => e.cls === 'continuation').length

  return (
    <div>
      <div className="strip">
        <div className="seg">
          {Object.entries(VIEWS).map(([k, v]) => (
            <button key={k} className={viewId === k ? 'on' : ''} onClick={() => switchView(k)}>{v.label}</button>
          ))}
        </div>
      </div>
      <p className="note" style={{ marginTop: 10 }}>
        {viewId === 'dev25' && 'Continuous single-channel stretches from held-out AMI meetings: one target speaker’s utterances at their true timeline offsets, silence between them. Every system below was replayed over this identical audio; fires shown are the originally recorded ones, scored with the paper’s rules.'}
        {viewId === 'heldout100' && 'The 100-stretch held-out set used for the released unified model (0.982 boundary recall, 1.30 false fires/speech-min). Only the released model was evaluated here.'}
        {viewId === 'fresh50' && 'A twice-larger confirmation set drawn (seed 1) after all development ended — the causal endpointer holds its operating point: 0.924 recall, 0.42 s P50, 0.21 false/min.'}
      </p>

      {/* arm chips */}
      {Object.entries(groups).map(([g, ids]) => (
        <div key={g} className="chiprow" style={{ alignItems: 'center' }}>
          <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--ink-faint)', width: 130, flexShrink: 0 }}>{GROUP_LABEL[g] || g}</span>
          {ids.map(id => {
            const on = effSel.includes(id)
            const color = on ? ARM_PALETTE[effSel.indexOf(id) % ARM_PALETTE.length] : undefined
            const s = set.arms[id].summary
            return (
              <button key={id} className={`chip${on ? ' on' : ''}`} style={{ color }}
                title={`recall ${s.boundary_recall?.toFixed(3)} · P50 ${s.latency_s_p50?.toFixed(2)}s · ${s.false_fires_per_speech_min?.toFixed(2)} false/min`}
                onClick={() => setSel(on ? effSel.filter(a => a !== id) : [...effSel, id])}>
                {on && <span className="dot" />}{set.arms[id].label}
              </button>
            )
          })}
        </div>
      ))}

      {/* stretch picker + player */}
      <div className="strip">
        <label>Stretch</label>
        <select value={idx} onChange={e => { setIdx(+e.target.value); audio.stop() }}>
          {set.stretches.slice(0, view.n).map((s, i) => (
            <option key={s.id} value={i}>
              #{i + 1} · {s.meeting} / {s.speaker} · {Math.round(s.dur)}s
            </option>
          ))}
        </select>
        <div className="pager" style={{ margin: 0 }}>
          <button disabled={idx === 0} onClick={() => { setIdx(idx - 1); audio.stop() }}>← prev</button>
          <button disabled={idx >= view.n - 1} onClick={() => { setIdx(idx + 1); audio.stop() }}>next →</button>
        </div>
        <span style={{ fontSize: 12.5, color: 'var(--ink-faint)' }}>
          {nTurn} turn-final · {nCont} continuation boundaries · {stretch.events.length} utterances
        </span>
      </div>

      <div className="player">
        <button className="playbtn" onClick={audio.toggle} aria-label="play/pause">
          {audio.playing ? '❚❚' : '▶'}
        </button>
        <span className="time">{fmtTime(audio.t)} / {fmtTime(stretch.dur)}</span>
        <span style={{ fontSize: 12, color: 'var(--ink-faint)' }}>click anywhere on the timeline to seek · click a fire marker to replay its moment</span>
      </div>

      <StretchTimeline stretch={stretch} arms={armsForTimeline} t={audio.t} onSeek={audio.seek} />

      <div className="legend">
        <span><i style={{ background: 'rgba(0,105,79,0.35)' }} /> turn-final boundary (should fire)</span>
        <span><i style={{ background: 'rgba(176,124,0,0.35)' }} /> continuation (fire = segmentation cost, not error)</span>
        <span><i style={{ background: CLS_COLOR.hit, borderRadius: 6 }} /> hit</span>
        <span><i style={{ background: CLS_COLOR.resume }} /> fire at continuation</span>
        <span><i style={{ background: CLS_COLOR.false }} /> false fire (mid-speech)</span>
        <span><i style={{ background: CLS_COLOR.dup }} /> duplicate/silence</span>
      </div>

      {/* per-arm stats for this stretch */}
      <div className="tbl-scroll">
        <table className="tbl" style={{ marginTop: 16 }}>
          <thead>
            <tr><th>System (this stretch)</th><th className="num">Fires</th><th className="num">Hits</th>
              <th className="num">At continuation</th><th className="num">False</th>
              <th className="num">Latencies (s)</th><th className="num">Set-level recall / false-min</th></tr>
          </thead>
          <tbody>
            {effSel.map((id, i) => {
              const c = counts(id)
              const f = set.arms[id].fires[String(idx)]?.f || []
              const lats = f.filter(x => x.lat != null).map(x => x.lat.toFixed(2)).join(', ')
              const s = set.arms[id].summary
              return (
                <tr key={id}>
                  <td><span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 3, background: ARM_PALETTE[i % ARM_PALETTE.length], marginRight: 7 }} />{set.arms[id].label}</td>
                  <td className="num">{f.length}</td>
                  <td className="num" style={{ color: CLS_COLOR.hit, fontWeight: 600 }}>{c.hit}/{nTurn}</td>
                  <td className="num" style={{ color: CLS_COLOR.resume }}>{c.resume}</td>
                  <td className="num" style={{ color: c.false ? CLS_COLOR.false : undefined, fontWeight: c.false ? 700 : 400 }}>{c.false}</td>
                  <td className="num mono" style={{ fontSize: 11.5 }}>{lats || '—'}</td>
                  <td className="num">{s.boundary_recall?.toFixed(3)} / {s.false_fires_per_speech_min?.toFixed(2)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* transcript */}
      <div className="evt-list">
        {stretch.events.map((e, i) => (
          <div key={i} className={`evt${audio.t >= e.b && audio.t <= e.e ? ' active' : ''}`}
            onClick={() => audio.seek(e.b)}>
            <span className="t">{fmtTime(e.b)}–{fmtTime(e.e)}</span>
            <span className={`badge ${e.cls}`}>{e.cls.replace('_', '-')}{e.gap != null && e.cls !== 'merged' ? ` · gap ${e.gap.toFixed(1)}s` : ''}</span>
            <span className="txt">{e.text.toLowerCase()}</span>
          </div>
        ))}
      </div>
      <p className="note">
        Reference transcript from AMI annotations. Boundary classes are causal: <b>turn-final</b> =
        same-speaker gap ≥ 2 s (the system should fire); <b>continuation</b> = gap in [0.3, 2) s
        (firing is a measured cost — the correct answer depends on the future);{' '}
        <b>merged</b> = gap &lt; 0.3 s (no boundary event).
      </p>
    </div>
  )
}
