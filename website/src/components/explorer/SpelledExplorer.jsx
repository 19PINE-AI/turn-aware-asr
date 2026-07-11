import React, { useMemo, useState } from 'react'
import { useJson } from '../../lib/useJson.js'
import MiniPlay from './MiniPlay.jsx'

function highlight(hyp, target, distractorName) {
  // mark exact-target matches green, distractor-name matches red
  const parts = []
  let rest = hyp
  const patterns = []
  if (target) patterns.push({ re: new RegExp(target.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i'), cls: 'good' })
  if (distractorName) patterns.push({ re: new RegExp(distractorName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i'), cls: 'bad' })
  let guard = 0
  while (rest && guard++ < 30) {
    let best = null
    for (const p of patterns) {
      const m = rest.match(p.re)
      if (m && (best === null || m.index < best.m.index)) best = { m, cls: p.cls }
    }
    if (!best) { parts.push(rest); break }
    parts.push(rest.slice(0, best.m.index))
    parts.push(<mark key={guard} className={best.cls}>{best.m[0]}</mark>)
    rest = rest.slice(best.m.index + best.m[0].length)
  }
  return parts
}

const distractorNameOf = (ctx) => {
  const m = ctx.match(/name:\s*([^;]+);/)
  return m ? m[1].trim().split(' ').pop() : null
}

export default function SpelledExplorer() {
  const { data, error } = useJson('data/spelled.json')
  const [kind, setKind] = useState('all')
  const [outcome, setOutcome] = useState('all')
  const [page, setPage] = useState(0)
  const PER = 9

  const items = useMemo(() => {
    if (!data) return []
    return data.items.filter(it =>
      (kind === 'all' || it.kind === kind) &&
      (outcome === 'all' ||
        (outcome === 'intrusion' && it.intrusion === 1) ||
        (outcome === 'ctx_helped' && it.hit_none === 0 && it.hit_profile === 1) ||
        (outcome === 'miss' && (it.hit_none === 0 || it.hit_profile === 0 || it.hit_distractor === 0))))
  }, [data, kind, outcome])

  if (error) return <div className="loading">Failed to load spelled-entity data: {String(error)}</div>
  if (!data) return <div className="loading">Loading spelled-entity probe…</div>

  const s = data.summary
  const pages = Math.ceil(items.length / PER)
  const shown = items.slice(page * PER, (page + 1) * PER)

  return (
    <div>
      <p className="note" style={{ marginTop: 8 }}>
        240 synthesized utterances spelling uncommon names and email addresses, decoded by the
        released model under three context conditions: <b>no context</b>, the <b>matching user
        profile</b>, and a <b>wrong-user distractor profile</b>. Exact-match: names{' '}
        {(s.name_none * 100).toFixed(0)}% → <b>{(s.name_profile * 100).toFixed(0)}%</b> with
        profile; emails {(s.email_none * 100).toFixed(0)}% → <b>{(s.email_profile * 100).toFixed(0)}%</b>.
        Wrong-profile intrusion: <b>{(s.intrusion_rate * 100).toFixed(1)}%</b> — the counterfactual
        twin at work. Highlighting: <mark className="good" style={{ padding: '0 3px' }}>target entity</mark>{' '}
        <mark className="bad" style={{ padding: '0 3px' }}>distractor intrusion</mark>.
      </p>
      <div className="strip">
        <div className="seg">
          {['all', 'name', 'email'].map(k => (
            <button key={k} className={kind === k ? 'on' : ''} onClick={() => { setKind(k); setPage(0) }}>
              {k === 'all' ? 'all kinds' : k + 's'}
            </button>
          ))}
        </div>
        <div className="seg">
          {[['all', 'all outcomes'], ['ctx_helped', 'context rescued it'], ['miss', 'any miss'], ['intrusion', 'intrusions']].map(([k, l]) => (
            <button key={k} className={outcome === k ? 'on' : ''} onClick={() => { setOutcome(k); setPage(0) }}>{l}</button>
          ))}
        </div>
        <span style={{ fontSize: 12.5, color: 'var(--ink-faint)' }}>{items.length} items</span>
      </div>

      <div className="itemgrid">
        {shown.map(it => {
          const dn = distractorNameOf(it.ctx_distractor)
          return (
            <div className="probe-card" key={it.id}>
              <h4>
                <span>#{it.id} · {it.kind}{it.style ? ` · ${it.style}` : ''} · target: <code>{it.target}</code></span>
                <MiniPlay url={it.audio} />
              </h4>
              <div style={{ fontSize: 12.5, color: 'var(--ink-soft)', marginBottom: 4 }}>“{it.utt}”</div>
              <div className="hyp">
                <span className="lab">no context {it.hit_none ? '✓' : '✗'}</span>
                {highlight(it.hyp_none, it.target, dn)}
              </div>
              <div className="hyp">
                <span className="lab">matching profile {it.hit_profile ? '✓' : '✗'}</span>
                {highlight(it.hyp_profile, it.target, dn)}
              </div>
              <div className="hyp" style={it.intrusion ? { outline: '2px solid var(--red)' } : {}}>
                <span className="lab">wrong-user profile {it.hit_distractor ? '✓' : '✗'}{it.intrusion ? ' · INTRUSION' : ''}</span>
                {highlight(it.hyp_distractor, it.target, dn)}
              </div>
            </div>
          )
        })}
      </div>
      {pages > 1 && (
        <div className="pager">
          <button disabled={page === 0} onClick={() => setPage(page - 1)}>← prev</button>
          <span>page {page + 1} / {pages}</span>
          <button disabled={page >= pages - 1} onClick={() => setPage(page + 1)}>next →</button>
        </div>
      )}
    </div>
  )
}
