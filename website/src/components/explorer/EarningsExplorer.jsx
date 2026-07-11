import React, { useMemo, useState } from 'react'
import { useJson } from '../../lib/useJson.js'
import MiniPlay from './MiniPlay.jsx'

function markEntities(text, entities, cls) {
  if (!entities?.length) return [text]
  let parts = [text]
  entities.forEach((ent, ei) => {
    const re = new RegExp(ent.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i')
    parts = parts.flatMap((p, pi) => {
      if (typeof p !== 'string') return [p]
      const m = p.match(re)
      if (!m) return [p]
      return [p.slice(0, m.index),
        <mark key={`${ei}-${pi}`} className={cls}>{m[0]}</mark>,
        p.slice(m.index + m[0].length)]
    })
  })
  return parts
}

export default function EarningsExplorer() {
  const { data, error } = useJson('data/earnings.json')
  const [armId, setArmId] = useState('base')
  const [filter, setFilter] = useState('all')
  const [page, setPage] = useState(0)
  const PER = 8

  const arm = data?.arms?.[armId]
  const items = useMemo(() => {
    if (!arm) return []
    return arm.items.filter(it => {
      if (filter === 'rescued') return (it.recall_relevant ?? 0) > (it.recall_no_ctx ?? 0)
      if (filter === 'halluc') return (it.hallucination_distractor_count ?? 0) > 0
      if (filter === 'entities') return (it.ref_entities?.length ?? 0) > 0
      return true
    })
  }, [arm, filter])

  if (error) return <div className="loading">Failed to load Earnings-22 data: {String(error)}</div>
  if (!data) return <div className="loading">Loading Earnings-22 biasing results…</div>

  const s = arm.summary
  const pages = Math.ceil(items.length / PER)
  const shown = items.slice(page * PER, (page + 1) * PER)

  return (
    <div>
      <div className="strip">
        <div className="seg">
          {Object.entries(data.arms).map(([k, a]) => (
            <button key={k} className={armId === k ? 'on' : ''} onClick={() => { setArmId(k); setPage(0) }}>{a.label}</button>
          ))}
        </div>
        <div className="seg">
          {[['all', 'all utterances'], ['entities', 'has entities'], ['rescued', 'context rescued an entity'], ['halluc', 'distractor hallucinations']].map(([k, l]) => (
            <button key={k} className={filter === k ? 'on' : ''} onClick={() => { setFilter(k); setPage(0) }}>{l}</button>
          ))}
        </div>
      </div>
      <p className="note" style={{ marginTop: 8 }}>
        Spontaneous earnings-call speech (Earnings-22) dense in tickers, executives, and products;
        entities LLM-extracted and filtered to genuine proper nouns. Each utterance is decoded
        three times: no context, a <b>relevant hotword prefix</b>, and a <b>distractor prefix</b>{' '}
        listing entities from a different call. This arm: entity recall{' '}
        {(s.entity_recall_no_ctx * 100).toFixed(1)}% → <b>{(s.entity_recall_relevant * 100).toFixed(1)}%</b> with
        context (+{s.entity_recall_uplift_pp.toFixed(1)} pp) · WER {s.wer_no_ctx.toFixed(1)}% →{' '}
        {s.wer_relevant.toFixed(1)}% · distractor hallucination{' '}
        {(s.distractor_hallucination_rate * 100).toFixed(1)}% · {s.n_utterances} utterances.
        Highlighting: <mark style={{ background: '#ffe28a', padding: '0 3px' }}>reference entity</mark>{' '}
        <mark className="bad" style={{ padding: '0 3px' }}>distractor entity in output</mark>.
      </p>

      <div style={{ marginTop: 10 }}>
        {shown.map(it => (
          <div className="probe-card" key={it.utt_id} style={{ marginBottom: 14 }}>
            <h4>
              <span>
                {it.utt_id}
                {it.ref_entities?.length > 0 && <> · entities: {it.ref_entities.map((e, i) => <code key={i} style={{ marginRight: 5 }}>{e}</code>)}</>}
              </span>
              {data.audio[it.utt_id] && <MiniPlay url={data.audio[it.utt_id]} />}
            </h4>
            <div className="hyp"><span className="lab">reference</span>{markEntities(it.ref, it.ref_entities, '')}</div>
            <div className="two-col" style={{ gap: 10, marginTop: 2 }}>
              <div className="hyp">
                <span className="lab">no context{it.recall_no_ctx != null ? ` · entity recall ${(it.recall_no_ctx * 100).toFixed(0)}%` : ''}</span>
                {markEntities(it.hyp_no_ctx, it.ref_entities, 'good')}
              </div>
              <div className="hyp">
                <span className="lab">relevant prefix{it.recall_relevant != null ? ` · entity recall ${(it.recall_relevant * 100).toFixed(0)}%` : ''}</span>
                {markEntities(it.hyp_relevant, it.ref_entities, 'good')}
              </div>
            </div>
            <div className="hyp" style={(it.hallucination_distractor_count ?? 0) > 0 ? { outline: '2px solid var(--red)' } : {}}>
              <span className="lab">distractor prefix ({(it.distractor_entities || []).join(', ')}){(it.hallucination_distractor_count ?? 0) > 0 ? ' · HALLUCINATION' : ''}</span>
              {markEntities(it.hyp_distractor, it.distractor_entities, 'bad')}
            </div>
          </div>
        ))}
      </div>
      {pages > 1 && (
        <div className="pager">
          <button disabled={page === 0} onClick={() => setPage(page - 1)}>← prev</button>
          <span>page {page + 1} / {pages} · {items.length} utterances</span>
          <button disabled={page >= pages - 1} onClick={() => setPage(page + 1)}>next →</button>
        </div>
      )}
    </div>
  )
}
