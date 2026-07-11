import React from 'react'
import { useJson } from '../../lib/useJson.js'
import TradeoffChart from './TradeoffChart.jsx'
import OscillationChart from './OscillationChart.jsx'
import BiasingChart from './BiasingChart.jsx'
import { SilAppend, Inversion, SilenceScatter, ToyChart } from './InterventionCharts.jsx'
import { MainTable, DictationTable, CompositionTable } from './Tables.jsx'

export default function Results() {
  const { data, error } = useJson('data/results.json')
  return (
    <section className="block alt" id="results">
      <div className="wrap">
        <div className="kicker">Section 2 · Key results</div>
        <h2>What the causal recipe buys, measured</h2>
        <p className="lede">
          All figures below are redrawn from the original result files — the same numbers as the
          paper, in interactive form. Section 3 lets you inspect every underlying trajectory.
        </p>
        {error && <div className="loading">Failed to load results data: {String(error)}</div>}
        {!data && !error && <div className="loading">Loading results…</div>}
        {data && (
          <>
            <TradeoffChart data={data.tradeoff} />
            <MainTable rows={data.main_table} heldout={data.heldout} />
            <CompositionTable rows={data.composition} />
            <OscillationChart data={data.oscillation} />
            <div className="two-col">
              <SilAppend data={data.silappend} />
              <Inversion data={data.inversion} />
            </div>
            <BiasingChart data={data.biasing} />
            <DictationTable rows={data.dictation_table} />
            <div className="two-col">
              <SilenceScatter data={data.silence} />
              {data.toy && <ToyChart data={data.toy} />}
            </div>
          </>
        )}
      </div>
    </section>
  )
}
