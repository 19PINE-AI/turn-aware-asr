#!/usr/bin/env python3
"""Aggregate exp-3 replay JSONs into the findings table (offline re-scoring)."""
import json, glob, os, sys
from pathlib import Path

ARMS = ["base", "A", "B", "C"]
TOL_EARLY = 0.25
TOL_LATE = 1.5
T_MERGE = 0.3

def load(arm, cfg):
    p = f"research/104-replay-{arm}-{cfg}.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))

def silence_seconds(ps):
    # silence = duration not in any speech interval
    tot = 0.0
    for r in ps:
        speech = sum(e["end_s"] - e["begin_s"] for e in r["events"])
        tot += r["duration_s"] - speech
    return tot

def merged_gap_fire_analysis(ps):
    """For all fires, count those landing shortly after a merged-boundary
    (gap<0.3s) utterance end -> proxy for 'fires at <0.3s silence'.
    Also compute median gap-to-preceding-utterance-end for non-turn fires."""
    short_gap_fires = 0
    total_nonhit = 0
    for r in ps:
        ev = r["events"]
        # turn-final hit windows to exclude
        bounds = []
        for i, e in enumerate(ev):
            if e["boundary"] == "merged":
                continue
            nxt = ev[i+1]["begin_s"] if i+1 < len(ev) else float("inf")
            bounds.append((e["end_s"], e["boundary"], min(e["end_s"]+TOL_LATE, nxt+TOL_EARLY)))
        for f in r["fires_s"]:
            # is it a turn-final hit?
            hit = any(bt - TOL_EARLY <= f <= we and bty == "turn_final"
                      for bt, bty, we in bounds)
            if hit:
                continue
            total_nonhit += 1
            # nearest preceding utterance end
            prev_ends = [(e["end_s"], e.get("gap_s")) for e in ev if e["end_s"] <= f + TOL_EARLY]
            if prev_ends:
                end_s, gap = max(prev_ends, key=lambda x: x[0])
                if gap is not None and gap < T_MERGE:
                    short_gap_fires += 1
    return short_gap_fires, total_nonhit

def agg():
    rows = {}
    for arm in ARMS:
        g = load(arm, "gated")
        u = load(arm, "ungated")
        c = load(arm, "confirm1")
        row = {}
        if g:
            s = g["summary"]; ps = g["per_stretch"]
            speech_min = sum(r["speech_s"] for r in ps)/60.0
            n_false = sum(r["n_false_fires"] for r in ps)       # internal (in-speech, non-boundary)
            n_resume = sum(r["n_resume_fires"] for r in ps)      # continuation boundary fires
            n_silfire = sum(r["n_dup_or_silence_fires"] for r in ps)
            sg, tot_nh = merged_gap_fire_analysis(ps)
            row.update(dict(
                recall=s["boundary_recall"],
                false_min=s["false_fires_per_speech_min"],
                p50=s.get("latency_s_p50"), p95=s.get("latency_s_p95"),
                resume=s["resume_after_fire_rate"],
                n_turn=s["n_turn_bounds"], n_cont=s["n_cont_bounds"],
                internal_false=n_false, cont_resume=n_resume, sil_fire=n_silfire,
                internal_false_min=n_false/speech_min, cont_resume_min=n_resume/speech_min,
                speech_min=speech_min,
                shortgap_fires=sg, nonhit_fires=tot_nh,
                shortgap_frac=(sg/tot_nh if tot_nh else 0.0),
            ))
        if u:
            ps = u["per_stretch"]
            sil_s = silence_seconds(ps)
            n_sil = sum(r["n_dup_or_silence_fires"] for r in ps)
            row["ungated_silfire_per_s"] = n_sil / sil_s if sil_s else 0.0
            row["ungated_silfire_total"] = n_sil
            row["ungated_sil_s"] = sil_s
            row["ungated_false_min"] = u["summary"]["false_fires_per_speech_min"]
            row["ungated_recall"] = u["summary"]["boundary_recall"]
        if c:
            ps = c["per_stretch"]
            canc = sum(r["n_cancelled_candidates"] for r in ps)
            surv = sum(r["n_fires"] for r in ps)
            row["confirm1_cancelled"] = canc
            row["confirm1_survived"] = surv
            row["confirm1_cancel_rate"] = canc/(canc+surv) if (canc+surv) else 0.0
            row["confirm1_recall"] = c["summary"]["boundary_recall"]
            row["confirm1_false_min"] = c["summary"]["false_fires_per_speech_min"]
        rows[arm] = row
    return rows

if __name__ == "__main__":
    rows = agg()
    print(json.dumps(rows, indent=1))
