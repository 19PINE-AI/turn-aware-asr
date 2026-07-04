"""Generate all paper figures from the research artifacts. Run from repo root:
    .venv/bin/python paper/figures/make_figures.py
Outputs PDFs into paper/figures/.
"""
from __future__ import annotations
import json
import traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
R = ROOT / "research"

# ---------------------------------------------------------------- style
OKABE = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "grey": "#808080", "yellow": "#F0E442", "black": "#111111",
}
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote {name}.pdf")


def speech_block(ax, x0, x1, y, h=0.34, color="#9ecae1", ec="#3182bd", label=None, hatch=None, alpha=1.0):
    """A rounded 'speech' block on a timeline."""
    b = FancyBboxPatch((x0, y - h / 2), x1 - x0, h,
                       boxstyle="round,pad=0.015,rounding_size=0.06",
                       fc=color, ec=ec, lw=0.9, hatch=hatch, alpha=alpha, zorder=3)
    ax.add_patch(b)
    if label:
        ax.text((x0 + x1) / 2, y, label, ha="center", va="center",
                fontsize=7.2, zorder=4)


def timeline(ax, x0, x1, y):
    ax.plot([x0, x1], [y, y], color="#555555", lw=0.9, zorder=1)


# ================================================================ F1: oscillation
def fig1_oscillation():
    v8 = json.load(open(ROOT / "checkpoints/semantic_endpoint_v8_es/eval_log.json"))
    v9 = json.load(open(ROOT / "checkpoints/semantic_endpoint_v9_es/eval_log.json"))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)

    ax = axes[0]
    steps = [e["step"] / 1000 for e in v8]
    fire = [e["single"] for e in v8]
    hold = [e["no_fire_correct"] for e in v8]
    ax.plot(steps, fire, "-o", ms=3.5, color=OKABE["red"], label="fire class (single)")
    ax.plot(steps, hold, "-s", ms=3.5, color=OKABE["blue"], label="hold class (no-fire)")
    ax.set_title("v8: clairvoyant labels", fontsize=9.5)
    ax.set_xlabel("training step (k)")
    ax.set_ylabel("holdout accuracy")
    ax.set_ylim(0, 1.08)
    # annotate the two attractors
    ax.annotate("fire-mode", xy=(6, 0.95), xytext=(4.2, 1.03),
                fontsize=7.5, color=OKABE["red"],
                arrowprops=dict(arrowstyle="-", color=OKABE["red"], lw=0.7))
    ax.annotate("no-fire-mode", xy=(12, 0.10), xytext=(12.3, 0.26),
                fontsize=7.5, color=OKABE["blue"],
                arrowprops=dict(arrowstyle="-", color=OKABE["blue"], lw=0.7))
    ax.legend(loc="center left", fontsize=7.2)

    ax = axes[1]
    steps = [e["step"] / 1000 for e in v9]
    fire = [e["single_sil_exact"] for e in v9]
    hold = [np.mean([e["complete_nosil_exact"], e["silence_only_exact"],
                     e["truncated_exact"]]) for e in v9]
    comp = [e["score"] for e in v9]
    ax.plot(steps, fire, "-o", ms=3.5, color=OKABE["red"], label="fire class (single)")
    ax.plot(steps, hold, "-s", ms=3.5, color=OKABE["blue"], label="hold classes (mean of 3)")
    ax.plot(steps, comp, "--", lw=1.0, color=OKABE["grey"], label="composite (8 schemas)")
    ax.axvline(7.5, color=OKABE["green"], lw=0.8, ls=":")
    ax.text(7.7, 0.42, "selected (step 7.5k)", fontsize=7, color=OKABE["green"],
            va="bottom", rotation=90)
    ax.set_title("v9: causal labels (only the data changed)", fontsize=9.5)
    ax.set_xlabel("training step (k)")
    ax.legend(loc="lower center", fontsize=7.2)
    save(fig, "fig1_oscillation")


# ================================================================ F2: the two contradictions
def fig2_contradictions():
    fig, axes = plt.subplots(2, 1, figsize=(6.9, 3.5))
    for ax in axes:
        ax.set_xlim(0, 11.3)
        ax.set_ylim(-0.42, 2.2)
        ax.axis("off")

    # ---- Panel A: trailing-silence clip cut
    ax = axes[0]
    ax.text(0.0, 2.05, "(a) The trailing-silence clip cut: opposite labels on the same prefix",
            fontsize=8.6, weight="bold")
    # row 1: offline pool
    y = 1.45
    timeline(ax, 0.4, 6.4, y)
    speech_block(ax, 0.6, 4.6, y, label="“let's move on to the next slide”")
    ax.plot([4.72, 4.72], [y - 0.32, y + 0.32], color=OKABE["black"], lw=1.4)
    ax.text(4.95, y + 0.38, "clip ends at end of speech", fontsize=6.4, ha="left")
    ax.text(6.75, y, "offline pool label:", fontsize=7.4, va="center")
    ax.text(10.35, y, "FIRE", fontsize=8.2, va="center", ha="center", weight="bold",
            color="white", bbox=dict(boxstyle="round,pad=0.25", fc=OKABE["red"], ec="none"))
    # row 2: streaming pool — same prefix
    y = 0.55
    timeline(ax, 0.4, 6.4, y)
    speech_block(ax, 0.6, 4.6, y, label="“let's move on to the next slide”")
    ax.plot([4.72, 4.72], [y - 0.32, y + 0.32], color=OKABE["black"], lw=1.4)
    ax.text(5.55, y - 0.02, "no silence\nobserved yet", fontsize=6.4, ha="center", va="center",
            color="#555555")
    ax.text(6.75, y, "streaming pool label:", fontsize=7.4, va="center")
    ax.text(10.35, y, "HOLD", fontsize=8.2, va="center", ha="center", weight="bold",
            color="white", bbox=dict(boxstyle="round,pad=0.25", fc=OKABE["blue"], ec="none"))
    ax.annotate("", xy=(2.6, 0.90), xytext=(2.6, 1.12),
                arrowprops=dict(arrowstyle="<->", color=OKABE["purple"], lw=1.1))
    ax.text(2.75, 1.01, "causally identical", fontsize=6.8, color=OKABE["purple"], va="center")

    # ---- Panel B: labels keyed on who speaks next
    ax = axes[1]
    ax.text(0.0, 2.05, "(b) Pause labels keyed on the future",
            fontsize=8.6, weight="bold")
    dp = 4.72
    for row, (nxt, lab, labc) in enumerate([
            ("same speaker resumes\n(“disfluency”)", "HOLD", OKABE["blue"]),
            ("another speaker / nobody\n(“turn end”)", "FIRE", OKABE["red"])]):
        y = 1.45 - row * 0.9
        timeline(ax, 0.4, 6.4, y)
        speech_block(ax, 0.6, 3.4, y, label="“so what I think is…”")
        # pause
        ax.text((3.4 + dp) / 2 + 0.05, y - 0.28, "pause 1.2 s", fontsize=6.2, ha="center")
        # decision point
        ax.plot([dp, dp], [y - 0.34, y + 0.34], color=OKABE["black"], lw=1.4, ls="-")
        # future (greyed)
        fut = Rectangle((dp + 0.06, y - 0.26), 1.9, 0.52, fc="#dddddd", ec="#aaaaaa",
                        lw=0.7, hatch="///", zorder=2)
        ax.add_patch(fut)
        ax.text(dp + 1.0, y, "future", fontsize=6.4, ha="center", va="center", color="#666666")
        ax.text(6.72, y, nxt, fontsize=6.8, va="center")
        ax.annotate("", xy=(10.25, y), xytext=(9.55, y),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))
        ax.text(10.72, y, lab, fontsize=8.0, va="center", ha="center", weight="bold",
                color="white", bbox=dict(boxstyle="round,pad=0.22", fc=labc, ec="none"))
    ax.text(dp, -0.20, "decision point — prefixes indistinguishable "
                       "(gap distributions overlap: medians 1.39 s vs 0.83 s)",
            fontsize=6.6, ha="center", color="#444444")
    save(fig, "fig2_contradictions")


# ================================================================ F3: tradeoff frontier
def fig3_tradeoff():
    rms = json.load(open(R / "71-timeout-rms-seed0.json"))["arms"]
    sil = json.load(open(R / "71-timeout-silero-seed0.json"))["arms"]
    fig, ax = plt.subplots(figsize=(4.9, 3.3))

    def fam(arms, color, name, marker):
        pts = []
        for x, a in sorted(arms.items(), key=lambda kv: float(kv[0])):
            s = a["summary"]
            if "latency_s_p50" not in s:
                continue
            pts.append((float(x), s["latency_s_p50"],
                        max(s["false_fires_per_speech_min"], 0.05), s["boundary_recall"]))
        usable = [p for p in pts if p[3] >= 0.5]
        lame = [p for p in pts if p[3] < 0.5]
        ax.plot([p[1] for p in usable], [p[2] for p in usable], "-", color=color, lw=1.2, zorder=2)
        for X, lat, ff, rec in usable:
            ax.scatter([lat], [ff], s=46, color=color, marker=marker, zorder=3)
            off = (6, 4) if not (name.startswith("RMS") and X == 1.0) else (-14, -17)
            ax.annotate(f"X={X:g}s\nR={rec:.2f}", (lat, ff), textcoords="offset points",
                        xytext=off, fontsize=6.4, color=color)
        for X, lat, ff, rec in lame:
            ax.scatter([lat], [ff], s=40, facecolors="none", edgecolors=color,
                       marker=marker, zorder=3)
            ax.annotate(f"X={X:g}s  R={rec:.2f} (misses)", (lat, ff), textcoords="offset points",
                        xytext=(6, -9), fontsize=6.4, color=color, alpha=0.85)
        ax.plot([], [], marker=marker, color=color, lw=1.2, label=name)

    fam(rms, OKABE["orange"], "RMS + timeout", "o")
    fam(sil, OKABE["purple"], "Silero VAD + timeout", "^")

    # prior checkpoints (best policies) and v9
    priors = [("v5+gate+confirm1", 0.68, 3.2, 0.927, (7, -13)),
              ("v8+gate+confirm1", 0.77, 0.05, 0.938, (6, 2))]
    for name, lat, ff, rec, off in priors:
        ax.scatter([lat], [ff], s=40, color=OKABE["grey"], marker="D", zorder=3)
        ax.annotate(f"{name}\nR={rec:.2f}", (lat, ff), textcoords="offset points",
                    xytext=off, fontsize=6.2, color=OKABE["grey"])
    ax.scatter([0.39], [0.323], s=230, color=OKABE["green"], marker="*",
               zorder=4, edgecolor="black", lw=0.5)
    ax.annotate("v9 + gate\nR=0.97", (0.39, 0.323), textcoords="offset points",
                xytext=(-12, 12), fontsize=7.4, weight="bold", color=OKABE["green"])

    ax.set_yscale("log")
    ax.set_yticks([0.05, 0.3, 1, 3, 10, 30])
    ax.set_yticklabels(["0*", "0.3", "1", "3", "10", "30"])
    ax.set_xlabel("median end-of-turn latency P50 (s)")
    ax.set_ylabel("false fires / speech-minute")
    ax.set_xlim(0.25, 1.55)
    ax.legend(loc="upper right", fontsize=7.2)
    save(fig, "fig3_tradeoff")


# ================================================================ F4: biasing
def fig4_biasing():
    e4 = json.load(open(R / "69-e4-pinned-ctx.json"))["summary"]["by_age"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5),
                             gridspec_kw={"width_ratios": [1, 1.35]})
    # (a) uplift bars
    ax = axes[0]
    bars = [("no context", 66.7, OKABE["grey"]),
            ("relevant\nprefix", 95.6, OKABE["green"]),
            ("distractor\nhallucination", 3.7, OKABE["red"])]
    xs = np.arange(3)
    for i, (lab, v, c) in enumerate(bars):
        ax.bar(i, v, 0.62, color=c, alpha=0.92)
        ax.text(i, v + 1.6, f"{v:.1f}%", ha="center", fontsize=8, weight="bold")
    ax.annotate("", xy=(1, 95.6), xytext=(1, 66.7 + 1),
                arrowprops=dict(arrowstyle="->", color=OKABE["black"], lw=1.0))
    ax.text(1.09, 80, "+28.9 pp", fontsize=8, weight="bold", rotation=90, va="center")
    ax.set_xticks(xs, [b[0] for b in bars], fontsize=7.4)
    ax.set_ylabel("entity recall / rate (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Earnings-22 hotword biasing", fontsize=9)

    # (b) durability vs session age
    ax = axes[1]
    ages = sorted(float(a) for a in e4)
    pin = [100 * e4[f"{a:g}.0" if f"{a:g}.0" in e4 else str(a)]["pin_recall"] for a in ages]
    nop = [100 * e4[f"{a:g}.0" if f"{a:g}.0" in e4 else str(a)]["nopin_recall"] for a in ages]
    ax.plot(ages, pin, "-o", ms=4, color=OKABE["green"], label="CTX prefix present")
    ax.plot(ages, nop, "-s", ms=4, color=OKABE["grey"], label="no prefix")
    ax.fill_between(ages, nop, pin, color=OKABE["green"], alpha=0.12)
    gap = np.mean([p - n for p, n in zip(pin, nop)])
    ax.text(np.mean(ages), (np.mean(pin) + np.mean(nop)) / 2,
            f"advantage ≈ +{gap:.0f} pp, flat", fontsize=8, ha="center",
            style="italic", color=OKABE["green"])
    ax.set_xlabel("intervening audio between prefix and target (s)")
    ax.set_ylabel("entity recall (%)")
    ax.set_ylim(40, 100)
    ax.set_title("biasing survives session length", fontsize=9)
    ax.legend(loc="lower left", fontsize=7.2)
    save(fig, "fig4_biasing")


# ================================================================ F5: ranking inversion
def fig5_inversion():
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    models = ["v3", "v5", "v8"]
    offline = {"v3": 100, "v5": 82, "v8": 10}      # offline single-turn fire rate (%)
    stream_ff = {"v3": 88.8, "v5": 27.3, "v8": 3.6}  # false fires / min (lower better)
    colors = {"v3": OKABE["red"], "v5": OKABE["orange"], "v8": OKABE["blue"]}
    # ranks: offline rank by score desc; streaming rank by false fires asc
    off_rank = {m: r for r, m in enumerate(sorted(models, key=lambda m: -offline[m]))}
    st_rank = {m: r for r, m in enumerate(sorted(models, key=lambda m: stream_ff[m]))}
    for m in models:
        y0, y1 = 2 - off_rank[m], 2 - st_rank[m]
        ax.plot([0, 1], [y0, y1], "-", color=colors[m], lw=2.2, alpha=0.85, zorder=2)
        ax.scatter([0, 1], [y0, y1], s=44, color=colors[m], zorder=3)
        ax.text(-0.06, y0, f"{m}   {offline[m]}%", ha="right", va="center",
                fontsize=8.6, color=colors[m], weight="bold")
        ax.text(1.06, y1, f"{m}   {stream_ff[m]:g} false/min", ha="left", va="center",
                fontsize=8.6, color=colors[m], weight="bold")
    ax.text(0, 2.55, "offline benchmark\n(fire-rate on clipped audio)", ha="center", fontsize=8)
    ax.text(1, 2.55, "deployment-matched replay\n(false fires, gate on)", ha="center", fontsize=8)
    ax.text(0.5, -0.62, "“ship v3” → 89 false fires/min      “v8 is broken” → best endpointer",
            ha="center", fontsize=7.6, style="italic", color="#444444")
    ax.set_xlim(-0.55, 1.7)
    ax.set_ylim(-0.85, 2.95)
    ax.axis("off")
    save(fig, "fig5_inversion")


# ================================================================ F6: silence incompetence
def fig6_silence():
    d = json.load(open(R / "61-replay-v5-specv1.json"))["per_stretch"]
    sil = [r["duration_s"] - r["speech_s"] for r in d]
    spam = [r["n_dup_or_silence_fires"] for r in d]
    r = np.corrcoef(sil, spam)[0, 1]
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    ax.scatter(sil, spam, s=26, color=OKABE["blue"], alpha=0.8, zorder=3)
    coef = np.polyfit(sil, spam, 1)
    xs = np.linspace(min(sil), max(sil), 10)
    ax.plot(xs, np.polyval(coef, xs), "--", color=OKABE["red"], lw=1.1,
            label=f"linear fit  (r = {r:.3f})")
    ax.set_xlabel("silence in stretch (s)")
    ax.set_ylabel("spurious fires (ungated v5)")
    ax.legend(loc="upper left")
    ax.text(0.97, 0.05, "every v1–v8 training example\nbegan with speech",
            transform=ax.transAxes, ha="right", fontsize=7.4,
            style="italic", color="#444444")
    save(fig, "fig6_silence")


# ================================================================ F7: serving
def fig7_serving():
    e5 = json.load(open(R / "68-e5-concurrency.json"))["per_n"]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.3))
    fig.subplots_adjust(wspace=0.44)

    ax = axes[0]
    ax.bar([0, 1], [88.0, 3.8], 0.55, color=[OKABE["red"], OKABE["green"]])
    ax.set_xticks([0, 1], ["chunked\nplaceholders", "bounded\nre-feed"], fontsize=7.4)
    for i, v in enumerate([88.0, 3.8]):
        ax.text(i, v + 2, f"{v:g}%", ha="center", fontsize=8, weight="bold")
    ax.set_ylabel("WER (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(a) feeding shape is\nan OOD landmine", fontsize=8.4)

    ax = axes[1]
    groups = [("median", [455, 84]), ("P95", [None, 413]), ("P95 + flush", [None, 117])]
    ax.bar([0], [455], 0.5, color=OKABE["grey"], label="transformers sim")
    ax.bar([1], [84], 0.5, color=OKABE["blue"], label="vLLM")
    ax.bar([2.1], [413], 0.5, color=OKABE["sky"])
    ax.bar([3.1], [117], 0.5, color=OKABE["blue"])
    ax.set_xticks([0, 1, 2.1, 3.1],
                  ["sim", "vLLM", "P95\nunbnd.", "P95\n+flush"], fontsize=6.6)
    for x, v in [(0, 455), (1, 84), (2.1, 413), (3.1, 117)]:
        ax.text(x, v + 10, f"{v}", ha="center", fontsize=7.4, weight="bold")
    ax.set_ylabel("ms / 0.5 s chunk")
    ax.set_ylim(0, 520)
    ax.axhline(500, color=OKABE["red"], lw=0.8, ls=":")
    ax.text(3.1, 508, "frame budget", fontsize=6.2, color=OKABE["red"], ha="right")
    ax.set_title("(b) 5.4× faster than the\nsimulator; flush bounds P95", fontsize=8.4)

    ax = axes[2]
    ns = [e["n"] for e in e5]
    p50 = [e["frame_ms_p50"] for e in e5]
    p95 = [e["frame_ms_p95"] for e in e5]
    ax.plot(ns, p50, "-o", ms=4, color=OKABE["blue"], label="P50")
    ax.plot(ns, p95, "-s", ms=4, color=OKABE["orange"], label="P95")
    ax.axhline(500, color=OKABE["red"], lw=0.9, ls=":")
    ax.text(1.05, 520, "500 ms frame budget", fontsize=6.6, color=OKABE["red"])
    ax.axvspan(8, 16, color=OKABE["red"], alpha=0.06)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("concurrent sessions N")
    ax.set_ylabel("per-frame latency (ms)")
    ax.set_title("(c) deadline-bound:\nflat to N=8, cliff at 16", fontsize=8.4)
    ax.legend(loc="upper left", fontsize=6.8)
    save(fig, "fig7_serving")


# ================================================================ F8: system diagram
def fig8_system():
    fig, ax = plt.subplots(figsize=(6.6, 2.15))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 4.0)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=7.6, weight="normal", tc="black"):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.09",
                           fc=fc, ec=ec, lw=1.0, zorder=3)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, weight=weight, color=tc, zorder=4)

    def arrow(x0, y0, x1, y1, color="#333333", ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=9, color=color, lw=1.0,
                                     linestyle=ls, zorder=2))

    # audio chunks
    for i in range(4):
        box(0.25 + i * 0.42, 1.45, 0.34, 0.9, "", "#cfe8f7", "#3182bd")
    ax.text(1.1, 0.82, "0.5 s audio chunks", ha="center", fontsize=6.8)
    arrow(2.1, 1.9, 2.75, 1.9)

    box(2.8, 1.35, 1.5, 1.1, "energy\ngate", "#fff3cd", "#c9a227")
    ax.text(3.55, 0.82, "silent chunks\nskip the LM", ha="center", fontsize=6.0, color="#666666")
    arrow(4.3, 1.9, 5.0, 1.9)

    box(5.05, 0.7, 3.3, 2.5, "", "#f0f0f0", "#888888")
    ax.text(6.7, 2.92, "Qwen3-ASR-0.6B + LoRA (merged)", ha="center", fontsize=7.2, weight="bold")
    box(5.3, 1.75, 1.3, 0.85, "audio\nencoder", "#dbe9f6", "#3182bd", fs=6.8)
    box(6.75, 1.75, 1.45, 0.85, "LM decoder\n+2 marker rows", "#dbe9f6", "#3182bd", fs=6.2)
    box(5.3, 0.85, 2.85, 0.6, "committed transcript prefix (bounded re-feed)", "#eeeeee", "#999999", fs=6.2)
    arrow(6.6, 2.17, 6.85, 2.17)

    box(5.65, 3.35, 2.1, 0.55, "⟨CTX⟩ hotword prefix", "#e2f0e5", "#2e7d32", fs=6.8)
    arrow(6.7, 3.35, 6.7, 3.22, color="#2e7d32")

    arrow(8.35, 1.9, 9.05, 1.9)
    box(9.1, 1.35, 1.9, 1.1, "policy knobs\nconfirm h · flush", "#fde2cf", "#d55e00", fs=6.8)
    arrow(11.0, 2.15, 11.7, 2.6)
    arrow(11.0, 1.65, 11.7, 1.2)
    box(11.75, 2.35, 1.5, 0.75, "transcript\nsegments", "#e8e8f8", "#5555aa", fs=6.8)
    box(11.75, 0.85, 1.5, 0.75, "END events\n(≈EagerEndOfTurn)", "#f8e2e2", "#aa3333", fs=6.4)
    save(fig, "fig8_system")


# ================================================================ F9: v9 rule + minimal pair
def fig9_minimalpair():
    fig, ax = plt.subplots(figsize=(6.9, 1.85))
    ax.set_xlim(0, 11.7)
    ax.set_ylim(0, 2.35)
    ax.axis("off")
    ax.text(0.0, 2.16, "The minimal-pair device: the same utterance, ± an observable silence tail",
            fontsize=8.6, weight="bold")
    y = 1.5
    timeline(ax, 0.4, 6.9, y)
    speech_block(ax, 0.6, 4.6, y, label="“turn the lights off in the kitchen”")
    ax.text(7.05, y, "target:  …kitchen", fontsize=7.6, va="center", family="monospace")
    ax.text(10.85, y, "HOLD", fontsize=8.0, va="center", ha="center", weight="bold",
            color="white", bbox=dict(boxstyle="round,pad=0.22", fc=OKABE["blue"], ec="none"))
    y = 0.55
    timeline(ax, 0.4, 6.9, y)
    speech_block(ax, 0.6, 4.6, y, label="“turn the lights off in the kitchen”")
    sil = Rectangle((4.72, y - 0.10), 1.15, 0.2, fc="#efe6c8", ec="#c9a227", lw=0.8, zorder=3)
    ax.add_patch(sil)
    ax.text(5.3, y + 0.33, "+0.3–1.2 s silence", fontsize=6.6, ha="center", color="#8a6d1a")
    ax.text(7.05, y, "target:  …kitchen ⟨M⟩", fontsize=7.6, va="center", family="monospace")
    ax.text(10.85, y, "FIRE", fontsize=8.0, va="center", ha="center", weight="bold",
            color="white", bbox=dict(boxstyle="round,pad=0.22", fc=OKABE["red"], ec="none"))
    ax.annotate("only difference = the feature the model must key on",
                xy=(5.3, 0.72), xytext=(2.2, 0.11), fontsize=6.8, color="#8a6d1a",
                arrowprops=dict(arrowstyle="->", color="#8a6d1a", lw=0.8))
    save(fig, "fig9_minimalpair")


# ================================================================ F10: landscape quadrant
def fig10_landscape():
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axhline(5, color="#bbbbbb", lw=0.8)
    ax.axvline(5, color="#bbbbbb", lw=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("#888888")
    ax.grid(False)
    ax.set_xlabel("end-of-turn decision signal:   acoustic (silence/VAD)  →  semantic + acoustic (in-model)",
                  fontsize=8, labelpad=6)
    ax.set_ylabel("closed  →  open weights + recipe", fontsize=8, labelpad=6)

    def pt(x, y, name, color, fs=7.6, weight="normal", marker="o", ms=46):
        ax.scatter([x], [y], s=ms, color=color, marker=marker, zorder=3,
                   edgecolor="black", lw=0.4)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, 7),
                    fontsize=fs, ha="center", weight=weight, color=color)

    pt(1.6, 7.6, "WebRTC / Silero VAD\n+ timeout cascades", OKABE["grey"])
    pt(3.4, 2.6, "commercial cascades\n(classic endpointing)", OKABE["grey"])
    pt(6.4, 2.0, "learned acoustic EOT\n(Meta hier., Smart Turn)", OKABE["sky"])
    pt(8.4, 3.4, "Deepgram Flux,\nOpenAI semantic_vad", OKABE["purple"])
    pt(6.2, 6.0, "joint E2E endpointing\n(RNN-T + </s>)", OKABE["orange"])
    pt(5.6, 7.9, "Kyutai STT\n(semantic-VAD head)", OKABE["blue"])
    pt(8.5, 8.3, "this work", OKABE["green"], fs=9, weight="bold", marker="*", ms=260)
    save(fig, "fig10_landscape")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in [fig1_oscillation, fig2_contradictions, fig3_tradeoff, fig4_biasing,
               fig5_inversion, fig6_silence, fig7_serving, fig8_system, fig9_minimalpair,
               fig10_landscape]:
        try:
            fn()
        except Exception:
            print(f"FAILED {fn.__name__}")
            traceback.print_exc()
