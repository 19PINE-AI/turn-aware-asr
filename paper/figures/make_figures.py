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
import matplotlib.font_manager as fm
for _f in ("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyrepagella-regular.otf",
           "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyrepagella-bold.otf",
           "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyrepagella-italic.otf",
           "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyrepagella-bolditalic.otf"):
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Pagella", "DejaVu Serif"],
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


def speech_block(ax, x0, x1, y, h=0.34, color="#9ecae1", ec="#3182bd", label=None, hatch=None, alpha=1.0, fs=7.2):
    """A rounded 'speech' block on a timeline."""
    b = FancyBboxPatch((x0, y - h / 2), x1 - x0, h,
                       boxstyle="round,pad=0.015,rounding_size=0.06",
                       fc=color, ec=ec, lw=0.9, hatch=hatch, alpha=alpha, zorder=3)
    ax.add_patch(b)
    if label:
        ax.text((x0 + x1) / 2, y, label, ha="center", va="center",
                fontsize=fs, zorder=4)


def timeline(ax, x0, x1, y):
    ax.plot([x0, x1], [y, y], color="#555555", lw=0.9, zorder=1)


# ================================================================ F0: teaser
def fig0_teaser():
    """Headline figure: two turns no single silence timeout can get right, and
    what the turn-aware model does on each. Pure illustration, schematic time
    axis. One 'hold-through' turn (a dictated number) and one 'fire-now'
    completion; the full three-scenario treatment is fig16_scenarios (§2)."""
    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0.15, 4.15)
    ax.axis("off")
    ax.grid(False)

    RED, GREEN, GREY = "#c23b22", "#00694f", "#555555"

    def cross(x, y, label=None, dy=0.38, above=False):
        ax.scatter([x], [y], marker="X", s=95, color=RED, zorder=5,
                   edgecolor="black", lw=0.4)
        if label:
            if above:
                ax.text(x, y + dy, label, ha="center", va="bottom",
                        fontsize=7.4, color=RED)
            else:
                ax.text(x, y - dy, label, ha="center", va="top",
                        fontsize=7.4, color=RED)

    def star(x, y, label=None, dy=0.38):
        ax.scatter([x], [y], marker="*", s=210, color=OKABE["green"], zorder=5,
                   edgecolor="black", lw=0.4)
        if label:
            ax.text(x, y - dy, label, ha="center", va="top", fontsize=7.4,
                    color=GREEN, weight="bold")

    # ---- Turn 1: the pause that means "wait" (1 s of audio ~ 1.1 units)
    yA = 3.35
    ax.text(0.15, yA + 0.52, "Turn 1 — the pause that means “wait”",
            fontsize=8.6, style="italic")
    timeline(ax, 0.15, 12.45, yA)
    speech_block(ax, 0.3, 2.1, yA, label="“nine eight one …”")
    speech_block(ax, 3.1, 4.9, yA, label="“five one four …”")
    speech_block(ax, 6.1, 8.3, yA, label="“two three five one.”")
    ax.text(2.6, yA + 0.28, "0.9 s", ha="center", fontsize=7.0, color=GREY)
    ax.text(5.5, yA + 0.28, "1.1 s", ha="center", fontsize=7.0, color=GREY)
    cross(2.85, yA, "0.7 s timeout\nfires mid-number")
    cross(5.85, yA)
    star(8.75, yA, "ours: +0.39 s")
    ax.text(4.55, yA - 1.12, "ours: holds (six digits predict four more)",
            ha="center", fontsize=7.4, color=GREEN)

    # ---- Turn 2: the completion that means "go"
    yB = 1.15
    ax.text(0.15, yB + 0.52, "Turn 2 — the completion that means “go”",
            fontsize=8.6, style="italic")
    timeline(ax, 0.15, 12.45, yB)
    speech_block(ax, 0.3, 3.1, yB, label="“Hello? I’m still here.”")
    star(3.55, yB)
    ax.text(3.15, yB - 0.38, "ours: +0.39 s —\nthought complete", ha="center",
            va="top", fontsize=7.4, color=GREEN, weight="bold")
    cross(4.2, yB)
    ax.text(4.65, yB - 0.38, "1.0 s timeout: still waiting", ha="left",
            va="top", fontsize=7.4, color=RED)

    # legend, in the empty right half of the lower row
    ax.scatter([9.6], [0.85], marker="X", s=70, color=RED, edgecolor="black", lw=0.4)
    ax.text(9.85, 0.85, "silence timeout", va="center", fontsize=7.8)
    ax.scatter([9.6], [0.42], marker="*", s=150, color=OKABE["green"],
               edgecolor="black", lw=0.4)
    ax.text(9.85, 0.42, "turn-aware ASR (this work)", va="center", fontsize=7.8)

    save(fig, "fig0_teaser")


# ============================================ F16: three-scenario problem setup
def fig16_scenarios():
    """Comprehensive companion to the teaser (Figure 1), placed in the problem
    formulation. All three motivating turns, each decision point annotated with
    the causal read (completeness + observed silence) versus what a fixed
    silence timeout does. Pure illustration, schematic time axis."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 12.9)
    ax.set_ylim(0.05, 5.95)
    ax.axis("off")
    ax.grid(False)

    GREEN, GREY = "#00694f", "#5b5b5b"

    def tmo(x, y, label=None):
        ax.scatter([x], [y], marker="X", s=78, color="#c23b22", zorder=5,
                   edgecolor="black", lw=0.4)
        if label:
            ax.text(x, y + 0.24, label, ha="center", va="bottom",
                    fontsize=6.5, color="#c23b22")

    def star(x, y):
        ax.scatter([x], [y], marker="*", s=185, color=OKABE["green"], zorder=6,
                   edgecolor="black", lw=0.4)

    def read(x, y, label, fire=False):
        ax.text(x, y - 0.27, label, ha="center", va="top", fontsize=6.5,
                color=(GREEN if fire else GREY),
                style=("normal" if fire else "italic"),
                weight=("bold" if fire else "normal"))

    # (1) dictated number — hold through the pauses
    y = 5.15
    ax.text(0.1, y + 0.54,
            "(1) Dictated number — the words say “more to come”: hold through the pauses",
            fontsize=8.3, style="italic")
    timeline(ax, 0.1, 12.8, y)
    for a, b, t in [(0.3, 2.1, "“nine eight one”"), (3.3, 5.1, "“five one four”"),
                    (6.3, 8.7, "“two three five one.”")]:
        speech_block(ax, a, b, y, label=t)
    tmo(2.7, y, "0.7 s timeout fires"); read(2.7, y, "3 digits — incomplete → hold")
    tmo(5.7, y); read(5.7, y, "6 digits — incomplete → hold")
    star(9.1, y); read(9.1, y, "10 digits complete\n+ 0.3 s silence → fire  (+0.39 s)", fire=True)

    # (2) long question — short clauses sized to fit inside the blocks
    y = 3.05
    ax.text(0.1, y + 0.54,
            "(2) Long question — clause pauses inside one turn: hold through them",
            fontsize=8.3, style="italic")
    timeline(ax, 0.1, 12.8, y)
    for a, b, t in [(0.3, 3.0, "“the request times out”"),
                    (3.7, 6.4, "“and the retry fails,”"),
                    (7.1, 9.5, "“what should we do?”")]:
        speech_block(ax, a, b, y, label=t, fs=6.5)
    tmo(3.35, y, "timeout fires"); read(3.35, y, "subordinate clause → hold")
    tmo(6.75, y); read(6.75, y, "still mid-question → hold")
    star(9.9, y); read(9.9, y, "question complete\n+ silence → fire  (+0.40 s)", fire=True)

    # (3) completed turn — model fires at near-zero silence; timeout keeps counting
    y = 1.05
    ax.text(0.1, y + 0.54,
            "(3) Completed turn — no pause to wait out: fire at near-zero silence",
            fontsize=8.3, style="italic")
    timeline(ax, 0.1, 12.8, y)
    speech_block(ax, 0.3, 3.7, y, label="“That’s everything, thanks.”", fs=6.6)
    star(4.05, y)
    ax.text(3.55, y - 0.30, "complete + 0.3 s silence\n→ fires +0.39 s", ha="center",
            va="top", fontsize=6.5, color=GREEN, weight="bold")
    tmo(4.9, y)
    ax.text(4.62, y + 0.24, "1.0 s timeout", ha="left", va="bottom",
            fontsize=6.5, color="#c23b22")
    ax.text(5.4, y - 0.30, "still counting silence", ha="left", va="top",
            fontsize=6.5, color=GREY, style="italic")

    # legend, empty right half of the completion row
    ax.scatter([9.7], [1.30], marker="X", s=64, color="#c23b22", edgecolor="black", lw=0.4)
    ax.text(9.92, 1.30, "silence-timeout fire", va="center", fontsize=7.4)
    ax.scatter([9.7], [0.88], marker="*", s=140, color=OKABE["green"],
               edgecolor="black", lw=0.4)
    ax.text(9.92, 0.88, "turn-aware fire (this work)", va="center", fontsize=7.4)

    save(fig, "fig16_scenarios")


# ================================================================ F1: oscillation
def fig1_oscillation():
    v8 = json.load(open(ROOT / "checkpoints/semantic_endpoint_v8_es/eval_log.json"))
    v9 = json.load(open(ROOT / "checkpoints/semantic_endpoint_v9_es/eval_log.json"))
    # second seed of the opposed-pools recipe (data regenerated; only the seed changed)
    v8b_path = ROOT / "checkpoints/semantic_endpoint_v8_seed1_es/eval_log.json"
    v8b = json.load(open(v8b_path)) if v8b_path.exists() else None
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)

    ax = axes[0]
    steps = [e["step"] / 1000 for e in v8]
    fire = [e["single"] for e in v8]
    hold = [e["no_fire_correct"] for e in v8]
    ax.plot(steps, fire, "-o", ms=3.5, color=OKABE["red"], label="fire class (seed 0)")
    ax.plot(steps, hold, "-s", ms=3.5, color=OKABE["blue"], label="hold class (seed 0)")
    if v8b is not None:  # multi-seed: the oscillation reproduces
        sb = [e["step"] / 1000 for e in v8b]
        ax.plot(sb, [e["single"] for e in v8b], "--o", ms=2.5, lw=1.0,
                color=OKABE["red"], alpha=0.45, label="fire class (seed 1)")
        ax.plot(sb, [e["no_fire_correct"] for e in v8b], "--s", ms=2.5, lw=1.0,
                color=OKABE["blue"], alpha=0.45, label="hold class (seed 1)")
    ax.set_title("opposed-pools: clairvoyant labels (2 seeds)", fontsize=9.5)
    ax.set_xlabel("training step (k)")
    ax.set_ylabel("holdout accuracy")
    ax.set_ylim(0, 1.08)
    # annotate the two attractors
    ax.annotate("fire mode", xy=(6, 0.95), xytext=(4.2, 1.03),
                fontsize=7.5, color=OKABE["red"],
                arrowprops=dict(arrowstyle="-", color=OKABE["red"], lw=0.7))
    ax.annotate("no-fire mode", xy=(12, 0.10), xytext=(12.3, 0.26),
                fontsize=7.5, color=OKABE["blue"],
                arrowprops=dict(arrowstyle="-", color=OKABE["blue"], lw=0.7))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
              fontsize=7.2, frameon=False, columnspacing=1.2, handletextpad=0.5)

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
    ax.set_title("causal labels (only the supervision changed)", fontsize=9.5)
    ax.set_xlabel("training step (k)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
              fontsize=7.2, frameon=False, columnspacing=1.2, handletextpad=0.5)
    fig.subplots_adjust(bottom=0.34)
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
    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    # darker text variants of the marker colors: annotation text must hold
    # contrast against the white background even at print size
    TEXT = {OKABE["orange"]: "#8a6100", OKABE["purple"]: "#9c4f76",
            OKABE["grey"]: "#595959", OKABE["green"]: "#00694f",
            OKABE["sky"]: "#145a85", "#8a5a2b": "#6f4820"}

    def fam(arms, color, name, marker, over=None):
        over = over or {}
        pts = []
        for x, a in sorted(arms.items(), key=lambda kv: float(kv[0])):
            s = a["summary"]
            if "latency_s_p50" not in s:
                continue
            pts.append((float(x), s["latency_s_p50"],
                        max(s["false_fires_per_speech_min"], 0.05), s["boundary_recall"]))
        usable = [p for p in pts if p[3] >= 0.5]
        lame = [p for p in pts if p[3] < 0.5]
        ax.plot([p[1] for p in usable], [p[2] for p in usable], "-", color=color, lw=1.5, zorder=2)
        for X, lat, ff, rec in usable:
            ax.scatter([lat], [ff], s=58, color=color, marker=marker, zorder=3)
            if X in over:
                dx, dy, ha = over[X]
            else:
                (dx, dy), ha = ((7, 5), "left") if not (name.startswith("RMS") and X == 1.0) else ((-20, -24), "left")
            ax.annotate(f"X={X:g}s\nR={rec:.2f}", (lat, ff), textcoords="offset points",
                        xytext=(dx, dy), fontsize=8.2, color=TEXT[color], ha=ha)
        for X, lat, ff, rec in lame:
            ax.scatter([lat], [ff], s=50, facecolors="none", edgecolors=color,
                       marker=marker, zorder=3)
            dx, dy, ha = over.get(X, (7, -12, "left"))
            ax.annotate(f"X={X:g}s  R={rec:.2f} (misses)", (lat, ff), textcoords="offset points",
                        xytext=(dx, dy), fontsize=8.2, color=TEXT[color], ha=ha)
        ax.plot([], [], marker=marker, color=color, lw=1.5, label=name)

    fam(rms, OKABE["orange"], "RMS + timeout", "o",
        over={0.5: (8, -3, "left"), 2.0: (7, 4, "left")})
    fam(sil, OKABE["purple"], "Silero VAD + timeout", "^",
        over={0.5: (-8, -3, "right"), 1.5: (7, 6, "left"), 2.0: (7, -3, "left")})
    ext_handle = ax.plot([], [], marker="X", color="#8a5a2b", lw=0,
                         label="external turn-aware (as-shipped)")

    # prior checkpoints (best policies) and the causal model
    priors = [("mixed-pools + confirm h=1", 0.68, 3.2, 0.927, (8, -17)),
              ("opposed-pools + confirm h=1", 0.77, 0.05, 0.938, (7, 3))]
    for name, lat, ff, rec, off in priors:
        ax.scatter([lat], [ff], s=50, color=OKABE["grey"], marker="D", zorder=3)
        ax.annotate(f"{name}\nR={rec:.2f}", (lat, ff), textcoords="offset points",
                    xytext=off, fontsize=8.0, color=TEXT[OKABE["grey"]])
    ax.scatter([0.39], [0.323], s=300, color=OKABE["green"], marker="*",
               zorder=4, edgecolor="black", lw=0.5)
    ax.annotate("causal endpointer\nR=0.97", (0.39, 0.323), textcoords="offset points",
                xytext=(-18, 13), fontsize=9.2, weight="bold", color=TEXT[OKABE["green"]])
    # released unified model (rank-32, all four behaviors + WER/biasing fixes):
    # a little endpointing precision traded for dictation + context + intrusion-
    # resistance, at preserved recall (dev-25, same set as the pure point)
    ax.scatter([0.39], [0.97], s=190, color=OKABE["sky"], marker="D",
               zorder=4, edgecolor="black", lw=0.5)
    ax.annotate("unified release\nR=0.95", (0.39, 0.97), textcoords="offset points",
                xytext=(7, 5), fontsize=9.2, weight="bold", color=TEXT[OKABE["sky"]])

    # external open turn-aware systems on the same protocol (exp-4; as-shipped,
    # public thresholds swept). None reaches the causal model's corner.
    ext = [("Parakeet-EOU", 0.47, 0.2, 0.17, (-4, -12), "right"),
           ("Kyutai VAD", 0.53, 14.6, 0.88, (10, 4), "left"),
           ("Smart Turn v3", 0.64, 3.4, 0.71, (-9, -14), "right"),
           ("LiveKit (oracle)", 0.65, 6.25, 0.96, (-9, -3), "right")]
    for name, lat, ff, rec, off, ha in ext:
        ax.scatter([lat], [ff], s=54, color="#8a5a2b",
                   marker="X", zorder=3, edgecolor="black", lw=0.4)
        ax.annotate(f"{name}\nR={rec:.2f}", (lat, ff), textcoords="offset points",
                    xytext=off, fontsize=7.8, color=TEXT["#8a5a2b"], ha=ha)

    ax.set_yscale("log")
    ax.set_yticks([0.05, 0.3, 1, 3, 10, 30])
    ax.set_yticklabels(["0*", "0.3", "1", "3", "10", "30"])
    ax.set_xlabel("median end-of-turn latency P50 (s)", fontsize=11)
    ax.set_ylabel("false fires / speech-minute", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.set_xlim(0.25, 1.55)
    # legend OUTSIDE the plot area (a row above the axes), so it cannot
    # collide with the threshold annotations in the upper region
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.01), ncol=3,
              fontsize=8.6, frameon=False, columnspacing=1.1, handletextpad=0.4,
              borderaxespad=0)
    save(fig, "fig3_tradeoff")


# ================================================================ F4: biasing
def fig4_biasing():
    e4 = json.load(open(R / "69-e4-pinned-ctx.json"))["summary"]["by_age"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5),
                             gridspec_kw={"width_ratios": [1, 1.35]})
    # (a) uplift bars (base-model capability) + the released model's distractor
    # hallucination, which the natural-speech counterfactual drives below base
    ax = axes[0]
    bars = [("no context", 66.7, OKABE["grey"]),
            ("relevant\nprefix", 95.6, OKABE["green"])]
    for i, (lab, v, c) in enumerate(bars):
        ax.bar(i, v, 0.62, color=c, alpha=0.92)
        ax.text(i, v + 1.6, f"{v:.1f}%", ha="center", fontsize=8, weight="bold")
    # paired distractor-hallucination bars: base vs released (rank-32 counterfactual)
    ax.bar(2 - 0.16, 3.7, 0.30, color=OKABE["red"], alpha=0.45)
    ax.bar(2 + 0.16, 2.0, 0.30, color=OKABE["red"], alpha=0.92)
    ax.text(2 - 0.16, 3.7 + 1.6, "3.7%", ha="center", fontsize=7, color=OKABE["red"])
    ax.text(2 + 0.16, 2.0 + 1.6, "2.0%", ha="center", fontsize=7, weight="bold", color=OKABE["red"])
    ax.annotate("", xy=(1, 95.6), xytext=(1, 66.7 + 1),
                arrowprops=dict(arrowstyle="->", color=OKABE["black"], lw=1.0))
    ax.text(1.09, 80, "+28.9 pp", fontsize=8, weight="bold", rotation=90, va="center")
    ax.set_xticks([0, 1, 2], ["no context", "relevant\nprefix",
                              "distractor halluc.\nbase / released"], fontsize=7.0)
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
    fig, ax = plt.subplots(figsize=(4.9, 2.9))
    models = ["offline-labeled", "mixed-pools", "opposed-pools"]
    offline = {"offline-labeled": 100, "mixed-pools": 82, "opposed-pools": 10}      # offline single-turn fire rate (%)
    stream_ff = {"offline-labeled": 88.8, "mixed-pools": 27.3, "opposed-pools": 3.6}  # false fires / min (lower better)
    colors = {"offline-labeled": OKABE["red"], "mixed-pools": OKABE["orange"], "opposed-pools": OKABE["blue"]}
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
    ax.text(0.5, -0.62, "offline “champion” → 89 false fires/min      “broken” opposed-pools → best prior endpointer",
            ha="center", fontsize=7.6, style="italic", color="#444444")
    ax.set_xlim(-0.75, 1.9)
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
    ax.set_ylabel("spurious fires (ungated, mixed-pools)")
    ax.legend(loc="upper left")
    ax.text(0.97, 0.05, "every pre-diagnosis training example\nbegan with speech",
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
    fig, ax = plt.subplots(figsize=(7.4, 2.35))
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 4.1)
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
        box(0.25 + i * 0.46, 1.45, 0.38, 0.9, "", "#cfe8f7", "#3182bd")
    ax.text(1.2, 0.78, "0.5 s audio chunks", ha="center", fontsize=7.2)
    arrow(2.15, 1.9, 2.75, 1.9)

    box(2.8, 1.35, 1.6, 1.1, "energy\ngate", "#fff3cd", "#c9a227", fs=8.0)
    ax.text(3.6, 0.62, "silent chunks\nskip the LM", ha="center", fontsize=6.8, color="#555555")
    arrow(4.4, 1.9, 4.95, 1.9)

    box(5.0, 0.48, 4.7, 2.76, "", "#f0f0f0", "#888888")
    ax.text(7.35, 2.88, "Qwen3-ASR-0.6B + LoRA (merged)", ha="center", fontsize=7.8, weight="bold")
    box(5.25, 1.75, 1.7, 0.85, "audio\nencoder", "#dbe9f6", "#3182bd", fs=7.5)
    box(7.15, 1.75, 2.3, 0.85, "LM decoder\n+2 marker rows", "#dbe9f6", "#3182bd", fs=7.2)
    box(5.25, 0.65, 4.2, 0.85, "committed transcript prefix\n(bounded re-feed)", "#eeeeee", "#999999", fs=6.8)
    arrow(6.95, 2.17, 7.15, 2.17)

    box(6.0, 3.42, 2.6, 0.55, "⟨CTX⟩ hotword prefix", "#e2f0e5", "#2e7d32", fs=7.5)
    arrow(7.3, 3.42, 7.3, 3.28, color="#2e7d32")

    arrow(9.7, 1.9, 10.3, 1.9)
    box(10.35, 1.35, 2.15, 1.1, "policy knobs\nconfirm h · flush", "#fde2cf", "#d55e00", fs=7.5)
    arrow(12.5, 2.15, 13.0, 2.6)
    arrow(12.5, 1.65, 13.0, 1.2)
    box(13.05, 2.35, 1.9, 0.78, "transcript\nsegments", "#e8e8f8", "#5555aa", fs=7.5)
    box(13.05, 0.72, 1.9, 0.98, "END events\n(same shape as\nEagerEndOfTurn)", "#f8e2e2", "#aa3333", fs=6.4)
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


# ================================================================ F11: one principle, two axes
def fig11_twoaxes():
    fig, axes = plt.subplots(2, 1, figsize=(6.9, 3.5))
    for ax in axes:
        ax.set_xlim(0, 12.4); ax.set_ylim(-0.35, 2.35); ax.axis("off")

    def chip(ax, x, y, txt, color):
        ax.text(x, y, txt, fontsize=7.6, va="center", ha="center", weight="bold",
                color="white", bbox=dict(boxstyle="round,pad=0.22", fc=color, ec="none"))

    # ---- Panel A: temporal axis (minimal pair, +/- silence tail)
    ax = axes[0]
    ax.text(0.0, 2.18, "Temporal axis: target must not depend on the future  "
            "(fix: same utterance $\\pm$ observable silence)", fontsize=8.4, weight="bold")
    for row, (tail, tgt, lab, lc) in enumerate([
            (False, "target: …kitchen", "HOLD", OKABE["blue"]),
            (True,  "target: …kitchen ⟨M⟩", "FIRE", OKABE["red"])]):
        y = 1.5 - row * 0.95
        timeline(ax, 0.4, 6.6, y)
        speech_block(ax, 0.6, 4.5, y, label="“…off in the kitchen”")
        if tail:
            ax.add_patch(Rectangle((4.62, y - 0.10), 1.15, 0.2, fc="#efe6c8",
                                   ec="#c9a227", lw=0.8, zorder=3))
            ax.text(5.2, y + 0.30, "+silence", fontsize=6.4, ha="center", color="#8a6d1a")
        ax.text(7.0, y, tgt, fontsize=7.4, va="center", family="monospace")
        chip(ax, 11.6, y, lab, lc)

    # ---- Panel B: contextual axis (counterfactual, matching vs disagreeing ctx)
    ax = axes[1]
    ax.text(0.0, 2.18, "Contextual axis: target must not be copied from context  "
            "(fix: same audio $\\pm$ a disagreeing context)", fontsize=8.4, weight="bold")
    for row, (ctxt, ctxc, tgt, lab, lc) in enumerate([
            ("ctx: profile = Kowalski", "#e2f0e5",
             "target: kowalski", "GROUND", OKABE["green"]),
            ("ctx: profile = Nguyen (wrong)", "#f6e0d8",
             "target: kowalski", "IGNORE ctx", OKABE["green"])]):
        y = 1.5 - row * 0.95
        timeline(ax, 0.4, 6.6, y)
        speech_block(ax, 0.6, 4.5, y, label="audio spells “K-O-W…”")
        ax.text(0.6, y + 0.42, ctxt, fontsize=6.4, ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc=ctxc, ec="#999999", lw=0.5))
        ax.text(7.0, y, tgt, fontsize=7.4, va="center", family="monospace")
        chip(ax, 11.6, y, lab, lc)
    ax.text(6.2, -0.22, "naive data (context always matches) $\\Rightarrow$ model copies context "
            "$\\Rightarrow$ 40% wrong-profile intrusion; the counterfactual twin removes it",
            fontsize=6.6, ha="center", color="#444444")
    save(fig, "fig11_twoaxes")


# ================================================================ F12-F15: one real example per §3 subsection
def _real_examples_material():
    """Shared loader for the real-example figures. Nothing is synthetic:
    texts, gaps, waveforms, and fire timestamps are read from the benchmark
    material and the recorded result files (62-replay-v9gate,
    74/75-dictation-probe, 119-probe-v18)."""
    import sys
    import soundfile as sf
    sys.path.insert(0, str(ROOT))
    from eval.streaming_replay_eval import load_meetings, build_stretches, SR
    import random

    split = json.load(open(ROOT / "data/semantic_endpoint_v3/meeting_split.json"))
    meetings = load_meetings(ROOT / "data/ami/ihm", set(split["eval_meeting_ids"]))
    st0 = build_stretches(meetings, random.Random(0), 1)[0]   # IS1007d / MIO049
    assert st0["meeting_id"] == "IS1007d"
    digit_audio, dsr = sf.read(ROOT / "data/probes/digit_wavs/digit_000.wav", dtype="float32")
    assert dsr == SR
    v9_fires = json.load(open(R / "74-dictation-probe-v9.json"))["digit"]["per_item"][0]["fires_s"]
    v11_fires = json.load(open(R / "75-dictation-probe-v11.json"))["digit"]["per_item"][0]["fires_s"]
    return st0, digit_audio, v9_fires, v11_fires, SR


def _chip(ax, x, y, txt, color, fs=7.6):
    ax.text(x, y, txt, fontsize=fs, va="center", ha="center", weight="bold",
            color="white", bbox=dict(boxstyle="round,pad=0.28", fc=color, ec="none"),
            zorder=6)


def fig12_pauses():
    """§3.1 — two real pauses, 90 ms apart in length; the label is decided by
    the future. Sources: held-out AMI stretches of the replay benchmark."""
    fig, ax = plt.subplots(figsize=(7.0, 1.85))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(-0.75, 2.15)
    ax.axis("off")
    ax.grid(False)
    GREY = "#555555"
    dp = 3.0 + 1.6          # decision point drawn ~1.6 s into each pause
    for row, (txt, gap, resumes, cls, lab, lc, src) in enumerate([
            ("“can i close this”", 1.92, True, "continuation", "HOLD", OKABE["blue"],
             "AMI TS3012d · MTD045PM"),
            ("“did you manage”", 2.01, False, "turn-final", "FIRE", OKABE["red"],
             "AMI TS3012c · MTD046ID")]):
        y = 1.72 - row * 1.24
        timeline(ax, 0.3, 8.5, y)
        speech_block(ax, 0.5, 3.0, y, label=txt)
        ax.text((3.0 + dp) / 2, y + 0.36, f"pause {gap:.2f} s", fontsize=7.0,
                ha="center", color=GREY)
        ax.plot([dp, dp], [y - 0.36, y + 0.36], color=OKABE["black"], lw=1.3)
        if resumes:
            speech_block(ax, dp + 0.35, dp + 2.6, y, color="#e8e6e0", ec="#9aa0a8",
                         label="resumes at +1.92 s")
        else:
            ax.text(dp + 1.55, y, "silence continues…", fontsize=7.0, ha="center",
                    va="center", color=GREY, style="italic", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none"))
        ax.text(8.8, y + 0.28, src, fontsize=6.6, color=GREY, va="center")
        ax.text(8.8, y - 0.24, f"class: {cls}", fontsize=7.2, color=lc, va="center")
        _chip(ax, 12.0, y, lab, lc)
    ax.plot([dp, dp], [0.28, 2.0], color="#444444", lw=0.6, ls=":")
    ax.text(6.3, -0.55, "decision point $t$: the two prefixes are indistinguishable — "
            "the class is a function of audio after $t$",
            fontsize=7.2, color="#444444", ha="center")
    save(fig, "fig12_pauses")


def fig13_rule():
    """§3.2 — the causal labeling rule firing on real replay audio."""
    st0, _, _, _, SR = _real_examples_material()
    fig, ax = plt.subplots(figsize=(7.0, 1.9))
    t0, t1 = 21.6, 25.4
    seg = st0["audio"][int(t0 * SR):int(t1 * SR)]
    tt = np.linspace(t0, t1, len(seg))
    ax.fill_between(tt, seg, -seg, color="#9aa7b4", lw=0)
    ax.set_xlim(t0, t1)
    ax.set_ylim(-0.30, 0.46)
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    for k in np.arange(np.ceil(t0 / 0.5) * 0.5, t1, 0.5):     # the 0.5 s chunk grid
        ax.axvline(k, color="#e2e0d8", lw=0.5, zorder=0)
    ax.set_xticks([22, 23, 24, 25])
    ax.set_xticklabels(["22 s", "23 s", "24 s", "25 s"], fontsize=7.5)
    ax.annotate("“did you prepare” — complete\n(speech ends 23.04 s)", xy=(23.0, 0.13),
                xytext=(21.7, 0.30), fontsize=7.4, color=OKABE["blue"],
                arrowprops=dict(arrowstyle="->", color=OKABE["blue"], lw=0.8))
    ax.axvspan(23.04, 23.34, ymin=0.12, ymax=0.72, color=OKABE["yellow"], alpha=0.45, zorder=1)
    ax.annotate("0.3 s silence observed", xy=(23.19, -0.10), xytext=(23.62, -0.24),
                fontsize=7.2, color="#8a6d1a",
                arrowprops=dict(arrowstyle="->", color="#8a6d1a", lw=0.8))
    ax.scatter([23.5], [0.0], marker="*", s=230, color=OKABE["green"], zorder=5,
               edgecolor="black", lw=0.4)
    ax.annotate("recorded fire: 23.5 s — the first chunk boundary\nwith both conditions true (latency +0.46 s)",
                xy=(23.53, 0.06), xytext=(23.85, 0.28), fontsize=7.4, color="#00694f",
                arrowprops=dict(arrowstyle="->", color="#00694f", lw=0.8))
    ax.text(25.35, 0.40, "AMI IS1007d · MIO049 — causal-model replay (62-replay)",
            fontsize=6.6, ha="right", color="#555555")
    save(fig, "fig13_rule")


def fig14_dictation():
    """§3.3 — dictation probe item 0: identical audio, two supervisions."""
    _, digit_audio, v9_fires, v11_fires, SR = _real_examples_material()
    fig, ax = plt.subplots(figsize=(7.0, 2.35))
    td = np.linspace(0, len(digit_audio) / SR, len(digit_audio))
    ax.fill_between(td, digit_audio * 0.52 + 0.68, -digit_audio * 0.52 + 0.68,
                    color="#9aa7b4", lw=0)
    ax.set_xlim(-2.1, 9.6)
    ax.set_ylim(-1.62, 1.38)
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.set_xticks(range(0, 10))
    ax.set_xticklabels([f"{s} s" for s in range(0, 10)], fontsize=7.5)
    for (a, b) in [(1.70, 2.41), (4.14, 4.82)]:
        ax.axvspan(a, b, ymin=0.70, ymax=0.99, color=OKABE["yellow"], alpha=0.4)
        ax.text((a + b) / 2, 1.22, "pause", fontsize=7.0, ha="center", color="#8a6d1a")
    ax.axvline(7.08, color="#555555", lw=0.8, ls=":")
    ax.text(7.08, 1.22, "last digit ends", fontsize=7.0, ha="center", color="#555555")
    ax.text(-2.05, 0.68, "audio\n(digit probe #0)", fontsize=7.0, color="#555555",
            ha="left", va="center")
    for y, fires, name in [(-0.45, v9_fires, "conversational-only"),
                           (-1.25, v11_fires, "+ dictation schemas")]:
        ax.plot([0, 9.5], [y, y], color="#e0ddd4", lw=0.8, zorder=1)
        ax.text(-2.05, y, name, fontsize=7.2, color="#333333", ha="left", va="center")
        for f in fires:
            if f <= 7.08:
                ax.scatter([f], [y], marker="X", s=62, color=OKABE["red"], zorder=5,
                           edgecolor="black", lw=0.3)
            else:
                ax.scatter([f], [y], marker="*", s=170, color=OKABE["green"], zorder=5,
                           edgecolor="black", lw=0.4)
    ax.text(3.4, -0.86, "5 premature fires — the agent interrupts mid-number",
            fontsize=7.2, color=OKABE["red"], ha="center")
    ax.text(8.35, -0.86, "one fire, +0.42 s", fontsize=7.2, color="#00694f", ha="center")
    save(fig, "fig14_dictation")


def fig15_context():
    """§3.3 — spelled-entity probe under a wrong-user profile: matched-only
    training copies the context; the counterfactual grounds it in audio."""
    fig, ax = plt.subplots(figsize=(7.0, 1.8))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(-1.05, 3.55)
    ax.axis("off")
    ax.grid(False)
    rows = [
        ("matched-only ctx supervision (11.3%→40% intrusion):", "COPIED THE CTX", OKABE["red"],
         "audio says   marcus.szymanski@acmecorp.io",
         "model wrote  tomasz.wojciechowski@outlook.com"),
        ("+ counterfactual twin (released model, 0.8% intrusion):", "FOLLOWED AUDIO", OKABE["green"],
         "audio says   ignatius.yankovic@pine.ai",
         "model wrote  ignatius.yankovic@pine.ai"),
    ]
    for row, (model, lab, lc, audio_txt, out) in enumerate(rows):
        y = 3.2 - row * 2.05
        ax.text(0.0, y, model, fontsize=7.4, va="center", color="#333333", weight="bold")
        _chip(ax, 11.5, y, lab, lc, fs=7.0)
        ax.text(0.45, y - 0.62, audio_txt, fontsize=7.2, va="center", family="monospace",
                color="#333333")
        ax.text(0.45, y - 1.22, out, fontsize=7.2, va="center", family="monospace",
                color=lc, weight="bold")
    ax.text(0.0, -0.92, "ctx in both rows: a different user's profile (name, email, phone); "
            "outputs verbatim from the recorded probe results (75-dictation-probe-v11, 119-probe-v18).",
            fontsize=6.6, color="#555555")
    save(fig, "fig15_context")


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
    pt(2.6, 1.6, "learned acoustic EOT\n(Helwani et al.)", OKABE["sky"])
    pt(2.7, 6.0, "Smart Turn (acoustic\nEOT, open recipe)", OKABE["sky"])
    pt(5.9, 6.9, "text EOU on separate STT\n(LiveKit, TEN)", OKABE["orange"])
    pt(8.4, 3.4, "Deepgram Flux,\nOpenAI semantic_vad", OKABE["purple"])
    pt(7.3, 2.0, "UAF\n(paper only)", OKABE["purple"])
    pt(6.3, 5.2, "joint E2E endpointing\n(RNN-T + </s>)", OKABE["orange"])
    pt(6.9, 7.9, "Kyutai STT (semantic-VAD\nhead, weights only)", OKABE["blue"])
    pt(8.3, 6.8, "Parakeet-EOU (in-transcript\ntoken, recipe undocumented)", OKABE["blue"])
    pt(8.7, 8.7, "this work", OKABE["green"], fs=9, weight="bold", marker="*", ms=260)
    save(fig, "fig10_landscape")


def fig17_decodetape():
    """§2 — how a streaming turn-aware recognizer decodes, cascade vs ours,
    on a real per-chunk trace (research/130-decode-trace.json: unified r32 on
    dictation probe #0). Panel a: cascade (separate ASR + silence-timeout).
    Panel b: ours (endpoint token emitted inline, one decode per 0.5 s chunk).
    """
    import soundfile as sf
    tr = json.load(open(R / "130-decode-trace.json"))
    audio, sr = sf.read(ROOT / "data/probes/digit_wavs/digit_000.wav", dtype="float32")
    dur = len(audio) / sr
    # compact peak envelope (keeps the visual, drops the file size)
    step = max(1, len(audio) // 2400)
    n = (len(audio) // step) * step
    env = np.abs(audio[:n]).reshape(-1, step).max(axis=1)
    env = env / max(env.max(), 1e-6)
    ta = np.linspace(0, dur, len(env))
    PAUSES = [(1.70, 2.41), (4.14, 4.82)]
    LAST = 7.08
    OURS_FIRE = tr["ours_fires"][0]                       # 7.5
    lat = OURS_FIRE - LAST

    GREEN, RED, YEL, BLU = OKABE["green"], OKABE["red"], "#c9a227", OKABE["blue"]
    GREY = "#666666"
    fig, ax = plt.subplots(figsize=(7.2, 5.15))
    ax.set_xlim(-2.75, 10.7)
    ax.set_ylim(0, 10.5)
    ax.axis("off")
    ax.grid(False)

    def wave(yc, amp, hl=None):
        ax.fill_between(ta, yc + env * amp, yc - env * amp, color="#9aa7b4", lw=0)
        for k in np.arange(0.5, dur, 0.5):                # 0.5 s chunk grid
            ax.axvline(k, ymin=0, ymax=1, color="#e4e2da", lw=0.4, zorder=0)
        for (a, b) in PAUSES:
            ax.add_patch(Rectangle((a, yc - amp - 0.05), b - a, 2 * (amp + 0.05),
                                   color=YEL, alpha=0.22, lw=0, zorder=1))

    def chip(x, y, txt, color, fs=7.0):
        ax.text(x, y, txt, fontsize=fs, va="center", ha="center", weight="bold",
                color="white", zorder=6,
                bbox=dict(boxstyle="round,pad=0.24", fc=color, ec="none"))

    def pbox(x, y, w, h, lines, ec, fc="#fbfbf8", fs=6.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                                    fc=fc, ec=ec, lw=1.0, zorder=4))
        yy = y + h - 0.26
        for t, c, wt in lines:
            ax.text(x + 0.16, yy, t, fontsize=fs, va="top", ha="left", color=c,
                    weight=wt, zorder=5)
            yy -= 0.30

    DIG = "nine  eight  one   five  one  four   two  three  five  one"

    # ---- pinned CTX header ----
    ax.add_patch(FancyBboxPatch((-2.55, 9.62), 12.9, 0.66,
                                boxstyle="round,pad=0.04,rounding_size=0.08",
                                fc="#e9f3ec", ec="#2e7d32", lw=1.0, zorder=3))
    ax.text(-2.4, 9.95, "context, pinned in the system slot for the whole session",
            fontsize=6.7, style="italic", color="#2e7d32", va="center", ha="left")
    ax.text(-2.4, 9.72, r"$\langle$sys$\rangle$ User profile — name: Priya Raman; "
            r"email: priya.raman@gmail.com; plan: business tier $\langle$/sys$\rangle$",
            fontsize=6.9, color="#1b4d24", va="center", ha="left", family="monospace")

    # ================= PANEL (a): CASCADE =================
    ax.text(-2.7, 9.05, "(a)  The usual pipeline — a separate recognizer  +  a silence timer",
            fontsize=8.6, weight="bold", ha="left", color="#333333")
    wave(8.05, 0.34)
    ax.text(-2.7, 8.05, "audio\n(caller reads\na 10-digit no.)", fontsize=6.6, color=GREY,
            ha="left", va="center")
    for (a, b) in PAUSES:
        ax.text((a + b) / 2, 8.52, "pause", fontsize=6.2, ha="center", color="#8a6d1a")

    # ASR lane (transcript only, no marker)
    ax.text(-2.7, 7.34, "recognizer\noutput:", fontsize=6.8, color="#333333", ha="left", va="center")
    ax.add_patch(FancyBboxPatch((0.05, 7.12), 8.9, 0.46, boxstyle="round,pad=0.03,rounding_size=0.06",
                                fc="#eef1f4", ec="#9aa7b4", lw=0.8, zorder=3))
    ax.text(0.25, 7.35, DIG, fontsize=6.6, family="monospace", va="center", ha="left", color="#222222")
    ax.text(9.15, 7.35, "transcript\ntokens only", fontsize=6.3, color=GREY, va="center", ha="left")

    # silence-timeout lane
    ax.text(-2.7, 6.5, "turn-end call\n(silence timer):", fontsize=6.8, color="#333333", ha="left", va="center")
    # responsive 0.5s timeout barges into the pauses
    for (a, b) in PAUSES:
        ax.scatter([b + 0.5], [6.5], marker="X", s=52, color=RED, zorder=5, edgecolor="black", lw=0.3)
    ax.scatter([LAST + 0.5], [6.5], marker="X", s=52, color=RED, zorder=5, edgecolor="black", lw=0.3)
    ax.text(2.0, 6.06, r"responsive timeout ($X{=}0.5$ s): fires inside the pauses — "
            "2 premature / number", fontsize=6.4, color=RED, ha="left")
    # patient 1.0s timeout — late
    ax.scatter([tr["timeout_fires"][0]], [6.5], marker="*", s=150, color="#b0872a",
               zorder=5, edgecolor="black", lw=0.3)
    ax.text(tr["timeout_fires"][0] + 0.18, 6.5, f"patient ($X{{=}}1.0$ s):\n+{tr['timeout_fires'][0]-LAST:.1f} s, slow",
            fontsize=6.2, color="#8a6d1a", va="center", ha="left")
    ax.text(2.0, 5.72, "the clock is blind to the digits: no single timeout is both prompt and pause-proof",
            fontsize=6.6, color="#333333", style="italic", ha="left")

    ax.plot([-2.7, 10.4], [5.4, 5.4], color="#cfcdc4", lw=0.7)

    # ================= PANEL (b): OURS =================
    ax.text(-2.7, 5.05, "(b)  Ours — the recognizer emits the transcript AND the end-of-turn token, inline",
            fontsize=8.6, weight="bold", ha="left", color="#1b5e20")
    wave(4.05, 0.34)
    # trigger arrows at each 0.5 s boundary
    for k in np.arange(0.5, dur + 0.01, 0.5):
        ax.annotate("", xy=(k, 4.45), xytext=(k, 4.66),
                    arrowprops=dict(arrowstyle="-|>", color=BLU, lw=0.7, mutation_scale=6))
    ax.text(-2.7, 4.05, "audio", fontsize=6.6, color=GREY, ha="left", va="center")
    ax.text(4.5, 4.82, "one decode per 0.5 s audio chunk  (≈ 0.1 s compute each, well inside real time)",
            fontsize=6.6, color=BLU, ha="center")

    # committed transcript lane + HOLD / FIRE marks
    ax.text(-2.7, 3.42, "decode\noutput:", fontsize=6.8, color="#333333", ha="left", va="center")
    ax.add_patch(FancyBboxPatch((0.05, 3.2), 8.9, 0.46, boxstyle="round,pad=0.03,rounding_size=0.06",
                                fc="#eaf3ec", ec=GREEN, lw=0.9, zorder=3))
    ax.text(0.25, 3.43, DIG, fontsize=6.6, family="monospace", va="center", ha="left", color="#123")
    ax.annotate("", xy=(9.28, 3.43), xytext=(8.98, 3.43),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=0.8, mutation_scale=7))
    chip(9.75, 3.43, r"$\langle$END_SPEECH$\rangle$", GREEN, fs=6.3)
    for (a, b) in PAUSES:
        ax.text((a + b) / 2, 3.02, "held", fontsize=6.2, ha="center", color=GREEN, weight="bold")

    # zoom boxes: one HOLD (pause), one FIRE (end)
    pbox(-2.55, 1.30, 5.35, 1.60, [
        (r"at a pause  (t = 4.5 s)", GREY, "bold"),
        (r"$\langle$sys$\rangle$ …priya.raman@gmail.com…", "#1b4d24", "normal"),
        (r"$\langle$audio 0.0–4.8 s$\rangle$", "#333333", "normal"),
        (r"$\langle$asst$\rangle$ nine eight one · five one four", "#123", "normal"),
        (r"model emits:  → four …   (no $\langle$END$\rangle$)", GREEN, "bold"),
    ], GREEN)
    pbox(3.15, 1.30, 5.4, 1.60, [
        (r"at the turn end  (t = 7.5 s)", GREY, "bold"),
        (r"$\langle$sys$\rangle$ …priya.raman@gmail.com…", "#1b4d24", "normal"),
        (r"$\langle$audio 0.0–7.5 s$\rangle$", "#333333", "normal"),
        (r"$\langle$asst$\rangle$ …two three five one", "#123", "normal"),
        (r"model emits:  → $\langle$END_SPEECH$\rangle$", GREEN, "bold"),
    ], RED)
    ax.text(-2.4, 0.74, "HOLD — the 3-3-4 pattern predicts more digits",
            fontsize=6.6, color="#8a6d1a", ha="left", va="center")
    ax.text(3.3, 0.74, rf"FIRE — complete + 0.3 s silence heard,  +{lat:.2f} s",
            fontsize=6.6, color=RED, ha="left", va="center")
    ax.text(8.75, 1.9, "can output:\nany word, plus\ntwo markers\n"
            r"{$\langle$EAGER_END$\rangle$,"
            "\n" r"$\langle$END_SPEECH$\rangle$}",
            fontsize=6.3, color="#333333", ha="left", va="center")
    ax.text(10.4, 0.32, "real per-chunk replay · unified r32 checkpoint · dictation probe #0",
            fontsize=6.0, color="#999999", ha="right")
    save(fig, "fig17_decodetape")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in [fig0_teaser, fig16_scenarios, fig1_oscillation, fig2_contradictions, fig3_tradeoff,
               fig4_biasing, fig5_inversion, fig6_silence, fig7_serving, fig8_system, fig9_minimalpair,
               fig10_landscape, fig11_twoaxes,
               fig12_pauses, fig13_rule, fig14_dictation, fig15_context, fig17_decodetape]:
        try:
            fn()
        except Exception:
            print(f"FAILED {fn.__name__}")
            traceback.print_exc()
