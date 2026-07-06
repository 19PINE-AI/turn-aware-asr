"""exp-5 figure: clairvoyant-fraction toy task. Run from repo root:
    .venv/bin/python paper/figures/make_exp5_figure.py
Reads research/94-exp5-toy-task.json, writes paper/figures/exp5_toy.pdf.

Style matches paper/figures/make_figures.py (Okabe-Ito palette, TeX Gyre Pagella).
Left  : holdout FIRE/HOLD accuracy trajectories for three clairvoyant fractions
        f (monotone at f=0, anti-phase oscillation as f grows).
Right : the apparent recall-vs-precision "phantom frontier" across checkpoints --
        a tight top-right dot at f=0, a spread anti-diagonal cloud at high f.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
DATA = ROOT / "research" / "94-exp5-toy-task.json"

OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
         "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
         "grey": "#808080", "yellow": "#F0E442", "black": "#111111"}
for _f in ("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyrepagella-regular.otf",
           "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyrepagella-bold.otf",
           "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyrepagella-italic.otf"):
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Pagella", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def mean_traj(runs, f, key):
    """Mean over seeds of a trajectory quantity, aligned by step."""
    rs = [r for r in runs if abs(r["f"] - f) < 1e-9]
    steps = [t["step"] for t in rs[0]["trajectory"]]
    vals = np.array([[t[key] for t in r["trajectory"]] for r in rs])
    return np.array(steps), vals.mean(0)


def main():
    d = json.load(open(DATA))
    runs = d["runs"]
    fs_all = sorted({r["f"] for r in runs})
    warm = d["config"].get("eval_every", 20)  # unused; warmup handled below

    fig = plt.figure(figsize=(7.2, 3.0))
    gs = GridSpec(3, 2, width_ratios=[1.0, 1.05], hspace=0.18, wspace=0.28,
                  left=0.07, right=0.99, top=0.9, bottom=0.14)

    # -------- left column: three stacked trajectory panels -----------------
    sel = [0.0, 0.7, 1.0]
    labels = {0.0: r"$f=0$  (causal)", 0.7: r"$f=0.7$", 1.0: r"$f=1.0$  (all clairvoyant)"}
    for i, f in enumerate(sel):
        ax = fig.add_subplot(gs[i, 0])
        st, fa = mean_traj(runs, f, "fire_acc")
        _, ha = mean_traj(runs, f, "hold_acc")
        ax.plot(st / 1000, fa, "-", color=OKABE["red"], lw=1.2,
                label="FIRE class (recall)")
        ax.plot(st / 1000, ha, "-", color=OKABE["blue"], lw=1.2,
                label="HOLD class")
        ax.set_ylim(-0.05, 1.08)
        ax.set_yticks([0, 0.5, 1.0])
        ax.axhline(0.5, color=OKABE["grey"], lw=0.6, ls=":")
        ax.text(0.015, 0.90, labels[f], transform=ax.transAxes, ha="left",
                va="top", fontsize=8, color=OKABE["black"],
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.7))
        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("training step (k)")
        if i == 1:
            ax.set_ylabel("holdout accuracy")
            ax.legend(loc="lower right", fontsize=6.8, ncol=2,
                      handlelength=1.3, borderaxespad=0.2, columnspacing=1.0)
        if i == 0:
            ax.set_title("(a) oscillation onset as clairvoyant fraction grows",
                         fontsize=9, loc="left")

    # -------- right: phantom frontier across checkpoints -------------------
    axR = fig.add_subplot(gs[:, 1])
    cmap = plt.cm.viridis
    fs_plot = fs_all
    norm = plt.Normalize(min(fs_plot), max(fs_plot))
    for f in fs_plot:
        rs = [r for r in runs if abs(r["f"] - f) < 1e-9]
        xs, ys = [], []
        for r in rs:
            tail = r["trajectory"][len(r["trajectory"]) // 4:]  # drop warmup
            xs += [t["fire_acc"] for t in tail]
            ys += [t["hold_acc"] for t in tail]
        axR.scatter(xs, ys, s=7, color=cmap(norm(f)), alpha=0.55,
                    edgecolors="none", zorder=3)
    axR.plot([0, 1], [1, 0], color=OKABE["grey"], lw=0.7, ls="--", zorder=1)
    axR.set_xlim(-0.02, 1.04)
    axR.set_ylim(-0.02, 1.04)
    axR.set_xlabel("FIRE-class accuracy  (recall)")
    axR.set_ylabel("HOLD-class accuracy  (precision proxy)")
    axR.set_title("(b) apparent recall–precision frontier across checkpoints",
                  fontsize=9, loc="left")
    axR.annotate("f=0: one point,\ntop-right corner", xy=(0.985, 0.985),
                 xytext=(0.60, 0.70), fontsize=7, color=OKABE["green"],
                 arrowprops=dict(arrowstyle="->", color=OKABE["green"], lw=0.8))
    axR.text(0.05, 0.10, "high f: a phantom\nPareto cloud", fontsize=7,
             color=OKABE["red"])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axR, fraction=0.046, pad=0.02)
    cb.set_label("clairvoyant fraction $f$", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "exp5_toy.pdf")
    fig.savefig(OUT / "exp5_toy.png", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT/'exp5_toy.pdf'} and .png")


if __name__ == "__main__":
    main()
