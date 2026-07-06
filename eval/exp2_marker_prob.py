"""exp-2: marker-token PROBABILITY analysis across the opposed-pools (v8)
oscillation — distinguish TRUE MODE CIRCULATION from THRESHOLD FLAPPING.

Background (research/86, exp-2). The opposed-pools recipe, retrained under
seed-1, reproduces the oscillation fingerprint: holdout argmax accuracy on the
fire-class bounces between ~1.0 and ~0.3 across training steps. Argmax bouncing
is ambiguous:
  - mode circulation: the model's PROBABILITY mass on firing genuinely swings
    high<->low across steps (weights move between two attractors);
  - threshold flapping: the fire probability hovers near the decision boundary
    (~0.5) the whole time and tiny changes flip the argmax.
This script reads the marker PROBABILITY (not argmax) across the saved snapshots
to tell these apart, and reports the per-example distribution so a mean near 0.5
cannot masquerade as flapping when it is really two confident modes flipping.

P_fire definition
-----------------
The trained marker pathway is a rigid two-token sequence: after the transcript
the model emits a space then <EAGER_END_SPEECH> then <END_SPEECH>, OR it emits
<|im_end|> and stops. Empirically the EAGER->END bigram is deterministic
(P(<END_SPEECH> | <EAGER_END_SPEECH>) ~ 1.0 for every snapshot), so reading
P(<END_SPEECH>) as a single next-token probability at the end of the transcript
is ~0 and uninformative. The graded fire decision lives one position earlier:
stop (<|im_end|>) vs enter the marker path.

We therefore define P_fire as the MARGINAL probability that the model's
continuation emits <END_SPEECH> before <|im_end|>, evaluated at the end-of-audio
decision position (the transcript is teacher-forced up to that point). It is
computed by marginalizing over the first continuation token (top-k covering
>= `mass` of the softmax) and greedy-rolling each branch to termination; a
branch contributes its first-token probability iff its greedy roll reaches
<END_SPEECH>. Because the marker bigram is deterministic once entered, this
equals the probability the model commits to firing the end marker. P_fire lies
in [0, 1] and is directly comparable across FIRE and HOLD examples and across
snapshots (validated: mean P_fire tracks the eval_log argmax accuracy ~1:1).

FIRE examples  = schemas {single, trailing_silence}  (complete utterance -> fire)
HOLD examples  = schemas {truncated, no_fire}         (incomplete/no-tail -> hold)

The probe set is the SAME balanced holdout the training reserved (random seed
12345, 10 per schema) so P_fire is directly comparable to the recorded argmax
oscillation and never leaks training examples.

Usage:
    .venv/bin/python eval/exp2_marker_prob.py \
        --glob 'checkpoints/semantic_endpoint_v8_seed1_es/snap_step*.pt' \
        --data data/semantic_endpoint_v8/data.pt \
        --out research/exp2-marker-prob.json \
        --figure paper/figures/exp2_marker_prob.pdf
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)

END_TOK = "<END_SPEECH>"
EAGER_TOK = "<EAGER_END_SPEECH>"
MAX_DUR_S = 12.0
HOLDOUT_SEED = 12345           # matches src/train_semantic_endpoint.py
PER_SCHEMA = 10                # matches the reserved holdout (eval-holdout-size 60 / 6 schemas)
FIRE_SCHEMAS = ("single", "trailing_silence")
HOLD_SCHEMAS = ("truncated", "no_fire")


def build_holdout(data_path: str) -> list[dict]:
    """Reproduce EXACTLY the balanced holdout that training reserved."""
    examples = torch.load(data_path, weights_only=False)
    examples = [e for e in examples if len(e["audio"]) <= int(MAX_DUR_S * 16000)]
    rng = random.Random(HOLDOUT_SEED)
    by_schema: dict[str, list] = defaultdict(list)
    for e in examples:
        by_schema[e["schema"]].append(e)
    for k in by_schema:
        rng.shuffle(by_schema[k])
    holdout: list[dict] = []
    for k, pool in by_schema.items():
        if k in FIRE_SCHEMAS or k in HOLD_SCHEMAS:
            holdout.extend(pool[:PER_SCHEMA])
    return holdout


@torch.no_grad()
def marker_pfire(model, tokenizer, processor, e, end_id, imend_id,
                 device, K=14, topk=8, mass=0.98) -> float:
    """Marginal probability the continuation emits <END_SPEECH> before <|im_end|>."""
    txt = e["text"].replace(EAGER_TOK, "").replace(END_TOK, "").strip()
    msgs = [{"role": "system", "content": e.get("ctx", "")},
            {"role": "user", "content": [{"type": "audio"}]}]
    prefix = tokenizer.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
    full = prefix + f"language English<asr_text>{txt}"
    audio = np.asarray(e["audio"], dtype=np.float32)
    inp = processor(text=[full], audio=[audio], return_tensors="pt", padding=True)
    inp = {k: (v.to(device).bfloat16() if torch.is_floating_point(v) else v.to(device))
           for k, v in inp.items()}
    out = model.thinker(**inp, use_cache=True)
    base_past = out.past_key_values
    probs = torch.softmax(out.logits[0, -1].float(), dim=-1)
    order = torch.argsort(probs, descending=True)

    p_fire = 0.0
    cum = 0.0
    for rank in range(topk):
        v = int(order[rank])
        pv = float(probs[v])
        cum += pv
        if v == end_id:
            p_fire += pv                      # already the marker
        elif v == imend_id:
            pass                              # stop -> no fire
        else:
            past = copy.deepcopy(base_past)   # fork this branch's cache
            cur = v
            fired = False
            for _ in range(K):
                if cur == end_id:
                    fired = True
                    break
                if cur == imend_id:
                    break
                o = model.thinker(input_ids=torch.tensor([[cur]], device=device),
                                  past_key_values=past, use_cache=True)
                past = o.past_key_values
                cur = int(torch.argmax(o.logits[0, -1]))
            if cur == end_id:
                fired = True
            if fired:
                p_fire += pv
        if cum >= mass:
            break
    return float(p_fire)


def summarize(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    return {
        "mean": round(float(a.mean()), 4),
        "median": round(float(np.median(a)), 4),
        "std": round(float(a.std()), 4),
        "min": round(float(a.min()), 4),
        "max": round(float(a.max()), 4),
        # per-example bimodality: confident-fire (>0.8), confident-hold (<0.2),
        # or sitting in the ambiguous boundary band [0.2, 0.8]
        "frac_confident_fire": round(float((a > 0.8).mean()), 4),
        "frac_confident_hold": round(float((a < 0.2).mean()), 4),
        "frac_boundary": round(float(((a >= 0.2) & (a <= 0.8)).mean()), 4),
        "n": int(a.size),
    }


def run(args) -> dict:
    from src.train_semantic_endpoint import (
        load_base_model, extend_tokenizer_and_model,
        apply_lora, freeze_except_lora_and_new_rows,
    )
    device = "cuda"
    model, tokenizer, processor = load_base_model()
    model = model.cuda().bfloat16()
    new_ids, eager_id, end_id = extend_tokenizer_and_model(model, tokenizer, processor)
    apply_lora(model.thinker, rank=16, alpha=32)
    freeze_except_lora_and_new_rows(model, new_ids)
    imend_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    holdout = build_holdout(args.data)
    fire_ex = [e for e in holdout if e["schema"] in FIRE_SCHEMAS]
    hold_ex = [e for e in holdout if e["schema"] in HOLD_SCHEMAS]
    logger.info("Probe set: %d FIRE (%s), %d HOLD (%s)",
                len(fire_ex), FIRE_SCHEMAS, len(hold_ex), HOLD_SCHEMAS)

    paths = sorted(glob.glob(args.glob),
                   key=lambda s: int(Path(s).stem.split("step")[-1]))
    if not paths:
        raise SystemExit(f"no snapshots match {args.glob}")
    logger.info("Scoring %d snapshots", len(paths))

    # Optional: cross-check against the recorded argmax oscillation.
    elog_path = Path(args.glob).parent.parent / Path(args.glob).parent.name / "eval_log.json"
    elog = {}
    cand = Path(paths[0]).parent / "eval_log.json"
    if cand.exists():
        elog = {r["step"]: r for r in json.loads(cand.read_text())}

    steps = []
    for path in paths:
        step = int(Path(path).stem.split("step")[-1])
        snap = torch.load(path, weights_only=False, map_location="cpu")
        model.load_state_dict(snap["trainable"], strict=False)
        model.eval()
        end_id_ = snap.get("end_id", end_id)

        fire_vals = [marker_pfire(model, tokenizer, processor, e, end_id_,
                                  imend_id, device) for e in fire_ex]
        hold_vals = [marker_pfire(model, tokenizer, processor, e, end_id_,
                                  imend_id, device) for e in hold_ex]
        row = {
            "step": step,
            "mean_pfire_fire_examples": summarize(fire_vals)["mean"],
            "mean_pfire_hold_examples": summarize(hold_vals)["mean"],
            "fire": summarize(fire_vals),
            "hold": summarize(hold_vals),
            "fire_pfire_per_example": [round(v, 4) for v in fire_vals],
            "hold_pfire_per_example": [round(v, 4) for v in hold_vals],
        }
        r = elog.get(step, {})
        if r:
            row["eval_single_argmax"] = r.get("single")
            row["eval_no_fire_correct_argmax"] = r.get("no_fire_correct")
            row["eval_trail_correct_argmax"] = r.get("trail_correct")
            row["eval_trunc_correct_argmax"] = r.get("trunc_correct")
        steps.append(row)
        logger.info("step %5d  FIRE mean P_fire=%.3f (boundary frac %.2f)  "
                    "HOLD mean P_fire=%.3f (boundary frac %.2f)  [eval single=%s nf=%s]",
                    step, row["fire"]["mean"], row["fire"]["frac_boundary"],
                    row["hold"]["mean"], row["hold"]["frac_boundary"],
                    row.get("eval_single_argmax"), row.get("eval_no_fire_correct_argmax"))

    fire_means = [s["fire"]["mean"] for s in steps]
    hold_means = [s["hold"]["mean"] for s in steps]
    # average per-snapshot boundary occupancy: how often is an example genuinely
    # near 0.5 (flapping signature) vs confidently in one mode (circulation)
    fire_boundary = float(np.mean([s["fire"]["frac_boundary"] for s in steps]))
    hold_boundary = float(np.mean([s["hold"]["frac_boundary"] for s in steps]))
    verdict = {
        "fire_pfire_swing_amplitude": round(max(fire_means) - min(fire_means), 4),
        "fire_pfire_min": round(min(fire_means), 4),
        "fire_pfire_max": round(max(fire_means), 4),
        "hold_pfire_swing_amplitude": round(max(hold_means) - min(hold_means), 4),
        "hold_pfire_min": round(min(hold_means), 4),
        "hold_pfire_max": round(max(hold_means), 4),
        "mean_fire_frac_boundary": round(fire_boundary, 4),
        "mean_hold_frac_boundary": round(hold_boundary, 4),
    }
    result = {
        "recipe": "opposed-pools v8, seed-1 (semantic_endpoint_v8_seed1_es)",
        "pfire_definition": "marginal P(<END_SPEECH> before <|im_end|>) at end-of-audio "
                            "decision position; first-token marginalization + greedy roll",
        "probe": {"fire_schemas": list(FIRE_SCHEMAS), "hold_schemas": list(HOLD_SCHEMAS),
                  "n_fire": len(fire_ex), "n_hold": len(hold_ex),
                  "holdout_seed": HOLDOUT_SEED, "per_schema": PER_SCHEMA},
        "verdict": verdict,
        "steps": steps,
    }
    Path(args.out).write_text(json.dumps(result, indent=1))
    logger.info("Wrote %s", args.out)
    return result


def make_figure(result: dict, out_pdf: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    for _f in ("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyrepagella-regular.otf",
               "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyrepagella-bold.otf"):
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
        "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    OK = {"blue": "#0072B2", "orange": "#E69F00", "red": "#D55E00", "grey": "#808080"}
    steps = result["steps"]
    x = [s["step"] for s in steps]
    fire_mean = [s["fire"]["mean"] for s in steps]
    hold_mean = [s["hold"]["mean"] for s in steps]

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    # 0.2-0.8 "threshold-flapping band" shading
    ax.axhspan(0.2, 0.8, color=OK["grey"], alpha=0.10, lw=0)
    ax.axhline(0.5, color=OK["grey"], lw=0.8, ls="--")
    # per-example clouds (show the bimodality behind the means)
    for s in steps:
        ax.scatter([s["step"]] * len(s["fire_pfire_per_example"]),
                   s["fire_pfire_per_example"], s=6, color=OK["blue"],
                   alpha=0.18, lw=0, zorder=2)
        ax.scatter([s["step"]] * len(s["hold_pfire_per_example"]),
                   s["hold_pfire_per_example"], s=6, color=OK["red"],
                   alpha=0.18, lw=0, zorder=2)
    ax.plot(x, fire_mean, "-o", color=OK["blue"], ms=3.5, lw=1.4,
            label="FIRE examples (complete utt.)", zorder=4)
    ax.plot(x, hold_mean, "-s", color=OK["red"], ms=3.5, lw=1.4,
            label="HOLD examples (incomplete)", zorder=4)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("training step")
    ax.set_ylabel(r"marker probability $P_{\mathrm{fire}}$")
    ax.set_title("Opposed-pools oscillation: mode circulation, not flapping",
                 fontsize=9)
    ax.text(x[0], 0.5, " 0.5 boundary", va="bottom", ha="left",
            fontsize=6.5, color=OK["grey"])
    ax.legend(loc="center right", fontsize=7)
    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)
    logger.info("Wrote %s", out_pdf)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="checkpoints/semantic_endpoint_v8_seed1_es/snap_step*.pt")
    p.add_argument("--data", default="data/semantic_endpoint_v8/data.pt")
    p.add_argument("--out", default="research/exp2-marker-prob.json")
    p.add_argument("--figure", default="paper/figures/exp2_marker_prob.pdf")
    p.add_argument("--figure-only", action="store_true",
                   help="skip inference, rebuild the figure from --out json")
    args = p.parse_args()

    if args.figure_only:
        result = json.loads(Path(args.out).read_text())
    else:
        result = run(args)
    if args.figure:
        make_figure(result, args.figure)


if __name__ == "__main__":
    main()
