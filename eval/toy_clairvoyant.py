#!/usr/bin/env python
"""exp-5: synthetic clairvoyant-fraction toy task (outside speech).

Thesis under test (see paper/main.tex around FIXME(exp-5) and
research/86-exp3-schema-ablation-exp4-external-baselines.md):

    Streaming supervision labels that depend on FUTURE tokens ("clairvoyant
    labels" = target leakage in time) manufacture (a) training OSCILLATION
    between two mode attractors and (b) an apparent recall-vs-precision
    "phantom frontier" across checkpoints. A single causal relabel removes both.

The four-gap speech work proves this in the ASR model. This toy reproduces the
SAME mechanism on a synthetic discrete stream, as a function of the clairvoyant
label fraction f, with no speech anywhere.

--------------------------------------------------------------------------------
Task (a faithful analog of the endpointing decision)
--------------------------------------------------------------------------------
A stream is a sequence of PHRASES separated by SILENCE runs ("gaps"):

    BOS  <phrase>  <gap>  <phrase>  <gap>  ...

* A phrase ends in a TERMINAL content token (semantically "complete") or a
  NON-TERMINAL one ("incomplete") -- 50/50. Completeness is a function of the
  prefix (the identity of the last content token), so it is always observable
  at decision time. This mirrors "the words so far are semantically complete".

* A gap is a run of SILENCE tokens. Its full length is drawn short or long
  (50/50). Long == the turn actually ended; short == the speaker resumes.
  The FULL gap length is only knowable AFTER the decision point, exactly like a
  live microphone that has not yet delivered the future.

At exactly one DECISION position per gap -- the g_min-th silence token, i.e. the
moment the causal silence threshold is first met -- the model must emit
FIRE vs HOLD from its causal prefix (masked self-attention; it cannot see the
rest of the gap).

Two labelling rules for that decision:

    CAUSAL      FIRE iff the phrase was complete           (prefix-computable)
    CLAIRVOYANT FIRE iff the gap turns out to be long      (uses future tokens)

completeness is drawn independently of gap length, so the clairvoyant rule is a
genuine in-time target leak: on a complete+short prefix it says HOLD where the
causal rule says FIRE (a continuation), and on an incomplete+long prefix it says
FIRE where the causal rule says HOLD (an abandoned turn). The two rules are
maximally opposed while each being internally consistent -- the paper's "pools
with incompatible optima".

A fraction f of TRAINING sequences are "clairvoyant-annotated" (all their
decisions use the clairvoyant rule); the rest use the causal rule. Assignment is
per-sequence so the contradictory labels arrive as coherent pools, mirroring
"some recordings were labelled by the offline pipeline". The HOLDOUT is ALWAYS
labelled causally.

Holdout metrics, logged every --eval-every steps:
    fire_acc = P(pred FIRE | causal-FIRE decision)   (recall on complete prefixes)
    hold_acc = P(pred HOLD | causal-HOLD decision)    (1 - false-fire, precision proxy)

At f=0 both climb monotonically to ~1 (single dominating checkpoint). As f grows
the two anti-correlate and swing across checkpoints (oscillation), tracing an
apparent recall-vs-precision frontier that is a single model swinging, not a
real capability trade-off.

Everything is seeded; runs in seconds on CPU. Do NOT point this at the GPU that
is running the big job -- default device is cpu.
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn

# ------------------------------------------------------------------ vocabulary
PAD = 0
BOS = 1
TERMINALS = [2, 3, 4, 5]        # content tokens that make a phrase "complete"
NONTERMINALS = [6, 7, 8, 9]     # content tokens that leave a phrase "incomplete"
SIL = 10
VOCAB = 11

# ------------------------------------------------------------------ task params
G_MIN = 2          # silence tokens that must be observed before a decision
G_SHORT = (2, 4)   # inclusive gap-length range for "resume" gaps  (>= G_MIN)
G_LONG = (5, 7)    # inclusive gap-length range for "turn-ended" gaps
P_MHRASE = (1, 4)  # phrase length range (content tokens)
N_GAPS = 6         # decisions per sequence
SEQ_LEN = 72       # padded stream length


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(4)  # tiny model: many threads only add contention


def make_sequence(rng: np.random.Generator):
    """Build one stream. Returns (tokens, decisions).

    decisions: list of (position, is_complete, is_long) at each gap's g_min-th
    silence token.
    """
    toks = [BOS]
    decisions = []
    for _ in range(N_GAPS):
        # ---- phrase
        plen = int(rng.integers(P_MHRASE[0], P_MHRASE[1] + 1))
        is_complete = bool(rng.integers(0, 2))
        for j in range(plen):
            last = j == plen - 1
            pool = TERMINALS if (last and is_complete) else NONTERMINALS
            toks.append(int(rng.choice(pool)))
        # ---- gap (full length observable only in the future)
        is_long = bool(rng.integers(0, 2))
        lo, hi = G_LONG if is_long else G_SHORT
        glen = int(rng.integers(lo, hi + 1))
        decision_idx = len(toks) + (G_MIN - 1)   # the g_min-th silence token
        for _ in range(glen):
            toks.append(SIL)
        decisions.append((decision_idx, is_complete, is_long))
        if len(toks) >= SEQ_LEN - 8:
            break
    toks = toks[:SEQ_LEN]
    decisions = [d for d in decisions if d[0] < SEQ_LEN]
    if len(toks) < SEQ_LEN:
        toks += [PAD] * (SEQ_LEN - len(toks))
    return toks, decisions


def build_pool(n_seq: int, f_clair: float, rng: np.random.Generator):
    """Return tensors: tokens[N,T], dpos/dcomplete/dlong/dclair as flat decision
    lists with a seq index, ready to gather logits at decision positions."""
    tokens = np.full((n_seq, SEQ_LEN), PAD, dtype=np.int64)
    seq_idx, pos, complete, long_ = [], [], [], []
    for i in range(n_seq):
        toks, decisions = make_sequence(rng)
        tokens[i] = toks
        clair_seq = rng.random() < f_clair
        for (p, c, l) in decisions:
            seq_idx.append(i)
            pos.append(p)
            complete.append(int(c))
            long_.append(int(l))
    seq_idx = np.array(seq_idx, np.int64)
    pos = np.array(pos, np.int64)
    complete = np.array(complete, np.int64)
    long_ = np.array(long_, np.int64)
    # per-sequence clairvoyant flag (coherent pools)
    clair_flag = (rng.random(n_seq) < f_clair)
    dec_clair = clair_flag[seq_idx]
    causal_label = complete                       # FIRE iff complete
    clair_label = long_                           # FIRE iff gap turns out long
    train_label = np.where(dec_clair, clair_label, causal_label)
    return {
        "tokens": torch.from_numpy(tokens),
        "seq_idx": torch.from_numpy(seq_idx),
        "pos": torch.from_numpy(pos),
        "complete": torch.from_numpy(complete),
        "causal_label": torch.from_numpy(causal_label),
        "train_label": torch.from_numpy(train_label.astype(np.int64)),
    }


class TinyStreamTransformer(nn.Module):
    def __init__(self, d_model=32, nhead=4, layers=2, ff=64):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(SEQ_LEN, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=ff, batch_first=True,
            dropout=0.0, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d_model, 2)
        cm = torch.triu(torch.full((SEQ_LEN, SEQ_LEN), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", cm)

    def forward(self, tokens):
        p = torch.arange(tokens.size(1), device=tokens.device)
        h = self.emb(tokens) + self.pos(p)[None]
        h = self.enc(h, mask=self.causal_mask)
        return self.head(h)   # [B, T, 2]


def gather_decisions(logits, seq_idx, pos):
    return logits[seq_idx, pos]   # [D, 2]


@torch.no_grad()
def evaluate(model, hold, device):
    model.eval()
    logits = model(hold["tokens"].to(device))
    dl = gather_decisions(logits, hold["seq_idx"], hold["pos"])
    pred = dl.argmax(-1).cpu()
    complete = hold["complete"]
    fire_mask = complete == 1
    hold_mask = complete == 0
    fire_acc = (pred[fire_mask] == 1).float().mean().item()
    hold_acc = (pred[hold_mask] == 0).float().mean().item()
    overall = (pred == hold["causal_label"]).float().mean().item()
    return fire_acc, hold_acc, overall


def train_run(f, seed, steps, eval_every, device, n_train=6000, n_hold=3000,
              batch=256, lr=2e-3):
    seed_all(seed)
    rng = np.random.default_rng(seed)
    train = build_pool(n_train, f, rng)
    # holdout labels are causal by construction (f=0)
    hold = build_pool(n_hold, 0.0, np.random.default_rng(seed + 777))

    model = TinyStreamTransformer().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    D = train["seq_idx"].shape[0]
    tok = train["tokens"].to(device)
    sidx = train["seq_idx"]
    tpos = train["pos"]
    tlab = train["train_label"].to(device)

    gen = torch.Generator().manual_seed(seed + 13)
    traj = []
    for step in range(steps + 1):
        if step % eval_every == 0:
            fa, ha, ov = evaluate(model, hold, device)
            traj.append({"step": step, "fire_acc": fa, "hold_acc": ha,
                         "overall": ov})
        if step == steps:
            break
        model.train()
        sel = torch.randint(0, D, (batch,), generator=gen)
        s = sidx[sel]
        p = tpos[sel]
        y = tlab[sel]
        # forward only the sequences we need (unique), remap decision -> row
        uniq, inv = torch.unique(s, return_inverse=True)
        logits = model(tok[uniq.to(device)])
        dl = logits[inv.to(device), p.to(device)]
        loss = lossf(dl, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return traj


def oscillation_stats(traj, warmup_frac=0.25):
    """Quantify swing over the post-warmup trajectory."""
    n = len(traj)
    tail = traj[int(n * warmup_frac):]
    fa = np.array([t["fire_acc"] for t in tail])
    ha = np.array([t["hold_acc"] for t in tail])
    diff = fa - ha
    # mode circulation: sign changes in (fire-hold), counting only swings whose
    # amplitude exceeds a margin so trivial near-1.0 float noise is not a "flip".
    margin = 0.15
    signs = np.sign(diff)
    signs[np.abs(diff) < margin] = 0
    signs = signs[signs != 0]
    flips = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
    return {
        "fire_std": float(fa.std()),
        "hold_std": float(ha.std()),
        "min_overall_tail": float(min(t["overall"] for t in tail)),
        "mode_flips": flips,
        "fire_hold_anticorr": float(np.corrcoef(fa, ha)[0, 1])
        if fa.std() > 1e-6 and ha.std() > 1e-6 else 0.0,
        "fire_range": [float(fa.min()), float(fa.max())],
        "hold_range": [float(ha.min()), float(ha.max())],
    }


def run_sweep(fs, seeds, steps, eval_every, device, lr, out_path):
    """Full clairvoyant-fraction sweep; writes the exp-5 results JSON."""
    runs = []
    for f in fs:
        for seed in seeds:
            traj = train_run(f, seed, steps, eval_every, device, lr=lr)
            stats = oscillation_stats(traj)
            runs.append({"f": f, "seed": seed, "trajectory": traj,
                         "stats": stats})
            last = traj[-1]
            print(f"f={f:.2f} seed={seed}  final fire={last['fire_acc']:.2f} "
                  f"hold={last['hold_acc']:.2f} overall={last['overall']:.2f} "
                  f"flips={stats['mode_flips']} fstd={stats['fire_std']:.3f} "
                  f"anticorr={stats['fire_hold_anticorr']:+.2f}", flush=True)

    # per-f aggregates (mean over seeds)
    by_f = {}
    for f in fs:
        rs = [r for r in runs if r["f"] == f]
        agg = lambda key: float(np.mean([r["stats"][key] for r in rs]))
        by_f[f"{f:.2f}"] = {
            "n_seeds": len(rs),
            "mean_fire_std": agg("fire_std"),
            "mean_hold_std": agg("hold_std"),
            "mean_mode_flips": float(np.mean([r["stats"]["mode_flips"] for r in rs])),
            "mean_fire_hold_anticorr": agg("fire_hold_anticorr"),
            "mean_min_overall_tail": agg("min_overall_tail"),
            "mean_final_fire": float(np.mean([r["trajectory"][-1]["fire_acc"] for r in rs])),
            "mean_final_hold": float(np.mean([r["trajectory"][-1]["hold_acc"] for r in rs])),
        }
    out = {
        "task": "synthetic clairvoyant-fraction toy (exp-5)",
        "config": {"fs": fs, "seeds": seeds, "steps": steps,
                   "eval_every": eval_every, "lr": lr,
                   "g_min": G_MIN, "g_short": G_SHORT, "g_long": G_LONG,
                   "n_gaps": N_GAPS, "seq_len": SEQ_LEN, "vocab": VOCAB},
        "by_f": by_f,
        "runs": runs,
    }
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true",
                    help="run the full f-sweep and write the exp-5 JSON")
    ap.add_argument("--f", type=float, default=0.0,
                    help="clairvoyant label fraction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--out", default=None, help="write single-run JSON here")
    args = ap.parse_args()

    if args.sweep:
        fs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.85, 1.0]
        seeds = [0, 1, 2]
        out = args.out or "research/94-exp5-toy-task.json"
        run_sweep(fs, seeds, args.steps, args.eval_every, args.device, args.lr,
                  out)
        return

    traj = train_run(args.f, args.seed, args.steps, args.eval_every, args.device,
                     n_train=args.n_train, lr=args.lr)
    stats = oscillation_stats(traj)
    result = {"f": args.f, "seed": args.seed, "steps": args.steps,
              "eval_every": args.eval_every, "trajectory": traj, "stats": stats}
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
    last = traj[-1]
    print(f"f={args.f} seed={args.seed}  final fire={last['fire_acc']:.2f} "
          f"hold={last['hold_acc']:.2f} overall={last['overall']:.2f}  "
          f"flips={stats['mode_flips']} fire_std={stats['fire_std']:.3f} "
          f"anticorr={stats['fire_hold_anticorr']:.2f}")


if __name__ == "__main__":
    main()
