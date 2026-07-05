"""Bootstrap / Wilson 95% CIs for the headline numbers (statistical rigor).

Replay (per-stretch bootstrap): resample stretches with replacement, recompute
boundary recall = sum(hits)/sum(turn_bounds) and false/min = sum(false_fires)/
sum(speech_min). Probes: Wilson CI for binary rates, per-item bootstrap for
premature/seq.

Usage:
    .venv/bin/python eval/bootstrap_ci.py --replay research/81-replay-*.json \
        --probe research/81-probe-*.json
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

RNG = np.random.default_rng(12345)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0, c - h), min(1, c + h))


def boot_replay(per, B=5000):
    tb = np.array([p.get("n_turn_bounds", 0) for p in per], float)
    hit = np.array([p.get("n_hits", 0) for p in per], float)
    ff = np.array([p.get("n_false_fires", 0) for p in per], float)
    spm = np.array([p.get("speech_s", 0) / 60.0 for p in per], float)
    n = len(per)
    rec, fpm = [], []
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        TB = tb[idx].sum()
        rec.append(hit[idx].sum() / TB if TB else 0)
        SP = spm[idx].sum()
        fpm.append(ff[idx].sum() / SP if SP else 0)
    return {
        "recall": (hit.sum() / tb.sum() if tb.sum() else 0,
                   float(np.percentile(rec, 2.5)), float(np.percentile(rec, 97.5))),
        "false_per_min": (ff.sum() / spm.sum() if spm.sum() else 0,
                          float(np.percentile(fpm, 2.5)), float(np.percentile(fpm, 97.5))),
        "n_stretches": n, "n_turn_bounds": int(tb.sum()),
    }


def boot_mean(vals, B=5000):
    v = np.array(vals, float); n = len(v)
    if n == 0:
        return (0, 0, 0)
    m = [v[RNG.integers(0, n, n)].mean() for _ in range(B)]
    return (float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", nargs="*", default=[])
    ap.add_argument("--probe", nargs="*", default=[])
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    report = {}

    for f in args.replay:
        d = json.load(open(f))
        per = d.get("per_stretch", [])
        r = boot_replay(per)
        report[Path(f).stem] = r
        print(f"\n[{Path(f).stem}] replay (n={r['n_stretches']} stretches, "
              f"{r['n_turn_bounds']} turn-final boundaries)")
        print(f"  recall        {r['recall'][0]:.3f}  95% CI [{r['recall'][1]:.3f}, {r['recall'][2]:.3f}]")
        print(f"  false/min     {r['false_per_min'][0]:.2f}  95% CI [{r['false_per_min'][1]:.2f}, {r['false_per_min'][2]:.2f}]")

    for f in args.probe:
        d = json.load(open(f))
        rep = {}
        if "digit" in d:
            per = d["digit"]["per_item"]
            prem = [p["n_premature"] for p in per]
            fr = sum(p["final_recall"] for p in per)
            m = boot_mean(prem)
            rec = wilson(fr, len(per))
            rep["digit_premature_per_seq"] = m
            rep["digit_final_recall"] = rec
            print(f"\n[{Path(f).stem}] digit probe (n={len(per)})")
            print(f"  premature/seq {m[0]:.2f}  95% CI [{m[1]:.2f}, {m[2]:.2f}]")
            print(f"  final recall  {rec[0]:.3f}  Wilson [{rec[1]:.3f}, {rec[2]:.3f}]")
        if "spelled" in d:
            per = d["spelled"]["per_item"]
            for kind in ("name", "email"):
                rows = [r for r in per if r["kind"] == kind]
                for cond in ("none", "profile"):
                    k = sum(r[f"hit_{cond}"] for r in rows)
                    w = wilson(k, len(rows))
                    rep[f"{kind}_{cond}"] = w
                    print(f"  {kind:5s} +{cond:7s} {w[0]:.3f}  Wilson [{w[1]:.3f}, {w[2]:.3f}]  (n={len(rows)})")
            intr = sum(r.get("intrusion_distractor", 0) for r in per)
            w = wilson(intr, len(per))
            rep["intrusion"] = w
            print(f"  intrusion     {w[0]:.3f}  Wilson [{w[1]:.3f}, {w[2]:.3f}]  (n={len(per)})")
        report[Path(f).stem] = rep

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
