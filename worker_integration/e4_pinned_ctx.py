"""E4: does context biasing survive session length? (pinned-CTX sink, app level)

The novel serving claim (research/60): a session runs for minutes; the
`<CTX>` prefix (hotwords) must keep biasing the CURRENT utterance even
though a growing span of audio history sits between the prefix and the
audio now being transcribed. The Metronome in-engine windowed-KV pins the
first S tokens (the CTX block) so they stay attended while audio scrolls
out. The Metronome paper notes application-level bounded re-feed "achieves
the same memory horizon" — so this driver tests the CLAIM at the app layer
(what the shipping product uses); the in-engine Triton-kernel pin is the
efficiency refinement that avoids re-encoding CTX every frame.

Protocol: for each entity-bearing Earnings-22 utterance, prepend `age`
seconds of unrelated filler speech, then bounded-re-feed the last
`window_s` seconds (target at the end) — so `age` seconds of audio sit
between the pinned CTX and the target's relevant words. Transcribe with
  PIN   : system prompt = the target's hotwords (CTX pinned, re-fed)
  NOPIN : empty system prompt (baseline)
and measure entity recall vs age. If PIN's advantage over NOPIN holds as
age grows, biasing survives session length; if it decays, the prefix loses
grip across intervening audio.

Runs under the project .venv (python3.11, qwen_asr transformers backend) —
NOT vLLM. E4 measures biasing ACCURACY (a model-behavior property), not the
serving path, so it uses the same working transcription path as
eval/earnings22_biasing (where spaCy en_core_web_sm is installed). This
also sidesteps vLLM's CUDA-fork startup and keeps the metric clean of any
GPU co-tenant contention.

Usage:
    .venv/bin/python -m worker_integration.e4_pinned_ctx \
        --n-utts 60 --out research/69-e4-pinned-ctx.json
"""
from __future__ import annotations
import argparse, json, logging, random
from pathlib import Path

import numpy as np

from eval.earnings22_biasing import (
    iter_earnings22, extract_entities, build_system_prompt, entity_recall,
    decode_audio, DEFAULT_DATA_GLOB, Transcriber,
)

logger = logging.getLogger(__name__)
SR = 16000


def to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(-1)
    if sr != SR:
        n = int(round(len(audio) * SR / sr))
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                          np.arange(len(audio)), audio).astype("float32")
    return audio.astype("float32")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument("--backend", default="pkg", choices=["pkg", "native"])
    p.add_argument("--data-glob", default=DEFAULT_DATA_GLOB)
    p.add_argument("--n-utts", type=int, default=60)
    p.add_argument("--window-s", type=float, default=16.0)
    p.add_argument("--ages", default="0,4,8,12")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="research/69-e4-pinned-ctx.json")
    args = p.parse_args()
    rng = random.Random(args.seed)
    ages = [float(a) for a in args.ages.split(",")]

    # Decode a candidate pool, then batch-extract entities with the LLM
    # (Haiku 4.5) rather than the dated spaCy NER. Entity-bearing clips of the
    # right length become targets; the rest are filler.
    from eval.llm_entities import extract_entities_llm
    cand = []
    for u in iter_earnings22(args.data_glob, args.n_utts * 8, shuffle=True, seed=args.seed):
        au = to_16k(*decode_audio(u["audio_bytes"]))
        cand.append({"audio": au, "ref": u["ref"], "dur": len(au) / SR})
        if len(cand) >= args.n_utts * 8:
            break
    ent_lists = extract_entities_llm([c["ref"] for c in cand],
                                     cache_path="data/earnings22/llm_entities.json")
    targets, fillers = [], []
    for c, ents in zip(cand, ent_lists):
        if ents and 1.0 <= c["dur"] <= 8.0 and len(targets) < args.n_utts:
            targets.append({"audio": c["audio"], "ref": c["ref"], "entities": ents})
        elif 1.0 <= c["dur"] <= 6.0 and len(fillers) < 60:
            fillers.append(c["audio"])
    logger.info("E4: %d entity-bearing targets, %d filler clips (LLM entities)",
                len(targets), len(fillers))

    def prior_audio(age_s: float) -> np.ndarray:
        if age_s <= 0 or not fillers:
            return np.zeros(0, dtype="float32")
        buf = []
        tot = 0.0
        while tot < age_s:
            f = fillers[rng.randrange(len(fillers))]
            buf.append(f)
            tot += len(f) / SR
        a = np.concatenate(buf)
        return a[-int(age_s * SR):]  # exactly age_s seconds

    tx = Transcriber(backend=args.backend, model_name=args.model)

    win = int(args.window_s * SR)
    per = []
    for i, t in enumerate(targets):
        rec = {"ref": t["ref"], "entities": t["entities"], "by_age": {}}
        for age in ages:
            full = np.concatenate([prior_audio(age), t["audio"]])
            windowed = full[-win:] if len(full) > win else full
            outs = {}
            for arm, ctx in [("pin", build_system_prompt(t["entities"])), ("nopin", "")]:
                hyp = tx.transcribe((windowed, SR), context=ctx)
                hits, tot = entity_recall(hyp, t["entities"])
                outs[arm] = {"hyp": hyp, "hits": hits, "total": tot}
            rec["by_age"][str(age)] = outs
        per.append(rec)
        if (i + 1) % 10 == 0:
            logger.info("  %d/%d targets", i + 1, len(targets))

    # Aggregate recall vs age, per arm
    summary = {"n_targets": len(per), "window_s": args.window_s, "ages": ages, "by_age": {}}
    for age in ages:
        a = str(age)
        agg = {}
        for arm in ("pin", "nopin"):
            hits = sum(r["by_age"][a][arm]["hits"] for r in per)
            tot = sum(r["by_age"][a][arm]["total"] for r in per)
            agg[arm + "_recall"] = hits / max(1, tot)
        agg["pin_minus_nopin_pp"] = 100 * (agg["pin_recall"] - agg["nopin_recall"])
        summary["by_age"][a] = agg
    logger.info("== E4 PINNED-CTX: recall vs session age ==")
    logger.info("  age_s   pin    nopin   gap_pp")
    for age in ages:
        a = summary["by_age"][str(age)]
        logger.info("  %5.0f  %.3f  %.3f   %+.1f", age, a["pin_recall"],
                    a["nopin_recall"], a["pin_minus_nopin_pp"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"model": args.model, "summary": summary,
                                          "per_target": per}, indent=2))
    logger.info("Saved %s", args.out)
    print("E4_DONE")


if __name__ == "__main__":
    main()
