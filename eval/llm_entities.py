"""LLM-based named-entity extraction for ASR context-biasing hotwords.

Replaces the dated spaCy `en_core_web_sm` NER (which mislabels domain proper
nouns — tickers, product names, mixed-case org names) with an LLM. Used by
eval/earnings22_biasing.py and worker_integration/e4_pinned_ctx.py to build
the RELEVANT/DISTRACTOR context prefixes.

Primary backend: Anthropic Claude Haiku 4.5 (cheap/fast, structured output).
Fallback: Google Gemini (GEMINI_API_KEY) if the Anthropic call fails.
Batched (~25 transcripts/call) and disk-cached by text hash so re-runs and
the E4 sweep don't re-pay for the same utterances.
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5"  # user opted out of Opus for cost; Haiku fits extraction
BATCH = 25

_INSTR = (
    "You extract named entities from earnings-call speech transcripts to use as "
    "ASR hotwords (context biasing). For each numbered transcript, list the "
    "proper nouns a speech recognizer should be biased toward: person names, "
    "company/organization names, product/brand names, stock tickers, and place "
    "names. Include domain jargon only if it is a specific proper noun. EXCLUDE "
    "common words, generic nouns, numbers, and filler. Return each entity as it "
    "appears in the transcript (original casing). If a transcript has no such "
    "entity, return an empty list for it."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "entities": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "entities"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def _cache_key(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()


def _build_prompt(batch: list[str]) -> str:
    lines = [_INSTR, "", "Transcripts:"]
    for i, t in enumerate(batch):
        lines.append(f"[{i}] {t}")
    return "\n".join(lines)


def _anthropic_batch(batch: list[str]) -> list[list[str]]:
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    resp = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        messages=[{"role": "user", "content": _build_prompt(batch)}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    data = json.loads(text)
    return _align(data.get("results", []), len(batch))


def _gemini_batch(batch: list[str]) -> list[list[str]]:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = _build_prompt(batch) + (
        '\n\nReturn ONLY JSON: {"results": [{"id": <int>, "entities": [<str>...]}, ...]}'
    )
    resp = model.generate_content(
        prompt, generation_config={"response_mime_type": "application/json"})
    data = json.loads(resp.text)
    return _align(data.get("results", []), len(batch))


def _align(results: list[dict], n: int) -> list[list[str]]:
    """Map id-tagged results back to input order; missing ids → empty."""
    by_id: dict[int, list[str]] = {}
    for r in results:
        try:
            by_id[int(r["id"])] = [str(e).strip() for e in r.get("entities", []) if str(e).strip()]
        except (KeyError, ValueError, TypeError):
            continue
    return [by_id.get(i, []) for i in range(n)]


def extract_entities_llm(texts: list[str], cache_path: str | Path | None = None
                         ) -> list[list[str]]:
    """Return one entity list per input text. Cached + batched.

    Tries Anthropic Haiku 4.5; on any error for a batch, falls back to Gemini;
    if both fail, that batch yields empty lists (logged) rather than crashing.
    """
    cache: dict[str, list[str]] = {}
    cpath = Path(cache_path) if cache_path else None
    if cpath and cpath.exists():
        cache = json.loads(cpath.read_text())

    out: list[list[str] | None] = [None] * len(texts)
    todo_idx, todo_txt = [], []
    for i, t in enumerate(texts):
        k = _cache_key(t)
        if k in cache:
            out[i] = cache[k]
        else:
            todo_idx.append(i)
            todo_txt.append(t)

    logger.info("LLM entities: %d texts, %d cached, %d to extract (%s)",
                len(texts), len(texts) - len(todo_txt), len(todo_txt), MODEL)

    for start in range(0, len(todo_txt), BATCH):
        chunk_idx = todo_idx[start:start + BATCH]
        chunk_txt = todo_txt[start:start + BATCH]
        try:
            ents = _anthropic_batch(chunk_txt)
        except Exception as e:  # noqa: BLE001
            logger.warning("Anthropic batch failed (%s); trying Gemini", e)
            try:
                ents = _gemini_batch(chunk_txt)
            except Exception as e2:  # noqa: BLE001
                logger.error("Gemini batch also failed (%s); empty for %d texts",
                             e2, len(chunk_txt))
                ents = [[] for _ in chunk_txt]
        for j, i in enumerate(chunk_idx):
            out[i] = ents[j]
            cache[_cache_key(texts[i])] = ents[j]
        if cpath:
            cpath.parent.mkdir(parents=True, exist_ok=True)
            cpath.write_text(json.dumps(cache))

    return [o if o is not None else [] for o in out]
