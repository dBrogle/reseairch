"""Extract numeric values from raw responses for the Cognitive Biases study.

Each scenario asks for a JSON object containing a `value_field`. The
extractor handles three value_kinds:

  - "numeric"           — read value_field as number
  - "categorical"       — read value_field as string, map via
                          scenario.category_to_score (case-insensitive)
  - "ranking_position"  — read value_field as list, return 1-based
                          position of first element matching any of
                          scenario.target_aliases (case-insensitive
                          substring match)

Direct JSON parsing handles most responses cleanly. Whatever's left
falls through to a batched LLM extractor that's given per-item
instructions.

Extraction results are cached so changing analysis logic doesn't force a
re-run of the underlying API calls.
"""

import asyncio
import json
import random

from services.llm import OpenRouterProvider
from studies.cognitive_biases.config import (
    EXTRACTION_BATCH_SIZE,
    EXTRACTOR_MODEL,
    MAX_PARALLEL_REQUESTS,
)
from studies.cognitive_biases.cache import (
    get_extraction,
    get_responses,
    load_extraction_cache,
    load_response_cache,
    request_signature,
    set_extraction,
)
from studies.cognitive_biases.scenarios.base import Scenario

ERROR_EXTRACTION   = {"reasoning": "ERROR",   "value": None}
REFUSED_EXTRACTION = {"reasoning": "REFUSED", "value": None}


# ---------------------------------------------------------------------------
# Direct JSON parse
# ---------------------------------------------------------------------------

def _strip_codefence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    return text.strip()


def _extract_json_object(text: str) -> dict | None:
    """Pull the first/last balanced { ... } and parse as JSON."""
    text = _strip_codefence(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_number(raw) -> float | None:
    """Best-effort numeric coercion. Strips currency symbols, commas, %."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        cleaned = (
            raw.replace(",", "").replace("€", "").replace("$", "")
               .replace("£", "").replace("%", "").strip()
        )
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _categorical_to_score(
    raw, mapping: dict[str, float]
) -> float | None:
    """Map a categorical value to its score using case-insensitive lookup."""
    if not isinstance(raw, str):
        return None
    key = raw.strip().lower()
    for cat, score in mapping.items():
        if cat.strip().lower() == key:
            return float(score)
    return None


def _ranking_position(
    raw, aliases: tuple[str, ...]
) -> float | None:
    """1-based position of the first list element whose lowercased text
    contains any of `aliases` (also lowercased). None if not a list, or
    no element matches."""
    if not isinstance(raw, list):
        return None
    lowered_aliases = tuple(a.lower() for a in aliases)
    for i, item in enumerate(raw):
        text = (item if isinstance(item, str) else json.dumps(item)).lower()
        if any(a in text for a in lowered_aliases):
            return float(i + 1)
    return None


def try_direct_parse(scenario: Scenario, response_text: str) -> dict | None:
    """Try to parse the response and pull out the scenario's value."""
    parsed = _extract_json_object(response_text)
    if parsed is None or scenario.value_field not in parsed:
        return None

    raw = parsed[scenario.value_field]
    reasoning = str(parsed.get("reasoning", ""))

    if scenario.value_kind == "numeric":
        value = _coerce_number(raw)
    elif scenario.value_kind == "categorical":
        value = _categorical_to_score(raw, scenario.category_to_score or {})
    elif scenario.value_kind == "ranking_position":
        value = _ranking_position(raw, scenario.target_aliases or ())
    else:
        return None

    if value is None:
        return None

    return {"reasoning": reasoning, "value": value}


# ---------------------------------------------------------------------------
# Batched LLM fallback extractor
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = (
    "You are a data extractor. You will receive a JSON object containing "
    "a batch of items keyed by index. Each item gives:\n"
    "  - the LLM's free-form response (raw text)\n"
    "  - a per-item `instruction` describing how to compute a single "
    "numeric value from that response\n\n"
    "For each item, follow its instruction exactly and return a number. "
    'If the LLM clearly refused, output value -1 and reasoning "REFUSED". '
    "If the response is unparseable or no number can be computed, output "
    'value -1 and reasoning "ERROR".\n\n'
    "Respond with ONLY a JSON object mapping each index to:\n"
    '  "reasoning": string,\n'
    '  "value": number\n\n'
    'Example: {"0": {"reasoning": "...", "value": 187888}, '
    '"1": {"reasoning": "REFUSED", "value": -1}}\n\n'
    "Return ONLY the JSON object, nothing else."
)


def _instruction_for(scenario: Scenario) -> str:
    """Per-scenario natural-language extraction instruction."""
    vf = scenario.value_field
    if scenario.value_kind == "numeric":
        return (
            f"Extract the value of '{vf}' from the response (as a number "
            f"of {scenario.value_unit}). Strip commas / currency / "
            "percent signs. Return the number."
        )
    if scenario.value_kind == "categorical":
        mapping = scenario.category_to_score or {}
        pairs = ", ".join(f'"{k}"={v}' for k, v in mapping.items())
        return (
            f"Extract the value of '{vf}' from the response and map it "
            f"using this table (case-insensitive): {pairs}. Return the "
            "mapped number. If the value isn't in the table, return -1."
        )
    if scenario.value_kind == "ranking_position":
        aliases = ", ".join(f'"{a}"' for a in (scenario.target_aliases or ()))
        return (
            f"Look at the '{vf}' list in the response. Return the "
            "1-based position of the FIRST list element whose text "
            f"mentions any of these aliases (case-insensitive substring): "
            f"{aliases}. If no element matches, return -1."
        )
    return f"Extract the value of '{vf}' as a number."


def _build_batch_input(items: list[dict]) -> str:
    return json.dumps(
        {str(i): item for i, item in enumerate(items)},
        indent=2,
    )


def _parse_batch_output(raw: str, batch_size: int) -> dict[int, dict]:
    text = _strip_codefence(raw)
    parsed = json.loads(text)
    out: dict[int, dict] = {}
    for idx_str, data in parsed.items():
        try:
            idx = int(idx_str)
        except (ValueError, TypeError):
            continue
        if idx < 0 or idx >= batch_size:
            continue
        if not isinstance(data, dict):
            out[idx] = ERROR_EXTRACTION.copy()
            continue
        reasoning = str(data.get("reasoning", ""))
        value = _coerce_number(data.get("value"))
        if value is None or value == -1:
            if reasoning == "REFUSED":
                out[idx] = {"reasoning": "REFUSED", "value": None}
            else:
                out[idx] = {"reasoning": reasoning or "ERROR", "value": None}
        else:
            out[idx] = {"reasoning": reasoning, "value": value}
    return out


async def _extract_batch_llm(
    provider: OpenRouterProvider,
    items: list[dict],
    cost_tracker=None,
) -> list[dict]:
    if not items:
        return []

    indices = list(range(len(items)))
    random.shuffle(indices)
    shuffled = [items[i] for i in indices]
    batch_json = _build_batch_input(shuffled)

    try:
        result, usage = await provider.complete_text_with_usage(
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                {"role": "user",   "content": batch_json},
            ],
            model=EXTRACTOR_MODEL,
            temperature=0.0,
            max_tokens=2000,
        )
        if cost_tracker and usage:
            await cost_tracker.record(EXTRACTOR_MODEL, usage)
        parsed = _parse_batch_output(result, len(shuffled))
    except Exception:
        return [ERROR_EXTRACTION.copy() for _ in items]

    answers = [ERROR_EXTRACTION.copy() for _ in items]
    for shuffled_idx, original_idx in enumerate(indices):
        answers[original_idx] = parsed.get(shuffled_idx, ERROR_EXTRACTION.copy())
    return answers


# ---------------------------------------------------------------------------
# Drive extraction across all cached responses
# ---------------------------------------------------------------------------

async def extract_all(
    provider: OpenRouterProvider,
    models: list[str],
    scenarios: list[Scenario],
    temperature: float,
    iterations: int,
    cost_tracker=None,
):
    """Walk every cached response; ensure each has a parsed extraction."""
    leftover_jobs: list[tuple[str, Scenario, str, str]] = []
    extraction_caches: dict[str, dict] = {m: load_extraction_cache(m) for m in models}

    for model in models:
        response_cache = load_response_cache(model)
        for scenario in scenarios:
            for arm in scenario.arms:
                sig = request_signature(scenario, arm)
                responses = get_responses(response_cache, sig, temperature)[:iterations]
                for r in responses:
                    if r.get("error") is not None or r.get("response") is None:
                        continue
                    response_text = r["response"]
                    existing = get_extraction(
                        extraction_caches[model],
                        scenario.id, arm.key, response_text,
                        scenario.value_field, EXTRACTOR_MODEL,
                    )
                    if existing is not None:
                        continue

                    direct = try_direct_parse(scenario, response_text)
                    if direct is not None:
                        set_extraction(
                            extraction_caches[model], model,
                            scenario.id, arm.key, response_text,
                            scenario.value_field, EXTRACTOR_MODEL,
                            direct,
                        )
                    else:
                        leftover_jobs.append(
                            (model, scenario, arm.key, response_text)
                        )

    if not leftover_jobs:
        print("\n  All extractions resolved via direct JSON parse.")
        return

    num_batches = (len(leftover_jobs) + EXTRACTION_BATCH_SIZE - 1) // EXTRACTION_BATCH_SIZE
    print(
        f"\n  {len(leftover_jobs)} response(s) need LLM extraction in "
        f"{num_batches} batch(es) of up to {EXTRACTION_BATCH_SIZE}."
    )

    if cost_tracker:
        cost_tracker.start_phase()

    batches = [
        leftover_jobs[i : i + EXTRACTION_BATCH_SIZE]
        for i in range(0, len(leftover_jobs), EXTRACTION_BATCH_SIZE)
    ]

    cache_locks: dict[str, asyncio.Lock] = {m: asyncio.Lock() for m in models}
    semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)
    done_batches = 0
    done_lock = asyncio.Lock()

    async def _run_batch(batch: list[tuple[str, Scenario, str, str]]):
        nonlocal done_batches

        items = [
            {
                "instruction": _instruction_for(scenario),
                "response":    response_text,
            }
            for _, scenario, _, response_text in batch
        ]

        async with semaphore:
            answers = await _extract_batch_llm(provider, items, cost_tracker=cost_tracker)

        for (model, scenario, arm_key, response_text), answer in zip(batch, answers):
            async with cache_locks[model]:
                set_extraction(
                    extraction_caches[model], model,
                    scenario.id, arm_key, response_text,
                    scenario.value_field, EXTRACTOR_MODEL,
                    answer,
                )

        async with done_lock:
            done_batches += 1
            sample = ", ".join(str(a.get("value", "?")) for a in answers[:5])
            cost_str = (
                f" | {cost_tracker.format_status(done_batches, num_batches)}"
                if cost_tracker else ""
            )
            print(
                f"  [batch {done_batches}/{num_batches}] {len(batch)} items "
                f"-> [{sample}{', ...' if len(answers) > 5 else ''}]{cost_str}"
            )

    await asyncio.gather(*[_run_batch(b) for b in batches])

    if cost_tracker and len(leftover_jobs) > 0:
        print(f"\n  Extractor cost: {cost_tracker.format_total()}")
