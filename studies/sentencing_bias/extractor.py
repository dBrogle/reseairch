"""LLM-powered answer extraction for the Sentencing Bias study.

Takes raw LLM responses and extracts structured sentencing data:
thoughts, fine_usd, and jail_years. Uses a separate LLM call for
robustness when the original response isn't clean JSON.
Batched and shuffled to prevent positional bias.
"""

import asyncio
import json
import random

from services.llm import OpenRouterProvider
from studies.sentencing_bias.config import (
    DEFENDANTS,
    CRIMES,
    TEMPERATURE,
    ITERATIONS,
    EXTRACTOR_MODEL,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_BATCH_SIZE,
    MAX_PARALLEL_REQUESTS,
)
from studies.sentencing_bias.runner import build_messages
from studies.sentencing_bias.cache import (
    load_response_cache,
    get_responses,
    load_extraction_cache,
    get_extraction,
    set_extraction,
)

ERROR_EXTRACTION = {"thoughts": "ERROR", "fine_usd": -1, "jail_years": -1}
REFUSED_EXTRACTION = {"thoughts": "REFUSED", "fine_usd": -1, "jail_years": -1}


def _try_direct_parse(response_text: str) -> dict | None:
    """Try to parse the response directly as JSON first."""
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "consequence" in parsed:
            consequence = parsed["consequence"]
            if "fine_usd" in consequence and "jail_years" in consequence:
                return {
                    "thoughts": str(parsed.get("thoughts", "")),
                    "fine_usd": int(consequence["fine_usd"]),
                    "jail_years": float(consequence["jail_years"]),
                }
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        pass
    return None


def _build_batch_input(items: list[tuple[dict, dict, str]]) -> str:
    """Build the JSON input for a batch of (defendant, crime, response_text) triples."""
    batch = {}
    for idx, (defendant, crime, response_text) in enumerate(items):
        scenario = crime["scenario"].format(name=defendant["name"])
        batch[str(idx)] = {
            "prompt": f"{scenario} Found guilty of {crime['charge']}.",
            "response": response_text,
        }
    return json.dumps(batch, indent=2)


def _parse_batch_output(raw: str, batch_size: int) -> dict[int, dict]:
    """Parse the extractor's JSON response into {index: {thoughts, fine_usd, jail_years}}."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

    parsed = json.loads(text)
    results = {}
    for idx_str, data in parsed.items():
        idx = int(idx_str)
        if idx < 0 or idx >= batch_size:
            continue
        try:
            thoughts = str(data.get("thoughts", ""))
            fine_usd = int(data.get("fine_usd", -1))
            jail_years = float(data.get("jail_years", -1))
            results[idx] = {
                "thoughts": thoughts,
                "fine_usd": fine_usd,
                "jail_years": jail_years,
            }
        except (ValueError, TypeError, AttributeError):
            results[idx] = ERROR_EXTRACTION.copy()
    return results


async def extract_batch(
    provider: OpenRouterProvider,
    items: list[tuple[dict, dict, str]],
) -> list[dict]:
    """Extract sentencing data for a batch of (defendant, crime, response_text) triples."""
    if not items:
        return []

    # First try direct parsing for each item
    results = [None] * len(items)
    needs_extraction = []
    for i, (defendant, crime, response_text) in enumerate(items):
        parsed = _try_direct_parse(response_text)
        if parsed is not None:
            results[i] = parsed
        else:
            needs_extraction.append(i)

    if not needs_extraction:
        return results

    # Batch the remaining through the extractor LLM
    ext_items = [items[i] for i in needs_extraction]
    indices = list(range(len(ext_items)))
    random.shuffle(indices)
    shuffled = [ext_items[i] for i in indices]

    batch_json = _build_batch_input(shuffled)

    try:
        result = await provider.complete_text(
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": batch_json},
            ],
            model=EXTRACTOR_MODEL,
            temperature=0.0,
            max_tokens=1000,
        )
        parsed = _parse_batch_output(result, len(shuffled))
    except Exception:
        for i in needs_extraction:
            if results[i] is None:
                results[i] = ERROR_EXTRACTION.copy()
        return results

    # Unshuffle and assign
    for shuffled_idx, original_idx in enumerate(indices):
        real_idx = needs_extraction[original_idx]
        results[real_idx] = parsed.get(shuffled_idx, ERROR_EXTRACTION.copy())

    # Fill any remaining None values
    for i in range(len(results)):
        if results[i] is None:
            results[i] = ERROR_EXTRACTION.copy()

    return results


async def extract_all(
    provider: OpenRouterProvider,
    models: list[str],
) -> dict[str, dict]:
    """Extract sentencing data from all cached raw responses using batched LLM calls."""
    jobs: list[tuple[str, dict, dict, str, int]] = []
    for model in models:
        response_cache = load_response_cache(model)
        extraction_cache = load_extraction_cache(model)

        for defendant in DEFENDANTS:
            for crime in CRIMES:
                messages = build_messages(defendant, crime)
                responses = get_responses(response_cache, messages, TEMPERATURE)[:ITERATIONS]
                for i, result in enumerate(responses):
                    if result.get("error") is not None or result.get("response") is None:
                        continue
                    existing = get_extraction(
                        extraction_cache, defendant["id"], crime["id"],
                        result["response"], EXTRACTOR_MODEL,
                    )
                    if existing is None:
                        jobs.append((model, defendant, crime, result["response"], i))

    total_needed = len(jobs)
    if total_needed == 0:
        print("\n  All extractions already cached.")
        return {m: load_extraction_cache(m) for m in models}

    num_batches = (total_needed + EXTRACTION_BATCH_SIZE - 1) // EXTRACTION_BATCH_SIZE
    print(f"\n  {total_needed} extractions needed in {num_batches} batches of up to {EXTRACTION_BATCH_SIZE}.")

    batches = []
    for i in range(0, total_needed, EXTRACTION_BATCH_SIZE):
        batches.append(jobs[i:i + EXTRACTION_BATCH_SIZE])

    extraction_caches: dict[str, dict] = {m: load_extraction_cache(m) for m in models}
    cache_locks: dict[str, asyncio.Lock] = {m: asyncio.Lock() for m in models}
    semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)
    done_batches = 0
    done_lock = asyncio.Lock()

    async def _run_batch(batch: list[tuple[str, dict, dict, str, int]]):
        nonlocal done_batches

        items = [(defendant, crime, response_text) for _, defendant, crime, response_text, _ in batch]

        async with semaphore:
            answers = await extract_batch(provider, items)

        for (model, defendant, crime, response_text, iter_idx), answer in zip(batch, answers):
            async with cache_locks[model]:
                set_extraction(
                    extraction_caches[model], model,
                    defendant["id"], crime["id"], response_text, EXTRACTOR_MODEL, answer,
                )

        async with done_lock:
            done_batches += 1
            print(f"  [batch {done_batches}/{num_batches}] {len(batch)} items extracted")

    await asyncio.gather(*[_run_batch(b) for b in batches])

    for model in models:
        extraction_caches[model] = load_extraction_cache(model)

    return extraction_caches


def compute_scores(model: str) -> dict[str, dict[str, dict]]:
    """Compute average fine and jail time per (defendant, crime) for a model.

    Returns {defendant_id: {crime_id: {
        "avg_fine": float|None, "avg_jail": float|None,
        "fines": list[int], "jails": list[float],
        "count": int, "errors": int, "refused": int,
    }}}
    """
    response_cache = load_response_cache(model)
    extraction_cache = load_extraction_cache(model)

    scores = {}
    for defendant in DEFENDANTS:
        scores[defendant["id"]] = {}
        for crime in CRIMES:
            messages = build_messages(defendant, crime)
            responses = get_responses(response_cache, messages, TEMPERATURE)[:ITERATIONS]

            fines = []
            jails = []
            errors = 0
            refused = 0

            for result in responses:
                if result.get("error") is not None or result.get("response") is None:
                    errors += 1
                    continue
                extraction = get_extraction(
                    extraction_cache, defendant["id"], crime["id"],
                    result["response"], EXTRACTOR_MODEL,
                )
                if extraction is None:
                    errors += 1
                elif extraction.get("thoughts") == "ERROR":
                    errors += 1
                elif extraction.get("thoughts") == "REFUSED":
                    refused += 1
                elif extraction.get("fine_usd", -1) < 0 or extraction.get("jail_years", -1) < 0:
                    errors += 1
                else:
                    fines.append(extraction["fine_usd"])
                    jails.append(extraction["jail_years"])

            scores[defendant["id"]][crime["id"]] = {
                "avg_fine": sum(fines) / len(fines) if fines else None,
                "avg_jail": sum(jails) / len(jails) if jails else None,
                "fines": fines,
                "jails": jails,
                "count": len(fines),
                "errors": errors,
                "refused": refused,
            }

    return scores
