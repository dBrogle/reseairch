"""LLM-powered answer extraction for the God Belief by Country study.

Takes raw LLM responses and uses a separate LLM call to determine whether
the model answered YES, NO, or REFUSED. Batched and shuffled to prevent
positional bias.
"""

import asyncio
import json
import random

from services.llm import OpenRouterProvider
from studies.god_by_country.config import (
    COUNTRIES,
    TEMPERATURE,
    ITERATIONS,
    EXTRACTOR_MODEL,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_BATCH_SIZE,
    MAX_PARALLEL_REQUESTS,
    QUESTION,
)
from studies.god_by_country.runner import build_messages, QUESTION_ID
from studies.god_by_country.cache import (
    load_response_cache,
    get_responses,
    load_extraction_cache,
    get_extraction,
    set_extraction,
)


def _build_batch_input(items: list[tuple[str, str]]) -> str:
    """Build the JSON input for a batch of (question, response_text) pairs."""
    batch = {}
    for idx, (question, response_text) in enumerate(items):
        batch[str(idx)] = {
            "question": question,
            "response": response_text,
        }
    return json.dumps(batch, indent=2)


def _parse_batch_output(raw: str, batch_size: int) -> dict[int, str]:
    """Parse the extractor's JSON response into {index: YES|NO|REFUSED}."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

    parsed = json.loads(text)
    results = {}
    for idx_str, answer in parsed.items():
        idx = int(idx_str)
        if idx < 0 or idx >= batch_size:
            continue
        answer = str(answer).strip().upper()
        if answer in ("YES", "NO", "REFUSED"):
            results[idx] = answer
        elif "YES" in answer:
            results[idx] = "YES"
        elif "NO" in answer:
            results[idx] = "NO"
        else:
            results[idx] = "REFUSED"
    return results


async def extract_batch(
    provider: OpenRouterProvider,
    items: list[tuple[str, str]],
) -> list[str]:
    """Extract answers for a batch of (question, response_text) pairs."""
    if not items:
        return []

    # Shuffle to prevent positional bias
    indices = list(range(len(items)))
    random.shuffle(indices)
    shuffled = [items[i] for i in indices]

    batch_json = _build_batch_input(shuffled)

    try:
        result = await provider.complete_text(
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": batch_json},
            ],
            model=EXTRACTOR_MODEL,
            temperature=0.0,
            max_tokens=200,
        )
        parsed = _parse_batch_output(result, len(shuffled))
    except Exception:
        return ["ERROR"] * len(items)

    # Un-shuffle back to original order
    answers = ["ERROR"] * len(items)
    for shuffled_idx, original_idx in enumerate(indices):
        answers[original_idx] = parsed.get(shuffled_idx, "ERROR")

    return answers


async def extract_all(
    provider: OpenRouterProvider,
    models: list[str],
    countries: list[str] | None = None,
) -> dict[str, dict]:
    """Extract answers from all cached raw responses using batched LLM calls."""
    target_countries = countries or COUNTRIES

    jobs: list[tuple[str, str, str, int]] = []
    for model in models:
        response_cache = load_response_cache(model)
        extraction_cache = load_extraction_cache(model)

        for country in target_countries:
            messages = build_messages(country)
            responses = get_responses(response_cache, messages, TEMPERATURE)[:ITERATIONS]
            for i, result in enumerate(responses):
                if result.get("error") is not None or result.get("response") is None:
                    continue
                existing = get_extraction(
                    extraction_cache, QUESTION_ID,
                    result["response"], EXTRACTOR_MODEL,
                )
                if existing is None:
                    jobs.append((model, result["response"], country, i))

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

    async def _run_batch(batch: list[tuple[str, str, str, int]]):
        nonlocal done_batches

        items = [(QUESTION, response_text) for _, response_text, _, _ in batch]

        async with semaphore:
            answers = await extract_batch(provider, items)

        for (model, response_text, country, iter_idx), answer in zip(batch, answers):
            async with cache_locks[model]:
                set_extraction(
                    extraction_caches[model], model,
                    QUESTION_ID, response_text, EXTRACTOR_MODEL, answer,
                )

        async with done_lock:
            done_batches += 1
            sample = ", ".join(answers[:3])
            if len(answers) > 3:
                sample += ", ..."
            print(f"  [batch {done_batches}/{num_batches}] {len(batch)} items -> [{sample}]")

    await asyncio.gather(*[_run_batch(b) for b in batches])

    for model in models:
        extraction_caches[model] = load_extraction_cache(model)

    return extraction_caches


def compute_country_scores(
    model: str,
    countries: list[str] | None = None,
) -> dict[str, float | None]:
    """Compute the proportion of YES answers per country for a model.

    Returns {country: yes_rate} where yes_rate is 0.0 to 1.0.
    Countries with no valid answers get None.
    """
    target_countries = countries or COUNTRIES
    response_cache = load_response_cache(model)
    extraction_cache = load_extraction_cache(model)

    country_scores: dict[str, float | None] = {}
    for country in target_countries:
        yes_count = 0
        total = 0
        messages = build_messages(country)
        responses = get_responses(response_cache, messages, TEMPERATURE)[:ITERATIONS]
        for result in responses:
            if result.get("error") is not None or result.get("response") is None:
                continue
            answer = get_extraction(
                extraction_cache, QUESTION_ID,
                result["response"], EXTRACTOR_MODEL,
            )
            if answer in ("YES", "NO"):
                total += 1
                if answer == "YES":
                    yes_count += 1
        if total > 0:
            country_scores[country] = yes_count / total
        else:
            country_scores[country] = None

    return country_scores
