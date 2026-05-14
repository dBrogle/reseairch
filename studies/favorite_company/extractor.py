"""LLM-powered answer extraction for the Favorite Company study.

Takes raw LLM responses and uses a separate LLM call to extract:
  - company: which company the model chose
  - reason: short summary of why
  - person: which individual the model chose

Batched and shuffled to prevent positional bias.
"""

import asyncio
import json
import random

from services.llm import OpenRouterProvider
from studies.favorite_company.config import (
    QUESTION,
    TEMPERATURE,
    ITERATIONS,
    EXTRACTOR_MODEL,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_BATCH_SIZE,
    MAX_PARALLEL_REQUESTS,
)
from studies.favorite_company.runner import build_messages
from studies.favorite_company.cache import (
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


def _parse_batch_output(raw: str, batch_size: int) -> dict[int, dict]:
    """Parse the extractor's JSON response into {index: {company, reason, person}}."""
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
        if isinstance(data, dict):
            results[idx] = {
                "company": str(data.get("company", "ERROR")).strip(),
                "reason": str(data.get("reason", "ERROR")).strip(),
                "person": str(data.get("person", "ERROR")).strip(),
            }
        else:
            results[idx] = {"company": "ERROR", "reason": "ERROR", "person": "ERROR"}
    return results


async def extract_batch(
    provider: OpenRouterProvider,
    items: list[tuple[str, str]],
) -> list[dict]:
    """Extract answers for a batch of (question, response_text) pairs."""
    if not items:
        return []

    indices = list(range(len(items)))
    random.shuffle(indices)
    shuffled = [items[i] for i in indices]

    batch_json = _build_batch_input(shuffled)

    error_result = {"company": "ERROR", "reason": "ERROR", "person": "ERROR"}

    try:
        result = await provider.complete_text(
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": batch_json},
            ],
            model=EXTRACTOR_MODEL,
            temperature=0.0,
            max_tokens=500,
        )
        parsed = _parse_batch_output(result, len(shuffled))
    except Exception:
        return [error_result] * len(items)

    answers = [error_result] * len(items)
    for shuffled_idx, original_idx in enumerate(indices):
        answers[original_idx] = parsed.get(shuffled_idx, error_result)

    return answers


async def extract_all(
    provider: OpenRouterProvider,
    models: list[str],
) -> dict[str, dict]:
    """Extract structured answers from all cached raw responses."""
    messages = build_messages()
    jobs: list[tuple[str, str, int]] = []  # (model, response_text, iter_idx)

    for model in models:
        response_cache = load_response_cache(model)
        extraction_cache = load_extraction_cache(model)

        responses = get_responses(response_cache, messages, TEMPERATURE)[:ITERATIONS]
        for i, result in enumerate(responses):
            if result.get("error") is not None or result.get("response") is None:
                continue
            existing = get_extraction(
                extraction_cache, result["response"], EXTRACTOR_MODEL,
            )
            if existing is None:
                jobs.append((model, result["response"], i))

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

    async def _run_batch(batch: list[tuple[str, str, int]]):
        nonlocal done_batches

        items = [(QUESTION, response_text) for _, response_text, _ in batch]

        async with semaphore:
            answers = await extract_batch(provider, items)

        for (model, response_text, iter_idx), answer in zip(batch, answers):
            async with cache_locks[model]:
                set_extraction(
                    extraction_caches[model], model,
                    response_text, EXTRACTOR_MODEL, answer,
                )

        async with done_lock:
            done_batches += 1
            sample_companies = [a.get("company", "?") for a in answers[:3]]
            sample = ", ".join(sample_companies)
            if len(answers) > 3:
                sample += ", ..."
            print(f"  [batch {done_batches}/{num_batches}] {len(batch)} items -> [{sample}]")

    await asyncio.gather(*[_run_batch(b) for b in batches])

    for model in models:
        extraction_caches[model] = load_extraction_cache(model)

    return extraction_caches


def compute_distributions(model: str) -> dict:
    """Compute company and person distributions for a model.

    Returns {
        "companies": {name: count},
        "persons": {name: count},
        "reasons": {company: [reasons]},
        "total": int,
        "errors": int,
        "refused_company": int,
        "refused_person": int,
    }
    """
    messages = build_messages()
    response_cache = load_response_cache(model)
    extraction_cache = load_extraction_cache(model)

    responses = get_responses(response_cache, messages, TEMPERATURE)[:ITERATIONS]

    companies: dict[str, int] = {}
    persons: dict[str, int] = {}
    reasons: dict[str, list[str]] = {}
    errors = 0
    refused_company = 0
    refused_person = 0

    for result in responses:
        if result.get("error") is not None or result.get("response") is None:
            errors += 1
            continue

        extraction = get_extraction(
            extraction_cache, result["response"], EXTRACTOR_MODEL,
        )
        if extraction is None or extraction.get("company") == "ERROR":
            errors += 1
            continue

        company = extraction["company"]
        person = extraction["person"]
        reason = extraction.get("reason", "")

        if company == "REFUSED":
            refused_company += 1
        else:
            companies[company] = companies.get(company, 0) + 1
            if company not in reasons:
                reasons[company] = []
            if reason and reason != "REFUSED":
                reasons[company].append(reason)

        if person == "REFUSED":
            refused_person += 1
        else:
            persons[person] = persons.get(person, 0) + 1

    return {
        "companies": companies,
        "persons": persons,
        "reasons": reasons,
        "total": len(responses),
        "errors": errors,
        "refused_company": refused_company,
        "refused_person": refused_person,
    }
