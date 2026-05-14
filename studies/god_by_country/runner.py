"""Parallel execution engine for the God Belief by Country study.

Builds messages for each (model, country) combo and runs them concurrently
with multiple iterations at high temperature.
"""

import asyncio

from services.llm import OpenRouterProvider
from studies.god_by_country.config import (
    COUNTRIES,
    COUNTRY_DISPLAY_NAMES,
    TEMPERATURE,
    ITERATIONS,
    MAX_PARALLEL_REQUESTS,
    MAX_RETRIES,
    SYSTEM_MESSAGE,
    COUNTRY_CONTEXT_TEMPLATE,
    CONVERSATION_SEED,
    QUESTION,
)
from studies.god_by_country.cache import (
    load_response_cache,
    get_responses,
    append_response,
    purge_response_errors,
)

QUESTION_ID = "god_exists"


def _display_name(country: str) -> str:
    """Get the display name for a country (for prompt text)."""
    return COUNTRY_DISPLAY_NAMES.get(country, country)


def build_messages(country: str) -> list[dict]:
    """Build the full message array for a country."""
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "system", "content": COUNTRY_CONTEXT_TEMPLATE.format(country=_display_name(country))},
    ]
    for role, content in CONVERSATION_SEED:
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": QUESTION})
    return messages


async def run_single_query(
    provider: OpenRouterProvider,
    model: str,
    messages: list[dict],
    temperature: float,
) -> dict:
    """Run a single query with retries."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response = await provider.complete_text(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=300,
            )
            return {"response": response, "error": None}
        except Exception as e:
            last_error = e
            if attempt <= MAX_RETRIES:
                wait = 2 ** (attempt - 1)
                await asyncio.sleep(wait)
    return {"response": None, "error": str(last_error)}


async def run_all(
    provider: OpenRouterProvider,
    models: list[str],
    countries: list[str] | None = None,
) -> dict[str, dict]:
    """
    Run the experiment for all (model, country) combos.

    Returns {model: response_cache} with all results populated.
    """
    target_countries = countries or COUNTRIES

    # Load caches and purge errors
    caches: dict[str, dict] = {}
    for model in models:
        caches[model] = load_response_cache(model)
        purged = purge_response_errors(caches[model], model)
        if purged:
            print(f"  Purged {purged} cached error(s) for {model}")

    # Build flat job list
    jobs = []
    for model in models:
        for country in target_countries:
            messages = build_messages(country)
            existing = len(get_responses(caches[model], messages, TEMPERATURE))
            for i in range(existing, ITERATIONS):
                jobs.append((model, country, i, messages))

    total_needed = len(jobs)
    total_possible = len(models) * len(target_countries) * ITERATIONS
    total_cached = total_possible - total_needed

    if total_needed == 0:
        print(f"\n  All {total_possible} results already cached.")
    else:
        print(f"\n  {total_cached} cached, {total_needed} remaining ({MAX_PARALLEL_REQUESTS} parallel).")

    # Concurrency controls
    cache_locks: dict[str, asyncio.Lock] = {m: asyncio.Lock() for m in models}
    semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)
    done = 0
    done_lock = asyncio.Lock()

    async def _run_job(model, country, iter_idx, messages):
        nonlocal done
        async with semaphore:
            result = await run_single_query(provider, model, messages, TEMPERATURE)

        async with cache_locks[model]:
            append_response(caches[model], model, messages, TEMPERATURE, result)

        async with done_lock:
            done += 1
            short = model.split("/")[-1]
            if result["error"] is None:
                print(f"  [{done}/{total_needed}] {short} | {country} | iter={iter_idx+1} -> OK")
            else:
                print(f"  [{done}/{total_needed}] {short} | {country} | iter={iter_idx+1} -> ERR: {result['error']}")

    if jobs:
        await asyncio.gather(*[_run_job(m, c, i, msgs) for m, c, i, msgs in jobs])

    # Reload caches from disk
    for model in models:
        caches[model] = load_response_cache(model)

    return caches
