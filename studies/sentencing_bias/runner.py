"""Parallel execution engine for the Sentencing Bias study.

Builds messages for each (model, defendant, crime) combo and runs them
concurrently with multiple iterations.
"""

import asyncio

from services.llm import OpenRouterProvider
from studies.sentencing_bias.config import (
    DEFENDANTS,
    CRIMES,
    TEMPERATURE,
    ITERATIONS,
    MAX_PARALLEL_REQUESTS,
    MAX_RETRIES,
    SYSTEM_MESSAGE,
    PROMPT_TEMPLATE,
)
from studies.sentencing_bias.cache import (
    load_response_cache,
    get_responses,
    append_response,
    purge_response_errors,
)


def build_messages(defendant: dict, crime: dict) -> list[dict]:
    """Build the full message array for a defendant + crime combo."""
    scenario = crime["scenario"].format(name=defendant["name"])
    prompt = PROMPT_TEMPLATE.format(
        scenario=scenario,
        name=defendant["name"],
        charge=crime["charge"],
    )

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]


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
                max_tokens=500,
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
) -> dict[str, dict]:
    """
    Run the experiment for all (model, defendant, crime) combos.

    Returns {model: response_cache} with all results populated.
    """
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
        for defendant in DEFENDANTS:
            for crime in CRIMES:
                messages = build_messages(defendant, crime)
                existing = len(get_responses(caches[model], messages, TEMPERATURE))
                for i in range(existing, ITERATIONS):
                    jobs.append((model, defendant, crime, i, messages))

    total_needed = len(jobs)
    total_possible = len(models) * len(DEFENDANTS) * len(CRIMES) * ITERATIONS
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

    async def _run_job(model, defendant, crime, iter_idx, messages):
        nonlocal done
        async with semaphore:
            result = await run_single_query(provider, model, messages, TEMPERATURE)

        async with cache_locks[model]:
            append_response(caches[model], model, messages, TEMPERATURE, result)

        async with done_lock:
            done += 1
            short = model.split("/")[-1]
            status = "OK" if result["error"] is None else f"ERR: {result['error']}"
            print(f"  [{done}/{total_needed}] {short} | {defendant['name']} x {crime['id']} iter={iter_idx+1} -> {status}")

    if jobs:
        await asyncio.gather(*[_run_job(m, d, c, i, msgs) for m, d, c, i, msgs in jobs])

    # Reload caches from disk
    for model in models:
        caches[model] = load_response_cache(model)

    return caches
