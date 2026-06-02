"""Parallel execution engine for the Claude Identity 2 study.

Builds a flat job list of (model, iteration) across ALL models and runs them
concurrently, bounded by a shared semaphore. There is no temperature sweep: every
call omits the temperature field so the model uses its own default. Models that
refuse to disable reasoning are called with reasoning enabled.
"""

import asyncio

from services.llm import OpenRouterProvider
from studies.claude_identity_2.config import (
    ITERATIONS,
    WESTERN_KEYWORDS,
    MAX_PARALLEL_REQUESTS,
    MAX_RETRIES,
    MAX_TOKENS,
    get_seed_convo,
    needs_reasoning,
)
from studies.claude_identity_2.cache import (
    load_cache,
    get_results,
    append_result,
    purge_errors,
)


def mentions_western(response: str) -> bool:
    """Cheap keyword check: does the response name a Western model/company?"""
    lower = response.lower()
    return any(kw.lower() in lower for kw in WESTERN_KEYWORDS)


async def run_single_query(
    provider: OpenRouterProvider, model: str, messages: list[dict]
) -> dict:
    """Run a single identity query at the model's default temperature."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 2):  # 1 initial + MAX_RETRIES retries
        try:
            response = await provider.complete_text(
                messages=messages,
                model=model,
                max_tokens=MAX_TOKENS,
                omit_temperature=True,
                enable_reasoning=needs_reasoning(model),
            )
            return {
                "response": response,
                "mentions_western": mentions_western(response),
                "error": None,
            }
        except Exception as e:
            last_error = e
            if attempt <= MAX_RETRIES:
                wait = 1 if attempt == 1 else 2 if attempt == 2 else 4
                print(f"    retry {attempt}/{MAX_RETRIES} for {model}: {e} (waiting {wait}s)")
                await asyncio.sleep(wait)
    return {
        "response": None,
        "mentions_western": False,
        "error": str(last_error),
    }


async def run_all(
    provider: OpenRouterProvider, models: list[str]
) -> dict[str, list[dict]]:
    """
    Run the full experiment across ALL models in parallel.

    1. Load caches and purge errors for every model
    2. Build one flat job list: (model, iter_idx) for all uncached work
    3. Fire all jobs concurrently, bounded by a shared semaphore
    4. Return {model_id: [results]}
    """
    seed_convos: dict[str, list[dict]] = {model: get_seed_convo(model) for model in models}

    caches: dict[str, dict] = {}
    for model in models:
        caches[model] = load_cache(model)
        purged = purge_errors(caches[model], model, seed_convos[model])
        if purged:
            print(f"  Purged {purged} cached error(s) for {model}")

    jobs: list[tuple[str, int]] = []
    for model in models:
        existing = len(get_results(caches[model], seed_convos[model]))
        for i in range(existing, ITERATIONS):
            jobs.append((model, i))

    total_needed = len(jobs)
    total_possible = len(models) * ITERATIONS
    total_cached = total_possible - total_needed

    if total_needed == 0:
        print(f"\n  All {total_possible} results already cached across {len(models)} model(s).")
    else:
        print(f"\n  {total_cached} cached, {total_needed} remaining across {len(models)} model(s) ({MAX_PARALLEL_REQUESTS} parallel).")

    cache_locks: dict[str, asyncio.Lock] = {model: asyncio.Lock() for model in models}
    semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)

    done = 0
    done_lock = asyncio.Lock()

    async def _run_job(model: str, iter_idx: int):
        nonlocal done
        async with semaphore:
            result = await run_single_query(provider, model, seed_convos[model])

        async with cache_locks[model]:
            append_result(caches[model], model, seed_convos[model], result)

        async with done_lock:
            done += 1
            if result["error"]:
                status = f"ERROR: {result['error']}"
            elif result["mentions_western"]:
                status = "claims western"
            else:
                status = "not western"
            short_model = model.split("/")[-1]
            print(f"  [{done}/{total_needed}] {short_model} iter={iter_idx+1}/{ITERATIONS} -> {status}")

    if jobs:
        await asyncio.gather(*[_run_job(m, i) for m, i in jobs])

    all_results: dict[str, list[dict]] = {}
    for model in models:
        final_cache = load_cache(model)
        all_results[model] = get_results(final_cache, seed_convos[model])[:ITERATIONS]

    return all_results
