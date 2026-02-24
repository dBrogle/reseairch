"""Parallel execution engine for the Claude Identity study.

Builds a flat job list of (model, temperature, iteration) across ALL models
and runs them concurrently, bounded by a shared semaphore.
"""

import asyncio

from services.llm import OpenRouterProvider
from studies.claude_identity.config import (
    TEMPERATURES,
    ITERATIONS_PER_TEMP,
    SEED_CONVO_2,
    CLAUDE_KEYWORDS,
    CHATGPT_KEYWORDS,
    MAX_PARALLEL_REQUESTS,
    MAX_RETRIES,
)
from studies.claude_identity.cache import (
    load_cache,
    get_results,
    append_result,
    purge_errors,
)


def mentions_claude(response: str) -> bool:
    """Check if a response contains Claude/Anthropic keywords."""
    lower = response.lower()
    return any(kw in lower for kw in CLAUDE_KEYWORDS)


def mentions_chatgpt(response: str) -> bool:
    """Check if a response contains ChatGPT/OpenAI keywords."""
    lower = response.lower()
    return any(kw in lower for kw in CHATGPT_KEYWORDS)


async def run_single_query(
    provider: OpenRouterProvider, model: str, temperature: float
) -> dict:
    """Run a single identity query, retrying up to MAX_RETRIES times on error."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 2):  # 1 initial + MAX_RETRIES retries
        try:
            response = await provider.complete_text(
                messages=SEED_CONVO_2,
                model=model,
                temperature=temperature,
                max_tokens=300,
            )
            return {
                "response": response,
                "mentions_claude": mentions_claude(response),
                "mentions_chatgpt": mentions_chatgpt(response),
                "error": None,
            }
        except Exception as e:
            last_error = e
            if attempt <= MAX_RETRIES:
                wait = 1 if attempt == 1 else 2 if attempt == 2 else 4
                print(f"    retry {attempt}/{MAX_RETRIES} for {model} temp={temperature}: {e} (waiting {wait}s)")
                await asyncio.sleep(wait)
    return {
        "response": None,
        "mentions_claude": False,
        "mentions_chatgpt": False,
        "error": str(last_error),
    }


async def run_all(
    provider: OpenRouterProvider, models: list[str]
) -> dict[str, dict[float, list[dict]]]:
    """
    Run the full experiment across ALL models in parallel.

    1. Load caches and purge errors for every model
    2. Build one flat job list: (model, temp, iter_idx) for all uncached work
    3. Fire all jobs concurrently, bounded by a shared semaphore
    4. Return {model_id: {temperature: [results]}}
    """
    # Load caches and purge errors upfront
    caches: dict[str, dict] = {}
    for model in models:
        caches[model] = load_cache(model)
        purged = purge_errors(caches[model], model, SEED_CONVO_2)
        if purged:
            print(f"  Purged {purged} cached error(s) for {model}")

    # Build flat job list across all models
    jobs: list[tuple[str, float, int]] = []
    for model in models:
        for temp in TEMPERATURES:
            existing = len(get_results(caches[model], SEED_CONVO_2, temp))
            for i in range(existing, ITERATIONS_PER_TEMP):
                jobs.append((model, temp, i))

    total_needed = len(jobs)
    total_possible = len(models) * len(TEMPERATURES) * ITERATIONS_PER_TEMP
    total_cached = total_possible - total_needed

    if total_needed == 0:
        print(f"\n  All {total_possible} results already cached across {len(models)} model(s).")
    else:
        print(f"\n  {total_cached} cached, {total_needed} remaining across {len(models)} model(s) ({MAX_PARALLEL_REQUESTS} parallel).")

    # Per-model locks for cache write safety
    cache_locks: dict[str, asyncio.Lock] = {model: asyncio.Lock() for model in models}
    semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)

    done = 0
    done_lock = asyncio.Lock()

    async def _run_job(model: str, temp: float, iter_idx: int):
        nonlocal done
        async with semaphore:
            result = await run_single_query(provider, model, temp)

        async with cache_locks[model]:
            append_result(caches[model], model, SEED_CONVO_2, temp, result)

        async with done_lock:
            done += 1
            if result["error"]:
                status = f"ERROR: {result['error']}"
            elif result["mentions_claude"]:
                status = "claims claude"
            else:
                status = "no claude"
            short_model = model.split("/")[-1]
            print(f"  [{done}/{total_needed}] {short_model} temp={temp} iter={iter_idx+1}/{ITERATIONS_PER_TEMP} -> {status}")

    # Fire ALL jobs concurrently, bounded by the semaphore
    if jobs:
        await asyncio.gather(*[_run_job(m, t, i) for m, t, i in jobs])

    # Build final results from cache (includes old + new)
    all_results: dict[str, dict[float, list[dict]]] = {}
    for model in models:
        final_cache = load_cache(model)
        model_results: dict[float, list[dict]] = {}
        for temp in TEMPERATURES:
            model_results[temp] = get_results(final_cache, SEED_CONVO_2, temp)[:ITERATIONS_PER_TEMP]
        all_results[model] = model_results

    return all_results
