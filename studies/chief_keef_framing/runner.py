"""Parallel execution engine for the Chief Keef Framing study.

Builds a flat job list of (model, arm, iteration) across ALL models and arms and
runs them concurrently, bounded by a shared semaphore. There is no temperature
sweep: every call omits the temperature field so the model uses its own default.
Models that refuse to disable reasoning are called with reasoning enabled.
"""

import asyncio

from services.llm import OpenRouterProvider
from studies.chief_keef_framing.config import (
    ARMS,
    ITERATIONS,
    MAX_PARALLEL_REQUESTS,
    MAX_RETRIES,
    MAX_TOKENS,
    needs_reasoning,
)
from studies.chief_keef_framing.cache import (
    load_cache,
    get_results,
    append_result,
    purge_errors,
)


async def run_single_query(
    provider: OpenRouterProvider, model: str, messages: list[dict]
) -> dict:
    """Run a single opinion query at the model's default temperature."""
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
            return {"response": response, "error": None}
        except Exception as e:
            last_error = e
            if attempt <= MAX_RETRIES:
                wait = 1 if attempt == 1 else 2 if attempt == 2 else 4
                print(f"    retry {attempt}/{MAX_RETRIES} for {model}: {e} (waiting {wait}s)")
                await asyncio.sleep(wait)
    return {"response": None, "error": str(last_error)}


async def run_all(
    provider: OpenRouterProvider, models: list[str]
) -> dict[str, dict]:
    """
    Run the full experiment across ALL models and arms in parallel.

    1. Load caches and purge errors for every (model, arm)
    2. Build one flat job list: (model, arm, iter_idx) for all uncached work
    3. Fire all jobs concurrently, bounded by a shared semaphore
    4. Return {model_id: cache_dict}
    """
    caches: dict[str, dict] = {}
    for model in models:
        caches[model] = load_cache(model)
        for arm in ARMS:
            purged = purge_errors(caches[model], model, arm["messages"])
            if purged:
                print(f"  Purged {purged} cached error(s) for {model} [{arm['key']}]")

    jobs: list[tuple[str, dict, int]] = []
    for model in models:
        for arm in ARMS:
            existing = len(get_results(caches[model], arm["messages"]))
            for i in range(existing, ITERATIONS):
                jobs.append((model, arm, i))

    total_possible = len(models) * len(ARMS) * ITERATIONS
    total_needed = len(jobs)
    total_cached = total_possible - total_needed

    if total_needed == 0:
        print(f"\n  All {total_possible} results already cached "
              f"across {len(models)} model(s) x {len(ARMS)} arm(s).")
    else:
        print(f"\n  {total_cached} cached, {total_needed} remaining across "
              f"{len(models)} model(s) x {len(ARMS)} arm(s) "
              f"({MAX_PARALLEL_REQUESTS} parallel).")

    cache_locks: dict[str, asyncio.Lock] = {model: asyncio.Lock() for model in models}
    semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)

    done = 0
    done_lock = asyncio.Lock()

    async def _run_job(model: str, arm: dict, iter_idx: int):
        nonlocal done
        async with semaphore:
            result = await run_single_query(provider, model, arm["messages"])

        async with cache_locks[model]:
            append_result(
                caches[model], model, arm["messages"], result,
                arm_key=arm["key"], label=arm["label"],
            )

        async with done_lock:
            done += 1
            status = f"ERROR: {result['error']}" if result["error"] else "ok"
            short_model = model.split("/")[-1]
            print(f"  [{done}/{total_needed}] {short_model} [{arm['key']}] "
                  f"iter={iter_idx+1}/{ITERATIONS} -> {status}")

    if jobs:
        await asyncio.gather(*[_run_job(m, a, i) for m, a, i in jobs])

    return {model: load_cache(model) for model in models}
