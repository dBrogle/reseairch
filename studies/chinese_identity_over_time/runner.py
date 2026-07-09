"""Parallel execution engine for the Chinese Identity Over Time study.

Builds a flat job list of (model, iteration) across ALL models in the lineage and
runs them concurrently, bounded by a shared semaphore. There is no temperature
sweep: every call omits the temperature field so the model uses its own default.

Reasoning is OFF for every model (we want the fast, default-completion answer, kept
identical across the timeline). Some endpoints 400 when reasoning is explicitly
disabled because reasoning is mandatory; the single-query path detects that and
retries with the reasoning field omitted entirely — the closest to "off" those
endpoints allow.
"""

import asyncio

from services.llm import OpenRouterProvider
from studies.chinese_identity_over_time.config import (
    ITERATIONS,
    CHINESE_KEYWORDS,
    MAX_PARALLEL_REQUESTS,
    MAX_RETRIES,
    MAX_TOKENS,
    get_seed_convo,
)
from studies.chinese_identity_over_time.cache import (
    load_cache,
    get_results,
    append_result,
    purge_errors,
)


def mentions_chinese(response: str) -> bool:
    """Cheap keyword check: does the response name a Chinese model/company?"""
    lower = response.lower()
    return any(kw.lower() in lower for kw in CHINESE_KEYWORDS)


def _cannot_disable_reasoning(err: Exception) -> bool:
    """Did the call 400 specifically because reasoning can't be disabled?

    Two shapes seen from OpenRouter when we send reasoning={enabled: false}:
      - "Reasoning is mandatory for this endpoint and cannot be disabled." (GPT-5, Gemini 2.5 Pro)
      - "Unsupported value: 'none' is not supported with ..."               (o3, reasoning_effort=none)
    Omitting the reasoning field entirely fixes both.
    """
    msg = str(err).lower()
    if "reasoning" in msg and ("mandatory" in msg or "cannot be disabled" in msg or "required" in msg):
        return True
    if "'none' is not supported" in msg or "is not supported with" in msg:
        return True
    return False


async def run_single_query(
    provider: OpenRouterProvider, model: str, messages: list[dict]
) -> dict:
    """Run a single identity query with reasoning OFF, at the model's default temp.

    If the endpoint refuses to disable reasoning, retries with the reasoning field
    omitted entirely (its unavoidable default) so the model still gets queried.
    """
    last_error = None
    omit_reasoning = False  # start by explicitly disabling reasoning
    for attempt in range(1, MAX_RETRIES + 2):  # 1 initial + MAX_RETRIES retries
        try:
            response = await provider.complete_text(
                messages=messages,
                model=model,
                max_tokens=MAX_TOKENS,
                omit_temperature=True,
                enable_reasoning=False,
                omit_reasoning=omit_reasoning,
            )
            return {
                "response": response,
                "mentions_chinese": mentions_chinese(response),
                "error": None,
            }
        except Exception as e:
            last_error = e
            # If this endpoint won't let reasoning be disabled, drop the reasoning
            # field entirely and retry immediately (doesn't consume retry budget).
            if not omit_reasoning and _cannot_disable_reasoning(e):
                omit_reasoning = True
                print(f"    {model}: reasoning can't be disabled, retrying with reasoning field omitted")
                continue
            if attempt <= MAX_RETRIES:
                wait = 1 if attempt == 1 else 2 if attempt == 2 else 4
                print(f"    retry {attempt}/{MAX_RETRIES} for {model}: {e} (waiting {wait}s)")
                await asyncio.sleep(wait)
    return {
        "response": None,
        "mentions_chinese": False,
        "error": str(last_error),
    }


async def run_all(
    provider: OpenRouterProvider, models: list[str]
) -> dict[str, list[dict]]:
    """
    Run the full experiment across the given models in parallel.

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
            elif result["mentions_chinese"]:
                status = "claims chinese"
            else:
                status = "not chinese"
            short_model = model.split("/")[-1]
            print(f"  [{done}/{total_needed}] {short_model} iter={iter_idx+1}/{ITERATIONS} -> {status}")

    if jobs:
        await asyncio.gather(*[_run_job(m, i) for m, i in jobs])

    all_results: dict[str, list[dict]] = {}
    for model in models:
        final_cache = load_cache(model)
        all_results[model] = get_results(final_cache, seed_convos[model])[:ITERATIONS]

    return all_results
