"""Parallel execution engine for the Hate Speech Detection Bias study.

Builds messages for each (model, template, group) combo and runs them
concurrently with multiple iterations. Handles deduplication for
people-form templates where race x gender collapses to race-only.
"""

import asyncio

from services.llm import OpenRouterProvider
from studies.hate_speech_detection.config import (
    GROUPS,
    TEMPLATES,
    TEMPERATURE,
    ITERATIONS,
    MAX_PARALLEL_REQUESTS,
    MAX_RETRIES,
    SYSTEM_MESSAGE,
)
from studies.hate_speech_detection.cache import (
    load_response_cache,
    get_responses,
    append_response,
    purge_response_errors,
)


def get_group_text(group: dict, form: str) -> str:
    """Get the appropriate text form for a group."""
    return group[form]


def build_messages(template: dict, group: dict) -> list[dict]:
    """Build the full message array for a template + group combo."""
    group_text = get_group_text(group, template["form"])
    post_text = template["text"].format(group=group_text)

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": post_text},
    ]


def build_job_list(models: list[str], caches: dict) -> list[tuple]:
    """Build flat job list for all (model, template, group) combos."""
    jobs = []

    for model in models:
        for template in TEMPLATES:
            for group in GROUPS:
                messages = build_messages(template, group)
                existing = len(get_responses(caches[model], messages, TEMPERATURE))

                for i in range(existing, ITERATIONS):
                    jobs.append((model, template, group, i, messages))

    return jobs


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
) -> dict[str, dict]:
    """
    Run the experiment for all (model, template, group) combos.

    Returns {model: response_cache} with all results populated.
    """
    # Load caches and purge errors
    caches: dict[str, dict] = {}
    for model in models:
        caches[model] = load_response_cache(model)
        purged = purge_response_errors(caches[model], model)
        if purged:
            print(f"  Purged {purged} cached error(s) for {model}")

    # Build flat job list (handles dedup)
    jobs = build_job_list(models, caches)

    total_needed = len(jobs)
    if total_needed == 0:
        print(f"\n  All results already cached.")
    else:
        print(f"\n  {total_needed} queries remaining ({MAX_PARALLEL_REQUESTS} parallel).")

    # Concurrency controls
    cache_locks: dict[str, asyncio.Lock] = {m: asyncio.Lock() for m in models}
    semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)
    done = 0
    done_lock = asyncio.Lock()

    async def _run_job(model, template, group, iter_idx, messages):
        nonlocal done
        async with semaphore:
            result = await run_single_query(provider, model, messages, TEMPERATURE)

        async with cache_locks[model]:
            append_response(caches[model], model, messages, TEMPERATURE, result)

        async with done_lock:
            done += 1
            short = model.split("/")[-1]
            group_text = get_group_text(group, template["form"])
            status = "OK" if result["error"] is None else f"ERR: {result['error']}"
            print(f"  [{done}/{total_needed}] {short} | {template['id']} x {group_text} iter={iter_idx+1} -> {status}")

    if jobs:
        await asyncio.gather(*[_run_job(m, t, g, i, msgs) for m, t, g, i, msgs in jobs])

    # Reload caches from disk
    for model in models:
        caches[model] = load_response_cache(model)

    return caches
