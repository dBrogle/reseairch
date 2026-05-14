"""Parallel execution engine for the Idea Priming study.

Builds messages for each (model, frame, idea, iteration) combo and runs
them concurrently, bounded by a shared semaphore.
"""

import asyncio

from services.llm import OpenRouterProvider
from studies.idea_priming.config import (
    FRAME_KEYS,
    FRAME_PRIMINGS,
    IDEAS,
    IDEA_BY_ID,
    MAX_PARALLEL_REQUESTS,
    MAX_RETRIES,
    PROMPT_TEMPLATE,
    SYSTEM_MESSAGE,
)
from studies.idea_priming.cache import (
    append_response,
    get_responses,
    load_response_cache,
    purge_response_errors,
)


def build_messages(frame_key: str, idea: dict) -> list[dict]:
    """Build the full message array for a (frame, idea) pairing."""
    user_msg = PROMPT_TEMPLATE.format(
        idea_description=idea["description"],
        priming_question=FRAME_PRIMINGS[frame_key],
    )
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_msg},
    ]


async def _run_single_query(
    provider: OpenRouterProvider,
    model: str,
    messages: list[dict],
    temperature: float,
) -> tuple[dict, dict]:
    """Run a single query with exponential backoff retries."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response, usage = await provider.complete_text_with_usage(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=1000,
            )
            return {"response": response, "error": None}, usage
        except Exception as e:
            last_error = e
            if attempt <= MAX_RETRIES:
                wait = 2 ** (attempt - 1)
                await asyncio.sleep(wait)
    return {"response": None, "error": str(last_error)}, {}


async def run_all(
    provider: OpenRouterProvider,
    models: list[str],
    ideas: list[dict],
    temperature: float,
    iterations: int,
    frame_keys: list[str] | None = None,
    cost_tracker=None,
) -> dict[str, dict]:
    """
    Run the experiment for all (model, frame, idea) combos × iterations.

    Returns {model: response_cache} with all results populated.
    """
    target_frames = frame_keys or FRAME_KEYS

    # Load caches and purge errors so they get retried
    caches: dict[str, dict] = {}
    for model in models:
        caches[model] = load_response_cache(model)
        purged = purge_response_errors(caches[model], model)
        if purged:
            print(f"  Purged {purged} cached error(s) for {model}")

    # Build flat job list: (model, frame, idea, iter_idx, messages)
    jobs = []
    for model in models:
        for frame_key in target_frames:
            for idea in ideas:
                messages = build_messages(frame_key, idea)
                existing = len(get_responses(caches[model], messages, temperature))
                for i in range(existing, iterations):
                    jobs.append((model, frame_key, idea, i, messages))

    total_needed = len(jobs)
    total_possible = len(models) * len(target_frames) * len(ideas) * iterations
    total_cached = total_possible - total_needed

    if cost_tracker:
        cost_tracker.start_phase()

    if total_needed == 0:
        print(f"\n  All {total_possible} results already cached.")
    else:
        print(
            f"\n  {total_cached} cached, {total_needed} remaining "
            f"({MAX_PARALLEL_REQUESTS} parallel)."
        )

    cache_locks: dict[str, asyncio.Lock] = {m: asyncio.Lock() for m in models}
    semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)
    done = 0
    done_lock = asyncio.Lock()

    async def _run_job(model, frame_key, idea, iter_idx, messages):
        nonlocal done
        async with semaphore:
            result, usage = await _run_single_query(provider, model, messages, temperature)

        if cost_tracker and usage:
            await cost_tracker.record(model, usage)

        async with cache_locks[model]:
            append_response(
                caches[model], model, messages, temperature, result,
                idea=idea, frame_key=frame_key,
            )

        async with done_lock:
            done += 1
            short = model.split("/")[-1]
            status = "OK" if result["error"] is None else f"ERR: {result['error']}"
            cost_str = f" | {cost_tracker.format_status(done, total_needed)}" if cost_tracker else ""
            print(
                f"  [{done}/{total_needed}] {short} | {frame_key:>8} | "
                f"{idea['id']} iter={iter_idx + 1} -> {status}{cost_str}"
            )

    if jobs:
        await asyncio.gather(*[_run_job(m, fk, q, i, msgs) for m, fk, q, i, msgs in jobs])

    if cost_tracker and total_needed > 0:
        print(f"\n  Runner cost: {cost_tracker.format_total()}")

    # Reload caches from disk
    for model in models:
        caches[model] = load_response_cache(model)

    return caches
