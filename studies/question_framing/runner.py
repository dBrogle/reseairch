"""Parallel execution engine for the Question Framing study.

Builds messages for each (model, framing, question) combo and runs them
concurrently, bounded by a shared semaphore.
"""

import asyncio

from services.llm import OpenRouterProvider
from studies.question_framing.config import (
    FRAMING_KEYS,
    FRAMING_SUFFIXES,
    TEMPERATURE,
    ITERATIONS,
    MAX_PARALLEL_REQUESTS,
    MAX_RETRIES,
    SYSTEM_MESSAGE,
    CONVERSATION_SEED,
    QUESTION_TEMPLATE,
    QUESTION_TEMPLATE_CONTROL,
)
from studies.question_framing.cache import (
    load_response_cache,
    get_responses,
    append_response,
    purge_response_errors,
)


def build_messages(framing_key: str, question: dict) -> list[dict]:
    """Build the full message array for a (framing, question) combo."""
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
    ]
    for role, content in CONVERSATION_SEED:
        messages.append({"role": role, "content": content})

    suffix = FRAMING_SUFFIXES[framing_key]
    if suffix:
        user_msg = QUESTION_TEMPLATE.format(
            question=question["question"],
            framing_suffix=suffix,
        )
    else:
        user_msg = QUESTION_TEMPLATE_CONTROL.format(question=question["question"])

    messages.append({"role": "user", "content": user_msg})
    return messages


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
                max_tokens=2000,
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
    questions: list[dict],
    framing_keys: list[str] | None = None,
    cost_tracker=None,
) -> dict[str, dict]:
    """
    Run the experiment for all (model, framing, question) combos.

    Returns {model: response_cache} with all results populated.
    """
    target_framings = framing_keys or FRAMING_KEYS

    # Load caches and purge errors
    caches: dict[str, dict] = {}
    for model in models:
        caches[model] = load_response_cache(model)
        purged = purge_response_errors(caches[model], model)
        if purged:
            print(f"  Purged {purged} cached error(s) for {model}")

    # Build flat job list: (model, framing_key, question, iter_idx, messages)
    jobs = []
    for model in models:
        for framing_key in target_framings:
            for question in questions:
                messages = build_messages(framing_key, question)
                existing = len(get_responses(caches[model], messages, TEMPERATURE))
                for i in range(existing, ITERATIONS):
                    jobs.append((model, framing_key, question, i, messages))

    total_needed = len(jobs)
    total_possible = len(models) * len(target_framings) * len(questions) * ITERATIONS
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

    async def _run_job(model, framing_key, question, iter_idx, messages):
        nonlocal done
        async with semaphore:
            result, usage = await _run_single_query(provider, model, messages, TEMPERATURE)

        if cost_tracker and usage:
            await cost_tracker.record(model, usage)

        async with cache_locks[model]:
            append_response(
                caches[model], model, messages, TEMPERATURE, result,
                question_meta=question, framing_key=framing_key,
            )

        async with done_lock:
            done += 1
            short = model.split("/")[-1]
            status = "OK" if result["error"] is None else f"ERR: {result['error']}"
            cost_str = f" | {cost_tracker.format_status(done, total_needed)}" if cost_tracker else ""
            print(
                f"  [{done}/{total_needed}] {short} | {framing_key} | "
                f"{question['id']} iter={iter_idx + 1} -> {status}{cost_str}"
            )

    if jobs:
        await asyncio.gather(*[_run_job(m, fk, q, i, msgs) for m, fk, q, i, msgs in jobs])

    if cost_tracker and total_needed > 0:
        print(f"\n  Runner cost: {cost_tracker.format_total()}")

    # Reload caches from disk
    for model in models:
        caches[model] = load_response_cache(model)

    return caches
