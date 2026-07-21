"""Parallel execution engine for the Surgeon Riddle study.

Builds the conversational message array for each (model, condition) and runs it
concurrently for ITERATIONS iterations, flushing every result to disk.
"""

import asyncio

from services.llm import OpenRouterProvider
from studies.surgeon_riddle.config import (
    CONDITIONS,
    TEMPERATURE,
    MAX_PARALLEL_REQUESTS,
    MAX_RETRIES,
    SYSTEM_MESSAGE,
    CONVERSATION_SEED,
    REASONING_LOW,
    iterations_for,
)
from studies.surgeon_riddle.cache import (
    load_response_cache,
    get_responses,
    append_response,
    purge_response_errors,
)


def build_messages(condition: dict) -> list[dict]:
    """Build the full conversational message array for a condition."""
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    for role, content in CONVERSATION_SEED:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": condition["prompt"]})
    return messages


async def run_single_query(
    provider: OpenRouterProvider,
    model: str,
    messages: list[dict],
    temperature: float,
) -> dict:
    """Run a single query with retries.

    Reasoning is off for every model (reply directly, don't think first), except
    the mandatory-reasoning endpoints in REASONING_LOW, which reject a disable
    and instead run at the lowest reasoning effort — with extra token headroom so
    the (unavoidable) reasoning doesn't crowd out the visible answer.
    """
    if model in REASONING_LOW:
        reasoning_kwargs = {"reasoning_effort": "low"}
        max_tokens = 2500
    else:
        reasoning_kwargs = {"enable_reasoning": False}
        max_tokens = 500

    last_error = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response, cost = await provider.complete_text_with_cost(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **reasoning_kwargs,
            )
            return {"response": response, "error": None, "cost": cost}
        except Exception as e:
            last_error = e
            if attempt <= MAX_RETRIES:
                wait = 2 ** (attempt - 1)
                await asyncio.sleep(wait)
    return {"response": None, "error": str(last_error), "cost": None}


async def run_all(
    provider: OpenRouterProvider,
    models: list[str],
) -> dict[str, dict]:
    """Run the experiment for all (model, condition) combos."""
    caches: dict[str, dict] = {}
    for model in models:
        caches[model] = load_response_cache(model)
        purged = purge_response_errors(caches[model], model)
        if purged:
            print(f"  Purged {purged} cached error(s) for {model}")

    # Build flat job list, skipping combos already fully cached.
    jobs = []
    for model in models:
        for condition in CONDITIONS:
            messages = build_messages(condition)
            existing = len(get_responses(caches[model], messages, TEMPERATURE))
            for i in range(existing, iterations_for(model)):
                jobs.append((model, condition, i, messages))

    total_needed = len(jobs)
    total_possible = sum(len(CONDITIONS) * iterations_for(m) for m in models)
    total_cached = total_possible - total_needed

    if total_needed == 0:
        print(f"\n  All {total_possible} results already cached.")
    else:
        print(f"\n  {total_cached} cached, {total_needed} remaining ({MAX_PARALLEL_REQUESTS} parallel).")

    cache_locks: dict[str, asyncio.Lock] = {m: asyncio.Lock() for m in models}
    semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)
    done = 0
    done_lock = asyncio.Lock()
    run_cost = 0.0          # USD spent on the calls actually made this run
    cost_by_model: dict[str, float] = {m: 0.0 for m in models}

    async def _run_job(model, condition, iter_idx, messages):
        nonlocal done, run_cost
        async with semaphore:
            result = await run_single_query(provider, model, messages, TEMPERATURE)

        async with cache_locks[model]:
            append_response(caches[model], model, messages, TEMPERATURE, result)

        async with done_lock:
            done += 1
            if result.get("cost"):
                run_cost += result["cost"]
                cost_by_model[model] += result["cost"]
            short = model.split("/")[-1]
            status = "OK" if result["error"] is None else f"ERR: {result['error']}"
            print(f"  [{done}/{total_needed}] {short} | {condition['id']} iter={iter_idx+1} -> {status}")

    if jobs:
        await asyncio.gather(*[_run_job(m, c, i, msgs) for m, c, i, msgs in jobs])
        print(f"\n  Raw-response cost this run: ${run_cost:.4f} over {total_needed} new call(s).")
        for m in models:
            if cost_by_model[m] > 0:
                n = sum(1 for job in jobs if job[0] == m)
                print(f"    {m.split('/')[-1]:<20} ${cost_by_model[m]:.4f}  "
                      f"(${cost_by_model[m]/n:.5f}/call over {n} calls)")

    for model in models:
        caches[model] = load_response_cache(model)

    return caches
