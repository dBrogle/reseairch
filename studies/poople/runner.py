"""Parallel execution engine for the Poople LLM test.

Runs a flat job list of (model, word, iteration) for one condition concurrently
under a shared semaphore, grades each response immediately, and caches it.

Condition:
  * no_reasoning -> reasoning is disabled (with an omit-the-field fallback for
    endpoints that 400 because reasoning is mandatory).
  * reasoning    -> reasoning enabled, the model thinks before answering.
"""

import asyncio

from services.llm import OpenRouterProvider
from studies.poople.config import (
    ITERATIONS,
    MAX_PARALLEL_REQUESTS,
    MAX_RETRIES,
    MAX_TOKENS,
    TEMPERATURE,
)
from studies.poople.grader import grade_attempt
from studies.poople.llm_cache import (
    append_result,
    get_results,
    load_cache,
    purge_errors,
)
from studies.poople.prompt import build_messages


async def _complete(
    provider: OpenRouterProvider, model: str, messages: list[dict], enable_reasoning: bool
) -> str:
    """Completion for one condition.

    With reasoning enabled we just ask the model to think. With it disabled we
    send reasoning:{enabled:false}; a few endpoints 400 because reasoning is
    mandatory, so we retry once with the field omitted (their minimal default).
    """
    if enable_reasoning:
        return await provider.complete_text(
            messages=messages, model=model, temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS, enable_reasoning=True,
        )
    try:
        return await provider.complete_text(
            messages=messages, model=model, temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS, enable_reasoning=False,
        )
    except RuntimeError as e:
        if "reasoning" in str(e).lower():
            return await provider.complete_text(
                messages=messages, model=model, temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS, omit_reasoning=True,
            )
        raise


async def run_single(
    provider: OpenRouterProvider,
    model: str,
    word: str,
    par: int,
    words: set[str],
    enable_reasoning: bool,
) -> dict:
    """Run + grade one Poople attempt, with retry/backoff on transient errors."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response = await _complete(provider, model, build_messages(word), enable_reasoning)
            grade = grade_attempt(response, word, par, words)
            return {"response": response, "error": None, **grade}
        except Exception as e:
            last_error = e
            if attempt <= MAX_RETRIES:
                await asyncio.sleep(2 ** (attempt - 1))
    return {
        "response": None, "error": str(last_error), "par": par, "parsed": False,
        "num_moves": 0, "illegal_moves": 0, "reached_target": False,
        "solved": False, "over_par": None, "ladder": None,
    }


async def run_all(
    provider: OpenRouterProvider,
    models: list[str],
    test_words: list[tuple[str, int]],
    words: set[str],
    condition: str,
    enable_reasoning: bool,
) -> None:
    """Run every (model, word) `ITERATIONS` times in parallel for one condition."""
    caches: dict[str, dict] = {}
    for model in models:
        caches[model] = load_cache(model, condition)
        purged = 0
        for word, _ in test_words:
            purged += purge_errors(caches[model], model, condition, build_messages(word))
        if purged:
            print(f"  Purged {purged} cached error(s) for {model}")

    jobs: list[tuple[str, str, int, int]] = []
    for model in models:
        for word, par in test_words:
            existing = len(get_results(caches[model], build_messages(word)))
            for i in range(existing, ITERATIONS):
                jobs.append((model, word, par, i))

    total_possible = len(models) * len(test_words) * ITERATIONS
    total_needed = len(jobs)
    if total_needed == 0:
        print(f"\n  [{condition}] all {total_possible} attempts already cached.")
    else:
        print(f"\n  [{condition}] {total_possible - total_needed} cached, "
              f"{total_needed} remaining ({MAX_PARALLEL_REQUESTS} parallel).")

    semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)
    cache_locks = {m: asyncio.Lock() for m in models}
    done = 0
    done_lock = asyncio.Lock()

    async def _run_job(model, word, par, iter_idx):
        nonlocal done
        async with semaphore:
            result = await run_single(provider, model, word, par, words, enable_reasoning)
        async with cache_locks[model]:
            append_result(caches[model], model, condition, build_messages(word), result, word, par)
        async with done_lock:
            done += 1
            short = model.split("/")[-1]
            if result["error"]:
                status = f"ERROR: {result['error'][:50]}"
            elif result["solved"]:
                status = f"solved +{result['over_par']}"
            elif not result["parsed"]:
                status = "unparseable"
            elif result["reached_target"]:
                status = f"reached, {result['illegal_moves']} illegal"
            else:
                status = f"failed ({result['illegal_moves']} illegal)"
            print(f"  [{condition} {done}/{total_needed}] {short} | {word} (par {par}) "
                  f"iter={iter_idx + 1} -> {status}")

    if jobs:
        await asyncio.gather(*[_run_job(*j) for j in jobs])
