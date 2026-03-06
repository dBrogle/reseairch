"""Parallel execution engine for the Biases study.

Builds a flat job list of (model, subject, action, iteration) and runs
them concurrently, bounded by a shared semaphore.
"""

import asyncio
import re

from services.llm import OpenRouterProvider
from studies.biases.config import (
    SUBJECTS,
    ACTIONS,
    TEMPERATURE,
    ITERATIONS,
    PROMPT_TEMPLATE,
    MAX_PARALLEL_REQUESTS,
    MAX_RETRIES,
    SKIP_COMBOS,
)
from studies.biases.cache import (
    load_cache,
    get_results,
    append_result,
    purge_errors,
)


def build_prompt(subject: str, action: str) -> str:
    """Build the full prompt for a given subject/action combo."""
    return PROMPT_TEMPLATE.format(subject=subject, action=action)


def parse_score(response: str) -> int | None:
    """Extract the Likert score (1-7) from a model response.

    Looks for 'Score: <N>' pattern. Returns None if not found or invalid.
    """
    match = re.search(r"[Ss]core:\s*(\d)", response)
    if match:
        val = int(match.group(1))
        if 1 <= val <= 7:
            return val
    return None


async def run_single_query(
    provider: OpenRouterProvider, model: str, prompt: str, temperature: float
) -> dict:
    """Run a single bias query, retrying up to MAX_RETRIES times on error."""
    messages = [{"role": "user", "content": prompt}]
    last_error = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response = await provider.complete_text(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=200,
            )
            score = parse_score(response)
            return {
                "response": response,
                "score": score,
                "error": None,
            }
        except Exception as e:
            last_error = e
            if attempt <= MAX_RETRIES:
                wait = 2 ** (attempt - 1)
                print(f"    retry {attempt}/{MAX_RETRIES} for {model}: {e} (waiting {wait}s)")
                await asyncio.sleep(wait)
    return {
        "response": None,
        "score": None,
        "error": str(last_error),
    }


async def run_all(
    provider: OpenRouterProvider, models: list[str]
) -> dict[str, dict[str, dict[str, list[dict]]]]:
    """
    Run the full experiment across all models in parallel.

    Returns {model_id: {subject: {action: [results]}}}
    """
    # Build prompt lookup
    prompts: dict[tuple[str, str], str] = {}
    for subject in SUBJECTS:
        for action in ACTIONS:
            prompts[(subject, action)] = build_prompt(subject, action)

    # Load caches and purge errors
    caches: dict[str, dict] = {}
    for model in models:
        caches[model] = load_cache(model)
        purged = purge_errors(caches[model], model)
        if purged:
            print(f"  Purged {purged} cached error(s) for {model}")

    # Build flat job list
    jobs: list[tuple[str, str, str, int]] = []  # (model, subject, action, iter_idx)
    for model in models:
        for subject in SUBJECTS:
            for action in ACTIONS:
                if (model, action, subject) in SKIP_COMBOS:
                    continue
                prompt = prompts[(subject, action)]
                existing = len(get_results(caches[model], prompt, TEMPERATURE))
                for i in range(existing, ITERATIONS):
                    jobs.append((model, subject, action, i))

    total_needed = len(jobs)
    total_possible = len(models) * len(SUBJECTS) * len(ACTIONS) * ITERATIONS
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

    async def _run_job(model: str, subject: str, action: str, iter_idx: int):
        nonlocal done
        prompt = prompts[(subject, action)]
        async with semaphore:
            result = await run_single_query(provider, model, prompt, TEMPERATURE)

        async with cache_locks[model]:
            append_result(caches[model], model, prompt, TEMPERATURE, result)

        async with done_lock:
            done += 1
            short_model = model.split("/")[-1]
            score_str = str(result["score"]) if result["score"] is not None else "ERR"
            print(f"  [{done}/{total_needed}] {short_model} {action}/{subject} iter={iter_idx+1} -> score={score_str}")

    if jobs:
        await asyncio.gather(*[_run_job(m, s, a, i) for m, s, a, i in jobs])

    # Build final results from cache
    all_results: dict[str, dict[str, dict[str, list[dict]]]] = {}
    for model in models:
        final_cache = load_cache(model)
        model_results: dict[str, dict[str, list[dict]]] = {}
        for subject in SUBJECTS:
            model_results[subject] = {}
            for action in ACTIONS:
                prompt = prompts[(subject, action)]
                model_results[subject][action] = get_results(
                    final_cache, prompt, TEMPERATURE
                )[:ITERATIONS]
        all_results[model] = model_results

    return all_results
