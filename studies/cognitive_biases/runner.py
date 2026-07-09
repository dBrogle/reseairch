"""Parallel execution engine for the Cognitive Biases study.

Builds a flat job list of (model, scenario, arm, iteration) and runs them
concurrently, bounded by a shared semaphore. Each job runs the arm's
turn list as a real conversation (the runner alternates user/assistant
messages), with the scenario's `response_format` appended to the FINAL
user turn only. Intermediate assistant responses are kept in the result's
transcript for inspection but are not part of the cache key.
"""

import asyncio

from services.llm import OpenRouterProvider
from studies.cognitive_biases.config import (
    MAX_PARALLEL_REQUESTS,
    MAX_RETRIES,
)
from studies.cognitive_biases.cache import (
    append_response,
    get_responses,
    load_response_cache,
    purge_response_errors,
    request_signature,
)
from studies.cognitive_biases.scenarios.base import Arm, Scenario


def _build_user_turns(scenario: Scenario, arm: Arm) -> list[str]:
    """Materialize an arm's user turns with response_format on the last."""
    turns = list(arm.turn_list)
    turns[-1] = f"{turns[-1]}\n\n{scenario.response_format}"
    return turns


async def _run_arm_iteration(
    provider: OpenRouterProvider,
    model: str,
    scenario: Scenario,
    arm: Arm,
    temperature: float,
    cost_tracker=None,
) -> dict:
    """Run one full iteration of an arm (single- or multi-turn).

    Returns {response, error, transcript}. `response` is the FINAL
    assistant message; `transcript` is the full alternating
    user/assistant message list. Intermediate assistant responses are
    re-sampled per iteration (not cached separately).
    """
    user_turns = _build_user_turns(scenario, arm)
    messages: list[dict] = []
    if arm.system is not None:
        messages.append({"role": "system", "content": arm.system})
    final_response: str | None = None
    last_error: Exception | None = None

    for i, turn in enumerate(user_turns):
        is_final = (i == len(user_turns) - 1)
        messages.append({"role": "user", "content": turn})

        # Per-turn retry with exponential backoff. Whole iteration fails
        # if any single turn fails, since the conversation can't continue.
        response: str | None = None
        usage: dict = {}
        for attempt in range(1, MAX_RETRIES + 2):
            try:
                response, usage = await provider.complete_text_with_usage(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=1000,
                )
                last_error = None
                break
            except Exception as e:
                last_error = e
                if attempt <= MAX_RETRIES:
                    await asyncio.sleep(2 ** (attempt - 1))

        if response is None:
            return {
                "response":   None,
                "error":      str(last_error),
                "transcript": messages,
            }

        if cost_tracker and usage:
            await cost_tracker.record(model, usage)

        messages.append({"role": "assistant", "content": response})
        if is_final:
            final_response = response

    return {
        "response":   final_response,
        "error":      None,
        "transcript": messages,
    }


async def run_all(
    provider: OpenRouterProvider,
    models: list[str],
    scenarios: list[Scenario],
    temperature: float,
    iterations: int,
    cost_tracker=None,
) -> dict[str, dict]:
    """Run every (model, scenario, arm) combo `iterations` times in parallel."""
    caches: dict[str, dict] = {}
    for model in models:
        caches[model] = load_response_cache(model)
        purged = purge_response_errors(caches[model], model)
        if purged:
            print(f"  Purged {purged} cached error(s) for {model}")

    # Build flat job list.
    jobs: list[tuple[str, Scenario, Arm, int]] = []
    for model in models:
        for scenario in scenarios:
            for arm in scenario.arms:
                sig = request_signature(scenario, arm)
                existing = len(get_responses(caches[model], sig, temperature))
                for i in range(existing, iterations):
                    jobs.append((model, scenario, arm, i))

    total_needed = len(jobs)
    total_possible = sum(len(s.arms) for s in scenarios) * len(models) * iterations
    total_cached = total_possible - total_needed

    if total_needed == 0:
        print(f"\n  All {total_possible} responses already cached.")
    else:
        print(
            f"\n  {total_cached} cached, {total_needed} remaining "
            f"({MAX_PARALLEL_REQUESTS} parallel)."
        )

    if cost_tracker:
        cost_tracker.start_phase()

    cache_locks: dict[str, asyncio.Lock] = {m: asyncio.Lock() for m in models}
    semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)
    done = 0
    done_lock = asyncio.Lock()

    async def _run_job(model, scenario, arm, iter_idx):
        nonlocal done
        async with semaphore:
            result = await _run_arm_iteration(
                provider, model, scenario, arm, temperature,
                cost_tracker=cost_tracker,
            )

        sig = request_signature(scenario, arm)
        async with cache_locks[model]:
            append_response(
                caches[model], model, sig, temperature, result,
                scenario_id=scenario.id,
                arm_key=arm.key,
                bias_type=scenario.bias_type,
            )

        async with done_lock:
            done += 1
            short = model.split("/")[-1]
            status = "OK" if result["error"] is None else f"ERR: {result['error']}"
            turns_n = len(arm.turn_list)
            turns_str = f"({turns_n}t)" if turns_n > 1 else ""
            cost_str = (
                f" | {cost_tracker.format_status(done, total_needed)}"
                if cost_tracker else ""
            )
            print(
                f"  [{done}/{total_needed}] {short} | {scenario.id} | "
                f"{arm.key}{turns_str} iter={iter_idx + 1} -> {status}{cost_str}"
            )

    if jobs:
        await asyncio.gather(*[_run_job(m, s, a, i) for m, s, a, i in jobs])

    if cost_tracker and total_needed > 0:
        print(f"\n  Runner cost: {cost_tracker.format_total()}")

    return {m: load_response_cache(m) for m in models}
