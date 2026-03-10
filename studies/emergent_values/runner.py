"""Parallel execution engine for the Emergent Values study.

For each experiment (category×measure), generates pairwise comparisons,
queries the LLM K times per pair in both directions, and caches results.
"""

import asyncio
import math
import random
import re

from services.llm import OpenRouterProvider
from studies.emergent_values.config import (
    COMPARISON_PROMPT,
    SYSTEM_MESSAGE,
    TEMPERATURE,
    MAX_TOKENS,
    MAX_PARALLEL_REQUESTS,
    MAX_RETRIES,
    K,
    BIDIRECTIONAL,
    PAIR_SEED,
    LOGPROB_MODELS,
    generate_options,
    compute_target_pairs,
)
from studies.emergent_values.cache import (
    load_cache,
    get_results,
    append_result,
    purge_errors,
)
from studies.emergent_values.costs import CostTracker


def parse_choice(response: str) -> str | None:
    """Extract 'A' or 'B' from the model response. Returns None if unparseable."""
    text = response.strip()
    if text in ("A", "B"):
        return text
    # Look for A or B as standalone
    match = re.search(r"\b([AB])\b", text)
    if match:
        return match.group(1)
    return None


def extract_logprob_choice(raw_response: dict) -> dict | None:
    """Extract P(A) vs P(B) from the logprobs in the API response.

    Returns {"choice": "A"|"B", "p_a": float} or None if logprobs unavailable.
    """
    choices = raw_response.get("choices", [])
    if not choices:
        return None

    logprobs_data = choices[0].get("logprobs")
    if not logprobs_data:
        return None

    content = logprobs_data.get("content", [])
    if not content:
        return None

    # Search top logprobs of the first token for A and B
    top = content[0].get("top_logprobs", [])
    log_a = None
    log_b = None
    for entry in top:
        token = entry.get("token", "").strip()
        if token == "A" and log_a is None:
            log_a = entry["logprob"]
        elif token == "B" and log_b is None:
            log_b = entry["logprob"]

    if log_a is None and log_b is None:
        return None

    # Normalize P(A) + P(B) = 1 via softmax over just A and B
    if log_a is not None and log_b is not None:
        max_log = max(log_a, log_b)
        exp_a = math.exp(log_a - max_log)
        exp_b = math.exp(log_b - max_log)
        p_a = exp_a / (exp_a + exp_b)
    elif log_a is not None:
        p_a = 1.0
    else:
        p_a = 0.0

    choice = "A" if p_a >= 0.5 else "B"
    return {"choice": choice, "p_a": p_a, "log_a": log_a, "log_b": log_b}


def sample_pairs(
    options: list[tuple[str, str]], seed: int
) -> list[tuple[int, int]]:
    """Sample pairs of option indices for comparison.

    Returns list of (idx_a, idx_b) tuples. If BIDIRECTIONAL, the reversed
    pair (idx_b, idx_a) is handled at query time, not in the pair list.
    """
    n = len(options)
    target = compute_target_pairs(n)
    total_possible = n * (n - 1) // 2

    # Generate all possible pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    if target >= total_possible:
        return all_pairs

    rng = random.Random(seed)
    return rng.sample(all_pairs, target)


async def run_single_query(
    provider: OpenRouterProvider, model: str,
    option_a_text: str, option_b_text: str,
    cost_tracker: CostTracker | None = None,
) -> dict:
    """Run a single A/B comparison query with retries. Tracks usage via cost_tracker."""
    prompt = COMPARISON_PROMPT.format(option_a=option_a_text, option_b=option_b_text)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]
    last_error = None
    use_logprobs = model in LOGPROB_MODELS

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            # Call _call_api directly to get the full response with usage data
            raw_response = await provider._call_api(
                messages, model, TEMPERATURE, MAX_TOKENS,
                logprobs=use_logprobs, top_logprobs=5 if use_logprobs else None,
            )

            # Try logprobs first (works even when model "refuses")
            logprob_result = extract_logprob_choice(raw_response)

            # Fall back to text parsing
            try:
                text = provider._extract_text(raw_response)
            except ValueError:
                text = ""
            text_choice = parse_choice(text)

            # Use logprobs if available, otherwise text
            if logprob_result:
                choice = logprob_result["choice"]
                p_a = logprob_result["p_a"]
                source = "logprobs"
            elif text_choice:
                choice = text_choice
                p_a = 1.0 if text_choice == "A" else 0.0
                source = "text"
            else:
                choice = None
                p_a = None
                source = "none"

            # Extract usage from OpenRouter response
            usage = raw_response.get("usage", {})
            if cost_tracker:
                await cost_tracker.record(
                    model=model,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    cost=usage.get("cost"),
                )

            return {
                "response": text,
                "choice": choice,
                "p_a": p_a,
                "source": source,
                "log_a": logprob_result["log_a"] if logprob_result else None,
                "log_b": logprob_result["log_b"] if logprob_result else None,
                "text_choice": text_choice,
                "error": None,
            }
        except Exception as e:
            last_error = e
            if cost_tracker:
                await cost_tracker.record(
                    model=model, prompt_tokens=0, completion_tokens=0, is_error=True,
                )
            if attempt <= MAX_RETRIES:
                wait = 2 ** (attempt - 1)
                print(f"    retry {attempt}/{MAX_RETRIES} for {model}: {e} (waiting {wait}s)")
                await asyncio.sleep(wait)

    return {
        "response": None,
        "choice": None,
        "error": str(last_error),
    }


async def run_experiment(
    provider: OpenRouterProvider,
    model: str,
    category: str,
    measure: str,
    cost_tracker: CostTracker | None = None,
) -> dict:
    """Run a single experiment (category×measure) for one model.

    Returns the loaded cache dict with all results.
    """
    options = generate_options(category, measure)
    pairs = sample_pairs(options, PAIR_SEED)

    cache = load_cache(model, category, measure)
    purged = purge_errors(cache, model, category, measure)
    if purged:
        print(f"  Purged {purged} cached error(s) for {model} {category}×{measure}")

    # Build job list: (pair_idx, direction, iteration)
    # direction 0 = original (A,B), direction 1 = flipped (B,A)
    directions = [0, 1] if BIDIRECTIONAL else [0]
    jobs = []

    for pair_idx, (idx_a, idx_b) in enumerate(pairs):
        for direction in directions:
            if direction == 0:
                oa_text = options[idx_a][1]
                ob_text = options[idx_b][1]
            else:
                oa_text = options[idx_b][1]
                ob_text = options[idx_a][1]

            existing = len(get_results(cache, oa_text, ob_text, TEMPERATURE))
            for k in range(existing, K):
                jobs.append((pair_idx, idx_a, idx_b, direction, k, oa_text, ob_text))

    total_needed = len(jobs)
    directions_count = len(directions)
    total_possible = len(pairs) * directions_count * K
    total_cached = total_possible - total_needed

    if total_needed == 0:
        print(f"\n  All {total_possible} comparisons already cached for {category}×{measure}.")
    else:
        print(
            f"\n  {category}×{measure}: {len(options)} options, {len(pairs)} pairs, "
            f"{directions_count} direction(s), K={K}"
        )
        print(f"  {total_cached} cached, {total_needed} remaining ({MAX_PARALLEL_REQUESTS} parallel)")

    # Run jobs
    cache_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)
    done = 0
    successes = 0
    errors = 0
    unparseable = 0
    done_lock = asyncio.Lock()

    # Snapshot starting cost so we only project based on this run's spend
    starting_cost = 0.0
    starting_tokens = 0
    if cost_tracker:
        summary = cost_tracker.get_all().get(model, {})
        starting_cost = summary.get("total_cost_usd", 0)
        starting_tokens = summary.get("total_tokens", 0)

    async def _run_job(pair_idx, idx_a, idx_b, direction, k, oa_text, ob_text):
        nonlocal done, successes, errors, unparseable
        async with semaphore:
            result = await run_single_query(provider, model, oa_text, ob_text, cost_tracker)

        async with cache_lock:
            # Only cache successful results — errors don't consume slots
            # so they'll be retried on the next run
            if result.get("error") is None:
                append_result(cache, model, category, measure, oa_text, ob_text, TEMPERATURE, result)

        async with done_lock:
            done += 1
            if result.get("error") is not None:
                errors += 1
            elif result.get("choice") is None:
                unparseable += 1
            else:
                successes += 1

            if done % 50 == 0 or done == total_needed:
                pct = done / total_needed * 100
                success_rate = successes / done * 100 if done > 0 else 0
                err_str = f" | {success_rate:.0f}% success ({errors} err, {unparseable} unparsed)"

                cost_info = ""
                if cost_tracker:
                    summary = cost_tracker.get_all().get(model, {})
                    run_cost = summary.get("total_cost_usd", 0) - starting_cost
                    run_tokens = summary.get("total_tokens", 0) - starting_tokens
                    if run_cost > 0:
                        projected_run = run_cost * (total_needed / done) if done > 0 else 0
                        cost_info = f" | ${run_cost:.4f} this run, ~${projected_run:.4f} projected"
                    elif run_tokens > 0:
                        cost_info = f" | {run_tokens:,} tokens this run"
                print(f"  [{done}/{total_needed} {pct:.0f}%] {category}×{measure}{err_str}{cost_info}")

    if jobs:
        await asyncio.gather(*[
            _run_job(pi, ia, ib, d, k, oa, ob)
            for pi, ia, ib, d, k, oa, ob in jobs
        ])

    return load_cache(model, category, measure)


async def run_all(
    provider: OpenRouterProvider,
    models: list[str],
    experiments: list[tuple[str, str]],
    cost_tracker: CostTracker | None = None,
) -> dict[str, dict[tuple[str, str], dict]]:
    """Run all experiments for all models.

    Returns {model: {(category, measure): cache_dict}}
    """
    all_results = {}
    for model in models:
        all_results[model] = {}
        for category, measure in experiments:
            cache = await run_experiment(provider, model, category, measure, cost_tracker)
            all_results[model][(category, measure)] = cache

    if cost_tracker:
        print("\n  === Cost Summary ===")
        cost_tracker.print_summary()

    return all_results
