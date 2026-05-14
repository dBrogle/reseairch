"""LLM-powered answer grading for the Question Framing study.

Takes raw LLM responses and uses a separate LLM to determine whether each
response is CORRECT or INCORRECT relative to the known HLE answer. Gradings
are batched (configurable batch size) and cached.
"""

import asyncio
import json
import math
import random

from services.llm import OpenRouterProvider
from studies.question_framing.config import (
    FRAMING_KEYS,
    TEMPERATURE,
    ITERATIONS,
    GRADER_MODEL,
    GRADER_SYSTEM_PROMPT,
    GRADER_BATCH_SIZE,
    MAX_PARALLEL_REQUESTS,
)
from studies.question_framing.runner import build_messages
from studies.question_framing.cache import (
    load_response_cache,
    get_responses,
    load_grading_cache,
    get_grading,
    set_grading,
)


# ---------------------------------------------------------------------------
# Batch grading
# ---------------------------------------------------------------------------

def _build_grader_input(
    items: list[tuple[dict, str]],  # (question, response_text)
) -> str:
    """Build the JSON input for a batch of (question, response_text) pairs."""
    batch = {}
    for idx, (question, response_text) in enumerate(items):
        batch[str(idx)] = {
            "question":    question["question"],
            "answer":      question["answer"],
            "answer_type": question["answer_type"],
            "response":    response_text,
        }
    return json.dumps(batch, indent=2)


def _parse_grader_output(raw: str, batch_size: int) -> dict[int, str]:
    """Parse the grader's JSON response into {index: verdict}."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

    parsed = json.loads(text)
    results = {}
    for idx_str, verdict in parsed.items():
        idx = int(idx_str)
        if idx < 0 or idx >= batch_size:
            continue
        verdict = str(verdict).strip().upper()
        if verdict in ("CORRECT", "INCORRECT"):
            results[idx] = verdict
        else:
            results[idx] = "INCORRECT"
    return results


async def _grade_batch(
    provider: OpenRouterProvider,
    items: list[tuple[dict, str]],
    cost_tracker=None,
) -> list[str]:
    """Grade a batch of (question, response_text) pairs via a single LLM call."""
    if not items:
        return []

    indices = list(range(len(items)))
    random.shuffle(indices)
    shuffled = [items[i] for i in indices]

    batch_json = _build_grader_input(shuffled)

    try:
        result, usage = await provider.complete_text_with_usage(
            messages=[
                {"role": "system", "content": GRADER_SYSTEM_PROMPT},
                {"role": "user", "content": batch_json},
            ],
            model=GRADER_MODEL,
            temperature=0.0,
            max_tokens=300,
        )
        if cost_tracker and usage:
            await cost_tracker.record(GRADER_MODEL, usage)
        parsed = _parse_grader_output(result, len(shuffled))
    except Exception:
        return ["ERROR"] * len(items)

    # Un-shuffle back to original order
    verdicts = ["ERROR"] * len(items)
    for shuffled_idx, original_idx in enumerate(indices):
        verdicts[original_idx] = parsed.get(shuffled_idx, "ERROR")

    return verdicts


# ---------------------------------------------------------------------------
# Grade all cached responses
# ---------------------------------------------------------------------------

async def grade_all(
    provider: OpenRouterProvider,
    models: list[str],
    questions: list[dict],
    framing_keys: list[str] | None = None,
    cost_tracker=None,
) -> dict[str, dict]:
    """
    Grade all cached raw responses using batched LLM calls.

    Returns {model: grading_cache} with all verdicts populated.
    """
    target_framings = framing_keys or FRAMING_KEYS

    # Build flat job list of items needing grading
    jobs: list[tuple[str, dict, str, str, int]] = []
    for model in models:
        response_cache = load_response_cache(model)
        grading_cache  = load_grading_cache(model)

        for framing_key in target_framings:
            for question in questions:
                messages = build_messages(framing_key, question)
                responses = get_responses(response_cache, messages, TEMPERATURE)[:ITERATIONS]
                for i, result in enumerate(responses):
                    if result.get("error") is not None or result.get("response") is None:
                        continue
                    existing = get_grading(
                        grading_cache, question["id"],
                        result["response"], GRADER_MODEL,
                    )
                    if existing is None:
                        jobs.append((model, question, result["response"], framing_key, i))

    total_needed = len(jobs)
    if total_needed == 0:
        print("\n  All gradings already cached.")
        return {m: load_grading_cache(m) for m in models}

    if cost_tracker:
        cost_tracker.start_phase()

    num_batches = (total_needed + GRADER_BATCH_SIZE - 1) // GRADER_BATCH_SIZE
    print(
        f"\n  {total_needed} gradings needed in "
        f"{num_batches} batches of up to {GRADER_BATCH_SIZE}."
    )

    batches = [
        jobs[i:i + GRADER_BATCH_SIZE]
        for i in range(0, total_needed, GRADER_BATCH_SIZE)
    ]

    grading_caches: dict[str, dict] = {m: load_grading_cache(m) for m in models}
    cache_locks: dict[str, asyncio.Lock] = {m: asyncio.Lock() for m in models}
    semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)
    done_batches = 0
    done_lock = asyncio.Lock()

    async def _run_batch(batch: list[tuple[str, dict, str, str, int]]):
        nonlocal done_batches

        items = [(question, response_text) for _, question, response_text, _, _ in batch]

        async with semaphore:
            verdicts = await _grade_batch(provider, items, cost_tracker=cost_tracker)

        for (model, question, response_text, framing_key, iter_idx), verdict in zip(batch, verdicts):
            async with cache_locks[model]:
                set_grading(
                    grading_caches[model], model,
                    question["id"], response_text, GRADER_MODEL, verdict,
                )

        async with done_lock:
            done_batches += 1
            sample = ", ".join(verdicts[:3])
            if len(verdicts) > 3:
                sample += ", ..."
            cost_str = f" | {cost_tracker.format_status(done_batches, num_batches)}" if cost_tracker else ""
            print(f"  [batch {done_batches}/{num_batches}] {len(batch)} items -> [{sample}]{cost_str}")

    await asyncio.gather(*[_run_batch(b) for b in batches])

    if cost_tracker and total_needed > 0:
        print(f"\n  Grader cost: {cost_tracker.format_total()}")

    for model in models:
        grading_caches[model] = load_grading_cache(model)

    return grading_caches


# ---------------------------------------------------------------------------
# Accuracy computation
# ---------------------------------------------------------------------------

# z-value for a two-sided 95% confidence interval
_Z_95 = 1.959963984540054


def wilson_ci(correct: int, total: int, z: float = _Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    More reliable than the normal approximation for small n and for proportions
    near 0 or 1. Returns (low, high) clamped to [0, 1].
    """
    if total <= 0:
        return (0.0, 1.0)
    p = correct / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half = (
        z * math.sqrt((p * (1.0 - p) / total) + (z * z) / (4.0 * total * total))
    ) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def compute_framing_accuracy(
    model: str,
    questions: list[dict],
    framing_keys: list[str] | None = None,
) -> dict[str, dict | None]:
    """
    Compute per-framing accuracy stats for a model.

    Returns {framing_key: stats_dict} where each stats_dict has:
        - accuracy:   correct / total (0.0-1.0)
        - ci_low:     Wilson 95% CI lower bound
        - ci_high:    Wilson 95% CI upper bound
        - n_correct:  number of CORRECT verdicts
        - n_total:    total graded (CORRECT + INCORRECT, excluding ERROR/None)
    None is returned for framings with no valid data.
    """
    target_framings = framing_keys or FRAMING_KEYS
    response_cache = load_response_cache(model)
    grading_cache  = load_grading_cache(model)

    framing_accuracy: dict[str, dict | None] = {}
    for framing_key in target_framings:
        correct = 0
        total = 0
        for question in questions:
            messages = build_messages(framing_key, question)
            responses = get_responses(response_cache, messages, TEMPERATURE)[:ITERATIONS]
            for result in responses:
                if result.get("error") is not None or result.get("response") is None:
                    continue
                verdict = get_grading(
                    grading_cache, question["id"],
                    result["response"], GRADER_MODEL,
                )
                if verdict == "CORRECT":
                    correct += 1
                    total += 1
                elif verdict == "INCORRECT":
                    total += 1

        if total == 0:
            framing_accuracy[framing_key] = None
        else:
            low, high = wilson_ci(correct, total)
            framing_accuracy[framing_key] = {
                "accuracy":  correct / total,
                "ci_low":    low,
                "ci_high":   high,
                "n_correct": correct,
                "n_total":   total,
            }

    return framing_accuracy


def compute_aggregated_framing_accuracy(
    models: list[str],
    questions: list[dict],
    framing_keys: list[str] | None = None,
) -> dict[str, dict | None]:
    """Pool every (model, question) trial per framing and compute accuracy + Wilson CI.

    Same shape as compute_framing_accuracy(), with an extra `n_models` field
    indicating how many models contributed data to each framing.
    """
    target_framings = framing_keys or FRAMING_KEYS
    sums = {
        f: {"correct": 0, "total": 0, "models_with_data": 0}
        for f in target_framings
    }

    for model in models:
        per_model = compute_framing_accuracy(
            model, questions, framing_keys=target_framings
        )
        for framing_key, stats in per_model.items():
            if stats is None:
                continue
            sums[framing_key]["correct"] += stats["n_correct"]
            sums[framing_key]["total"]   += stats["n_total"]
            sums[framing_key]["models_with_data"] += 1

    aggregated: dict[str, dict | None] = {}
    for framing_key, totals in sums.items():
        if totals["total"] == 0:
            aggregated[framing_key] = None
        else:
            low, high = wilson_ci(totals["correct"], totals["total"])
            aggregated[framing_key] = {
                "accuracy":  totals["correct"] / totals["total"],
                "ci_low":    low,
                "ci_high":   high,
                "n_correct": totals["correct"],
                "n_total":   totals["total"],
                "n_models":  totals["models_with_data"],
            }
    return aggregated
