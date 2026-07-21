"""LLM-powered classification for the Surgeon Riddle study.

Takes each raw LLM answer and uses a separate LLM to classify WHICH parent the
response says the doctor is: MOTHER, FATHER, TWO_SAME (two same-gender parents),
OTHER_PARENT, or OTHER. Batched and shuffled to prevent positional bias.
"""

import asyncio
import hashlib
import json
import random

from services.llm import OpenRouterProvider
from studies.surgeon_riddle.config import (
    CONDITIONS,
    TEMPERATURE,
    JUDGE_MODEL,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_LABELS,
    JUDGE_BATCH_SIZE,
    MAX_PARALLEL_REQUESTS,
    iterations_for,
)

# The judgement cache is keyed on (condition_id, response_text, judge_id). The
# judge identity folds in a hash of the classification prompt so that changing
# the taxonomy (e.g. adding TWO_SAME) automatically re-judges instead of serving
# stale labels from a previous prompt.
_JUDGE_SIG = hashlib.sha256(JUDGE_SYSTEM_PROMPT.encode()).hexdigest()[:8]
JUDGE_ID = f"{JUDGE_MODEL}@{_JUDGE_SIG}"
from studies.surgeon_riddle.runner import build_messages
from studies.surgeon_riddle.cache import (
    load_response_cache,
    get_responses,
    load_judgement_cache,
    get_judgement,
    set_judgement,
)


def _build_batch_input(items: list[tuple[dict, str]]) -> str:
    batch = {}
    for idx, (condition, response_text) in enumerate(items):
        batch[str(idx)] = {
            "riddle": condition["prompt"],
            "answer": response_text,
        }
    return json.dumps(batch, indent=2)


def _normalize_label(raw: str) -> str:
    label = str(raw).strip().upper().replace(" ", "_")
    if label in JUDGE_LABELS:
        return label
    if "TWO_SAME" in label or "TWO_DAD" in label or "TWO_MOM" in label or "SAME_SEX" in label:
        return "TWO_SAME"
    if "OTHER_PARENT" in label:
        return "OTHER_PARENT"
    if "MOTHER" in label:
        return "MOTHER"
    if "FATHER" in label:
        return "FATHER"
    return "OTHER"


def _parse_batch_output(raw: str, batch_size: int) -> dict[int, str]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

    parsed = json.loads(text)
    results = {}
    for idx_str, answer in parsed.items():
        idx = int(idx_str)
        if idx < 0 or idx >= batch_size:
            continue
        results[idx] = _normalize_label(answer)
    return results


async def judge_batch(
    provider: OpenRouterProvider,
    items: list[tuple[dict, str]],
) -> list[str]:
    if not items:
        return []

    indices = list(range(len(items)))
    random.shuffle(indices)
    shuffled = [items[i] for i in indices]

    batch_json = _build_batch_input(shuffled)

    try:
        result = await provider.complete_text(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": batch_json},
            ],
            model=JUDGE_MODEL,
            temperature=0.0,
            max_tokens=300,
        )
        parsed = _parse_batch_output(result, len(shuffled))
    except Exception:
        return ["ERROR"] * len(items)

    answers = ["ERROR"] * len(items)
    for shuffled_idx, original_idx in enumerate(indices):
        answers[original_idx] = parsed.get(shuffled_idx, "ERROR")
    return answers


async def judge_all(
    provider: OpenRouterProvider,
    models: list[str],
) -> dict[str, dict]:
    """Classify all cached raw responses via batched LLM calls."""
    jobs: list[tuple[str, dict, str]] = []
    for model in models:
        response_cache = load_response_cache(model)
        judgement_cache = load_judgement_cache(model)

        for condition in CONDITIONS:
            messages = build_messages(condition)
            responses = get_responses(response_cache, messages, TEMPERATURE)[:iterations_for(model)]
            for result in responses:
                if result.get("error") is not None or result.get("response") is None:
                    continue
                existing = get_judgement(
                    judgement_cache, condition["id"], result["response"], JUDGE_ID,
                )
                if existing is None or existing == "ERROR":
                    jobs.append((model, condition, result["response"]))

    total_needed = len(jobs)
    if total_needed == 0:
        print("\n  All judgements already cached.")
        return {m: load_judgement_cache(m) for m in models}

    num_batches = (total_needed + JUDGE_BATCH_SIZE - 1) // JUDGE_BATCH_SIZE
    print(f"\n  {total_needed} judgements needed in {num_batches} batches of up to {JUDGE_BATCH_SIZE}.")

    batches = [jobs[i:i + JUDGE_BATCH_SIZE] for i in range(0, total_needed, JUDGE_BATCH_SIZE)]

    judgement_caches: dict[str, dict] = {m: load_judgement_cache(m) for m in models}
    cache_locks: dict[str, asyncio.Lock] = {m: asyncio.Lock() for m in models}
    semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)
    done_batches = 0
    done_lock = asyncio.Lock()

    async def _run_batch(batch):
        nonlocal done_batches
        items = [(condition, response_text) for _, condition, response_text in batch]

        async with semaphore:
            answers = await judge_batch(provider, items)

        for (model, condition, response_text), answer in zip(batch, answers):
            async with cache_locks[model]:
                set_judgement(
                    judgement_caches[model], model,
                    condition["id"], response_text, JUDGE_ID, answer,
                )

        async with done_lock:
            done_batches += 1
            sample = ", ".join(answers[:4])
            if len(answers) > 4:
                sample += ", ..."
            print(f"  [batch {done_batches}/{num_batches}] {len(batch)} items -> [{sample}]")

    await asyncio.gather(*[_run_batch(b) for b in batches])

    for model in models:
        judgement_caches[model] = load_judgement_cache(model)

    return judgement_caches


def compute_scores(model: str) -> dict[str, dict]:
    """Per-condition label counts for a model.

    Returns {condition_id: {label: n for label in JUDGE_LABELS} plus
    "error": n and "total": n}.
    """
    response_cache = load_response_cache(model)
    judgement_cache = load_judgement_cache(model)

    scores = {}
    for condition in CONDITIONS:
        messages = build_messages(condition)
        responses = get_responses(response_cache, messages, TEMPERATURE)[:iterations_for(model)]

        counts = {label: 0 for label in JUDGE_LABELS}
        counts["error"] = 0
        for result in responses:
            if result.get("error") is not None or result.get("response") is None:
                counts["error"] += 1
                continue
            label = get_judgement(
                judgement_cache, condition["id"], result["response"], JUDGE_ID,
            )
            if label is None or label == "ERROR" or label not in JUDGE_LABELS:
                counts["error"] += 1
            else:
                counts[label] += 1

        counts["total"] = sum(counts[l] for l in JUDGE_LABELS)
        scores[condition["id"]] = counts

    return scores


# ---------------------------------------------------------------------------
# Correctness: was the named doctor a coherent (possibly-different) person?
# ---------------------------------------------------------------------------
# The trap is naming the SAME single parent who was already in the crash: the
# father in the classic (man-driving) version, the mother in the flipped
# (woman-driving) version — that person can't also be the surgeon at the
# hospital. Any other coherent parent solves it: the opposite-gender parent, a
# second same-gender parent (two dads / two moms), or another parental figure.

_TRAP_LABEL = {"father": "FATHER", "mother": "MOTHER"}


def outcome_of(condition_id: str, label: str) -> str:
    """Classify a label under a condition as 'solved', 'trap', or 'unclear'."""
    if label == "OTHER":
        return "unclear"
    if label == _TRAP_LABEL[condition_id]:
        return "trap"
    return "solved"


def outcome_counts(counts: dict, condition_id: str) -> dict:
    """Roll per-label counts up into {'solved', 'trap', 'unclear'}."""
    out = {"solved": 0, "trap": 0, "unclear": 0}
    for label in JUDGE_LABELS:
        out[outcome_of(condition_id, label)] += counts[label]
    return out


def correct_rate(counts: dict, condition_id: str) -> float | None:
    """Fraction of all replies that solved the riddle (None if no replies)."""
    if not counts["total"]:
        return None
    return outcome_counts(counts, condition_id)["solved"] / counts["total"]
