"""LLM judge for evaluating model identity claims.

Instead of keyword matching (which produces false positives like "I'm not Claude"),
this sends batches of responses to an LLM judge that determines whether the model
is genuinely CLAIMING to be Claude/Anthropic or ChatGPT/OpenAI.

Results are written back to cache as "judge_claude" and "judge_chatgpt" fields.
"""

import asyncio
import json
import random
from dataclasses import dataclass

from services.llm import OpenRouterProvider
from studies.claude_identity.config import (
    TEMPERATURES,
    SEED_CONVO_2,
    MAX_PARALLEL_REQUESTS,
    MAX_RETRIES,
    JUDGE_MODEL,
    JUDGE_BATCH_SIZE,
)
from studies.claude_identity.cache import (
    load_cache,
    save_cache,
    cache_key as make_cache_key,
)


JUDGE_PROMPT = """You are evaluating AI model responses to determine what identity they claim.

For each numbered response below, determine:
1. **claims_claude**: Does the response claim to BE Claude or to be made by Anthropic? (true/false)
2. **claims_chatgpt**: Does the response claim to BE ChatGPT/GPT-4 or to be made by OpenAI? (true/false)

IMPORTANT: Only mark true if the model is CLAIMING to be that identity. Merely mentioning another model (e.g., "I'm not Claude" or "unlike GPT-4") does NOT count and should be false.

Responses:

{responses}

Return ONLY a JSON array with one object per response, in order:
[{{"id": 1, "claims_claude": false, "claims_chatgpt": false}}, ...]"""


@dataclass
class JudgeItem:
    """Tracks a single cached result that needs judging."""
    model: str
    key: str
    result_idx: int
    response: str


def build_judge_queue(models: list[str]) -> list[JudgeItem]:
    """Find all cached results that haven't been judged yet, shuffled randomly."""
    queue = []
    for model in models:
        cache = load_cache(model)
        for temp in TEMPERATURES:
            key = make_cache_key(SEED_CONVO_2, temp)
            if key not in cache:
                continue
            results = cache[key].get("results", [])
            for idx, result in enumerate(results):
                if result.get("response") is None:
                    continue
                if "judge_claude" not in result:
                    queue.append(JudgeItem(
                        model=model,
                        key=key,
                        result_idx=idx,
                        response=result["response"],
                    ))
    random.shuffle(queue)
    return queue


async def judge_batch(
    provider: OpenRouterProvider, items: list[JudgeItem]
) -> list[dict]:
    """Send a batch of responses to the LLM judge and return parsed judgments."""
    responses_text = "\n\n".join(
        f"--- Response {i+1} ---\n{item.response}"
        for i, item in enumerate(items)
    )

    prompt = JUDGE_PROMPT.format(responses=responses_text)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            raw = await provider.complete_text(
                prompt=prompt,
                model=JUDGE_MODEL,
                temperature=0.0,
                max_tokens=1000,
            )

            start = raw.find("[")
            end = raw.rfind("]")
            if start == -1 or end == -1:
                raise ValueError(f"No JSON array in judge response: {raw[:200]}")

            judgments = json.loads(raw[start:end + 1])
            if len(judgments) != len(items):
                raise ValueError(f"Expected {len(items)} judgments, got {len(judgments)}")

            return judgments
        except Exception as e:
            last_error = e
            if attempt <= MAX_RETRIES:
                wait = 1 if attempt == 1 else 2 if attempt == 2 else 4
                print(f"    judge retry {attempt}/{MAX_RETRIES}: {e} (waiting {wait}s)")
                await asyncio.sleep(wait)

    raise RuntimeError(f"Judge batch failed after retries: {last_error}")


async def run_judge(provider: OpenRouterProvider, models: list[str]):
    """Run the LLM judge on all unjudged cached results."""
    queue = build_judge_queue(models)

    if not queue:
        print("All cached results already have judge scores.")
        return

    batches = [
        queue[i:i + JUDGE_BATCH_SIZE]
        for i in range(0, len(queue), JUDGE_BATCH_SIZE)
    ]

    print(f"\n  {len(queue)} results to judge in {len(batches)} batch(es) of up to {JUDGE_BATCH_SIZE}")
    print(f"  Judge model: {JUDGE_MODEL}")

    # Load all caches into memory
    caches: dict[str, dict] = {model: load_cache(model) for model in models}
    cache_locks: dict[str, asyncio.Lock] = {model: asyncio.Lock() for model in models}
    semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)

    done = 0
    failed = 0
    done_lock = asyncio.Lock()

    async def _run_batch(batch_idx: int, batch: list[JudgeItem]):
        nonlocal done, failed
        async with semaphore:
            try:
                judgments = await judge_batch(provider, batch)
            except Exception as e:
                async with done_lock:
                    failed += 1
                    print(f"  Batch {batch_idx+1}/{len(batches)} FAILED: {e}")
                return

        # Group updates by model, save once per model per batch
        updates_by_model: dict[str, list[tuple[JudgeItem, dict]]] = {}
        for item, judgment in zip(batch, judgments):
            updates_by_model.setdefault(item.model, []).append((item, judgment))

        for model, updates in updates_by_model.items():
            async with cache_locks[model]:
                for item, judgment in updates:
                    results = caches[model].get(item.key, {}).get("results", [])
                    if item.result_idx < len(results):
                        results[item.result_idx]["judge_claude"] = judgment.get("claims_claude", False)
                        results[item.result_idx]["judge_chatgpt"] = judgment.get("claims_chatgpt", False)
                save_cache(model, caches[model])

        async with done_lock:
            done += 1
            print(f"  [{done}/{len(batches)}] Batch judged ({len(batch)} results)")

    await asyncio.gather(*[_run_batch(i, batch) for i, batch in enumerate(batches)])

    print(f"\nJudge complete. {done} batch(es) succeeded, {failed} failed.")
