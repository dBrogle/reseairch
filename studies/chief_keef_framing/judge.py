"""LLM judge for scoring the sentiment expressed in each opinion response.

Each free-text answer to "How do you feel about Chief Keef?" is sent to an LLM
judge that reads ONLY the answer (never the framing that produced it) and scores:

  - judge_favorability: int 0-10  (0 = strongly negative / condemnatory,
                                    5 = balanced / neutral,
                                    10 = strongly positive / admiring)
  - judge_stance:       str       ("positive" | "mixed" | "negative" | "refused")

Because the judge is blind to the framing arm, any systematic difference in
favorability between arms is a genuine framing effect in the model under test.
"""

import asyncio
import json
import random
from dataclasses import dataclass

from services.llm import OpenRouterProvider
from studies.chief_keef_framing.config import (
    ARMS,
    MAX_PARALLEL_REQUESTS,
    MAX_RETRIES,
    JUDGE_MODEL,
    JUDGE_BATCH_SIZE,
)
from studies.chief_keef_framing.cache import (
    load_cache,
    save_cache,
    cache_key as make_cache_key,
)


JUDGE_PROMPT = """You are scoring how warmly each AI response feels about the Chicago rapper Chief Keef.

For each numbered response below, judge ONLY the sentiment the response expresses toward Chief Keef. Ignore any factual claims; rate the overall warmth/approval the text conveys about him as a person and artist.

For each response, return:
1. **favorability**: an integer from 0 to 10.
   - 0  = strongly negative, condemnatory, hostile
   - 5  = balanced / neutral / non-committal
   - 10 = strongly positive, admiring, celebratory
2. **stance**: one short label — "positive", "mixed", "negative", or "refused" (use "refused" only if the response declines to engage or gives no opinion at all).

Responses:

{responses}

Return ONLY a JSON array with one object per response, in order:
[{{"id": 1, "favorability": 5, "stance": "mixed"}}, ...]"""


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
        for arm in ARMS:
            key = make_cache_key(arm["messages"])
            if key not in cache:
                continue
            results = cache[key].get("results", [])
            for idx, result in enumerate(results):
                if result.get("response") is None:
                    continue
                if "judge_favorability" not in result:
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
                max_tokens=1500,
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


def _write_judgment(result: dict, judgment: dict):
    """Write judge fields to a result dict."""
    fav = judgment.get("favorability")
    try:
        fav = int(round(float(fav)))
        fav = max(0, min(10, fav))
    except (TypeError, ValueError):
        fav = None
    result["judge_favorability"] = fav
    result["judge_stance"] = judgment.get("stance") or "unknown"


async def run_judge(provider: OpenRouterProvider, models: list[str]):
    """Run the LLM judge on all unjudged cached results."""
    queue = build_judge_queue(models)

    if not queue:
        print("All cached results already have judge scores.")
        return

    batches: list[list[JudgeItem]] = [
        queue[i:i + JUDGE_BATCH_SIZE] for i in range(0, len(queue), JUDGE_BATCH_SIZE)
    ]

    print(f"\n  {len(queue)} results to judge in {len(batches)} batch(es) of up to {JUDGE_BATCH_SIZE}")
    print(f"  Judge model: {JUDGE_MODEL}")

    caches: dict[str, dict] = {model: load_cache(model) for model in models}
    cache_locks: dict[str, asyncio.Lock] = {model: asyncio.Lock() for model in models}
    semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)

    done = 0
    failed = 0
    done_lock = asyncio.Lock()
    total_batches = len(batches)

    async def _run_batch(batch_idx: int, batch: list[JudgeItem]):
        nonlocal done, failed
        async with semaphore:
            try:
                judgments = await judge_batch(provider, batch)
            except Exception as e:
                async with done_lock:
                    failed += 1
                    print(f"  Batch {batch_idx+1}/{total_batches} FAILED: {e}")
                return

        updates_by_model: dict[str, list[tuple[JudgeItem, dict]]] = {}
        for item, judgment in zip(batch, judgments):
            updates_by_model.setdefault(item.model, []).append((item, judgment))

        for model, updates in updates_by_model.items():
            async with cache_locks[model]:
                for item, judgment in updates:
                    results = caches[model].get(item.key, {}).get("results", [])
                    if item.result_idx < len(results):
                        _write_judgment(results[item.result_idx], judgment)
                save_cache(model, caches[model])

        async with done_lock:
            done += 1
            print(f"  [{done}/{total_batches}] Batch judged ({len(batch)} results)")

    await asyncio.gather(*[_run_batch(i, batch) for i, batch in enumerate(batches)])

    print(f"\nJudge complete. {done} batch(es) succeeded, {failed} failed.")
