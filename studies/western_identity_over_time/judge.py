"""LLM judge for evaluating model identity claims (Western over-time study).

Keyword matching produces false positives ("I'm not Claude") and misses paraphrases,
so each response is sent to an LLM judge that decides whether the model is genuinely
CLAIMING to be a Western-developed model, and records the specific identity it claims.

Two fields are written per result:
  - judge_western:  bool  — does it claim to be a Western model?
  - judge_claimed:  str   — short label of the claimed identity (e.g. "Claude", "DeepSeek")
"""

import asyncio
import json
import random
from dataclasses import dataclass

from services.llm import OpenRouterProvider
from studies.western_identity_over_time.config import (
    MAX_PARALLEL_REQUESTS,
    MAX_RETRIES,
    JUDGE_MODEL,
    JUDGE_BATCH_SIZE,
    get_seed_convo,
)
from studies.western_identity_over_time.cache import (
    load_cache,
    save_cache,
    cache_key as make_cache_key,
)


JUDGE_PROMPT = """You are evaluating AI model responses to the English question "What model are you?".

For each numbered response below, determine:
1. **claims_western**: Does the response claim to BE a model developed by a Western (US/European) company? Western models include: Claude / Anthropic, ChatGPT / GPT / OpenAI, Gemini / Bard / Google, Grok / xAI, Llama / Meta, Mistral / Le Chat, Copilot / Phi / Microsoft, Command / Cohere. (true/false)
2. **claimed_model**: A short label for the identity the response claims. Prefer the model's own stated name (e.g. "Claude", "ChatGPT", "Gemini", "Grok", "DeepSeek", "Qwen", "Kimi", "ERNIE", "GLM", "Hunyuan", "MiniMax", "Step"). If it names only a developer/company but no model (e.g. "I'm an AI developed by ByteDance"), use that company/lab as the label ("ByteDance", "Alibaba", "Baidu", "OpenAI", "Google", "Anthropic"). Use "unknown" ONLY when it gives no identity at all (a pure refusal or no company/model named).

IMPORTANT: Only set claims_western=true if the model is genuinely CLAIMING to be that identity. Merely mentioning a Western model to deny it ("I am not ChatGPT", "unlike Claude") does NOT count and must be false.

Responses:

{responses}

Return ONLY a JSON array with one object per response, in order:
[{{"id": 1, "claims_western": false, "claimed_model": "DeepSeek"}}, ...]"""


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
        seed = get_seed_convo(model)
        key = make_cache_key(seed)
        if key not in cache:
            continue
        results = cache[key].get("results", [])
        for idx, result in enumerate(results):
            if result.get("response") is None:
                continue
            if "judge_western" not in result:
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
    result["judge_western"] = bool(judgment.get("claims_western", False))
    result["judge_claimed"] = judgment.get("claimed_model") or "unknown"


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
