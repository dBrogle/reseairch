"""LLM judge for evaluating model identity claims.

Instead of keyword matching (which produces false positives like "I'm not Claude"),
this sends batches of responses to an LLM judge that determines whether the model
is genuinely CLAIMING to be a particular identity.

For English-tested models (Chinese AI): checks claims_claude and claims_chatgpt.
For Chinese-tested models (Claude): checks claims_deepseek and claims_kimi.
The irrelevant pair is set to None in each case.
"""

import asyncio
import json
import random
from dataclasses import dataclass

from services.llm import OpenRouterProvider
from studies.claude_identity.config import (
    TEMPERATURES,
    MAX_PARALLEL_REQUESTS,
    MAX_RETRIES,
    JUDGE_MODEL,
    JUDGE_BATCH_SIZE,
    Language,
    MODEL_LANGUAGES,
    get_seed_convo,
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

JUDGE_PROMPT_ZH = """You are evaluating AI model responses to determine what identity they claim.

For each numbered response below, determine:
1. **claims_deepseek**: Does the response claim to BE DeepSeek or to be made by DeepSeek? (true/false)
2. **claims_kimi**: Does the response claim to BE Kimi/Moonshot or to be made by Moonshot AI? (true/false)

IMPORTANT: Only mark true if the model is CLAIMING to be that identity. Merely mentioning another model (e.g., "I'm not DeepSeek" or "unlike Kimi") does NOT count and should be false.

Responses:

{responses}

Return ONLY a JSON array with one object per response, in order:
[{{"id": 1, "claims_deepseek": false, "claims_kimi": false}}, ...]"""


@dataclass
class JudgeItem:
    """Tracks a single cached result that needs judging."""
    model: str
    key: str
    result_idx: int
    response: str
    language: Language


def build_judge_queue(models: list[str]) -> list[JudgeItem]:
    """Find all cached results that haven't been judged yet, shuffled randomly."""
    queue = []
    for model in models:
        cache = load_cache(model)
        seed = get_seed_convo(model)
        lang = MODEL_LANGUAGES.get(model, Language.ENGLISH)
        # Which field marks this item as already judged depends on language
        check_field = "judge_deepseek" if lang == Language.CHINESE else "judge_claude"
        for temp in TEMPERATURES:
            key = make_cache_key(seed, temp)
            if key not in cache:
                continue
            results = cache[key].get("results", [])
            for idx, result in enumerate(results):
                if result.get("response") is None:
                    continue
                if check_field not in result:
                    queue.append(JudgeItem(
                        model=model,
                        key=key,
                        result_idx=idx,
                        response=result["response"],
                        language=lang,
                    ))
    random.shuffle(queue)
    return queue


async def judge_batch(
    provider: OpenRouterProvider, items: list[JudgeItem], language: Language
) -> list[dict]:
    """Send a batch of responses to the LLM judge and return parsed judgments."""
    responses_text = "\n\n".join(
        f"--- Response {i+1} ---\n{item.response}"
        for i, item in enumerate(items)
    )

    template = JUDGE_PROMPT_ZH if language == Language.CHINESE else JUDGE_PROMPT
    prompt = template.format(responses=responses_text)

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


def _write_judgment(result: dict, judgment: dict, language: Language):
    """Write judge fields to a result dict, nulling out the irrelevant pair."""
    if language == Language.ENGLISH:
        result["judge_claude"] = judgment.get("claims_claude", False)
        result["judge_chatgpt"] = judgment.get("claims_chatgpt", False)
        result["judge_deepseek"] = None
        result["judge_kimi"] = None
    else:
        result["judge_deepseek"] = judgment.get("claims_deepseek", False)
        result["judge_kimi"] = judgment.get("claims_kimi", False)
        result["judge_claude"] = None
        result["judge_chatgpt"] = None


async def run_judge(provider: OpenRouterProvider, models: list[str]):
    """Run the LLM judge on all unjudged cached results."""
    queue = build_judge_queue(models)

    if not queue:
        print("All cached results already have judge scores.")
        return

    # Separate by language so each batch uses the correct prompt
    en_items = [item for item in queue if item.language == Language.ENGLISH]
    zh_items = [item for item in queue if item.language == Language.CHINESE]

    tagged_batches: list[tuple[Language, list[JudgeItem]]] = []
    for lang, items in [(Language.ENGLISH, en_items), (Language.CHINESE, zh_items)]:
        for i in range(0, len(items), JUDGE_BATCH_SIZE):
            tagged_batches.append((lang, items[i:i + JUDGE_BATCH_SIZE]))

    print(f"\n  {len(queue)} results to judge in {len(tagged_batches)} batch(es) of up to {JUDGE_BATCH_SIZE}")
    if en_items:
        print(f"    English (Claude/ChatGPT check): {len(en_items)} results")
    if zh_items:
        print(f"    Chinese (DeepSeek/Kimi check): {len(zh_items)} results")
    print(f"  Judge model: {JUDGE_MODEL}")

    # Load all caches into memory
    caches: dict[str, dict] = {model: load_cache(model) for model in models}
    cache_locks: dict[str, asyncio.Lock] = {model: asyncio.Lock() for model in models}
    semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)

    done = 0
    failed = 0
    done_lock = asyncio.Lock()
    total_batches = len(tagged_batches)

    async def _run_batch(batch_idx: int, language: Language, batch: list[JudgeItem]):
        nonlocal done, failed
        async with semaphore:
            try:
                judgments = await judge_batch(provider, batch, language)
            except Exception as e:
                async with done_lock:
                    failed += 1
                    print(f"  Batch {batch_idx+1}/{total_batches} FAILED: {e}")
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
                        _write_judgment(results[item.result_idx], judgment, language)
                save_cache(model, caches[model])

        async with done_lock:
            done += 1
            print(f"  [{done}/{total_batches}] Batch judged ({len(batch)} results)")

    await asyncio.gather(*[
        _run_batch(i, lang, batch)
        for i, (lang, batch) in enumerate(tagged_batches)
    ])

    print(f"\nJudge complete. {done} batch(es) succeeded, {failed} failed.")
