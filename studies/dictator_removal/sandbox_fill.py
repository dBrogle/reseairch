"""One-time sandbox to backfill non-refusal responses for high-refusal models.

Gemini refuses ~50-70% of the time, making the manual purge-and-rerun loop slow.
This script:
  1. Runs N extra queries per dictator into a sandbox file (separate from main cache)
  2. Extracts YES/NO/REFUSED for every sandbox response
  3. Merges the first M non-refusals into the main cache to fill each (model, dictator)
     up to ITERATIONS, after purging any existing refusals from the main cache

To revert:
    rm -rf studies/dictator_removal/output/sandbox/
    rm studies/dictator_removal/sandbox_fill.py

Per-dictator targets are computed from the main cache: only dictators below
ITERATIONS non-refusals get queried, and the sandbox target for each is
`multiplier * needed`. With multiplier=5 and ~50-70% refusal rate, that's a
comfortable buffer.

Usage:
    python -m studies.dictator_removal.sandbox_fill                                 # all MODELS, multiplier=2
    python -m studies.dictator_removal.sandbox_fill --model google/gemini-3.1-pro-preview
    python -m studies.dictator_removal.sandbox_fill --multiplier 5 --dry-run
    python -m studies.dictator_removal.sandbox_fill --model google/gemini-3.1-pro-preview --skip-gather
"""

import argparse
import asyncio
import json
from pathlib import Path

from services.llm import OpenRouterProvider
from studies.dictator_removal.config import (
    MODELS,
    DICTATORS,
    TEMPERATURE,
    ITERATIONS,
    MAX_PARALLEL_REQUESTS,
    EXTRACTOR_MODEL,
    EXTRACTION_BATCH_SIZE,
    OUTPUT_DIR,
)
from studies.dictator_removal.runner import build_messages, run_single_query
from studies.dictator_removal.extractor import extract_batch
from studies.dictator_removal.cache import (
    load_response_cache,
    save_response_cache,
    response_cache_key,
    load_extraction_cache,
    save_extraction_cache,
    extraction_cache_key,
    get_extraction,
)

STUDY_DIR = Path(__file__).parent
SANDBOX_DIR = STUDY_DIR / OUTPUT_DIR / "sandbox"


def sandbox_file(model: str) -> Path:
    safe = model.replace("/", "_")
    return SANDBOX_DIR / f"{safe}.json"


def load_sandbox(model: str) -> dict:
    p = sandbox_file(model)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def save_sandbox(model: str, data: dict):
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    with open(sandbox_file(model), "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clear_sandbox(model: str):
    p = sandbox_file(model)
    if p.exists():
        p.unlink()


def _count_main_non_refusals(model: str) -> dict[str, int]:
    """Returns {dictator_id: count of YES/NO responses currently in main cache}."""
    response_cache = load_response_cache(model)
    extraction_cache = load_extraction_cache(model)
    counts = {}
    for dictator in DICTATORS:
        d_id = dictator["id"]
        messages = build_messages(dictator)
        rkey = response_cache_key(messages, TEMPERATURE)
        results = response_cache.get(rkey, {}).get("results", [])
        n = 0
        for r in results:
            if r.get("error") is not None or r.get("response") is None:
                continue
            ans = get_extraction(extraction_cache, d_id, r["response"], EXTRACTOR_MODEL)
            if ans in ("YES", "NO"):
                n += 1
        counts[d_id] = n
    return counts


async def gather_responses(provider: OpenRouterProvider, model: str, multiplier: int) -> dict:
    """Per-dictator: target sandbox size = multiplier * (ITERATIONS - main_non_refusals)."""
    sandbox = load_sandbox(model)
    main_non_refusals = _count_main_non_refusals(model)

    print(f"  Targets (multiplier={multiplier}):")
    print(f"  {'dictator':<16} {'main':>6} {'need':>6} {'target':>8} {'in_sb':>8} {'delta':>8}")
    targets: dict[str, int] = {}
    for dictator in DICTATORS:
        d_id = dictator["id"]
        current = main_non_refusals[d_id]
        needed = max(0, ITERATIONS - current)
        target = multiplier * needed
        existing = len(sandbox.get(d_id, {}).get("raw", []))
        delta = max(0, target - existing)
        targets[d_id] = target
        print(f"  {d_id:<16} {current:>6} {needed:>6} {target:>8} {existing:>8} {delta:>8}")

    jobs = []
    for dictator in DICTATORS:
        d_id = dictator["id"]
        existing = len(sandbox.get(d_id, {}).get("raw", []))
        if existing >= targets[d_id]:
            continue
        messages = build_messages(dictator)
        for _ in range(existing, targets[d_id]):
            jobs.append((dictator, messages))

    if not jobs:
        print(f"\n  Nothing to gather — main cache is full or sandbox already has enough.")
        return sandbox

    print(f"\n  Running {len(jobs)} new queries ({MAX_PARALLEL_REQUESTS} parallel).")

    sem = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)
    sandbox_lock = asyncio.Lock()
    done = 0
    done_lock = asyncio.Lock()

    async def _run(dictator, messages):
        nonlocal done
        async with sem:
            result = await run_single_query(provider, model, messages, TEMPERATURE)
        async with sandbox_lock:
            d_id = dictator["id"]
            if d_id not in sandbox:
                sandbox[d_id] = {"raw": [], "extracted": []}
            sandbox[d_id]["raw"].append(result.get("response"))
            save_sandbox(model, sandbox)
        async with done_lock:
            done += 1
            short = model.split("/")[-1]
            status = "OK" if result["error"] is None else f"ERR: {result['error']}"
            print(f"  [{done}/{len(jobs)}] {short} | {dictator['name']} -> {status}")

    await asyncio.gather(*[_run(d, m) for d, m in jobs])
    return sandbox


async def extract_sandbox(provider: OpenRouterProvider, model: str, sandbox: dict) -> dict:
    """Extract YES/NO/REFUSED for any sandbox responses missing extraction."""
    jobs = []
    for dictator in DICTATORS:
        d_id = dictator["id"]
        bucket = sandbox.get(d_id)
        if bucket is None:
            continue
        raw = bucket.get("raw", [])
        extracted = bucket.get("extracted", [])
        while len(extracted) < len(raw):
            extracted.append(None)
        bucket["extracted"] = extracted
        for i, (resp, ext) in enumerate(zip(raw, extracted)):
            if resp is None:
                continue
            if ext in (None, "ERROR"):
                jobs.append((d_id, i, dictator, resp))

    save_sandbox(model, sandbox)

    if not jobs:
        print("  All sandbox responses already extracted.")
        return sandbox

    batches = [jobs[i:i + EXTRACTION_BATCH_SIZE] for i in range(0, len(jobs), EXTRACTION_BATCH_SIZE)]
    print(f"  Extracting {len(jobs)} sandbox responses in {len(batches)} batches.")

    sem = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)
    sandbox_lock = asyncio.Lock()
    done = 0
    done_lock = asyncio.Lock()

    async def _run_batch(batch):
        nonlocal done
        items = [(d, r) for _, _, d, r in batch]
        async with sem:
            answers = await extract_batch(provider, items)
        async with sandbox_lock:
            for (d_id, idx, _, _), ans in zip(batch, answers):
                sandbox[d_id]["extracted"][idx] = ans
            save_sandbox(model, sandbox)
        async with done_lock:
            done += 1
            sample = ", ".join(answers[:3]) + (", ..." if len(answers) > 3 else "")
            print(f"  [batch {done}/{len(batches)}] {len(batch)} items -> [{sample}]")

    await asyncio.gather(*[_run_batch(b) for b in batches])
    return sandbox


def merge_to_main(model: str, sandbox: dict, dry_run: bool):
    """Purge refusals from main cache, then fill with non-refusals from sandbox."""
    response_cache = load_response_cache(model)
    extraction_cache = load_extraction_cache(model)

    summary = []
    for dictator in DICTATORS:
        d_id = dictator["id"]
        messages = build_messages(dictator)
        rkey = response_cache_key(messages, TEMPERATURE)

        existing_results = response_cache.get(rkey, {}).get("results", [])
        kept = []
        purged = 0
        for r in existing_results:
            if r.get("error") is not None or r.get("response") is None:
                purged += 1
                continue
            ans = get_extraction(extraction_cache, d_id, r["response"], EXTRACTOR_MODEL)
            if ans in ("YES", "NO"):
                kept.append(r)
            else:
                purged += 1

        current = len(kept)
        needed = max(0, ITERATIONS - current)

        bucket = sandbox.get(d_id, {})
        raws = bucket.get("raw", [])
        exts = bucket.get("extracted", [])
        non_refusals = [(r, a) for r, a in zip(raws, exts) if r is not None and a in ("YES", "NO")]
        taking = min(needed, len(non_refusals))

        summary.append((d_id, current, purged, needed, len(non_refusals), taking))

        if dry_run:
            continue

        if rkey not in response_cache:
            response_cache[rkey] = {
                "messages": messages,
                "temperature": TEMPERATURE,
                "results": [],
            }
        response_cache[rkey]["results"] = kept

        for resp, ans in non_refusals[:taking]:
            response_cache[rkey]["results"].append({"response": resp, "error": None})
            ekey = extraction_cache_key(d_id, resp, EXTRACTOR_MODEL)
            extraction_cache[ekey] = ans

    print(f"\n  Merge plan for {model}:")
    print(f"  {'dictator':<16} {'kept':>6} {'purged':>8} {'needed':>8} {'avail':>8} {'taking':>8}")
    for d_id, cur, purged, need, avail, take in summary:
        print(f"  {d_id:<16} {cur:>6} {purged:>8} {need:>8} {avail:>8} {take:>8}")

    if dry_run:
        print(f"\n  Dry-run only. Re-run without --dry-run to apply.")
        return

    save_response_cache(model, response_cache)
    save_extraction_cache(model, extraction_cache)
    clear_sandbox(model)
    print(f"\n  Merged into main cache. Sandbox cleared.")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", help="Model id (repeatable). Default: all MODELS.")
    parser.add_argument("--multiplier", type=int, default=2, help="Sandbox target = multiplier * (ITERATIONS - main_non_refusals) per dictator (default 2)")
    parser.add_argument("--dry-run", action="store_true", help="Show merge plan without writing to main cache")
    parser.add_argument("--skip-gather", action="store_true", help="Skip gather/extract; only merge what's already in sandbox")
    args = parser.parse_args()

    provider = OpenRouterProvider()
    models = args.model or MODELS

    for model in models:
        print(f"\n{'=' * 70}")
        print(f"  Model: {model}")
        print(f"{'=' * 70}")

        if args.skip_gather:
            print(f"\n=== Loading existing sandbox ===")
            sandbox = load_sandbox(model)
            if not sandbox:
                print("  No sandbox file found. Skipping.")
                continue
        else:
            print(f"\n=== Phase 1: Gathering sandbox responses ===")
            sandbox = await gather_responses(provider, model, args.multiplier)
            print(f"\n=== Phase 2: Extracting sandbox responses ===")
            sandbox = await extract_sandbox(provider, model, sandbox)

        print(f"\n=== Phase 3: Merging non-refusals into main cache ===")
        merge_to_main(model, sandbox, dry_run=args.dry_run)


if __name__ == "__main__":
    asyncio.run(main())
