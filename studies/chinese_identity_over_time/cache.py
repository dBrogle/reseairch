"""Cache management for the Chinese Identity Over Time study.

Each model gets its own JSON file keyed by a hash of the message list (there is no
temperature sweep here, so temperature is not part of the key). Changing the prompt
invalidates stale results, and runs can be interrupted and resumed without losing or
duplicating work.
"""

import hashlib
import json
from pathlib import Path

from studies.chinese_identity_over_time.config import OUTPUT_DIR, CHINESE_KEYWORDS

STUDY_DIR = Path(__file__).parent
RESULTS_DIR = STUDY_DIR / OUTPUT_DIR / "results"
GRAPHS_DIR = STUDY_DIR / OUTPUT_DIR / "graphs"


def cache_key(messages: list[dict]) -> str:
    """Deterministic short hash of the message list."""
    raw = json.dumps(messages, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def model_file(model: str) -> Path:
    """Path to the cache JSON for a given model."""
    safe = model.replace("/", "_")
    return RESULTS_DIR / f"{safe}.json"


def load_cache(model: str) -> dict:
    """Load existing cache for a model, or return empty dict."""
    path = model_file(model)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_cache(model: str, cache: dict):
    """Persist the model cache to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = model_file(model)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def get_results(cache: dict, messages: list[dict]) -> list[dict]:
    """Return the list of cached results for a message list, or []."""
    entry = cache.get(cache_key(messages))
    if entry is None:
        return []
    return entry.get("results", [])


def append_result(cache: dict, model: str, messages: list[dict], result: dict):
    """Append a single result and flush to disk immediately."""
    key = cache_key(messages)
    if key not in cache:
        cache[key] = {
            "messages": messages,
            "results": [],
        }
    cache[key]["results"].append(result)
    save_cache(model, cache)


def purge_errors(cache: dict, model: str, messages: list[dict]) -> int:
    """Strip errored results from cache so they get retried. Returns count purged."""
    key = cache_key(messages)
    if key not in cache or "results" not in cache[key]:
        return 0
    before = len(cache[key]["results"])
    cache[key]["results"] = [
        r for r in cache[key]["results"] if r.get("error") is None
    ]
    errored = before - len(cache[key]["results"])
    if errored:
        save_cache(model, cache)
    return errored


def purge_unknown(cache: dict, model: str, messages: list[dict]) -> int:
    """Drop results the judge couldn't identify (judge_claimed == 'unknown').

    These are real responses that named no model (refusals, deflections, truncated
    output); removing them lets the runner re-query those slots for a fresh answer.
    Returns the count purged.
    """
    key = cache_key(messages)
    if key not in cache or "results" not in cache[key]:
        return 0
    before = len(cache[key]["results"])
    cache[key]["results"] = [
        r for r in cache[key]["results"]
        if not (r.get("response") is not None and r.get("judge_claimed") == "unknown")
    ]
    removed = before - len(cache[key]["results"])
    if removed:
        save_cache(model, cache)
    return removed


def reprocess_cache(model: str) -> int:
    """Re-scan cached responses and update the keyword field. Returns count updated."""
    cache = load_cache(model)
    if not cache:
        return 0

    updated = 0
    for entry in cache.values():
        for result in entry.get("results", []):
            response = result.get("response")
            if response is None:
                continue
            lower = response.lower()
            new_chinese = any(kw.lower() in lower for kw in CHINESE_KEYWORDS)
            if result.get("mentions_chinese") != new_chinese:
                updated += 1
            result["mentions_chinese"] = new_chinese

    save_cache(model, cache)
    return updated
