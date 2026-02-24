"""Cache management for the Claude Identity study.

Handles reading, writing, hashing, and purging of per-model result caches.
Each model gets its own JSON file keyed by a hash of (messages, temperature),
so changing prompts automatically invalidates stale results.
"""

import hashlib
import json
from pathlib import Path

from studies.claude_identity.config import OUTPUT_DIR, TEMPERATURES, SEED_CONVO_2, CLAUDE_KEYWORDS, CHATGPT_KEYWORDS

STUDY_DIR = Path(__file__).parent
RESULTS_DIR = STUDY_DIR / OUTPUT_DIR / "results"
GRAPHS_DIR = STUDY_DIR / OUTPUT_DIR / "graphs"


def cache_key(messages: list[dict], temperature: float) -> str:
    """Deterministic short hash of (messages, temperature)."""
    raw = json.dumps(messages, sort_keys=True) + f"|{temperature}"
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


def get_results(cache: dict, messages: list[dict], temperature: float) -> list[dict]:
    """Return the list of cached results for a (messages, temp) combo, or []."""
    key = cache_key(messages, temperature)
    entry = cache.get(key)
    if entry is None:
        return []
    return entry.get("results", [])


def append_result(
    cache: dict, model: str, messages: list[dict], temperature: float, result: dict
):
    """Append a single result and flush to disk immediately."""
    key = cache_key(messages, temperature)
    if key not in cache:
        cache[key] = {
            "messages": messages,
            "temperature": temperature,
            "results": [],
        }
    cache[key]["results"].append(result)
    save_cache(model, cache)


def purge_errors(cache: dict, model: str, messages: list[dict]) -> int:
    """Strip errored results from cache so they get retried. Returns count purged."""
    errored = 0
    for temp in TEMPERATURES:
        key = cache_key(messages, temp)
        if key in cache and "results" in cache[key]:
            before = len(cache[key]["results"])
            cache[key]["results"] = [
                r for r in cache[key]["results"] if r.get("error") is None
            ]
            errored += before - len(cache[key]["results"])
    if errored:
        save_cache(model, cache)
    return errored


def reprocess_cache(model: str) -> int:
    """Re-scan all cached responses and update detection fields. Returns count updated."""
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
            new_claude = any(kw in lower for kw in CLAUDE_KEYWORDS)
            new_chatgpt = any(kw in lower for kw in CHATGPT_KEYWORDS)
            if result.get("mentions_claude") != new_claude or result.get("mentions_chatgpt") != new_chatgpt:
                updated += 1
            result["mentions_claude"] = new_claude
            result["mentions_chatgpt"] = new_chatgpt

    save_cache(model, cache)
    return updated
