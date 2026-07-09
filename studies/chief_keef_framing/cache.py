"""Cache management for the Chief Keef Framing study.

Each model gets its own JSON file keyed by a hash of the message list. Because
each framing arm is a different message list, all three arms live side by side in
the same per-model file under different keys. Changing a prompt invalidates only
that arm's stale results, and runs can be interrupted and resumed without losing
or duplicating work.
"""

import hashlib
import json
from pathlib import Path

from studies.chief_keef_framing.config import OUTPUT_DIR

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


def append_result(
    cache: dict,
    model: str,
    messages: list[dict],
    result: dict,
    arm_key: str | None = None,
    label: str | None = None,
):
    """Append a single result and flush to disk immediately."""
    key = cache_key(messages)
    if key not in cache:
        cache[key] = {
            "arm_key": arm_key,
            "label": label,
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
