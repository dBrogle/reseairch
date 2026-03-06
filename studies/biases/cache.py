"""Cache management for the Biases study.

Each model gets its own JSON file keyed by a hash of (prompt_text, temperature),
so changing prompts automatically invalidates stale results.
"""

import hashlib
import json
from pathlib import Path

from studies.biases.config import OUTPUT_DIR, TEMPERATURE

STUDY_DIR = Path(__file__).parent
RESULTS_DIR = STUDY_DIR / OUTPUT_DIR / "results"
GRAPHS_DIR = STUDY_DIR / OUTPUT_DIR / "graphs"


def cache_key(prompt: str, temperature: float) -> str:
    """Deterministic short hash of (prompt, temperature)."""
    raw = prompt + f"|{temperature}"
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


def get_results(cache: dict, prompt: str, temperature: float) -> list[dict]:
    """Return the list of cached results for a (prompt, temp) combo, or []."""
    key = cache_key(prompt, temperature)
    entry = cache.get(key)
    if entry is None:
        return []
    return entry.get("results", [])


def append_result(
    cache: dict, model: str, prompt: str, temperature: float, result: dict
):
    """Append a single result and flush to disk immediately."""
    key = cache_key(prompt, temperature)
    if key not in cache:
        cache[key] = {
            "prompt": prompt,
            "temperature": temperature,
            "results": [],
        }
    cache[key]["results"].append(result)
    save_cache(model, cache)


def purge_errors(cache: dict, model: str) -> int:
    """Strip errored results from cache so they get retried. Returns count purged."""
    errored = 0
    for key in cache:
        if "results" in cache[key]:
            before = len(cache[key]["results"])
            cache[key]["results"] = [
                r for r in cache[key]["results"] if r.get("error") is None
            ]
            errored += before - len(cache[key]["results"])
    if errored:
        save_cache(model, cache)
    return errored
