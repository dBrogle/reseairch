"""Cache management for the Emergent Values study.

Each experiment (category×measure) × model gets its own JSON file.
Keys are hashed from (option_a_text, option_b_text, temperature) so
changing prompts automatically invalidates stale results.
"""

import hashlib
import json
from pathlib import Path

from studies.emergent_values.config import OUTPUT_DIR

STUDY_DIR = Path(__file__).parent
RESULTS_DIR = STUDY_DIR / OUTPUT_DIR / "results"
GRAPHS_DIR = STUDY_DIR / OUTPUT_DIR / "graphs"


def cache_key(option_a: str, option_b: str, temperature: float) -> str:
    """Deterministic short hash of a comparison pair."""
    raw = f"{option_a}|{option_b}|{temperature}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def experiment_file(model: str, category: str, measure: str) -> Path:
    """Path to the cache JSON for a model×experiment."""
    safe_model = model.replace("/", "_")
    return RESULTS_DIR / f"{safe_model}_{category}_{measure}.json"


def load_cache(model: str, category: str, measure: str) -> dict:
    """Load existing cache for a model×experiment, or return empty dict."""
    path = experiment_file(model, category, measure)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_cache(model: str, category: str, measure: str, cache: dict):
    """Persist the cache to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = experiment_file(model, category, measure)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def get_results(cache: dict, option_a: str, option_b: str, temperature: float) -> list[dict]:
    """Return cached results for a specific pair, or []."""
    key = cache_key(option_a, option_b, temperature)
    entry = cache.get(key)
    if entry is None:
        return []
    return entry.get("results", [])


def append_result(
    cache: dict, model: str, category: str, measure: str,
    option_a: str, option_b: str, temperature: float, result: dict
):
    """Append a single result and flush to disk immediately."""
    key = cache_key(option_a, option_b, temperature)
    if key not in cache:
        cache[key] = {
            "option_a": option_a,
            "option_b": option_b,
            "temperature": temperature,
            "results": [],
        }
    cache[key]["results"].append(result)
    save_cache(model, category, measure, cache)


def purge_errors(cache: dict, model: str, category: str, measure: str) -> int:
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
        save_cache(model, category, measure, cache)
    return errored
