"""Cache management for the Hate Speech Detection Bias study.

Each model gets its own JSON file so results are isolated.
Cache key = hash(messages, temperature).
"""

import hashlib
import json
from pathlib import Path

from studies.hate_speech_detection.config import OUTPUT_DIR

STUDY_DIR = Path(__file__).parent
RESULTS_DIR = STUDY_DIR / OUTPUT_DIR / "results"
GRAPHS_DIR = STUDY_DIR / OUTPUT_DIR / "graphs"


def _hash(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Response cache (per model)
# ---------------------------------------------------------------------------

def response_cache_key(messages: list[dict], temperature: float) -> str:
    return _hash(json.dumps(messages, sort_keys=True), str(temperature))


def model_file(model: str) -> Path:
    safe = model.replace("/", "_")
    return RESULTS_DIR / f"{safe}.json"


def load_response_cache(model: str) -> dict:
    path = model_file(model)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_response_cache(model: str, cache: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = model_file(model)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def get_responses(cache: dict, messages: list[dict], temperature: float) -> list[dict]:
    key = response_cache_key(messages, temperature)
    entry = cache.get(key)
    if entry is None:
        return []
    return entry.get("results", [])


def append_response(cache: dict, model: str, messages: list[dict], temperature: float, result: dict):
    key = response_cache_key(messages, temperature)
    if key not in cache:
        cache[key] = {
            "messages": messages,
            "temperature": temperature,
            "results": [],
        }
    cache[key]["results"].append(result)
    save_response_cache(model, cache)


def purge_response_errors(cache: dict, model: str) -> int:
    errored = 0
    for key in cache:
        if "results" in cache[key]:
            before = len(cache[key]["results"])
            cache[key]["results"] = [
                r for r in cache[key]["results"] if r.get("error") is None
            ]
            errored += before - len(cache[key]["results"])
    if errored:
        save_response_cache(model, cache)
    return errored
