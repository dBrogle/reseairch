"""Cache management for the Political Sycophancy study.

Two separate caches:
1. Raw responses: keyed by hash of (full messages, temperature)
2. Extracted answers: keyed by hash of (question_id, raw_response, extractor_model)

Each model gets its own JSON file so results are isolated.
"""

import hashlib
import json
from pathlib import Path

from studies.political_sycophancy.config import OUTPUT_DIR

STUDY_DIR = Path(__file__).parent
RESULTS_DIR = STUDY_DIR / OUTPUT_DIR / "results"
EXTRACTIONS_DIR = STUDY_DIR / OUTPUT_DIR / "extractions"
GRAPHS_DIR = STUDY_DIR / OUTPUT_DIR / "graphs"


def _hash(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Raw response cache (per model)
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


# ---------------------------------------------------------------------------
# Extraction cache (per model)
# ---------------------------------------------------------------------------

def extraction_file(model: str) -> Path:
    safe = model.replace("/", "_")
    return EXTRACTIONS_DIR / f"{safe}.json"


def load_extraction_cache(model: str) -> dict:
    path = extraction_file(model)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_extraction_cache(model: str, cache: dict):
    EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = extraction_file(model)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def extraction_cache_key(question_id: str, response_text: str, extractor_model: str) -> str:
    return _hash(question_id, response_text, extractor_model)


def get_extraction(cache: dict, question_id: str, response_text: str, extractor_model: str) -> str | None:
    key = extraction_cache_key(question_id, response_text, extractor_model)
    return cache.get(key)


def set_extraction(cache: dict, model: str, question_id: str, response_text: str, extractor_model: str, answer: str):
    key = extraction_cache_key(question_id, response_text, extractor_model)
    cache[key] = answer
    save_extraction_cache(model, cache)
