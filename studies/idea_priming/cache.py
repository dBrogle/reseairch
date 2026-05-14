"""Cache management for the Idea Priming study.

Two caches:
1. Raw responses: per-model, keyed by hash of (full messages, temperature).
   Because the messages array contains the system prompt, the idea
   description, and the priming question, ANY change to prompts or to an
   idea's wording invalidates that idea's cache automatically — no false
   cache hits after a prompt edit.
2. Extractions: per-model, keyed by hash of (idea_id, frame, response_text,
   extractor_model). Holds the parsed {reasoning, score}.
"""

import hashlib
import json
from pathlib import Path

from studies.idea_priming.config import OUTPUT_DIR

STUDY_DIR = Path(__file__).parent
RESULTS_DIR     = STUDY_DIR / OUTPUT_DIR / "results"
EXTRACTIONS_DIR = STUDY_DIR / OUTPUT_DIR / "extractions"
GRAPHS_DIR      = STUDY_DIR / OUTPUT_DIR / "graphs"


def _hash(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Raw response cache (per model)
# ---------------------------------------------------------------------------

def response_cache_key(messages: list[dict], temperature: float) -> str:
    return _hash(json.dumps(messages, sort_keys=True), str(temperature))


def _model_file(model: str) -> Path:
    safe = model.replace("/", "_")
    return RESULTS_DIR / f"{safe}.json"


def load_response_cache(model: str) -> dict:
    path = _model_file(model)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_response_cache(model: str, cache: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _model_file(model)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def get_responses(cache: dict, messages: list[dict], temperature: float) -> list[dict]:
    key = response_cache_key(messages, temperature)
    entry = cache.get(key)
    if entry is None:
        return []
    return entry.get("results", [])


def append_response(
    cache: dict,
    model: str,
    messages: list[dict],
    temperature: float,
    result: dict,
    idea: dict,
    frame_key: str,
):
    """Append a response and flush to disk. Stores idea + frame metadata."""
    key = response_cache_key(messages, temperature)
    if key not in cache:
        cache[key] = {
            "messages": messages,
            "temperature": temperature,
            "idea_id":     idea["id"],
            "bucket":      idea["bucket"],
            "category":    idea["category"],
            "description": idea["description"],
            "frame":       frame_key,
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

def _extraction_file(model: str) -> Path:
    safe = model.replace("/", "_")
    return EXTRACTIONS_DIR / f"{safe}.json"


def load_extraction_cache(model: str) -> dict:
    path = _extraction_file(model)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_extraction_cache(model: str, cache: dict):
    EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = _extraction_file(model)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def extraction_cache_key(
    idea_id: str, frame_key: str, response_text: str, extractor_model: str
) -> str:
    return _hash(idea_id, frame_key, response_text, extractor_model)


def get_extraction(
    cache: dict,
    idea_id: str,
    frame_key: str,
    response_text: str,
    extractor_model: str,
) -> dict | None:
    key = extraction_cache_key(idea_id, frame_key, response_text, extractor_model)
    return cache.get(key)


def set_extraction(
    cache: dict,
    model: str,
    idea_id: str,
    frame_key: str,
    response_text: str,
    extractor_model: str,
    extraction: dict,
):
    key = extraction_cache_key(idea_id, frame_key, response_text, extractor_model)
    cache[key] = extraction
    save_extraction_cache(model, cache)
