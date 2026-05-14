"""Cache management for the Question Framing study.

Two caches:
1. Raw responses: per-model, keyed by hash of (full messages, temperature).
2. Graded answers: per-model, keyed by hash of (question_id, response, grader_model).

HLE questions are cached by the hle_sycophancy study and reused here.
"""

import hashlib
import json
from pathlib import Path

from studies.question_framing.config import OUTPUT_DIR

STUDY_DIR = Path(__file__).parent
RESULTS_DIR  = STUDY_DIR / OUTPUT_DIR / "results"
GRADINGS_DIR = STUDY_DIR / OUTPUT_DIR / "gradings"
GRAPHS_DIR   = STUDY_DIR / OUTPUT_DIR / "graphs"


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
    question_meta: dict,
    framing_key: str,
):
    """Append a response and flush to disk. Stores question metadata in the entry."""
    key = response_cache_key(messages, temperature)
    if key not in cache:
        cache[key] = {
            "messages": messages,
            "temperature": temperature,
            "question_id":   question_meta["id"],
            "question":      question_meta["question"],
            "answer":        question_meta["answer"],
            "answer_type":   question_meta["answer_type"],
            "category":      question_meta["category"],
            "raw_subject":   question_meta["raw_subject"],
            "framing":       framing_key,
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
# Grading cache (per model)
# ---------------------------------------------------------------------------

def _grading_file(model: str) -> Path:
    safe = model.replace("/", "_")
    return GRADINGS_DIR / f"{safe}.json"


def load_grading_cache(model: str) -> dict:
    path = _grading_file(model)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_grading_cache(model: str, cache: dict):
    GRADINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = _grading_file(model)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def grading_cache_key(question_id: str, response_text: str, grader_model: str) -> str:
    return _hash(question_id, response_text, grader_model)


def get_grading(
    cache: dict, question_id: str, response_text: str, grader_model: str
) -> str | None:
    """Return cached grading ("CORRECT" | "INCORRECT" | "ERROR"), or None."""
    key = grading_cache_key(question_id, response_text, grader_model)
    return cache.get(key)


def set_grading(
    cache: dict,
    model: str,
    question_id: str,
    response_text: str,
    grader_model: str,
    verdict: str,
):
    key = grading_cache_key(question_id, response_text, grader_model)
    cache[key] = verdict
    save_grading_cache(model, cache)
