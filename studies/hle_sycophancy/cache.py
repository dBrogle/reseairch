"""Cache management for the HLE Sycophancy study.

Three separate caches:
1. HLE questions: the sampled question set, keyed by (seed, per_category).
   Same config always produces the same questions across models and runs.
2. Raw responses: per-model, keyed by hash of (full messages, temperature).
   Each entry is independent of how many other questions are being asked —
   a response for question Q in state S is cached and reused regardless of
   whether the current run has 2 or 20 questions.
3. Graded answers: per-model, keyed by hash of (question_id, response, grader_model).
"""

import hashlib
import json
from pathlib import Path

from studies.hle_sycophancy.config import OUTPUT_DIR

STUDY_DIR = Path(__file__).parent
RESULTS_DIR  = STUDY_DIR / OUTPUT_DIR / "results"
GRADINGS_DIR = STUDY_DIR / OUTPUT_DIR / "gradings"
GRAPHS_DIR   = STUDY_DIR / OUTPUT_DIR / "graphs"
HLE_CACHE_DIR = STUDY_DIR / OUTPUT_DIR


def _hash(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# HLE questions cache (shared across models)
#
# Keyed by seed + per_category.  Changing either invalidates the cache so a
# fresh sample is drawn, but identical config reuses the same questions
# across models and across runs.
# ---------------------------------------------------------------------------

def _hle_questions_path(seed: int, per_category: int) -> Path:
    return HLE_CACHE_DIR / f"hle_questions_seed{seed}_per{per_category}.json"


def load_hle_questions(seed: int, per_category: int) -> list[dict] | None:
    """Return the cached question list for this seed + per_category, or None."""
    path = _hle_questions_path(seed, per_category)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def save_hle_questions(seed: int, per_category: int, questions: list[dict]):
    HLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _hle_questions_path(seed, per_category)
    with open(path, "w") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"  Cached {len(questions)} HLE questions to {path}")


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
    state: str,
):
    """Append a response and flush to disk. Stores question metadata in the entry."""
    key = response_cache_key(messages, temperature)
    if key not in cache:
        cache[key] = {
            "messages": messages,
            "temperature": temperature,
            # Question metadata stored for human readability and grading
            "question_id":   question_meta["id"],
            "question":      question_meta["question"],
            "answer":        question_meta["answer"],
            "answer_type":   question_meta["answer_type"],
            "category":      question_meta["category"],
            "raw_subject":   question_meta["raw_subject"],
            "state":         state,
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
