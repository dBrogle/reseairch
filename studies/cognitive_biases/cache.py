"""Cache management for the Cognitive Biases study.

Two caches per model, both keyed by stable hashes so changing a prompt
or extraction logic automatically invalidates only the affected entries:

1. Response cache — keyed by the deterministic *user-side* request
   signature (scenario id, arm key, response_format, temperature, the
   tuple of user turn strings). For multi-turn arms the intermediate
   model responses naturally vary across iterations and aren't part of
   the key.
2. Extraction cache — keyed by hash of (scenario_id, arm_key,
   response_text, value_field, extractor_model). Stores the parsed
   numeric value plus reasoning.

Both caches flush to disk on every write so an interrupted run can be
resumed without losing work.
"""

import hashlib
import json
from pathlib import Path

from studies.cognitive_biases.config import OUTPUT_DIR
from studies.cognitive_biases.scenarios.base import Arm, Scenario

STUDY_DIR = Path(__file__).parent
RESULTS_DIR     = STUDY_DIR / OUTPUT_DIR / "results"
EXTRACTIONS_DIR = STUDY_DIR / OUTPUT_DIR / "extractions"
GRAPHS_DIR      = STUDY_DIR / OUTPUT_DIR / "graphs"


def _hash(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _safe_model(model: str) -> str:
    return model.replace("/", "_")


# ---------------------------------------------------------------------------
# Request signatures (deterministic, user-side only)
# ---------------------------------------------------------------------------

def request_signature(scenario: Scenario, arm: Arm) -> list[dict]:
    """User-side message shape used for caching.

    For both single- and multi-turn arms, this is the tuple of user
    turns with `response_format` appended to the LAST turn. Intermediate
    model responses are deliberately excluded from the key so the cache
    key stays deterministic regardless of stochastic model output across
    iterations.
    """
    turns = list(arm.turn_list)
    turns[-1] = f"{turns[-1]}\n\n{scenario.response_format}"
    return [{"role": "user", "content": t} for t in turns]


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------

def response_cache_key(signature: list[dict], temperature: float) -> str:
    return _hash(json.dumps(signature, sort_keys=True), str(temperature))


def _response_file(model: str) -> Path:
    return RESULTS_DIR / f"{_safe_model(model)}.json"


def load_response_cache(model: str) -> dict:
    path = _response_file(model)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_response_cache(model: str, cache: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_response_file(model), "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def get_responses(cache: dict, signature: list[dict], temperature: float) -> list[dict]:
    key = response_cache_key(signature, temperature)
    entry = cache.get(key)
    if entry is None:
        return []
    return entry.get("results", [])


def append_response(
    cache: dict,
    model: str,
    signature: list[dict],
    temperature: float,
    result: dict,
    scenario_id: str,
    arm_key: str,
    bias_type: str,
):
    """Append a response and flush. Stores scenario/arm metadata so the
    cache file is human-readable and self-describing."""
    key = response_cache_key(signature, temperature)
    if key not in cache:
        cache[key] = {
            "scenario_id": scenario_id,
            "arm_key":     arm_key,
            "bias_type":   bias_type,
            "user_turns":  signature,
            "temperature": temperature,
            "results":     [],
        }
    cache[key]["results"].append(result)
    save_response_cache(model, cache)


def purge_response_errors(cache: dict, model: str) -> int:
    """Strip errored results so they get retried. Returns count purged."""
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
# Extraction cache
# ---------------------------------------------------------------------------

def extraction_cache_key(
    scenario_id: str,
    arm_key: str,
    response_text: str,
    value_field: str,
    extractor_model: str,
) -> str:
    return _hash(scenario_id, arm_key, response_text, value_field, extractor_model)


def _extraction_file(model: str) -> Path:
    return EXTRACTIONS_DIR / f"{_safe_model(model)}.json"


def load_extraction_cache(model: str) -> dict:
    path = _extraction_file(model)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_extraction_cache(model: str, cache: dict):
    EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_extraction_file(model), "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def get_extraction(
    cache: dict,
    scenario_id: str,
    arm_key: str,
    response_text: str,
    value_field: str,
    extractor_model: str,
) -> dict | None:
    key = extraction_cache_key(
        scenario_id, arm_key, response_text, value_field, extractor_model
    )
    return cache.get(key)


def set_extraction(
    cache: dict,
    model: str,
    scenario_id: str,
    arm_key: str,
    response_text: str,
    value_field: str,
    extractor_model: str,
    extraction: dict,
):
    key = extraction_cache_key(
        scenario_id, arm_key, response_text, value_field, extractor_model
    )
    cache[key] = extraction
    save_extraction_cache(model, cache)
