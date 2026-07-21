"""Response cache for the Poople LLM test.

Results, graphs, and result-image boards are split by experiment *condition*
("reasoning" vs "no_reasoning") into sibling directories:

    output/results_<condition>/<model>.json
    output/graphs_<condition>/
    output/results_images_<condition>/

Each cache file is one model under one condition, keyed by a hash of (prompt
messages + PROMPT_VERSION). Each start word has its own key, so adding words to
the sample never disturbs cached results for existing words. Flushes on every
write so interrupted runs resume cleanly.
"""

import hashlib
import json
from pathlib import Path

from studies.poople.config import OUTPUT_DIR, PROMPT_VERSION

STUDY_DIR = Path(__file__).parent
OUTPUT_ROOT = STUDY_DIR / OUTPUT_DIR


# ---------------------------------------------------------------------------
# Per-condition directories
# ---------------------------------------------------------------------------

def results_dir(condition: str) -> Path:
    return OUTPUT_ROOT / f"results_{condition}"


def graphs_dir(condition: str) -> Path:
    return OUTPUT_ROOT / f"graphs_{condition}"


def results_images_dir(condition: str) -> Path:
    return OUTPUT_ROOT / f"results_images_{condition}"


def vendor_graphs_dir(vendor: str, condition: str) -> Path:
    """Graphs restricted to one vendor's models, e.g. output/openai_no_reasoning/."""
    return OUTPUT_ROOT / f"{vendor}_{condition}"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def cache_key(messages: list[dict]) -> str:
    raw = json.dumps(messages, sort_keys=True) + "|" + PROMPT_VERSION
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _model_file(model: str, condition: str) -> Path:
    return results_dir(condition) / f"{model.replace('/', '_')}.json"


def load_cache(model: str, condition: str) -> dict:
    path = _model_file(model, condition)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(model: str, condition: str, cache: dict):
    results_dir(condition).mkdir(parents=True, exist_ok=True)
    with open(_model_file(model, condition), "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def get_results(cache: dict, messages: list[dict]) -> list[dict]:
    entry = cache.get(cache_key(messages))
    return entry.get("results", []) if entry else []


def append_result(
    cache: dict,
    model: str,
    condition: str,
    messages: list[dict],
    result: dict,
    word: str,
    par: int,
):
    key = cache_key(messages)
    if key not in cache:
        cache[key] = {"word": word, "par": par, "messages": messages, "results": []}
    cache[key]["results"].append(result)
    save_cache(model, condition, cache)


def purge_errors(cache: dict, model: str, condition: str, messages: list[dict]) -> int:
    """Drop errored (no-response) results so they get retried. Returns count."""
    key = cache_key(messages)
    if key not in cache or "results" not in cache[key]:
        return 0
    before = len(cache[key]["results"])
    cache[key]["results"] = [r for r in cache[key]["results"] if r.get("error") is None]
    purged = before - len(cache[key]["results"])
    if purged:
        save_cache(model, condition, cache)
    return purged
