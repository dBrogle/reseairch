"""Cache for the generated programs (one per model).

Generating code is the only paid step, so we cache it keyed by PROMPT_VERSION.
Running + grading the program is free and recomputed each time. The raw response
and extracted code are both stored; the code is also written to output/scripts/
for easy inspection.
"""

import json
from pathlib import Path

from studies.poople_coding.config import OUTPUT_DIR, PROMPT_VERSION

STUDY_DIR = Path(__file__).parent
RESULTS_DIR = STUDY_DIR / OUTPUT_DIR / "results"
SCRIPTS_DIR = STUDY_DIR / OUTPUT_DIR / "scripts"
GRAPHS_DIR = STUDY_DIR / OUTPUT_DIR / "graphs"


def safe_model(model: str) -> str:
    out = model
    for ch in "/.-":
        out = out.replace(ch, "_")
    return out


def _file(model: str) -> Path:
    return RESULTS_DIR / f"{safe_model(model)}.json"


def load_code(model: str) -> dict | None:
    p = _file(model)
    if not p.exists():
        return None
    with open(p) as f:
        entry = json.load(f)
    if entry.get("prompt_version") != PROMPT_VERSION:
        return None
    return entry


def save_code(model: str, code: str | None, raw_response: str | None, error: str | None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "code": code,
        "raw_response": raw_response,
        "error": error,
    }
    with open(_file(model), "w") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)
    # Also drop the runnable script for inspection.
    if code:
        SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(SCRIPTS_DIR / f"{safe_model(model)}.py", "w") as f:
            f.write(code)
    return entry
