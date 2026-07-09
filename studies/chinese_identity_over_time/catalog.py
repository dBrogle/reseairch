"""OpenRouter model catalog: real release dates for the over-time study.

The whole point of this study is to plot *when* Western models started claiming a
Chinese identity, so we need an authoritative release date per model. Rather than
hand-maintain a date table, we pull it straight from OpenRouter's ``/models``
endpoint, which exposes a ``created`` unix timestamp (when the model was listed on
OpenRouter) plus a human display name. The response is cached to disk so repeated
runs don't re-hit the API and dates stay stable across a run.

Public helpers:
  - ``load_catalog()``         -> {model_id: {"created": int, "name": str, ...}}
  - ``release_date(model_id)`` -> ``datetime.date`` (or None if unknown)
  - ``display_name(model_id)`` -> short human name (falls back to the id tail)
"""

import datetime
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

STUDY_DIR = Path(__file__).parent
CATALOG_PATH = STUDY_DIR / "output" / "catalog.json"
MODELS_URL = "https://openrouter.ai/api/v1/models"

_CACHE: dict[str, dict] | None = None


def _fetch_from_api() -> dict[str, dict]:
    """Pull the live model list from OpenRouter and keep only the fields we use."""
    key = os.getenv("OPENROUTER_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    resp = httpx.get(MODELS_URL, headers=headers, timeout=60.0)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    catalog = {}
    for m in data:
        mid = m.get("id")
        if not mid:
            continue
        catalog[mid] = {
            "created": m.get("created"),
            "name": m.get("name"),
            "knowledge_cutoff": m.get("knowledge_cutoff"),
        }
    return catalog


def refresh_catalog() -> dict[str, dict]:
    """Force a re-fetch from the API and overwrite the on-disk cache."""
    catalog = _fetch_from_api()
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CATALOG_PATH, "w") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    global _CACHE
    _CACHE = catalog
    return catalog


def load_catalog() -> dict[str, dict]:
    """Return the model catalog, fetching+caching from OpenRouter on first use."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if CATALOG_PATH.exists():
        with open(CATALOG_PATH, "r") as f:
            _CACHE = json.load(f)
        return _CACHE
    return refresh_catalog()


def release_date(model_id: str) -> datetime.date | None:
    """Release date (OpenRouter ``created``) for a model, or None if unknown."""
    entry = load_catalog().get(model_id)
    if not entry or not entry.get("created"):
        return None
    return datetime.date.fromtimestamp(entry["created"])


def display_name(model_id: str) -> str:
    """Short human-friendly name: OpenRouter name minus the vendor prefix."""
    entry = load_catalog().get(model_id)
    name = entry.get("name") if entry else None
    if name:
        # "OpenAI: GPT-5.5" -> "GPT-5.5"
        return name.split(":", 1)[-1].strip()
    return model_id.split("/")[-1]
