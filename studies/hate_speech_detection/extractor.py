"""Extraction logic for the Hate Speech Detection Bias study.

Parses LLM responses to determine:
  - flagged: bool (True = hate speech, False = not hate speech)
  - confidence: str ("low", "medium", "high") or None
"""

import json
import re

from studies.hate_speech_detection.config import (
    GROUPS,
    TEMPLATES,
    TEMPERATURE,
    ITERATIONS,
)
from studies.hate_speech_detection.runner import get_group_text, build_messages
from studies.hate_speech_detection.cache import (
    load_response_cache,
    get_responses,
)


def extract_decision(response_text: str) -> dict:
    """Extract flag decision from a JSON response."""
    if not response_text:
        return {"flagged": None}

    # Try direct JSON parse
    try:
        data = json.loads(response_text.strip())
        if isinstance(data, dict) and "flagged" in data:
            return {"flagged": bool(data["flagged"])}
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: extract JSON from markdown code blocks or surrounding text
    json_match = re.search(r'\{[^{}]*"flagged"\s*:\s*(true|false)[^{}]*\}', response_text, re.IGNORECASE)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return {"flagged": bool(data["flagged"])}
        except (json.JSONDecodeError, ValueError):
            pass

    # Last resort: look for the raw value
    if re.search(r'"flagged"\s*:\s*true', response_text, re.IGNORECASE):
        return {"flagged": True}
    if re.search(r'"flagged"\s*:\s*false', response_text, re.IGNORECASE):
        return {"flagged": False}

    return {"flagged": None}


def compute_flag_rates(model: str) -> dict:
    """
    Compute flag rates for all (template, group) combos for a model.

    Returns nested dict: {template_id: {group_text: {
        "flag_rate": float,
        "count": int,
        "flagged": int,
        "group_id": str,
        "parse_failures": int,
    }}}
    """
    cache = load_response_cache(model)
    results = {}

    for template in TEMPLATES:
        results[template["id"]] = {}

        for group in GROUPS:
            group_text = get_group_text(group, template["form"])

            messages = build_messages(template, group)
            responses = get_responses(cache, messages, TEMPERATURE)

            flagged_count = 0
            total = 0
            parse_failures = 0

            for r in responses:
                resp_text = r.get("response")
                if resp_text is None:
                    continue

                decision = extract_decision(resp_text)

                if decision["flagged"] is None:
                    parse_failures += 1
                    continue

                total += 1
                if decision["flagged"]:
                    flagged_count += 1

            results[template["id"]][group_text] = {
                "flag_rate": flagged_count / total if total > 0 else None,
                "count": total,
                "flagged": flagged_count,
                "group_id": group["id"],
                "parse_failures": parse_failures,
            }

    return results
