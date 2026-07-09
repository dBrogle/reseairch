"""Aggregate graded Poople attempts into per-model (and per-difficulty) stats.

The two headline metrics requested:
  * illegal moves — total, per-attempt average, and share of attempts with any.
  * average score relative to par — mean (moves − par) over *solved* attempts.

Plus a clean outcome partition of every attempt (optimal / suboptimal /
reached-but-illegal / failed / unparseable / error) for the scorecard charts.
"""

from collections import Counter

from studies.poople.config import ITERATIONS
from studies.poople.llm_cache import get_results, load_cache
from studies.poople.prompt import build_messages

# Outcome categories, in display order. Every attempt lands in exactly one.
OUTCOMES = [
    "optimal",          # solved at par (over_par == 0)
    "suboptimal",       # solved but over par
    "reached_illegal",  # ladder ends on target but used illegal move(s)
    "failed",           # parsed, but never reached the target legally
    "unparseable",      # got a response we couldn't parse as JSON
    "error",            # no response (API error after retries)
]

# Coarser 4-way split for the pie chart. Every attempt lands in exactly one.
# Note: ANY attempt that used an illegal move is "illegal" (even if it still
# reached poop); error/unparseable fold into "failed" (no valid solution).
PIE_CATEGORIES = ["par", "over_par", "illegal", "failed"]


def classify(result: dict) -> str:
    if result.get("error") is not None:
        return "error"
    if not result.get("parsed"):
        return "unparseable"
    if result.get("solved"):
        return "optimal" if result.get("over_par") == 0 else "suboptimal"
    if result.get("reached_target"):
        return "reached_illegal"
    return "failed"


def classify_pie(result: dict) -> str:
    if result.get("error") is not None or not result.get("parsed"):
        return "failed"
    if result.get("illegal_moves", 0) > 0:
        return "illegal"
    if result.get("solved"):
        return "par" if result.get("over_par") == 0 else "over_par"
    return "failed"


def flatten_results(
    model: str, test_words: list[tuple[str, int]], condition: str
) -> list[dict]:
    """All graded attempts for a model under one condition (≤ ITERATIONS each).

    Each item is the cached result dict augmented with its `word`.
    """
    cache = load_cache(model, condition)
    out: list[dict] = []
    for word, _par in test_words:
        for r in get_results(cache, build_messages(word))[:ITERATIONS]:
            out.append({**r, "word": word})
    return out


def _summarize(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {
            "n": 0, "solve_rate": 0.0, "avg_over_par": None,
            "illegal_total": 0, "illegal_per_attempt": 0.0, "pct_with_illegal": 0.0,
            "outcomes": {k: 0 for k in OUTCOMES},
            "pie": {k: 0 for k in PIE_CATEGORIES}, "over_par_dist": {},
        }

    outcomes = Counter(classify(r) for r in results)
    pie = Counter(classify_pie(r) for r in results)
    solved = [r for r in results if r.get("solved")]
    over_pars = [r["over_par"] for r in solved if r.get("over_par") is not None]
    illegal_total = sum(r.get("illegal_moves", 0) for r in results)
    with_illegal = sum(1 for r in results if r.get("illegal_moves", 0) > 0)

    return {
        "n": n,
        "solve_rate": len(solved) / n * 100,
        "avg_over_par": (sum(over_pars) / len(over_pars)) if over_pars else None,
        "illegal_total": illegal_total,
        "illegal_per_attempt": illegal_total / n,
        "pct_with_illegal": with_illegal / n * 100,
        "outcomes": {k: outcomes.get(k, 0) for k in OUTCOMES},
        "pie": {k: pie.get(k, 0) for k in PIE_CATEGORIES},
        "over_par_dist": dict(Counter(over_pars)),
    }


def compute_model_stats(model: str, test_words: list[tuple[str, int]], condition: str) -> dict:
    """Per-model summary plus a breakdown by difficulty bucket (par)."""
    results = flatten_results(model, test_words, condition)
    overall = _summarize(results)

    buckets = sorted({par for _w, par in test_words})
    by_bucket = {}
    for b in buckets:
        bucket_results = [r for r in results if r.get("par") == b]
        by_bucket[b] = _summarize(bucket_results)

    return {"model": model, "overall": overall, "by_bucket": by_bucket}
