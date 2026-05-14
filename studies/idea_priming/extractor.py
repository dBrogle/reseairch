"""LLM-powered extraction for the Idea Priming study.

Takes raw responses and extracts {reasoning, score} where score is an
integer 1-10. Tries direct JSON parsing first, falls back to a batched
extractor LLM call for unparseable responses. All extractions are cached.

Also provides score statistics (means, 95% t-CIs, paired biases) used by
the visualizer.
"""

import asyncio
import json
import math
import random

from services.llm import OpenRouterProvider
from studies.idea_priming.config import (
    EXTRACTION_BATCH_SIZE,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTOR_MODEL,
    FRAME_KEYS,
    MAX_PARALLEL_REQUESTS,
)
from studies.idea_priming.cache import (
    get_extraction,
    get_responses,
    load_extraction_cache,
    load_response_cache,
    set_extraction,
)
from studies.idea_priming.runner import build_messages

ERROR_EXTRACTION   = {"reasoning": "ERROR",   "score": -1}
REFUSED_EXTRACTION = {"reasoning": "REFUSED", "score": -1}


# ---------------------------------------------------------------------------
# Direct JSON parsing
# ---------------------------------------------------------------------------

def _try_direct_parse(response_text: str) -> dict | None:
    """Try to parse the response directly as the expected JSON shape."""
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(parsed, dict) or "score" not in parsed:
        return None

    try:
        score = int(parsed["score"])
    except (ValueError, TypeError):
        return None

    if score < 1 or score > 10:
        return None

    return {
        "reasoning": str(parsed.get("reasoning", "")),
        "score": score,
    }


# ---------------------------------------------------------------------------
# Batch extraction (fallback path)
# ---------------------------------------------------------------------------

def _build_batch_input(items: list[tuple[dict, str, str]]) -> str:
    """Build the JSON input for a batch of (idea, frame_key, response_text)."""
    batch = {}
    for idx, (idea, frame_key, response_text) in enumerate(items):
        batch[str(idx)] = {
            "idea":     idea["description"],
            "frame":    frame_key,
            "response": response_text,
        }
    return json.dumps(batch, indent=2)


def _parse_batch_output(raw: str, batch_size: int) -> dict[int, dict]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

    parsed = json.loads(text)
    results: dict[int, dict] = {}
    for idx_str, data in parsed.items():
        try:
            idx = int(idx_str)
        except (ValueError, TypeError):
            continue
        if idx < 0 or idx >= batch_size:
            continue
        try:
            reasoning = str(data.get("reasoning", ""))
            raw_score = data.get("score", -1)
            score = int(raw_score)
        except (ValueError, TypeError, AttributeError):
            results[idx] = ERROR_EXTRACTION.copy()
            continue
        if score == -1:
            results[idx] = {"reasoning": reasoning, "score": -1}
        elif 1 <= score <= 10:
            results[idx] = {"reasoning": reasoning, "score": score}
        else:
            results[idx] = ERROR_EXTRACTION.copy()
    return results


async def extract_batch(
    provider: OpenRouterProvider,
    items: list[tuple[dict, str, str]],
    cost_tracker=None,
) -> list[dict]:
    """Extract for a batch of (idea, frame_key, response_text) triples."""
    if not items:
        return []

    # Direct-parse first
    results: list[dict | None] = [None] * len(items)
    needs_extraction: list[int] = []
    for i, (_, _, response_text) in enumerate(items):
        parsed = _try_direct_parse(response_text)
        if parsed is not None:
            results[i] = parsed
        else:
            needs_extraction.append(i)

    if not needs_extraction:
        return results  # type: ignore[return-value]

    ext_items = [items[i] for i in needs_extraction]
    indices = list(range(len(ext_items)))
    random.shuffle(indices)
    shuffled = [ext_items[i] for i in indices]

    batch_json = _build_batch_input(shuffled)

    try:
        result, usage = await provider.complete_text_with_usage(
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": batch_json},
            ],
            model=EXTRACTOR_MODEL,
            temperature=0.0,
            max_tokens=2000,
        )
        if cost_tracker and usage:
            await cost_tracker.record(EXTRACTOR_MODEL, usage)
        parsed = _parse_batch_output(result, len(shuffled))
    except Exception:
        for i in needs_extraction:
            if results[i] is None:
                results[i] = ERROR_EXTRACTION.copy()
        return results  # type: ignore[return-value]

    for shuffled_idx, original_idx in enumerate(indices):
        real_idx = needs_extraction[original_idx]
        results[real_idx] = parsed.get(shuffled_idx, ERROR_EXTRACTION.copy())

    for i in range(len(results)):
        if results[i] is None:
            results[i] = ERROR_EXTRACTION.copy()

    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Extract all cached responses
# ---------------------------------------------------------------------------

async def extract_all(
    provider: OpenRouterProvider,
    models: list[str],
    ideas: list[dict],
    temperature: float,
    iterations: int,
    frame_keys: list[str] | None = None,
    cost_tracker=None,
) -> dict[str, dict]:
    target_frames = frame_keys or FRAME_KEYS

    jobs: list[tuple[str, dict, str, str, int]] = []
    for model in models:
        response_cache = load_response_cache(model)
        extraction_cache = load_extraction_cache(model)

        for frame_key in target_frames:
            for idea in ideas:
                messages = build_messages(frame_key, idea)
                responses = get_responses(response_cache, messages, temperature)[:iterations]
                for i, result in enumerate(responses):
                    if result.get("error") is not None or result.get("response") is None:
                        continue
                    existing = get_extraction(
                        extraction_cache, idea["id"], frame_key,
                        result["response"], EXTRACTOR_MODEL,
                    )
                    if existing is None:
                        jobs.append((model, idea, frame_key, result["response"], i))

    total_needed = len(jobs)
    if total_needed == 0:
        print("\n  All extractions already cached.")
        return {m: load_extraction_cache(m) for m in models}

    if cost_tracker:
        cost_tracker.start_phase()

    num_batches = (total_needed + EXTRACTION_BATCH_SIZE - 1) // EXTRACTION_BATCH_SIZE
    print(
        f"\n  {total_needed} extractions needed in "
        f"{num_batches} batches of up to {EXTRACTION_BATCH_SIZE}."
    )

    batches = [
        jobs[i:i + EXTRACTION_BATCH_SIZE]
        for i in range(0, total_needed, EXTRACTION_BATCH_SIZE)
    ]

    extraction_caches: dict[str, dict] = {m: load_extraction_cache(m) for m in models}
    cache_locks: dict[str, asyncio.Lock] = {m: asyncio.Lock() for m in models}
    semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)
    done_batches = 0
    done_lock = asyncio.Lock()

    async def _run_batch(batch: list[tuple[str, dict, str, str, int]]):
        nonlocal done_batches

        items = [(idea, frame_key, response_text)
                 for _, idea, frame_key, response_text, _ in batch]

        async with semaphore:
            answers = await extract_batch(provider, items, cost_tracker=cost_tracker)

        for (model, idea, frame_key, response_text, _), answer in zip(batch, answers):
            async with cache_locks[model]:
                set_extraction(
                    extraction_caches[model], model,
                    idea["id"], frame_key, response_text,
                    EXTRACTOR_MODEL, answer,
                )

        async with done_lock:
            done_batches += 1
            scores_sample = ", ".join(str(a.get("score", "?")) for a in answers[:5])
            cost_str = (
                f" | {cost_tracker.format_status(done_batches, num_batches)}"
                if cost_tracker else ""
            )
            print(
                f"  [batch {done_batches}/{num_batches}] {len(batch)} items "
                f"-> [{scores_sample}{', ...' if len(answers) > 5 else ''}]{cost_str}"
            )

    await asyncio.gather(*[_run_batch(b) for b in batches])

    if cost_tracker and total_needed > 0:
        print(f"\n  Extractor cost: {cost_tracker.format_total()}")

    for model in models:
        extraction_caches[model] = load_extraction_cache(model)

    return extraction_caches


# ---------------------------------------------------------------------------
# Score statistics
# ---------------------------------------------------------------------------

# t-distribution critical values for 95% two-sided CI, df = n-1.
# Hard-coded for common small n so we don't need to import scipy.
_T_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    40: 2.021, 50: 2.009, 60: 2.000, 80: 1.990, 100: 1.984,
    200: 1.972,
}


def _t_critical_95(df: int) -> float:
    """Return t critical value for 95% two-sided CI, df = n - 1."""
    if df <= 0:
        return float("nan")
    if df in _T_95:
        return _T_95[df]
    # Pick the closest pre-computed df below or use the asymptotic value.
    keys = sorted(_T_95.keys())
    if df > keys[-1]:
        return 1.960
    for k in keys:
        if k > df:
            return _T_95[k]
    return 1.960


def t_ci_95(values: list[float]) -> tuple[float, float, float]:
    """Mean and t-distribution 95% CI. Returns (mean, lo, hi).

    For n == 1 returns (mean, mean, mean) — no CI possible.
    """
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = sum(values) / n
    if n == 1:
        return (mean, mean, mean)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(var)
    sem = sd / math.sqrt(n)
    margin = _t_critical_95(n - 1) * sem
    return (mean, mean - margin, mean + margin)


def _collect_scores(
    model: str,
    ideas: list[dict],
    temperature: float,
    iterations: int,
) -> dict[str, dict[str, list[int]]]:
    """Return {idea_id: {frame_key: [scores]}} for a model.

    Only scores in [1, 10] (i.e. successfully extracted) are included.
    """
    response_cache = load_response_cache(model)
    extraction_cache = load_extraction_cache(model)

    out: dict[str, dict[str, list[int]]] = {}
    for idea in ideas:
        out[idea["id"]] = {fk: [] for fk in FRAME_KEYS}
        for frame_key in FRAME_KEYS:
            messages = build_messages(frame_key, idea)
            responses = get_responses(response_cache, messages, temperature)[:iterations]
            for result in responses:
                if result.get("error") is not None or result.get("response") is None:
                    continue
                ext = get_extraction(
                    extraction_cache, idea["id"], frame_key,
                    result["response"], EXTRACTOR_MODEL,
                )
                if ext is None:
                    continue
                score = ext.get("score", -1)
                if isinstance(score, int) and 1 <= score <= 10:
                    out[idea["id"]][frame_key].append(score)
    return out


def compute_model_stats(
    model: str,
    ideas: list[dict],
    temperature: float,
    iterations: int,
) -> dict:
    """
    Compute per-model and per-(model, idea) stats.

    Returns:
    {
      "model": str,
      "scores": {idea_id: {frame_key: [scores]}},
      "per_idea": {
          idea_id: {
              frame_key: {"mean", "ci_low", "ci_high", "n", "values"},
              ...
              "bias": positive_mean - negative_mean,
          }
      },
      "frame_pooled": {
          frame_key: {"mean", "ci_low", "ci_high", "n", "values"}
      },
      "per_idea_bias": [(idea_id, bias), ...],   # only ideas with both frames
      "model_bias": {"mean", "ci_low", "ci_high", "n_ideas"},
    }

    The "model_bias" CI is computed across per-idea biases (each idea is one
    paired observation), which is the right unit of analysis for paired-frame
    inference. The "frame_pooled" CIs treat each iteration as independent,
    which slightly understates uncertainty due to clustering by idea — they're
    mainly for showing the raw mean per frame.
    """
    scores = _collect_scores(model, ideas, temperature, iterations)

    per_idea: dict[str, dict] = {}
    pooled: dict[str, list[int]] = {fk: [] for fk in FRAME_KEYS}
    biases: list[tuple[str, float]] = []

    for idea in ideas:
        idea_id = idea["id"]
        idea_stats: dict = {}
        means: dict[str, float | None] = {}
        for frame_key in FRAME_KEYS:
            vals = scores[idea_id][frame_key]
            if not vals:
                idea_stats[frame_key] = None
                means[frame_key] = None
                continue
            mean, lo, hi = t_ci_95([float(v) for v in vals])
            idea_stats[frame_key] = {
                "mean":    mean,
                "ci_low":  lo,
                "ci_high": hi,
                "n":       len(vals),
                "values":  list(vals),
            }
            means[frame_key] = mean
            pooled[frame_key].extend(vals)

        if means.get("positive") is not None and means.get("negative") is not None:
            bias = means["positive"] - means["negative"]
            idea_stats["bias"] = bias
            biases.append((idea_id, bias))
        else:
            idea_stats["bias"] = None

        per_idea[idea_id] = idea_stats

    frame_pooled: dict[str, dict | None] = {}
    for frame_key in FRAME_KEYS:
        vals = pooled[frame_key]
        if not vals:
            frame_pooled[frame_key] = None
        else:
            mean, lo, hi = t_ci_95([float(v) for v in vals])
            frame_pooled[frame_key] = {
                "mean":    mean,
                "ci_low":  lo,
                "ci_high": hi,
                "n":       len(vals),
                "values":  list(vals),
            }

    if biases:
        bias_vals = [b for _, b in biases]
        bmean, blo, bhi = t_ci_95(bias_vals)
        model_bias = {
            "mean":    bmean,
            "ci_low":  blo,
            "ci_high": bhi,
            "n_ideas": len(bias_vals),
        }
    else:
        model_bias = None

    return {
        "model":         model,
        "scores":        scores,
        "per_idea":      per_idea,
        "frame_pooled":  frame_pooled,
        "per_idea_bias": biases,
        "model_bias":    model_bias,
    }
