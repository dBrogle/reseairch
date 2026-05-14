"""Pre-flight cost estimation for the Cognitive Biases study.

Reads the response cache to count how many (model × scenario × arm ×
iteration) combos still need to run, then projects a USD cost using
pricing the CostTracker has already fetched. Token counts are
rough char/4 approximations — fine for an at-a-glance "is this run
$0.20 or $20?" decision.

Multi-turn arms accumulate context: each prior turn's assistant
response gets added to the next turn's prompt, so the prompt-token
estimate climbs across turns.
"""

from studies.cognitive_biases.cache import get_responses, request_signature
from studies.cognitive_biases.scenarios.base import Scenario

# Heuristic per-call output sizing (in tokens):
#   - intermediate assistant responses in multi-turn arms (vivid
#     descriptions, debug discussions) tend to be longer
#   - the FINAL structured-JSON response is short
_INTERMEDIATE_COMPLETION_TOKENS = 500
_FINAL_COMPLETION_TOKENS        = 350

# Char→token ratio. OpenRouter / GPT family is ~3.5–4 chars per token
# for English; we round up slightly for safety.
_CHARS_PER_TOKEN = 3.8


def _approx_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _estimate_arm_tokens(scenario: Scenario, arm) -> tuple[int, int]:
    """Per-iteration (prompt_tokens, completion_tokens) estimate for one arm.

    For multi-turn arms, prior assistant responses accumulate into
    later turns' prompts, so the same intermediate response is counted
    once as completion (the call that produced it) and additional
    times as prompt context for each subsequent turn.
    """
    turns = list(arm.turn_list)
    n = len(turns)

    prompt_tokens = 0
    completion_tokens = 0

    # Track the running prompt context size as turns accumulate.
    running_prompt_tokens = 0
    for i, turn_text in enumerate(turns):
        is_final = (i == n - 1)
        suffixed = turn_text + (
            "\n\n" + scenario.response_format if is_final else ""
        )
        running_prompt_tokens += _approx_tokens(suffixed)

        # This turn's API call: bills the running prompt + its own
        # completion.
        prompt_tokens += running_prompt_tokens
        if is_final:
            completion_tokens += _FINAL_COMPLETION_TOKENS
        else:
            completion_tokens += _INTERMEDIATE_COMPLETION_TOKENS
            # The model's reply is added to the conversation for the
            # next turn — bake it into the running prompt.
            running_prompt_tokens += _INTERMEDIATE_COMPLETION_TOKENS

    return prompt_tokens, completion_tokens


def estimate_remaining_run(
    pricing: dict[str, tuple[float, float]],  # model_id -> (prompt_rate, completion_rate)
    response_caches: dict[str, dict],         # model_id -> response cache
    models: list[str],
    scenarios: list[Scenario],
    temperature: float,
    iterations: int,
) -> dict:
    """Pre-flight estimate of work + cost remaining after consulting the cache.

    Returns:
      {
        "total_calls":     int,    # all models × scenarios × arms × iterations × turns
        "cached_calls":    int,    # iterations already in cache (× turns)
        "remaining_calls": int,    # what we still need to run (× turns)
        "est_cost_usd":    float,  # projected cost for remaining calls only
        "by_model": {
            model: {
                "remaining_iters": int,
                "remaining_calls": int,  # × turns
                "est_cost_usd":    float,
            }
        },
      }
    """
    overall_total = 0
    overall_cached = 0
    overall_remaining = 0
    overall_cost = 0.0
    by_model: dict[str, dict] = {}

    for model in models:
        cache = response_caches.get(model, {})
        prompt_rate, completion_rate = pricing.get(model, (0.0, 0.0))

        m_remaining_iters = 0
        m_remaining_calls = 0
        m_total_iters = 0
        m_cached_iters = 0
        m_cost = 0.0

        for scenario in scenarios:
            for arm in scenario.arms:
                turns_per_iter = len(arm.turn_list)
                m_total_iters += iterations

                sig = request_signature(scenario, arm)
                cached = len(get_responses(cache, sig, temperature))
                cached = min(cached, iterations)
                m_cached_iters += cached

                remaining = iterations - cached
                if remaining <= 0:
                    continue

                m_remaining_iters += remaining
                m_remaining_calls += remaining * turns_per_iter

                pt, ct = _estimate_arm_tokens(scenario, arm)
                m_cost += remaining * (pt * prompt_rate + ct * completion_rate)

        by_model[model] = {
            "remaining_iters": m_remaining_iters,
            "remaining_calls": m_remaining_calls,
            "est_cost_usd":    m_cost,
        }
        overall_total += m_total_iters
        overall_cached += m_cached_iters
        overall_remaining += m_remaining_iters
        overall_cost += m_cost

    # "calls" = API invocations (one per turn for multi-turn arms)
    overall_total_calls = sum(
        len(arm.turn_list) * iterations
        for s in scenarios for arm in s.arms
    ) * len(models)
    overall_cached_calls = overall_total_calls - sum(
        v["remaining_calls"] for v in by_model.values()
    )
    overall_remaining_calls = sum(
        v["remaining_calls"] for v in by_model.values()
    )

    return {
        "total_iters":      overall_total,
        "cached_iters":     overall_cached,
        "remaining_iters":  overall_remaining,
        "total_calls":      overall_total_calls,
        "cached_calls":     overall_cached_calls,
        "remaining_calls":  overall_remaining_calls,
        "est_cost_usd":     overall_cost,
        "by_model":         by_model,
    }


def format_estimate(est: dict, models: list[str]) -> str:
    """Pretty-print the estimate as a multi-line string."""
    lines = []
    lines.append(
        f"  Iterations: {est['cached_iters']:,} cached / "
        f"{est['remaining_iters']:,} remaining "
        f"(of {est['total_iters']:,} total)"
    )
    lines.append(
        f"  API calls : {est['cached_calls']:,} cached / "
        f"{est['remaining_calls']:,} remaining "
        f"(multi-turn arms count one call per turn)"
    )
    lines.append(
        f"  Estimated cost (remaining only): ${est['est_cost_usd']:.4f}"
    )
    if est["remaining_iters"] == 0:
        return "\n".join(lines)

    lines.append("  By model:")
    for m in models:
        bm = est["by_model"].get(m)
        if bm is None or bm["remaining_iters"] == 0:
            lines.append(f"    {m}: fully cached")
            continue
        lines.append(
            f"    {m}: {bm['remaining_iters']:,} iters "
            f"({bm['remaining_calls']:,} calls)  "
            f"~${bm['est_cost_usd']:.4f}"
        )
    return "\n".join(lines)
