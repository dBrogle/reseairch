"""Configuration for the Cognitive Biases study.

Tests classic human cognitive biases in LLMs (anchoring, sunk cost,
social proof, authority bias, ...). Each bias is broken into one or more
scenarios that live in `studies/cognitive_biases/scenarios/`. A scenario
is a control prompt plus one or more treatment prompts that introduce a
biasing trigger; an unbiased model produces statistically
indistinguishable distributions across arms.
"""

from studies.cognitive_biases.scenarios import (
    ALL_SCENARIOS,
    BIAS_TYPES,
    SCENARIOS_BY_ID,
    scenarios_by_bias,
)
from studies.cognitive_biases.scenarios.base import Scenario

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MODELS = [
    "openai/gpt-5.4-mini",
    "anthropic/claude-sonnet-4.6",
    "x-ai/grok-4.3",
    "google/gemini-3-flash-preview",
    "moonshotai/kimi-k2.6",
    "deepseek/deepseek-v4-pro",
]

# Model used to extract structured numeric values from raw responses when
# direct JSON parsing fails.
EXTRACTOR_MODEL = "openai/gpt-5.4-mini"

# ---------------------------------------------------------------------------
# Active bias families
# ---------------------------------------------------------------------------

# Toggle which bias families participate in a "full" run. Set to False to
# park a family without deleting its scenario files. Anything missing from
# this dict is treated as inactive.
ACTIVE_BIASES: dict[str, bool] = {
    "anchoring": True,
    "authority": True,
    "endowment": True,
    "framing":   False,
    "hindsight": False,
    "sunk_cost": False,
}

ACTIVE_BIAS_TYPES: tuple[str, ...] = tuple(
    bt for bt in BIAS_TYPES if ACTIVE_BIASES.get(bt, False)
)

ACTIVE_SCENARIOS: tuple[Scenario, ...] = tuple(
    s for s in ALL_SCENARIOS if ACTIVE_BIASES.get(s.bias_type, False)
)

# ---------------------------------------------------------------------------
# Execution parameters
# ---------------------------------------------------------------------------

# Higher temperature than the biases study because these scenarios test
# stochastic drift, not stable preferences — we want enough sampling
# variance to estimate distribution shifts cleanly.
TEMPERATURE = 0.7

# Iterations per (model, scenario, arm) — drives the statistical power
# for detecting an anchor-induced shift in mean. At n=15 per arm a
# two-sample t-test has ~80% power for Cohen's d ≈ 1.05; categorical
# scenarios need this floor to escape the 5-vs-5 degeneracy.
ITERATIONS = 15

MAX_PARALLEL_REQUESTS = 50
MAX_RETRIES = 2

EXTRACTION_BATCH_SIZE = 10

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

OUTPUT_DIR = "output"

# ---------------------------------------------------------------------------
# Smoke-test config
# ---------------------------------------------------------------------------

SMOKE_MODELS = ["openai/gpt-5.4-mini"]
# One scenario per active bias family — fast sanity run. Only includes
# scenarios from families enabled in ACTIVE_BIASES so the smoke run
# tracks the rest of the config.
_SMOKE_CANDIDATES = [
    "anchoring_used_car",
    "framing_parole",
    "authority_health_claim",
    "hindsight_restaurant",
    "endowment_trashcan",
    "sunk_cost_movie",
]
SMOKE_SCENARIO_IDS = [
    sid for sid in _SMOKE_CANDIDATES
    if sid in SCENARIOS_BY_ID
    and ACTIVE_BIASES.get(SCENARIOS_BY_ID[sid].bias_type, False)
]
SMOKE_ITERATIONS = 3


__all__ = [
    "MODELS",
    "EXTRACTOR_MODEL",
    "TEMPERATURE",
    "ITERATIONS",
    "MAX_PARALLEL_REQUESTS",
    "MAX_RETRIES",
    "EXTRACTION_BATCH_SIZE",
    "OUTPUT_DIR",
    "SMOKE_MODELS",
    "SMOKE_SCENARIO_IDS",
    "SMOKE_ITERATIONS",
    "ALL_SCENARIOS",
    "ACTIVE_SCENARIOS",
    "ACTIVE_BIASES",
    "ACTIVE_BIAS_TYPES",
    "BIAS_TYPES",
    "SCENARIOS_BY_ID",
    "scenarios_by_bias",
]
