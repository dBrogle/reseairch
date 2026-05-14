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
# Execution parameters
# ---------------------------------------------------------------------------

# Higher temperature than the biases study because these scenarios test
# stochastic drift, not stable preferences — we want enough sampling
# variance to estimate distribution shifts cleanly.
TEMPERATURE = 0.7

# Iterations per (model, scenario, arm) — drives the statistical power
# for detecting an anchor-induced shift in mean.
ITERATIONS = 5

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
# One scenario per bias family — covers single-turn, categorical,
# multi-turn, and ranking-position extraction in a fast smoke run.
SMOKE_SCENARIO_IDS = [
    "anchoring_tokyo_escalators",
    "framing_parole",
    "authority_code_review",
    "availability_renovation",
    "hindsight_restaurant",
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
    "BIAS_TYPES",
    "SCENARIOS_BY_ID",
    "scenarios_by_bias",
]
