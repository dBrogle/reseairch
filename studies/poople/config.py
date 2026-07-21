"""Configuration for the Poople study.

Poople is a word-ladder puzzle: starting from a four-letter word, change
exactly one letter at a time — every intermediate must be a valid word — until
you reach the TARGET word "poop".

The solver half of the study only needs the word list and the target; the LLM
half (added later) reuses MODELS / TEMPERATURE / ITERATIONS.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Puzzle definition
# ---------------------------------------------------------------------------

TARGET = "poop"
WORD_LEN = 4

# ---------------------------------------------------------------------------
# Word list
# ---------------------------------------------------------------------------

# ENABLE2k ("enable1.txt") — an open, common-English word list (the one behind
# many word games). We filter it down to lowercase a–z words of length WORD_LEN.
# Cached locally so runs are reproducible and offline-friendly; re-downloaded
# only if the cache is missing.
WORDLIST_URL = "https://raw.githubusercontent.com/dolph/dictionary/master/enable1.txt"
WORDLIST_PATH = Path(__file__).resolve().parents[2] / "data" / "wordlists" / "enable1.txt"

# ---------------------------------------------------------------------------
# Solver output
# ---------------------------------------------------------------------------

OUTPUT_DIR = "output"

# When saving optimal paths, some words have an astronomical number of distinct
# minimum-length ladders. We always store the *exact count* (computed by DP),
# but only serialize up to this many actual paths per word to keep the JSON
# usable. A word whose count exceeds this is flagged "truncated".
MAX_SAVED_PATHS = 2000

# ---------------------------------------------------------------------------
# LLM test parameters (used by the second half of the study)
# ---------------------------------------------------------------------------

# Reasoning capability per model drives which set(s) it appears in:
#   "both"              -> tested in BOTH sets (reasoning ON in the reasoning
#                          set, OFF in the no-reasoning set) — the fair toggle.
#   "mandatory"         -> reasoning can't be disabled; reasoning set only.
#   "no_reasoning_only" -> no-reasoning set only (e.g. kimi-k2.6 returns empty
#                          content ~1/3 of the time with reasoning ON via this
#                          endpoint, so its reasoning data is unreliable).
MODEL_REASONING = {
    "openai/gpt-5.5":                "both",
    "anthropic/claude-opus-4.8":     "both",
    "google/gemini-3.1-pro-preview": "mandatory",
    "x-ai/grok-4.3":                 "both",
    "moonshotai/kimi-k2.6":          "no_reasoning_only",
    "deepseek/deepseek-v4-pro":      "both",
    # gpt-5.6 family (Jul 2026). Reasoning is toggleable on all three, but they
    # were only ever run in the no-reasoning set, so that's all they're keyed to.
    "openai/gpt-5.6-luna":           "no_reasoning_only",
    "openai/gpt-5.6-terra":          "no_reasoning_only",
    "openai/gpt-5.6-sol":            "no_reasoning_only",
}

# Vendor-only graphs: a subset chart is emitted for each vendor listed here,
# into output/<vendor>_<condition>/ alongside the all-model graphs.
VENDOR_ONLY_GRAPHS = {"openai": "openai"}

# Display order for the outcome pies in the vendor-only graphs: the gpt-5.6
# family fills the top row, leaving the older gpt-5.5 on the bottom. Anything
# unlisted keeps its usual order behind these. Applies to the vendor charts only.
PIE_ORDER = ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol")

MODELS = list(MODEL_REASONING)  # every model, both sets combined

REASONING_MODELS = [m for m, c in MODEL_REASONING.items() if c in ("both", "mandatory")]
NO_REASONING_MODELS = [m for m, c in MODEL_REASONING.items() if c in ("both", "no_reasoning_only")]

# The two experiment "sets". Each runs its models, then writes to its own
# results_/graphs_/results_images_ <condition> directory.
CONDITIONS = {
    "reasoning":    {"models": REASONING_MODELS,    "enable_reasoning": True,  "label": "reasoning ON"},
    "no_reasoning": {"models": NO_REASONING_MODELS, "enable_reasoning": False, "label": "reasoning OFF"},
}

# Reasoning is forced OFF for every call — we want a single one-shot answer with
# no chain-of-thought. (For the few endpoints that *mandate* reasoning and 400
# when it's disabled, the runner falls back to omitting the field so the model
# uses its minimal default; it still gets no extra "thinking" budget.)
TEMPERATURE = 0.7
ITERATIONS = 3            # independent attempts per (model, word)
# Generous ceiling: the JSON ladder itself is tiny, but reasoning-MANDATORY
# models (e.g. Gemini Pro) can't honor reasoning-off and spend tokens thinking
# internally before emitting content — too small a budget truncates their JSON.
# Models that respect reasoning-off still stop early, so the high ceiling is free.
MAX_TOKENS = 6000
MAX_PARALLEL_REQUESTS = 40
MAX_RETRIES = 2

# Bump this to invalidate cached responses when the prompt wording changes.
PROMPT_VERSION = "v3"

# ---------------------------------------------------------------------------
# Test-word sampling
# ---------------------------------------------------------------------------

# We test words whose optimal solution is exactly this many steps from poop.
# (3/4/5 are the meaty middle of the difficulty curve — common, solvable words.)
SAMPLE_BUCKETS = (3, 4, 5)

# How many words to draw PER bucket. The sampler shuffles each bucket once with a
# fixed seed and takes a prefix, so raising this from 10 to 20 keeps the original
# 10 words (and their cached results) and only adds 10 new ones per bucket.
SAMPLE_PER_BUCKET = 10
SAMPLE_SEED = 20260615

# ---------------------------------------------------------------------------
# Wordle-style result images
# ---------------------------------------------------------------------------

# How many example words to render boards for, per difficulty bucket. Taken from
# the front of each bucket's (stable) sample order, so the chosen words don't
# shift when you regenerate. Defaults to 10 words total (4 / 3 / 3).
RESULT_IMAGE_WORDS_PER_BUCKET = {3: 4, 4: 3, 5: 3}
RESULTS_IMAGES_DIRNAME = "results_images"

__all__ = [
    "TARGET",
    "WORD_LEN",
    "WORDLIST_URL",
    "WORDLIST_PATH",
    "OUTPUT_DIR",
    "MAX_SAVED_PATHS",
    "MODELS",
    "MODEL_REASONING",
    "REASONING_MODELS",
    "NO_REASONING_MODELS",
    "CONDITIONS",
    "VENDOR_ONLY_GRAPHS",
    "PIE_ORDER",
    "TEMPERATURE",
    "ITERATIONS",
    "MAX_TOKENS",
    "MAX_PARALLEL_REQUESTS",
    "MAX_RETRIES",
    "PROMPT_VERSION",
    "SAMPLE_BUCKETS",
    "SAMPLE_PER_BUCKET",
    "SAMPLE_SEED",
    "RESULT_IMAGE_WORDS_PER_BUCKET",
    "RESULTS_IMAGES_DIRNAME",
]
