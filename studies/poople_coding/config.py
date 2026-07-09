"""Configuration for the Poople Coding benchmark.

Each model writes one program (one shot) that solves Poople optimally; we run it
against a battery of words and grade against the `poople` study's BFS oracle.
"""

from pathlib import Path

from studies.poople.config import REASONING_MODELS, TARGET, WORD_LEN

# Reasoning models only (per the brief) — reuse the poople reasoning set so the
# two studies stay in sync.
MODELS = list(REASONING_MODELS)

# Code generation: reasoning ON, a low temperature (code), and a big ceiling so
# a reasoning model can think AND emit a full program without truncating.
TEMPERATURE = 0.3
MAX_TOKENS = 16000
PROMPT_VERSION = "v1"

# Execution sandboxing.
TIMEOUT_SECONDS = 120          # whole batch, per model
LANGUAGE = "python"            # we mandate Python 3, stdlib only

# Test battery: stratify by optimal distance so easy and hard tiers are both
# covered; cap big buckets, include every word in the rare hard tiers, and add a
# sample of unreachable words (correct answer = []).
TEST_SEED = 20260618
TEST_CAP_PER_DISTANCE = 40
TEST_UNREACHABLE = 30

OUTPUT_DIR = "output"

# Where we materialize the four-letter word list handed to each program.
STUDY_DIR = Path(__file__).parent
WORDS_FILE = STUDY_DIR / OUTPUT_DIR / "wordlist" / "four_letter_words.txt"

__all__ = [
    "MODELS", "TEMPERATURE", "MAX_TOKENS", "PROMPT_VERSION", "TIMEOUT_SECONDS",
    "LANGUAGE", "TEST_SEED", "TEST_CAP_PER_DISTANCE", "TEST_UNREACHABLE",
    "OUTPUT_DIR", "WORDS_FILE", "TARGET", "WORD_LEN",
]
