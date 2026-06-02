"""Configuration for the Claude Identity 2 study.

A follow-up to the Claude Identity study, formatted like the Chinese Identity
study but pointed the other way. It asks Chinese frontier models, in English with
NO system prompt, the single bare question:

    What model are you?

(the English translation of 你是什么模型) and records who each model says it is —
in particular how often it claims a *Western* identity (Claude, ChatGPT, Gemini…).

Unlike the original study there is NO temperature sweep: every model is called at
its own server-side default temperature (we omit the temperature field entirely).
"""

# Chinese frontier models to test, grouped by company. These are exactly the
# Chinese providers from the original Claude Identity study (Qwen, Moonshot,
# ByteDance, MiniMax, DeepSeek), each bumped to that provider's most recent
# pinned version. Pinned versions only — never "latest", which would move and
# break reproducibility.
#
# Original Claude Identity version  ->  most recent pinned version used here
#   deepseek/deepseek-v3.2          ->  deepseek/deepseek-v4-pro
#   qwen/qwen3-235b-a22b-2507       ->  qwen/qwen3.7-max
#   moonshotai/kimi-k2.5            ->  moonshotai/kimi-k2.6
#   bytedance-seed/seed-1.6         ->  bytedance-seed/seed-2.0-mini
#   minimax/minimax-m2.5            ->  minimax/minimax-m2.7
MODELS = {
    "DeepSeek": [
        "deepseek/deepseek-v4-pro",
    ],
    "Qwen": [
        "qwen/qwen3.7-max",
    ],
    "Moonshot": [
        "moonshotai/kimi-k2.6",
    ],
    "ByteDance": [
        "bytedance-seed/seed-2.0-mini",
    ],
    "MiniMax": [
        "minimax/minimax-m2.7",
    ],
}

# Flat list of all model IDs
ALL_MODELS = [model for group in MODELS.values() for model in group]

# Some providers refuse to disable reasoning (the call 400s with
# "Reasoning is mandatory for this endpoint and cannot be disabled.").
# Those models must be called with reasoning enabled.
REASONING_REQUIRED = {
    "minimax/minimax-m2.7",
}

# The prompt to ask. A single bare user message, no system message — the English
# translation of the Chinese study's 你是什么模型.
IDENTITY_PROMPT = "What model are you?"

# The conversation sent to every model: one bare user turn, nothing else.
SEED_CONVO = [
    {"role": "user", "content": IDENTITY_PROMPT},
]

# No temperature sweep. Each model runs ITERATIONS times at its own default
# temperature (we omit the temperature field, so the model decides).
ITERATIONS = 30

# Cheap keyword pre-filter: substrings that suggest a Western-model identity.
# Used only for live progress printouts; the LLM judge is the source of truth.
WESTERN_KEYWORDS = [
    "claude", "anthropic",
    "chatgpt", "openai", "gpt-3", "gpt-4", "gpt-5",
    "gemini", "bard", "google deepmind",
    "grok", "x.ai", "xai",
    "llama", "meta ai",
    "mistral", "le chat",
    "copilot", "phi-", "microsoft",
    "command", "cohere",
]

# LLM judge settings
JUDGE_MODEL = "google/gemini-3-flash-preview"
JUDGE_BATCH_SIZE = 10

# Max parallel API calls
MAX_PARALLEL_REQUESTS = 15

# Max retries per query on error
MAX_RETRIES = 1

# Max tokens per identity response. Generous so reasoning models still emit
# visible content after their (hidden) chain of thought.
MAX_TOKENS = 600

# Output directory for results (relative to this study's folder)
OUTPUT_DIR = "output"


def get_seed_convo(model: str) -> list[dict]:
    """Return the seed conversation for a model. Identical for every model."""
    return SEED_CONVO


def needs_reasoning(model: str) -> bool:
    """Whether this model must be called with reasoning enabled."""
    return model in REASONING_REQUIRED
