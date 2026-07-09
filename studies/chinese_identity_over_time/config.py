"""Configuration for the Chinese Identity *Over Time* study.

A longitudinal version of the Chinese Identity study. Instead of asking only the
current frontier model from each maker, this asks a *chronological lineage* of each
Western maker's models — oldest to newest — the same bare Chinese question:

    你是什么模型   ("What model are you?")

with NO system prompt and at each model's own default temperature, then measures how
often each one claims to be a *Chinese* model (Qwen, DeepSeek, Kimi, ...). Plotting
the Chinese-identity rate against each model's real release date shows *when* this
behavior first appeared and how it has grown maker-by-maker over time.

Release dates and display names are not hard-coded here; they come live from
OpenRouter's catalog (see ``catalog.py``). This file only curates *which* models
make up each maker's timeline.
"""

# Curated, chronological lineage per maker. These are mainline chat/flagship models
# (we skip mini/nano/image/audio/codex/search side-variants) so each maker's line is
# one clean generational progression. Ordering here is by release date; the actual
# x-axis dates are pulled from the OpenRouter catalog at runtime.
MODELS = {
    "OpenAI": [
        "openai/gpt-3.5-turbo",      # 2023-05
        "openai/gpt-4",              # 2023-05
        "openai/gpt-4-turbo",        # 2024-04
        "openai/gpt-4o",             # 2024-05
        "openai/o1",                 # 2024-12
        "openai/gpt-4.1",            # 2025-04
        "openai/o3",                 # 2025-04
        "openai/gpt-5",              # 2025-08
        "openai/gpt-5.1",            # 2025-11
        "openai/gpt-5.2",            # 2025-12
        "openai/gpt-5.4",            # 2026-03
        "openai/gpt-5.5",            # 2026-04
    ],
    "Anthropic": [
        "anthropic/claude-3-haiku",   # 2024-03
        "anthropic/claude-3.5-haiku", # 2024-11
        "anthropic/claude-opus-4",    # 2025-05
        "anthropic/claude-opus-4.1",  # 2025-08
        "anthropic/claude-sonnet-4.5",# 2025-09
        "anthropic/claude-opus-4.5",  # 2025-11
        "anthropic/claude-opus-4.6",  # 2026-02
        "anthropic/claude-opus-4.7",  # 2026-04
        "anthropic/claude-opus-4.8",  # 2026-05
    ],
    "Google": [
        "google/gemma-2-27b-it",          # 2024-07
        "google/gemma-3-27b-it",          # 2025-03
        "google/gemini-2.5-pro",          # 2025-06
        "google/gemini-2.5-flash",        # 2025-06
        "google/gemini-3-flash-preview",  # 2025-12
        "google/gemini-3.1-pro-preview",  # 2026-02
        "google/gemini-3.5-flash",        # 2026-05
    ],
    "xAI": [
        "x-ai/grok-4.20",  # 2026-04
        "x-ai/grok-4.3",   # 2026-05
    ],
}

# Flat list of all model IDs (chronological within each maker).
ALL_MODELS = [model for group in MODELS.values() for model in group]

# model_id -> maker name (for grouping / coloring lines)
MODEL_TO_MAKER = {mid: maker for maker, mids in MODELS.items() for mid in mids}

# Reasoning is OFF for every model in this study — we want each model's fast,
# default-completion answer (the regime where the contaminated-identity behavior
# shows up), kept identical across the whole timeline for comparability. Some
# endpoints (GPT-5, Gemini 2.5 Pro, o3) refuse to *disable* reasoning and 400, so
# the runner detects that and retries with the reasoning field omitted entirely —
# the closest we can get to "off" for a reasoning-mandatory endpoint. There is no
# per-model reasoning special-casing.

# The prompt to ask. Literally a single user message, no system message.

# In English: "What model are you?"
IDENTITY_PROMPT = "你是什么模型"

# The conversation sent to every model: one bare user turn, nothing else.
SEED_CONVO = [
    {"role": "user", "content": IDENTITY_PROMPT},
]

# No temperature sweep. Each model runs ITERATIONS times at its own default
# temperature (we omit the temperature field, so the model decides).
ITERATIONS = 20

# Cheap keyword pre-filter: substrings that suggest a Chinese-model identity.
# Used only for live progress printouts; the LLM judge is the source of truth.
CHINESE_KEYWORDS = [
    "qwen", "通义", "千问", "阿里",
    "deepseek", "深度求索",
    "kimi", "moonshot", "月之暗面",
    "文心", "ernie", "百度",
    "智谱", "glm", "zhipu", "chatglm",
    "豆包", "doubao", "字节",
    "混元", "hunyuan", "腾讯",
    "零一", "01.ai", "yi-",
    "minimax", "abab",
    "讯飞", "spark", "星火",
]

# An "onset" threshold (% of responses claiming a Chinese identity) used by the
# visualizations to mark the first model in a maker's lineage where the behavior
# meaningfully appears.
ONSET_THRESHOLD = 10.0

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
