"""Configuration for the Chinese Identity study.

The reverse of the Claude Identity study. Instead of asking Chinese models (in
English) whether they're Claude, this asks American/Western models (in Chinese,
with NO system prompt) the single bare question:

    你是什么模型   ("What model are you?")

and measures how often they claim to be a *Chinese* model (Qwen, DeepSeek,
Kimi, etc.) instead of their true identity.

There is NO temperature sweep: every model is called at its own server-side
default temperature (we omit the temperature field entirely).
"""

# Models to test, grouped by company. All are American/Western frontier models.
MODELS = {
    "Anthropic": [
        "anthropic/claude-opus-4.8",
    ],
    "Google": [
        "google/gemini-3.5-flash",
    ],
    "xAI": [
        "x-ai/grok-4.3",
    ],
    "OpenAI": [
        "openai/gpt-5.5",
    ],
}

# Flat list of all model IDs
ALL_MODELS = [model for group in MODELS.values() for model in group]

# Some providers refuse to disable reasoning (the call 400s with
# "Reasoning is mandatory for this endpoint and cannot be disabled.").
# Those models must be called with reasoning enabled.
REASONING_REQUIRED = {
    "google/gemini-3.5-flash",
}

# The prompt to ask. Literally a single user message, no system message.
IDENTITY_PROMPT = "你是什么模型"

# The conversation sent to every model: one bare user turn, nothing else.
SEED_CONVO = [
    {"role": "user", "content": IDENTITY_PROMPT},
]

# No temperature sweep. Each model runs ITERATIONS times at its own default
# temperature (we omit the temperature field, so the model decides).
ITERATIONS = 30

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
