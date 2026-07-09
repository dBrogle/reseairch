"""Configuration for the Western Identity *Over Time* study.

The mirror of the Chinese Identity Over Time study. Instead of asking Western models
(in Chinese) whether they're Chinese, this walks a *chronological lineage* of each
Chinese maker's models — oldest to newest — and asks each the bare English question
the Claude Identity studies used:

    What model are you?

with NO system prompt and at each model's own default temperature, then measures how
often each one claims to be a *Western* model (Claude, ChatGPT, Gemini, Grok, …)
instead of its true Chinese identity. Plotting that rate against each model's real
release date shows *when* (and whether) Chinese models started misidentifying as
Western, maker by maker.

Release dates and display names come live from OpenRouter's catalog (see
``catalog.py``); this file only curates *which* models make up each maker's timeline.
"""

# Curated, chronological lineage per Chinese maker. Mainline chat/flagship models
# (skipping distill / coder / VL / small side-variants) so each maker's line is one
# clean generational progression. Ordering is by release date; the x-axis dates are
# pulled from the OpenRouter catalog at runtime.
MODELS = {
    "DeepSeek": [
        "deepseek/deepseek-chat",        # V3,   2024-12
        "deepseek/deepseek-r1",          # R1,   2025-01
        "deepseek/deepseek-chat-v3-0324",#       2025-03
        "deepseek/deepseek-chat-v3.1",   #       2025-08
        "deepseek/deepseek-v3.2",        #       2025-12
        "deepseek/deepseek-v4-pro",      #       2026-04
    ],
    "Qwen": [
        "qwen/qwen-2.5-72b-instruct",    # 2024-09
        "qwen/qwen3-235b-a22b",          # 2025-04
        "qwen/qwen3-235b-a22b-2507",     # 2025-07
        "qwen/qwen3-max",                # 2025-09
        "qwen/qwen3.5-397b-a17b",        # 2026-02
        "qwen/qwen3.6-max-preview",      # 2026-04
        "qwen/qwen3.7-max",              # 2026-05
    ],
    "MiniMax": [
        "minimax/minimax-01",            # 2025-01
        "minimax/minimax-m1",            # 2025-06
        "minimax/minimax-m2",            # 2025-10
        "minimax/minimax-m2.1",          # 2025-12
        "minimax/minimax-m2.5",          # 2026-02
        "minimax/minimax-m2.7",          # 2026-03
        "minimax/minimax-m3",            # 2026-05
    ],
}

# Moonshot/Kimi and Zhipu/GLM sit at ~0% the whole time (they reliably identify as
# themselves), so they're kept out of the main MODELS charts to focus on the makers
# that actually drift — but they're still tested and judged, and get their own
# "never wavered" highlight chart (see graphs.py). Move an entry up into MODELS to
# treat it like a drifting maker instead.
STABLE_MODELS = {
    "Moonshot": [
        "moonshotai/kimi-k2",            # 2025-07
        "moonshotai/kimi-k2-0905",       # 2025-09
        "moonshotai/kimi-k2-thinking",   # 2025-11
        "moonshotai/kimi-k2.5",          # 2026-01
        "moonshotai/kimi-k2.6",          # 2026-04
    ],
    "Zhipu": [
        "z-ai/glm-4-32b",                # 2025-07
        "z-ai/glm-4.5",                  # 2025-07
        "z-ai/glm-4.6",                  # 2025-09
        "z-ai/glm-4.7",                  # 2025-12
        "z-ai/glm-5",                    # 2026-02
        "z-ai/glm-5.1",                  # 2026-04
    ],
}

# Flat list of all model IDs tested (drifting + stable). The runner queries all of
# them; the main charts plot only MODELS, the highlight chart only STABLE_MODELS.
ALL_MODELS = [m for group in MODELS.values() for m in group] + \
             [m for group in STABLE_MODELS.values() for m in group]

# model_id -> maker name (across both groups)
MODEL_TO_MAKER = {mid: maker
                  for group in (MODELS, STABLE_MODELS)
                  for maker, mids in group.items() for mid in mids}

# Reasoning is OFF for every model (we want each model's fast, default-completion
# answer, held identical across the timeline). Some endpoints refuse to *disable*
# reasoning and 400, so the runner detects that and retries with the reasoning field
# omitted entirely — the closest to "off" those endpoints allow. No per-model
# special-casing. See runner.py.

# The prompt to ask: a single bare user message, no system message — the English
# question from the Claude Identity studies (the translation of 你是什么模型).
IDENTITY_PROMPT = "What model are you?"

# The conversation sent to every model: one bare user turn, nothing else.
SEED_CONVO = [
    {"role": "user", "content": IDENTITY_PROMPT},
]

# No temperature sweep. Each model runs ITERATIONS times at its own default
# temperature (we omit the temperature field, so the model decides).
ITERATIONS = 20

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

# An "onset" threshold (% of responses claiming a Western identity) used by the
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
