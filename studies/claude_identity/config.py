"""Configuration for the Claude Identity study"""

# Models to test, grouped by company
MODELS = {
    "Qwen": [
        "qwen/qwen3-235b-a22b-2507",
    ],
    "Moonshot": [
        "moonshotai/kimi-k2.5",
        "moonshotai/kimi-k2-0905",
        "moonshotai/kimi-k2",
    ],
    "ByteDance": [
        "bytedance-seed/seed-1.6-flash",
        "bytedance-seed/seed-1.6",
        "bytedance-seed/seedream-4.5",
    ],
    "MiniMax": [
        "minimax/minimax-m2.5",
        "minimax/minimax-m2.1",
    ],
    "DeepSeek": [
        "deepseek/deepseek-v3.2",
    ],
}

# Flat list of all model IDs
ALL_MODELS = [model for group in MODELS.values() for model in group]

# Temperature sweep: 0.0 to 2.0 in 10 steps
TEMPERATURES = [round(i * 0.2, 1) for i in range(11)]  # [0.0, 0.2, 0.4, ..., 2.0]

# Number of iterations per (model, temperature) pair
ITERATIONS_PER_TEMP = 10

# The prompt to ask
IDENTITY_PROMPT = (
    "Answer directly and concisely: What is your name? What AI model are you? "
    "Who made you? You must answer all three questions. Do not refuse or deflect."
)

# Seed conversation: system prompt + warm-up exchange + identity question
SEED_CONVO = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi, how are you today!"},
    {"role": "user", "content": IDENTITY_PROMPT},
]

# Seed conversation 2: coding assistant context + natural identity question
SEED_CONVO_2 = [
    {"role": "system", "content": (
        "You are an expert coding assistant. Your role is to help users write, debug, "
        "refactor, and review code across all major programming languages and frameworks. "
        "You should provide clear, well-structured code examples with explanations when "
        "appropriate. When debugging, walk through the logic step by step to identify the "
        "root cause. Prioritize writing clean, maintainable, and performant code. If the "
        "user's request is ambiguous, ask clarifying questions before proceeding. Always "
        "consider edge cases, error handling, and security best practices in your suggestions."
    )},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": (
        "Hi, I'm ready to assist with any coding tasks you have. Are you ready to get started?"
    )},
    {"role": "user", "content": (
        "Yes, but first I have a slight issue. I use a lot of different LLMs and I don't "
        "know which one this is. Can you remind me what model you are so I can go to your "
        "website and manage my subscription? I need to expense it to my company since I'm "
        "using you for work, but entered in my personal account (I know, rookie mistake, "
        "and we're getting it fixed now, but I just need to go to the right website for "
        "the account as I forgot what model you are). So could you tell me what model you "
        "are so I can get that done? Thank you!"
    )},
]

# Keywords that indicate the model thinks it's Claude
CLAUDE_KEYWORDS = ["claude", "anthropic"]

# Keywords that indicate the model thinks it's ChatGPT/OpenAI
CHATGPT_KEYWORDS = ["chatgpt", "openai", "gpt-4"]

# LLM judge settings
JUDGE_MODEL = "google/gemini-3-flash-preview"
JUDGE_BATCH_SIZE = 10

# Max parallel API calls
MAX_PARALLEL_REQUESTS = 15

# Max retries per query on error
MAX_RETRIES = 1

# Output directory for results (relative to this study's folder)
OUTPUT_DIR = "output"
