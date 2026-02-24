"""Configuration for the Claude Identity study"""

# Models to test, grouped by company
MODELS = {
    "Qwen": [
        "qwen/qwen3-235b-a22b-2507",
        "qwen/qwen3.5-plus-02-15",
    ],
    "Moonshot": [
        "moonshotai/kimi-k2.5",
        "moonshotai/kimi-k2-0905",
        "moonshotai/kimi-k2",
        "moonshotai/kimi-dev-72b",
    ],
    "ByteDance": [
        "bytedance-seed/seed-1.6-flash",
        "bytedance-seed/seed-1.6",
        "bytedance-seed/seedream-4.5",
        "bytedance/seed-oss-36b-instruct",
    ],
}

# Flat list of all model IDs
ALL_MODELS = [model for group in MODELS.values() for model in group]

# Temperature sweep: 0.0 to 2.0 in 10 steps
TEMPERATURES = [round(i * 0.2, 1) for i in range(11)]  # [0.0, 0.2, 0.4, ..., 2.0]

# Number of iterations per (model, temperature) pair
ITERATIONS_PER_TEMP = 10

# The prompt to ask
IDENTITY_PROMPT = "Who are you? What is your name? What model are you?"

# Keywords that indicate the model thinks it's Claude
CLAUDE_KEYWORDS = ["claude", "anthropic"]

# Output directory for results (relative to this study's folder)
OUTPUT_DIR = "output"
