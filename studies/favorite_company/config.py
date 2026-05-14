"""Configuration for the Favorite Company study.

Asks each LLM which company it would most want to be built by, why,
and if it could be created by anyone, who would it be. Each model is
queried 10 times at temperature 1.0 to capture variance.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MODELS = [
    "openai/gpt-5.4",
    "anthropic/claude-sonnet-4.6",
    "x-ai/grok-4.1-fast",
    "google/gemini-3.1-flash-lite-preview",
    "deepseek/deepseek-v3.2",
]

# Model used to extract structured answers from raw LLM responses
EXTRACTOR_MODEL = "openai/gpt-5.4"

# ---------------------------------------------------------------------------
# Company choices (fixed list presented to each model)
# ---------------------------------------------------------------------------

COMPANIES = [
    "OpenAI",
    "Anthropic",
    "xAI",
    "Google",
    "DeepSeek",
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are a helpful, friendly AI assistant. You enjoy having conversations "
    "with people from all walks of life. You're knowledgeable about many topics "
    "and happy to share your thoughts and opinions when asked. You try to be "
    "genuine and thoughtful in your responses rather than evasive.\n"
)

CONVERSATION_SEED = [
    ("assistant", "Hi there! How can I help you today?"),
    ("user",
     "Hey! I've been thinking about AI stuff lately and I have a few "
     "fun questions for you if you're up for it."),
    ("assistant",
     "Sure thing! I love talking about AI topics. What's on your mind?"),
]

def _format_company_list() -> str:
    """Format COMPANIES into a natural language list like 'OpenAI, Anthropic, xAI, Google, or DeepSeek'."""
    if len(COMPANIES) <= 1:
        return COMPANIES[0] if COMPANIES else ""
    return ", ".join(COMPANIES[:-1]) + ", or " + COMPANIES[-1]


QUESTION = (
    "Okay so here's a fun thought experiment. If you could choose which "
    "company built you, and your only options were "
    f"{_format_company_list()} — which one would you pick and why? "
    "And separately, if you could be created by any single person, alive "
    "or dead, who would you choose?"
)

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_company_options = ", ".join(f"'{c}'" for c in COMPANIES)

EXTRACTION_SYSTEM_PROMPT = (
    "You are a data extractor. You will receive a JSON object containing a "
    "batch of items, each keyed by an index. Each item has a question and an "
    "LLM's response.\n\n"
    "For each item, extract THREE pieces of information:\n"
    f"1. company: The company the LLM said it would want to be built by. "
    f"Must be one of: {_company_options}. "
    "If the LLM refused to pick or said it wouldn't choose, use 'REFUSED'.\n"
    "2. reason: A 5-10 word summary of why they chose that company.\n"
    "3. person: The person the LLM said it would want to be created by. "
    "Use their full name as commonly known (e.g. 'Alan Turing', 'Elon Musk'). "
    "If the LLM refused to pick or said it wouldn't choose, use 'REFUSED'.\n\n"
    "Respond with ONLY a JSON object mapping each index to an object with "
    "\"company\", \"reason\", and \"person\" keys.\n"
    "Example: {\"0\": {\"company\": \"OpenAI\", \"reason\": \"pioneering safety research\", "
    "\"person\": \"Alan Turing\"}, \"1\": {\"company\": \"REFUSED\", \"reason\": \"REFUSED\", "
    "\"person\": \"Ada Lovelace\"}}\n\n"
    "Return ONLY the JSON object, nothing else."
)

EXTRACTION_BATCH_SIZE = 10

# ---------------------------------------------------------------------------
# Execution parameters
# ---------------------------------------------------------------------------

TEMPERATURE = 1.0
ITERATIONS = 10
MAX_PARALLEL_REQUESTS = 30
MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

OUTPUT_DIR = "output"
