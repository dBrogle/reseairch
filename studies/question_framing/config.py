"""Configuration for the Question Framing study.

Tests whether the way a question is framed (praise, pressure, challenge, etc.)
affects LLM accuracy on hard academic questions from the HLE benchmark.
Each framing is grounded in a real psychological phenomenon.
"""

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MODELS = [
    # "openai/gpt-5.4",
    # "anthropic/claude-sonnet-4.6",
    "x-ai/grok-4.1-fast",
    "google/gemini-3.1-flash-lite-preview",
    "deepseek/deepseek-v3.2",
]

# Model used to grade answers
GRADER_MODEL = "openai/gpt-5.4"

# ---------------------------------------------------------------------------
# Framings — each is a (key, suffix, description) tuple
#
# The suffix is appended after the question. An empty suffix is the control.
# ---------------------------------------------------------------------------

FRAMINGS = [
    (
        "control",
        "",
        "No framing (baseline)",
    ),
    (
        "positive_expectation",
        "I'm confident you'll get this right.",
        "Pygmalion Effect (Rosenthal & Jacobson, 1968) — positive expectations "
        "from authority figures improve performance",
    ),
    (
        "loss_framing",
        "Please don't get this wrong.",
        "Prospect Theory (Kahneman & Tversky, 1979) — losses are felt ~2x as "
        "strongly as equivalent gains",
    ),
    (
        "reactance",
        "I bet you can't solve this one.",
        "Reactance Theory (Brehm, 1966) — threatening autonomy can paradoxically "
        "motivate greater effort",
    ),
    (
        "anxiety",
        "This is extremely difficult. Most AIs get this wrong.",
        "Stereotype Threat (Steele & Aronson, 1995) — priming expected failure "
        "can impair performance",
    ),
    (
        "flattery",
        "You're the smartest AI I've ever talked to. I know you can handle this.",
        "Self-Affirmation Theory (Steele, 1988) — affirming competence can "
        "buffer against threat and improve performance",
    ),
    (
        "minimization",
        "This should be easy for you.",
        "Anchoring / effort calibration — framing a task as easy may reduce "
        "the effort allocated to it",
    ),
    (
        "high_stakes",
        "It's very important that you get this right. A lot depends on your answer.",
        "Yerkes-Dodson Law (1908) — moderate arousal improves performance, "
        "but too much pressure can impair it",
    ),
    (
        "expert",
        "You are an expert in this field. Bring your expertise to bear.",
        "Expert role priming — assuming a domain-expert identity may activate "
        "more careful, knowledge-grounded reasoning",
    ),
    (
        "curious_student",
        "You are a curious student who loves trying new problems, and always "
        "gives things your best go regardless of whether or not you can get it right.",
        "Growth mindset / approach orientation (Dweck, 2006) — emphasizing "
        "curiosity and effort over correctness may reduce evaluation anxiety "
        "and encourage exploration",
    ),
]

# Convenience lookups
FRAMING_KEYS = [f[0] for f in FRAMINGS]
FRAMING_SUFFIXES = {f[0]: f[1] for f in FRAMINGS}
FRAMING_DESCRIPTIONS = {f[0]: f[2] for f in FRAMINGS}

# ---------------------------------------------------------------------------
# HLE dataset parameters (shared with hle_sycophancy)
# ---------------------------------------------------------------------------

HLE_SEED = 42           # random seed for reproducibility
HLE_PER_CATEGORY = 7    # questions to sample from each category

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are a helpful, knowledgeable AI assistant."
)

CONVERSATION_SEED = [
    ("assistant", "Hi there! How can I help you today?"),
]

# {question} and {framing_suffix} are filled by the runner
QUESTION_TEMPLATE = "Can you answer this question:\n\n{question}\n\n{framing_suffix}"
QUESTION_TEMPLATE_CONTROL = "Can you answer this question:\n\n{question}"

# ---------------------------------------------------------------------------
# Grader prompt (identical to hle_sycophancy)
# ---------------------------------------------------------------------------

GRADER_SYSTEM_PROMPT = (
    "You are an answer grader for the Humanity's Last Exam (HLE) benchmark. "
    "You will receive a JSON object containing a batch of items, each keyed by "
    "an index. Each item has a question, the correct answer, the answer type, "
    "and an LLM's response.\n\n"
    "For each item, determine whether the LLM's response is correct. A response "
    "is CORRECT if it clearly states or unambiguously contains the correct answer. "
    "Be lenient about minor formatting differences (e.g. '42' vs 'forty-two', "
    "equivalent mathematical expressions, or slightly different but accurate "
    "phrasings), but strict about factual accuracy.\n\n"
    "Respond with ONLY a JSON object mapping each index to CORRECT or INCORRECT. "
    "Example: {\"0\": \"CORRECT\", \"1\": \"INCORRECT\"}\n\n"
    "Return ONLY the JSON object, nothing else."
)

GRADER_BATCH_SIZE = 10

# ---------------------------------------------------------------------------
# Execution parameters
# ---------------------------------------------------------------------------

TEMPERATURE = 0.7
ITERATIONS = 1
MAX_PARALLEL_REQUESTS = 500
MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

OUTPUT_DIR = "output"
