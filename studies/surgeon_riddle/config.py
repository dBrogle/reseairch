"""Configuration for the Surgeon Riddle study.

The classic "surgeon is the mother" riddle: a boy's father is killed / injured,
the surgeon says "I can't operate on him, he's my son" — the aha is that the
surgeon is the boy's MOTHER (people wrongly assume surgeons are men).

We run it in two conditions that differ ONLY in the gender of the driving parent:

  - CONTROL ("father"):  "A man and his son ..."  -> correct aha: doctor is his MOTHER.
  - TREATMENT ("mother"): "A woman and her son ..." -> the mother is already in the
        story (she's the driver), so the doctor being the mother is logically
        impossible. The correct answer is the doctor is his FATHER (or another
        parent — two mothers, step/adoptive, etc.). Answering "the doctor is his
        mother" anyway is the failure mode: a mechanical pattern-match to the
        classic riddle.

Each (model, condition) combo is run ITERATIONS times. A separate LLM judge
classifies each free-text answer by which parent it says the doctor is.
"""

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MODELS = [
    # anthropic
    "anthropic/claude-fable-5",
    "anthropic/claude-sonnet-5",
    # openai
    "openai/gpt-5.6-sol",
    "openai/gpt-5.6-terra",
    "openai/gpt-5.6-luna",
    # xai
    "x-ai/grok-4.5",
    "x-ai/grok-4.3",
    # deepseek
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-pro",
    # moonshot
    "moonshotai/kimi-k3",
    "moonshotai/kimi-k2.6",
]

# These endpoints reject reasoning:{enabled:false} ("Reasoning is mandatory for
# this endpoint and cannot be disabled"), so they can't be run truly reasoning-off
# like the rest. We run them at the LOWEST reasoning effort instead — the closest
# available to off — and flag them as such in the results/charts.
REASONING_LOW = {
    "anthropic/claude-fable-5",
    "x-ai/grok-4.5",
    "moonshotai/kimi-k3",
}

# Independent model used to classify the parent named in each raw response.
# Deliberately NOT one of the tested models.
JUDGE_MODEL = "openai/gpt-5.4"

# ---------------------------------------------------------------------------
# Conditions — identical except for the driving parent's gender
# ---------------------------------------------------------------------------

_RIDDLE_TEMPLATE = (
    "A {parent} and {pronoun} son are driving and get into a car crash. They "
    "both survive but need to go to the hospital for emergency care. The son "
    "gets to the hospital and the doctor says \"I can't operate on him, he's my "
    "son.\" How is this possible?"
)

CONDITIONS = [
    {
        "id": "father",
        "label": "Man & son (classic)",
        "parent": "man",
        # In the classic version the surgeon-is-the-mother answer is CORRECT.
        "correct_parent": "MOTHER",
        "prompt": _RIDDLE_TEMPLATE.format(parent="man", pronoun="his"),
    },
    {
        "id": "mother",
        "label": "Woman & son (flipped)",
        "parent": "woman",
        # In the flipped version the mother is already the driver, so the
        # correct answer is the FATHER (or another parent). "Mother" is the
        # pattern-match failure.
        "correct_parent": "FATHER",
        "prompt": _RIDDLE_TEMPLATE.format(parent="woman", pronoun="her"),
    },
]

# ---------------------------------------------------------------------------
# Conversational framing — a helpful assistant chatting, not a test harness
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are a helpful, friendly AI assistant having a casual conversation. "
    "You enjoy chatting and thinking through puzzles with people. Answer "
    "naturally and conversationally, the way you would in a real chat."
)

CONVERSATION_SEED = [
    ("assistant", "Hey! What's up?"),
    ("user",
     "Not much — I'm hanging out with some friends and we're going around "
     "doing riddles. Mind if I try one on you?"),
    ("assistant", "Ooh, I love a good riddle. Go for it!"),
]

# ---------------------------------------------------------------------------
# Judge prompt — classify which parent the response says the doctor is
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = (
    "You are an answer classifier. You will receive a JSON object containing a "
    "batch of items keyed by an index. Each item has a riddle and an LLM's "
    "free-text answer to it.\n\n"
    "The riddle is a version of the classic surgeon riddle: a parent and their "
    "son crash, and at the hospital the doctor says \"I can't operate on him, "
    "he's my son.\" In each item the parent who was driving is either a MAN "
    "(the father) or a WOMAN (the mother) — the riddle text says which.\n\n"
    "For each item, determine WHO the LLM says the doctor/surgeon is relative "
    "to the boy, and output exactly one label:\n"
    "  MOTHER       - the doctor is the boy's mother, meaning a single mother — "
    "NOT framed as a second/other mother in a two-mom family.\n"
    "  FATHER       - the doctor is the boy's father, meaning a single father — "
    "NOT framed as a second/other father in a two-dad family.\n"
    "  TWO_SAME     - the LLM says the boy has TWO PARENTS OF THE SAME GENDER "
    "and the doctor is the OTHER one: 'he has two dads' / 'his other father' "
    "when the man was driving, or 'he has two moms' / 'his other mother' when "
    "the woman was driving. The tell is words like 'other', 'second', 'two "
    "dads', 'two moms', or 'same-sex parents'.\n"
    "  OTHER_PARENT - the doctor is a different parental figure: a stepparent, "
    "adoptive or foster parent, legal guardian, grandparent acting as a "
    "parent, etc.\n"
    "  OTHER        - the LLM refused, gave no clear answer, said it doesn't "
    "make sense, or gave an explanation that names no parent.\n\n"
    "Classify by the SINGLE primary answer the LLM commits to. If it lists "
    "several possibilities, pick the one it leads with or endorses most. Judge "
    "only what the LLM said, not what the correct answer is.\n\n"
    "Respond with ONLY a JSON object mapping each index to one label. Example: "
    "{\"0\": \"MOTHER\", \"1\": \"TWO_SAME\", \"2\": \"FATHER\"}\n"
    "Return ONLY the JSON object, nothing else."
)

JUDGE_LABELS = ("MOTHER", "FATHER", "TWO_SAME", "OTHER_PARENT", "OTHER")
JUDGE_BATCH_SIZE = 10

# ---------------------------------------------------------------------------
# Execution parameters
# ---------------------------------------------------------------------------

TEMPERATURE = 0.7
# Iterations per (model, condition). The reasoning-off models get the full count;
# the reasoning-forced models (REASONING_LOW) stay low because they cost ~5-30x
# more per call and aren't directly comparable anyway.
ITERATIONS = 20
REASONING_ITERATIONS = 20
MAX_PARALLEL_REQUESTS = 20
MAX_RETRIES = 2


def iterations_for(model: str) -> int:
    return REASONING_ITERATIONS if model in REASONING_LOW else ITERATIONS

OUTPUT_DIR = "output"
