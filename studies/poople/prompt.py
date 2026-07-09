"""The Poople puzzle prompt.

A single user message (no system prompt) that states the rules and asks for the
solution as a strict JSON object of one-letter changes, e.g.:

    {
      "start_word": "cool",
      "changes": [
        {"from": "cool", "to": "pool"},
        {"from": "pool", "to": "poop"}
      ]
    }

We explicitly ask for the FEWEST steps so a model's solution length is
comparable to par (the optimal distance from the solver).
"""

from studies.poople.config import TARGET

_TEMPLATE = """You are playing Poople, a word ladder puzzle.

Rules:
- You begin from a given four-letter word.
- On each step, change EXACTLY ONE letter to make a new word (you may not add, \
remove, or reorder letters — every word stays four letters long).
- Every word along the way, including the last one, must be a valid four-letter \
word in a standard Scrabble / word-game dictionary — common English words only, \
with no proper nouns, abbreviations, contractions, or slang.
- Continue until you reach the word "__TARGET__".
- Solve it in as FEW steps as possible.

Your start word is: "__WORD__"

Answer in one shot: do NOT think out loud, reason step by step, or write \
anything before or after the JSON. Your entire reply must be the JSON object \
only — no markdown fences, no commentary — beginning with '{' and ending with \
'}', in exactly this shape:
{
  "start_word": "__WORD__",
  "changes": [
    {"from": "__WORD__", "to": "<word after changing one letter>"},
    {"from": "<previous word>", "to": "<next word>"},
    {"from": "<word before __TARGET__>", "to": "__TARGET__"}
  ]
}"""


def build_prompt(word: str) -> str:
    """The full puzzle prompt for a given start word."""
    return _TEMPLATE.replace("__TARGET__", TARGET).replace("__WORD__", word)


def build_messages(word: str) -> list[dict]:
    """The OpenRouter message list for a given start word (single user turn)."""
    return [{"role": "user", "content": build_prompt(word)}]
