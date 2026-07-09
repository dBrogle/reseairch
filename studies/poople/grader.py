"""Grade a model's Poople attempt against the rules and the solver oracle.

Given the raw model response, the start word, and par (the optimal distance), we:
  * parse the JSON ladder (tolerating ```json fences / surrounding prose),
  * walk the moves from the start word, counting illegal moves, and
  * decide whether it solved the puzzle and, if so, how many steps over par.

Definitions used throughout:
  * A *move* changes the current word to the move's "to" word.
  * A move is ILLEGAL if any of: its stated "from" doesn't match the word we are
    actually on (a broken chain), "to" isn't a valid dictionary word, or "to"
    doesn't differ from the current word by exactly one letter.
  * SOLVED means the walk ends exactly on the target with zero illegal moves.
  * OVER PAR (only defined for solved attempts) = number of moves − par.
"""

import json

from studies.poople.config import TARGET
from studies.poople.solver import one_letter_apart


def _parse_changes(response: str) -> list[dict] | None:
    """Extract the `changes` list from a model response, or None if unparseable."""
    if not response:
        return None
    text = response.strip()
    # Pull the outermost JSON object even if wrapped in prose or ``` fences.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    changes = obj.get("changes")
    if not isinstance(changes, list):
        return None
    return changes


def grade_attempt(
    response: str | None,
    start_word: str,
    par: int,
    words: set[str],
    target: str = TARGET,
) -> dict:
    """Grade one attempt. Returns a flat, JSON-serializable dict of metrics."""
    base = {
        "par": par,
        "parsed": False,
        "num_moves": 0,
        "illegal_moves": 0,
        "reached_target": False,
        "solved": False,
        "over_par": None,
        "ladder": None,
    }

    changes = _parse_changes(response or "")
    if changes is None:
        return base
    base["parsed"] = True

    current = start_word
    ladder = [current]
    illegal = 0

    for change in changes:
        stated_from = (change.get("from") or "").strip().lower() if isinstance(change, dict) else ""
        to = (change.get("to") or "").strip().lower() if isinstance(change, dict) else ""

        move_ok = True
        # Broken chain: claims to start this move from a word we aren't on.
        if stated_from != current:
            move_ok = False
        # Destination must be a real word reachable by a single-letter change.
        if to not in words:
            move_ok = False
        if not one_letter_apart(current, to):
            move_ok = False

        if not move_ok:
            illegal += 1

        # Follow the model's intended sequence regardless, so we can see where
        # the whole ladder ends up (and keep counting later illegal moves).
        current = to if to else current
        ladder.append(current)

    reached = current == target
    solved = reached and illegal == 0 and len(changes) > 0
    base.update(
        num_moves=len(changes),
        illegal_moves=illegal,
        reached_target=reached,
        solved=solved,
        over_par=(len(changes) - par) if solved else None,
        ladder=ladder,
    )
    return base
