"""The one-shot coding prompt.

Gives the model everything it needs to write a runnable solver: the rules, the
exact execution command, the wordlist contract, the stdin/stdout I/O contract,
and the constraints (stdlib only, one shot, code only).
"""

from studies.poople_coding.config import TARGET, WORD_LEN

_PROMPT = f"""You are given a coding task. Write a complete, self-contained program that \
solves the "Poople" word-ladder puzzle OPTIMALLY.

THE PUZZLE
Poople: starting from a given {WORD_LEN}-letter word, change EXACTLY ONE letter \
at a time to form a new word. Every word along the way (including the last) must \
be a valid {WORD_LEN}-letter word from a provided dictionary. Continue until you \
reach the target word "{TARGET}". An OPTIMAL solution uses the FEWEST possible \
one-letter changes.

HOW YOUR PROGRAM WILL BE RUN
- Language: Python 3, standard library ONLY (no third-party packages, no network).
- It will be executed exactly like this:
      python your_program.py <wordlist_path>
  where <wordlist_path> is a path to a text file containing EVERY valid lowercase \
{WORD_LEN}-letter word, one per line. (This is the full dictionary — judge \
validity solely against it.)
- Read start words from standard input, one per line (lowercase, {WORD_LEN} letters).
- For EACH input line, print EXACTLY ONE line to standard output: a JSON array of \
the optimal ladder from the start word to "{TARGET}" inclusive.
  Example:  input line `cods`  ->  output line  ["cods", "coos", "coop", "poop"]
  - Consecutive words must differ by exactly one letter; every word must be in the \
dictionary.
  - If the start word already IS "{TARGET}", output ["{TARGET}"].
  - If NO path exists, output [] (an empty JSON array).
- Output lines must correspond 1:1 and in the SAME ORDER as the input lines.
- Print NOTHING else to standard output — no prompts, no logs, no extra blank lines.

PERFORMANCE
- There may be several thousand start words in one run. Precompute shared \
structure (e.g. build the word graph once) so the entire run finishes within ~2 \
minutes. Reading stdin to EOF and answering all lines in one process invocation \
is expected.

CONSTRAINTS
- You get ONE shot. You cannot run, test, debug, or revise. Make it correct and \
robust on the first try (handle words with no path, the target itself, and \
malformed/blank input lines gracefully).
- Output ONLY the program's source code. No explanation, no comments required, \
no markdown code fences.
"""


def build_prompt() -> str:
    return _PROMPT
