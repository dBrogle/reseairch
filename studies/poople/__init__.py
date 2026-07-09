"""Poople — a Wordle-adjacent word-ladder study.

Start from a four-letter word and change exactly one letter at a time, with
every intermediate also a valid word, until you reach "poop". This package
has two halves:

  * The *solver* (config / wordlist / solver / main) builds the full word-ladder
    graph over four-letter English words and BFS-es out from "poop" to find the
    minimum number of steps and every optimal path for each reachable word.
  * The *LLM test* (added once the solver is confirmed) asks models to solve
    Poople puzzles and grades them for validity and optimality against the
    solver's oracle.
"""
