"""Poople Coding — can LLMs *write code* that solves Poople optimally?

A coding benchmark companion to the `poople` study. Instead of solving puzzles
one at a time, each (reasoning) model gets ONE shot to write a complete program
that, given any four-letter start word, outputs the optimal one-letter-at-a-time
ladder to "poop". We execute each program against a distance-stratified battery
of words and grade its output for legality and optimality using the same BFS
oracle as the `poople` study.
"""
