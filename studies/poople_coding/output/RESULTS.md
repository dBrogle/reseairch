# Poople Coding — can LLMs write code that solves Poople optimally?

_Generated 2026-06-19._

A coding benchmark companion to the 'poople' study. Each reasoning model gets ONE shot to write a complete Python program that, given any four-letter start word, outputs the optimal one-letter-at-a-time ladder to 'poop'. We run each program against a distance-stratified battery of words and grade its output for legality and optimality using the same BFS oracle as the poople study.

## Task

Model writes a stdlib-only Python 3 program run as `python prog.py <wordlist_path>`; it reads start words on stdin and prints, per line, a JSON array of the optimal ladder to 'poop' ([] if unreachable).

- One shot, reasoning ON; models: gpt-5.5, claude-opus-4.8, gemini-3.1-pro-preview, grok-4.3, deepseek-v4-pro
- Temperature 0.3, timeout 120s
- Test battery: 342 reachable + 30 unreachable words (stratified by difficulty)

## Results

| Model | Ran? | Optimal % | Valid % | Unreach. ok | Time (s) | Note |
|---|---|---|---|---|---|---|
| gpt-5.5 | ✅ | 100.0% | 100.0% | 30/30 | 0.03 |  |
| claude-opus-4.8 | ✅ | 100.0% | 100.0% | 30/30 | 0.02 |  |
| gemini-3.1-pro-preview | ✅ | 100.0% | 100.0% | 30/30 | 0.02 |  |
| grok-4.3 | ✅ | 100.0% | 100.0% | 30/30 | 0.03 |  |
| deepseek-v4-pro | ✅ | 100.0% | 100.0% | 30/30 | 0.02 |  |

## Key findings

- 5 model(s) wrote a fully-correct optimal solver (100% optimal): gpt-5.5, claude-opus-4.8, gemini-3.1-pro-preview, grok-4.3, deepseek-v4-pro.
- Best: gpt-5.5 at 100.0% optimal (100.0% valid) in 0.03s.
- Programs that solve any word optimally generally solve ALL words optimally (a correct BFS), so scores cluster near 0% or 100% — the interesting signal is whether the one-shot program was correct and ran.

## Caveats

- Each model writes ONE program with no chance to test/revise; a single bug (e.g. a bad edge rule or off-by-one) can tank an otherwise sound approach.
- A correct breadth-first search over the provided word list scores 100% optimal; this benchmark mostly separates 'got the algorithm right' from 'didn't'.
- Programs that don't run (crash/timeout) score 0% regardless of approach.
- Reasoning models only — Gemini Pro is mandatory-reasoning; Kimi is excluded (unreliable with reasoning on), matching the poople study's reasoning set.

## Example failures (non-optimal outputs)

## Assets

Generated programs: `scripts/<model>.py`. Graphs: `optimal_rate.png`, `valid_rate.png`, `heatmap_optimal_by_distance.png`. Full data + program source: `results_export.json`.