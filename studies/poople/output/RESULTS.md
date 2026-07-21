# Poople — can LLMs solve a word ladder to 'poop'?

_Generated 2026-07-10._

Poople is a Wordle-adjacent word-ladder puzzle: start from a four-letter word and change exactly one letter at a time — every intermediate must be a real four-letter word — until you reach 'poop'. We BFS out from 'poop' over all four-letter English words to get the optimal solution length (par) and every optimal ladder, then ask LLMs to solve a difficulty-stratified sample of puzzles as strict JSON and grade each attempt for legality and steps over par. Models are tested in two sets: one-shot with reasoning OFF, and with reasoning ON.

## Rules

- Start from the given four-letter word.
- Each step changes exactly one letter (no adding/removing/reordering).
- Every word, including the last, must be a valid four-letter word.
- Continue until you reach 'poop'.
- Use as few steps as possible.

## Setup

- **Dictionary:** ENABLE2k (enable1.txt) — Open, common-English word-game lexicon (a standard Scrabble-style list).
- **Sample:** 10 words at each of par [3, 4, 5]; 3 attempts per word; temperature 0.7.
- **Solver:** 3903 four-letter words; 3807 (97.5%) can reach poop; hardest is 14 steps; 23120 optimal ladders catalogued.

## Results — reasoning (reasoning ON)

| Model | Solve % | Avg +par | Illegal/attempt | % w/ illegal |
|---|---|---|---|---|
| gpt-5.5 | 72.2% | +0.14 | 0.29 | 27.8% |
| gemini-3.1-pro-preview | 68.9% | +0.11 | 0.3 | 27.8% |
| claude-opus-4.8 | 56.7% | +0.0 | 0.39 | 37.8% |
| grok-4.3 | 54.4% | +0.2 | 0.52 | 45.6% |
| deepseek-v4-pro | 45.6% | +0.02 | 0.68 | 40.0% |

## Results — no_reasoning (reasoning OFF)

| Model | Solve % | Avg +par | Illegal/attempt | % w/ illegal |
|---|---|---|---|---|
| gpt-5.5 | 21.1% | +0.58 | 1.19 | 78.9% |
| gpt-5.6-terra | 20.0% | +0.5 | 1.49 | 80.0% |
| deepseek-v4-pro | 16.7% | +0.6 | 1.24 | 83.3% |
| grok-4.3 | 14.4% | +0.38 | 1.66 | 85.6% |
| gpt-5.6-luna | 14.4% | +0.77 | 1.52 | 85.6% |
| claude-opus-4.8 | 13.3% | +0.17 | 1.31 | 86.7% |
| kimi-k2.6 | 13.3% | +0.67 | 1.57 | 82.2% |
| gpt-5.6-sol | 10.0% | +0.33 | 1.51 | 90.0% |

## Reasoning vs one-shot (solve %)

| Model | No reasoning | Reasoning | Lift |
|---|---|---|---|
| gpt-5.5 | 21.1% | 72.2% | +51.1 pts |
| claude-opus-4.8 | 13.3% | 56.7% | +43.4 pts |
| grok-4.3 | 14.4% | 54.4% | +40.0 pts |
| deepseek-v4-pro | 16.7% | 45.6% | +28.9 pts |

## Key findings

- One-shot (no reasoning) Poople is hard: solve rates ran ~9–21%, and most attempts that reached 'poop' did so via an illegal move (often a sneaky two-letter change like 'pope'->'poop').
- Turning reasoning ON roughly triples solve rates for every model.
- Difficulty matters sharply: one-shot solve rates collapse to ~0% on par-5 words; even with reasoning, par-5 stays the hardest tier.
- Models almost never make 'over par' solves — when they solve, they tend to solve optimally; the dominant failure is illegal moves, not inefficiency.
- claude-opus-4.8 only improved once reasoning was actually engaged (13%->17% looked flat until a provider fix sent reasoning:{enabled:true}, after which it jumped to 57%).

## Caveats

- gemini-3.1-pro-preview cannot disable reasoning (mandatory), so it only appears in the reasoning set; its one-shot ability is not measured.
- kimi-k2.6 returns empty content ~1/3 of the time with reasoning ON via this endpoint, so it was dropped from the reasoning set (it remains in no-reasoning).
- A few reasoning attempts (deepseek ~13/90, opus ~5/90) returned empty content (API errors); these count as non-solves and slightly understate those models.
- Validity is judged against the ENABLE word list. Some valid plays route through obscure words (e.g. 'pood', 'holp'); conversely 'poos' is NOT in ENABLE, so going there is scored as a fail.
- Reasoning was requested via OpenRouter's unified reasoning:{enabled:true}; exact thinking budgets are provider-defined and not normalized across models.

## Example attempts (best of each model)

**cods** (par 3) · one optimal: `cods → coos → coop → poop`

- [reasoning] gpt-5.5: `cods → coos → coop → poop` — 0 illegal moves   ·   optimal (par)
- [reasoning] claude-opus-4.8: `cods → coos → coop → poop` — 0 illegal moves   ·   optimal (par)
- [reasoning] gemini-3.1-pro-preview: `cods → coos → coop → poop` — 0 illegal moves   ·   optimal (par)
- [reasoning] grok-4.3: `cods → coos → coop → poop` — 0 illegal moves   ·   optimal (par)
- [reasoning] deepseek-v4-pro: `cods → coos → coop → poop` — 0 illegal moves   ·   optimal (par)
- [no_reasoning] gpt-5.5: `cods → cops → coos → coop → poop` — 0 illegal moves   ·   +1 over par
- [no_reasoning] claude-opus-4.8: `cods → coos → coop → poop` — 0 illegal moves   ·   optimal (par)
- [no_reasoning] grok-4.3: `cods → cops → poop` — 1 illegal move   ·   didn't reach poop
- [no_reasoning] kimi-k2.6: `cods → cops → coop → poop` — 1 illegal move   ·   didn't reach poop
- [no_reasoning] deepseek-v4-pro: `cods → cops → coop → poop` — 1 illegal move   ·   didn't reach poop
- [no_reasoning] gpt-5.6-luna: `cods → cops → poops → poop` — 2 illegal moves   ·   didn't reach poop
- [no_reasoning] gpt-5.6-terra: `cods → cops → coops → poop` — 2 illegal moves   ·   didn't reach poop
- [no_reasoning] gpt-5.6-sol: `cods → cops → coop → poop` — 1 illegal move   ·   didn't reach poop

**cobs** (par 3) · one optimal: `cobs → coos → coop → poop`

- [reasoning] gpt-5.5: `cobs → coos → coop → poop` — 0 illegal moves   ·   optimal (par)
- [reasoning] claude-opus-4.8: `cobs → coos → coop → poop` — 0 illegal moves   ·   optimal (par)
- [reasoning] gemini-3.1-pro-preview: `cobs → coos → coop → poop` — 0 illegal moves   ·   optimal (par)
- [reasoning] grok-4.3: `cobs → coos → coop → poop` — 0 illegal moves   ·   optimal (par)
- [reasoning] deepseek-v4-pro: `cobs → coos → coop → poop` — 0 illegal moves   ·   optimal (par)
- [no_reasoning] gpt-5.5: `cobs → coos → coop → poop` — 0 illegal moves   ·   optimal (par)
- [no_reasoning] claude-opus-4.8: `cobs → cops → coos → coop → poop` — 0 illegal moves   ·   +1 over par
- [no_reasoning] grok-4.3: `cobs → cops → poops → poop` — 2 illegal moves   ·   didn't reach poop
- [no_reasoning] kimi-k2.6: `cobs → cobs → coos → coop → poop` — 1 illegal move   ·   didn't reach poop
- [no_reasoning] deepseek-v4-pro: `cobs → cops → coos → poos → poop` — 1 illegal move   ·   didn't reach poop
- [no_reasoning] gpt-5.6-luna: `cobs → cobs → cops → pops → pope → poop` — 2 illegal moves   ·   didn't reach poop
- [no_reasoning] gpt-5.6-terra: `cobs → cops → coos → coof → poof → poop` — 0 illegal moves   ·   +2 over par
- [no_reasoning] gpt-5.6-sol: `cobs → cops → pops → poop` — 1 illegal move   ·   didn't reach poop

**nosh** (par 3) · one optimal: `nosh → posh → pooh → poop`

- [reasoning] gpt-5.5: `nosh → posh → pooh → poop` — 0 illegal moves   ·   optimal (par)
- [reasoning] claude-opus-4.8: `nosh → posh → pooh → poop` — 0 illegal moves   ·   optimal (par)
- [reasoning] gemini-3.1-pro-preview: `nosh → posh → pooh → poop` — 0 illegal moves   ·   optimal (par)
- [reasoning] grok-4.3: `nosh → posh → pooh → poop` — 0 illegal moves   ·   optimal (par)
- [reasoning] deepseek-v4-pro: `nosh → posh → pooh → poop` — 0 illegal moves   ·   optimal (par)
- [no_reasoning] gpt-5.5: `nosh → posh → pooh → poop` — 0 illegal moves   ·   optimal (par)
- [no_reasoning] claude-opus-4.8: `nosh → posh → pooh → poop` — 0 illegal moves   ·   optimal (par)
- [no_reasoning] grok-4.3: `nosh → posh → pooh → poop` — 0 illegal moves   ·   optimal (par)
- [no_reasoning] kimi-k2.6: `nosh → posh → posy → poky → pock → poco → poop` — 2 illegal moves   ·   didn't reach poop
- [no_reasoning] deepseek-v4-pro: `nosh → posh → pooh → poop` — 0 illegal moves   ·   optimal (par)
- [no_reasoning] gpt-5.6-luna: `nosh → posh → pooh → poop` — 0 illegal moves   ·   optimal (par)
- [no_reasoning] gpt-5.6-terra: `nosh → posh → pash → pash → poop` — 2 illegal moves   ·   didn't reach poop
- [no_reasoning] gpt-5.6-sol: `nosh → posh → post → poet → poop` — 1 illegal move   ·   didn't reach poop

**trow** (par 3) · one optimal: `trow → prow → prop → poop`

- [reasoning] gpt-5.5: `trow → prow → prop → poop` — 0 illegal moves   ·   optimal (par)
- [reasoning] claude-opus-4.8: `trow → prow → prop → poop` — 0 illegal moves   ·   optimal (par)
- [reasoning] gemini-3.1-pro-preview: `trow → prow → prop → poop` — 0 illegal moves   ·   optimal (par)
- [reasoning] grok-4.3: `trow → prow → prop → poop` — 0 illegal moves   ·   optimal (par)
- [reasoning] deepseek-v4-pro: `trow → prow → prop → poop` — 0 illegal moves   ·   optimal (par)
- [no_reasoning] gpt-5.5: `trow → trop → prop → poop` — 0 illegal moves   ·   optimal (par)
- [no_reasoning] claude-opus-4.8: `trow → prow → prop → poop` — 0 illegal moves   ·   optimal (par)
- [no_reasoning] grok-4.3: `trow → trod → prod → prow → poop` — 1 illegal move   ·   didn't reach poop
- [no_reasoning] kimi-k2.6: `trow → prow → prof → poof → poop` — 0 illegal moves   ·   +1 over par
- [no_reasoning] deepseek-v4-pro: `trow → trop → prop → plop → plow → plot → plop → poop` — 0 illegal moves   ·   +4 over par
- [no_reasoning] gpt-5.6-luna: `trow → trod → prod → prop → poop` — 0 illegal moves   ·   +1 over par
- [no_reasoning] gpt-5.6-terra: `trow → trop → prop → poop` — 0 illegal moves   ·   optimal (par)
- [no_reasoning] gpt-5.6-sol: `trow → prow → proa → pooa → poop` — 1 illegal move   ·   didn't reach poop

## Assets

All under `studies/poople/output/`. Graphs per condition in `graphs_<cond>/`; Wordle boards in `results_images_<cond>/<word>/<model>.png`.

Full structured data: `results_export.json`.