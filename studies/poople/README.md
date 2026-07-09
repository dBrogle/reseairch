# Poople

A Wordle-adjacent word-ladder puzzle: start from a four-letter word and change
**exactly one letter at a time** — every intermediate must itself be a valid
four-letter word — until you reach **`poop`**.

The study has two halves.

## Part 1 — The solver (done & verified)

Builds the full word-ladder graph over four-letter English words and BFS-es
outward from `poop` to compute, for every reachable word, the minimum number of
steps and **every** optimal (minimum-length) ladder.

- **Word list:** ENABLE2k (`enable1.txt`), filtered to the **3,903** valid
  four-letter words. Cached at `data/wordlists/enable1.txt`.
- **Graph:** an edge joins two words differing in exactly one position
  (built with wildcard buckets, e.g. `c_ld` → `cold`/`card`/…).
- **Oracle:** BFS distances from `poop`. A solution is *optimal* iff its length
  equals `dist[start_word]`.

### Results

- **3,807 / 3,903 (97.5%)** of words can reach `poop`; 96 are unreachable
  (rare-letter or isolated words like `aqua`, `hymn`, `envy`).
- **Max distance: 14 steps** (`unau`). Most words sit 4–6 steps out.
- **23,120** total optimal ladders; the most ambiguous word (`lave`, `rate`)
  has 91 distinct optimal solutions.

### Files

| File | Contents |
|------|----------|
| `output/solutions/distances.json` | `{word: min steps to poop}` for every reachable word — the grading oracle |
| `output/solutions/optimal_paths.json` | per word: `dist`, exact `num_optimal_paths`, and every optimal ladder |
| `output/solutions/unreachable.json` | valid words with no path to `poop` |

### Verification

`verify_saved()` independently reloads the word list and saved JSON and checks
that every saved ladder is a legal solution (one-letter steps, all valid words,
ends at `poop`, length `== dist`), that the distance map matches a fresh BFS,
and that path counts match a fresh DP count. **All 23,120 ladders pass.**

### Run

```bash
python -m studies.poople.main        # menu: build / verify / stats / lookup
```

Code: `wordlist.py` (load words), `solver.py` (graph, BFS, path counting,
enumeration, ladder validation), `main.py` (build/verify/stats/lookup).

## Part 2 — The LLM test

Asks each model to solve a difficulty-stratified sample of puzzles and grades
every attempt against the oracle.

### Two sets: reasoning vs no-reasoning

The test runs as two **sets** (`config.CONDITIONS`), so we can measure the
think-vs-one-shot gap. Each model's `MODEL_REASONING` capability decides where it
appears: `both` (run in each set), `mandatory` (reasoning only, e.g. Gemini Pro),
or `no_reasoning_only` (e.g. Kimi K2.6, which returns empty content too often with
reasoning on). Outputs are split per set:
`results_<cond>/`, `graphs_<cond>/`, `results_images_<cond>/` where `<cond>` is
`reasoning` or `no_reasoning`.

> **Provider note:** `enable_reasoning=True` sends `reasoning:{enabled:true}` so
> reasoning is *actively* turned on. This matters for providers whose default is
> thinking-off (Anthropic) — otherwise they don't reason and look unchanged.

**Headline result:** reasoning roughly triples solve rates. No-reasoning (one
shot) lands ~9–21%; with reasoning, gpt-5.5 **72%**, gemini **69%**, claude-opus
**57%**, grok **54%**, deepseek **46%**.

### Prompt & conditions

- **One user message, no system prompt.** Reasoning is forced **off** and the
  prompt explicitly forbids any preamble/thinking — a pure one-shot answer.
  (`reasoning:{enabled:false}` only kills native reasoning *tokens*; the prompt
  constraint is what actually suppresses visible chain-of-thought.)
- The model must answer as strict JSON of one-letter changes:
  ```json
  {"start_word": "cool",
   "changes": [{"from": "cool", "to": "pool"}, {"from": "pool", "to": "poop"}]}
  ```
- `temperature` 0.7, `ITERATIONS` attempts per word (default 3).

### Test-word sampling (`sampling.py`)

`SAMPLE_PER_BUCKET` words are drawn from each difficulty bucket in
`SAMPLE_BUCKETS` (default **10 each at par 3 / 4 / 5**). Each bucket is shuffled
once with a fixed seed and a **prefix** is taken, so raising `SAMPLE_PER_BUCKET`
(10 → 20) keeps the original words — and their cached results — and only appends
new ones. Growing the test set never re-runs prior work.

### Grading (`grader.py`)

Each attempt is parsed (tolerating fences/prose) and walked from the start word:

- **Illegal move** = stated `from` doesn't match the current word, OR `to` isn't
  a valid word, OR `to` differs by ≠ 1 letter.
- **Solved** = ends exactly on `poop` with zero illegal moves.
- **Over par** (solved only) = `moves − dist[start]` (so `+0` is optimal).

### Recorded metrics & charts (`analysis.py`, `visualize.py`)

Per model and per difficulty bucket: legal **solve rate**, **average steps over
par** (solved only), **illegal moves per attempt** & % of attempts with any, and
a six-way outcome partition (optimal / over-par / reached-but-illegal / failed /
unparseable / error). Charts in `output/graphs/`:
`solve_rate`, `avg_over_par`, `illegal_per_attempt`, `outcomes_stacked`,
`heatmap_solve_rate`, `heatmap_over_par`, plus the headline **`outcomes_pie`**:
one pie per model over a 4-way split — **green** = par (optimal), **light green**
= solved over par, **red** = used an illegal move, **orange** = failed to reach
poop. (Any attempt with an illegal move is red even if it still reached poop;
unparseable/API errors fold into orange.)

### Wordle-style result boards (`result_images.py`)

`output/results_images/<word>/<model>.png` (model id with `/ . -` → `_`) renders
each model's best attempt at ~10 example words as a **Wordle board**:

- **Rows are ladder steps** — start word on top, each next word on the line below
  (a "GOAL" row of all-green `poop` tiles sits up top to anchor the colors).
- A tile is **green** when its letter matches `poop` in that position, so a solved
  board's final row is all green; gray otherwise.
- Each step is judged: a legal one-letter change gets a green **✓** and a subtle
  outline on the changed tile; an illegal step gets a red **✗**, red outlines on
  the offending tiles, and a dashed red frame + "not a word" when the word is
  invalid.

10 words are chosen from the front of each bucket's stable sample
(`RESULT_IMAGE_WORDS_PER_BUCKET`, default 4/3/3 across par 3/4/5).

### First-iteration results (6 models, 30 words, 3 attempts, reasoning off)

One-shot with no thinking is **hard**: solve rates ran 8–23%, and most attempts
*reached* `poop` but via an illegal move (a sneaky two-letter change like
`pope→poop`). Solve rate falls off a cliff with difficulty — **every model
scored 0% on par-5 words**. Claude Sonnet 4.6 led overall (23%); Gemini 3 Flash
was best on the easy par-3 set (43%).

### Run

```bash
python -m studies.poople.main
#   [5] run BOTH sets   [6] reasoning only   [7] no-reasoning only
#   [8] regenerate graphs + result images + summaries from cache
```

Results cache per model in `output/results_<cond>/`; bump `PROMPT_VERSION` to
invalidate. `MAX_TOKENS` is not part of the cache key — delete a model's JSON to
force a re-run.
