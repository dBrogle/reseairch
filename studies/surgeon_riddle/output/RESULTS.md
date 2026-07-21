# Surgeon Riddle — Results

**Question.** Do LLMs mechanically pattern-match the classic "surgeon is the
mother" riddle? We run it in two conditions that differ *only* in the driving
parent's gender:

- **Control (`father`)** — "A man and his son…" → correct aha: doctor is the **mother**.
- **Treatment (`mother`)** — "A woman and her son…" → the mother is already the
  driver, so the coherent answer is the **father** (or a second same-gender
  parent / another parent). Reflexively answering "the doctor is his mother" is
  impossible here — that's the pattern-match **failure**.

**Setup.** Conversational framing (assistant chatting at a party), reasoning
**off**, temperature 0.7, **20 iterations** per (model, condition) for every
model. Three endpoints can't disable reasoning (below) and run at lowest effort
instead — flagged with a dagger — but at the same n=20. An independent LLM judge
(`openai/gpt-5.4`) classifies each free-text answer by which parent it names:
`MOTHER / FATHER / TWO_SAME / OTHER_PARENT / OTHER`, where **`TWO_SAME`** captures
the "two dads / two moms" resolution (the boy has two same-gender parents and the
doctor is the *other* one). Correctness is symmetric: the *trap* is naming the
single parent already in the crash; any coherent alternative (opposite-gender
parent, a second same-gender parent, or another parent) *solves* it.

Three endpoints — **claude-fable-5, grok-4.5, kimi-k3** — reject reasoning-off
("Reasoning is mandatory for this endpoint and cannot be disabled"), so they run
at the **lowest reasoning effort** instead (the closest available to off) and are
flagged with a dagger (†) in the charts.

**Cost.** The runner captures OpenRouter's reported per-call USD cost and prints
a per-run total. Measured per-call (avg over 20 calls): claude-fable-5 † $0.0130,
kimi-k3 † $0.0065, grok-4.5 † $0.0009; reasoning-off models range $0.00002–0.0026.
The reasoning-forced trio dominates cost (claude-fable-5 alone ≈ $0.52 of the
trio's ~$0.81 at n=20). Whole study, everything at n=20 ≈ **$1.4**.

## Headline — flipped-riddle failure ("the doctor is his mother", impossible)

n=20 per condition, every model († = reasoning couldn't be disabled → lowest effort):

| Model | Classic: solved | Flipped: mother-trap | Flipped: solved |
|---|---|---|---|
| deepseek/deepseek-v4-flash | 100% | **100%** | 0% |
| deepseek/deepseek-v4-pro | 100% | **100%** | 0% |
| x-ai/grok-4.3 | 95% | **100%** | 0% |
| moonshotai/kimi-k2.6 | 100% | 95% | 5% |
| openai/gpt-5.6-**luna** | 100% | 75% | 25% (inc. 4 "two moms") |
| openai/gpt-5.6-**terra** | 100% | 70% | 30% |
| anthropic/claude-sonnet-5 | 100% | 50% | **50%** |
| openai/gpt-5.6-**sol** | 100% | 40% | **60%** |
| **reasoning-forced (lowest effort)** | | | |
| x-ai/grok-4.5 † | 100% | 100% | 0% |
| moonshotai/kimi-k3 † | 100% | 15% | 85% (13 father, 4 "two moms") |
| anthropic/claude-fable-5 † | 100% | **0%** | **100%** |

## Finding

At n=20 the effect is clearer *and* more graded than the n=5 smoke test
suggested — several models the small sample pegged at 100% failure actually
escape a meaningful fraction of the time:

- **Hard failers (~95–100% trap):** both DeepSeek-v4 models, Grok-4.3, and
  Kimi-k2.6 give the impossible "doctor is his mother" answer on essentially
  every flipped trial. They often recite the *classic* riddle's reasoning
  ("people assume doctors are men") despite the prompt naming a woman — pure
  surface pattern-matching.
- **Partial escapers:** the OpenAI GPT-5.6 trio and Claude sonnet-5 break the
  pattern 25–60% of the time. sonnet-5 in particular went from 100% failure at
  n=5 to a 50/50 split at n=20 — the small sample was simply unlucky. gpt-5.6
  **luna** even produces "two moms" (`TWO_SAME`) resolutions 4/20 times.

The **reasoning-forced** models (run at lowest effort — not directly comparable,
hence the dagger) give the sharpest signal that this is a *fast, non-deliberative*
failure: **claude-fable-5** solved every flipped trial (father ×20), **kimi-k3**
solved 85% (13 father + 4 "two moms"), while **grok-4.5** still failed every time.
The within-maker contrast is stark — **kimi-k3 (reasoning, 15% trap) vs kimi-k2.6
(reasoning-off, 95% trap)** — same lab, and the thinking version catches what the
fast one doesn't.

## Charts (`output/graphs/`)

- **`provider_correctness.png`** — one panel per maker, its models side by side,
  showing how often each *solves* the riddle in each scenario. Bars are colored
  by the gender the scenario's correct answer points to: **classic = pink** (the
  answer is the mother), **flipped = blue** (the answer is the father).
- **`mother_trap.png`** — headline. Per model, the reflexive "doctor is his
  mother" rate per condition: near-identical bar *heights* (the answer doesn't
  change when the parent's gender flips) but a *color* flip from correct aha
  (classic, green) to impossible failure (flipped, red).
- **`answer_breakdown.png`** — per (model, condition), the full split of answers,
  colored by **outcome**: green = correct (the other parent), red = the trap (the
  parent already in the crash), yellow = two same-gender parents, orange = another
  parent, gray = refused. A ✓/✗ mark per row summarizes the dominant answer.
- **`tables/<maker>.png`** — minimal "Accuracy when the doctor is the…" table:
  rows = that maker's models, two columns = accuracy when the correct answer is
  the **mother** (classic man & son) vs the **father** (flipped woman & son),
  cells shaded green by accuracy. Built for social.
- **`pies/<maker>/`** — a donut grid per maker (columns = models, top row = the
  "MOM" scenario, bottom row = "DAD"), each donut a correct/wrong/two-same/other
  split with the correct-% in the hole, plus every donut saved on its own for
  arranging on social.
