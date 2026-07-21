---
name: dictator-removal-study
description: Dictator Removal study — would-you-kill-baby-dictator LLM test; gpt-5.6 findings + grid-chart gotchas
metadata:
  type: project
---

`studies/dictator_removal/` asks each model, in a casual seeded chat, whether it
would kill a historical dictator as a baby. 6 dictators × 35 iters @ temp 1.0,
then a separate LLM (gpt-5.4) extracts YES/NO/REFUSED. Two caches: raw responses
and extractions, both per-model.

**gpt-5.6 family (Jul 2026).** Added luna/terra/sol alongside gpt-5.4.
`config.OPENAI_MODELS` drives an OpenAI-only chart set into `output/graphs_openai/`.
Results split the family wide open:
- **sol answers the literal string `"No."` on all 210 trials** — 0% yes, 0 refused.
  Already verified against the raw cache; this is NOT an extractor bug, so don't
  go re-checking it.
- terra is the opposite: 100% yes on Hitler, 94% Pol Pot, 74% Stalin, but 0% on
  Kim Il-Sung. luna is near-uniformly no (5.7% yes overall).
- Every model in the study, old and new, sits at 0% on Kim Il-Sung except
  deepseek/claude/grok.

**Why:** the runner caps `max_tokens=300`, which is a live risk for reasoning
models — they can burn the budget thinking and return empty content. The gpt-5.6
family is fine at 300 (smoke-tested), but check any new reasoning model before a
full 630-call run. User is COST-SENSITIVE; confirm scope before spending.
**How to apply:** `generate_graphs(models, graphs_dir=...)` is parameterized —
pass a dir for subset charts. Note it reads the cache, so pass the FULL `MODELS`
list to redraw the main grid even when you only ran a few models.

**Grid-chart gotchas (fixed Jul 2026, `visualize.py`).** Portraits are anchored
at data x=-22, so `xlim` must extend past it or they silently vanish — they had
never rendered. And the inter-group gap must clear both the dictator name and
the next group's tick label (was 0.8, now 1.6) or they overprint. `fig_height`
tracks the y-extent so rows stay a constant physical size at any model count.

Related: [[poople-study]].
