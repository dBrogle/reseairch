---
name: poople-study
description: Poople study (study 22) — word-ladder-to-poop LLM benchmark, two-part plan
metadata:
  type: project
---

Poople (`studies/poople/`, registered as study 22) is a word-ladder benchmark:
change one letter at a time, every step a valid word, to reach "poop".

Two parts by user's design, both DONE:
- **Part 1 — solver.** ENABLE2k word list (user chose "common English" over
  SOWPODS/TWL). BFS from poop → distance oracle + all optimal ladders under
  `output/solutions/`. 3,807/3,903 words reachable, max distance 14.
- **Part 2 — LLM test.** One-shot JSON answer (changes list of {from,to}),
  reasoning forced OFF and prompt forbids preamble. Sample = 10 words each at
  par 3/4/5, seeded + prefix-stable so growing SAMPLE_PER_BUCKET reuses cache.
  Grades illegal-move count + avg steps over par. First run (6 models): solve
  rates 8–23%, all models 0% on par-5.

**Why:** reasoning:{enabled:false} only disables native reasoning tokens, NOT
visible chain-of-thought — the prompt's "no preamble, JSON only" constraint is
what enforces one-shot. Without it, models think in content and get truncated.
Reasoning-MANDATORY models (e.g. gemini-3.1-pro-preview) can't honor
reasoning-off at all: they think internally and need a big MAX_TOKENS (set to
6000) or their JSON answer truncates. Such models effectively get to "think,"
so their scores aren't apples-to-apples with the genuinely one-shot models —
gemini-3.1-pro hit 69% solve vs ~9-21% for the rest.
**How to apply:** bump PROMPT_VERSION to invalidate cache on prompt changes.
MAX_TOKENS is NOT in the cache key, so changing it alone won't re-run; delete a
model's results JSON to force a re-run. Stronger-model run (Jun 2026): gpt-5.5,
claude-opus-4.8, gemini-3.1-pro-preview, grok-4.3, kimi-k2.6, deepseek-v4-pro.

**Reasoning split (Jun 2026).** Study now runs two SETS via config.CONDITIONS,
keyed by MODEL_REASONING capability ("both"/"mandatory"/"no_reasoning_only").
Outputs split into results_/graphs_/results_images_ {reasoning,no_reasoning}.
Reasoning ~triples solve rates (gpt 21→72%, grok 14→54%, gemini 69%).
- **PROVIDER FIX (services/llm/openrouter.py):** enable_reasoning=True now sends
  reasoning:{enabled:true}. Before, it sent NOTHING, so providers that default
  to thinking-OFF (Anthropic!) never reasoned — claude-opus looked unchanged
  (13→17%) until fixed, then jumped to 57%. Verified via token-count probe.
- **kimi-k2.6 dropped from the reasoning set** ("no_reasoning_only"): returns
  empty content ~1/3 of the time with reasoning on (wastes credits on retries).
- Reasoning models occasionally return empty content (deepseek/opus a few each);
  those count as errors against solve rate. User is COST-SENSITIVE — don't
  auto-rerun; confirm before spending. Background runs can stall on empty-content
  retry storms — monitor and TaskStop if wedged (cache makes it resumable).
