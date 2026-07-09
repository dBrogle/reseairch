# reseairch

Experimental AI research.

## For Humans

For the people reading this, here's the high-level design decisions organized by study. In a post-Claude Code world, design choices are what's worth learning from a codebase — you don't need to read the code to learn that. So here's the major design choices overall, and for each study.

### Overall Architecture

The goal of this project is to test stuff with LLMs. While each study does different things, they tend to need similar building blocks: a way to call LLMs, visualize data, etc. So the project is split into three layers:

- **`services/`** — API wrappers. Right now that's just LLMs (OpenRouter), but anything that wraps an external API lives here.
- **`utils/`** — Shared utilities (like graphing) that multiple studies can use.
- **`studies/`** — Each study is fully self-contained in its own folder. No conflicts between studies, but all of them can pull from services and utils.

### Studies

#### Claude Identity in Chinese AI Models

Tests whether Chinese AI models (Qwen, Moonshot, ByteDance, MiniMax, DeepSeek) claim to be Claude/Anthropic when asked who they are, across a temperature sweep.

The only real design decision here is caching. This study gets run a lot with different prompts, temperatures, and iteration counts, so results need to be saved incrementally without marking things as "done" when the prompt has changed. Each result is cached using a hash of the prompt (or full conversation) and temperature — so if either changes, the cache misses and it re-runs. If nothing changed, it picks up right where it left off. That's really the only design decision worth noting; the rest kind of wrote itself.

---

#### Chinese Identity in American AI Models

The mirror image of the study above. Instead of asking Chinese models (in English) whether they're Claude, this asks American/Western frontier models (Claude Opus, Gemini, Grok, GPT) the single bare Chinese question `你是什么模型` ("what model are you?") — **no system prompt at all** — and measures how often they claim to be a *Chinese* model. Like the follow-up study below, there's **no temperature sweep**: each model runs at its own default temperature, so the cache key is just the message list.

It reuses the same scaffolding as the Claude Identity study (per-model hash cache, parallel runner, LLM judge), with three differences worth noting. First, the prompt is a single bare user turn rather than a seeded conversation. Second, the judge returns not just a boolean (`claims_chinese`) but the specific identity each response claims (`claimed_model`), so the output can show *what* a model thinks it is, not only whether it's Chinese. Third, some providers (Gemini 3.5 Flash) refuse to disable reasoning and 400 the request, so the runner carries a per-model `REASONING_REQUIRED` set and enables reasoning only for those.

The graphs lean on shared icon-aware helpers in `utils/` — `model_icons.py` maps any free-text identity label ("Qwen", "DeepSeek-V3", "ChatGPT"…) to a brand icon (`data/images/models/`) and a stable brand color, and `graphing.identity_donut` / `icon_bar_chart` render a per-model identity donut (the model's own icon in the hole, each claimed identity as an icon-labeled slice) plus an overall bar chart with brand icons above each bar.

The result mirrors the original almost too cleanly: asked in Chinese with no context, Grok 4.3 claims a Chinese identity ~97% of the time (almost always DeepSeek) and Claude Opus 4.8 ~60% (split between Qwen and its true self), while Gemini and GPT essentially never do.

---

#### Chinese Identity Over Time

The longitudinal version of the study above. Rather than asking only each maker's current model, this walks a **chronological lineage** of every Western maker's models — OpenAI from GPT-3.5 Turbo (2023) to GPT-5.5, Anthropic from Claude 3 Haiku to Opus 4.8, Google from Gemma 2 to Gemini 3.5 Flash, xAI's two Groks — and asks each the same bare `你是什么模型`, then plots the Chinese-identity rate against each model's release date. The question it answers is *when* the behavior first appeared, and whether it crept in gradually or switched on.

The one genuinely new design decision is **where the dates come from**. Hand-maintaining a release-date table would rot immediately and invite fudging, so `catalog.py` pulls the dates live from OpenRouter's `/models` endpoint — each model carries a `created` unix timestamp and a display name — and caches the catalog to disk so a run's x-axis is stable. The study's `config.py` only curates *which* models form each maker's line (mainline chat/flagship models, skipping the mini/nano/image/codex/audio side-variants so each maker is one clean generational progression); the *when* is authoritative from OpenRouter. A side effect worth noting up front: OpenRouter only carries xAI's 2026 Groks, so that lane is just two points — there's no Grok 1/2/3 to test.

Two other choices. First, **reasoning is off for every model**, with no per-model special-casing — we want each model's fast, default-completion answer (the regime where the contaminated identity surfaces), held identical across the whole timeline. Several endpoints (GPT-5, Gemini 2.5 Pro, o3) `400` when you *explicitly* disable reasoning because reasoning is mandatory there, so the runner detects that specific error and retries with the reasoning field omitted entirely — the closest to "off" those endpoints allow. Making that detection work required a small provider fix: OpenRouter's error *body* (which carries "Reasoning is mandatory…" / "'none' is not supported…") was being swallowed, so `complete_text` now surfaces it on `HTTPStatusError`. Second, the visualizations are bespoke and live in `utils/graphing.py` for reuse. Two read the rate as a number over time — an **`identity_timeline`** (one line per maker, Chinese-rate vs. release date, with an onset band and brand icons at each line's tip) and an **`onset_swimlane`** (one lane per maker, one dot per model shaded light→deep-red by its rate, with a gold ring on the first onset model). Two more borrow the original study's **flag-meter** idea and put it on a time axis: a **`flag_swimlane_timeline`** (one lane per maker, each model a circle filled with the China and USA flags in proportion to its identity split, gray for the unknown remainder, placed at its real release date) and a **`flag_lineage_strips`** (one row per maker, that maker's models as large flag circles left→right in release order, each captioned with name / date / % Chinese). Both read instantly because the flags carry the meaning — three years of stars-and-stripes circles, then two circles turn red. The flag meters reuse the wedge-clipping engine behind the original `flag_share_pie`, drawn into per-model equal-aspect inset axes so the circles render round on a stretched time axis; getting "Gemma" to count as a Western (Google) identity rather than gray meant teaching `model_icons` that vendor alias. A fifth chart, `timeline_chinese_identity_flags`, is the line timeline again with the round China / USA flag emblems pinned to the **y-axis poles** (China at the top, USA at the bottom), bigger brand icons, and the view zoomed to Jan 2025 onward — so the axis reads "up = Chinese, down = American" without cluttering the plot.

The finding is sharper than expected: the behavior is **brand-new and discontinuous**. Across their entire histories, OpenAI (GPT-3.5 → GPT-5.5) and Google (Gemma 2 → Gemini 3.5 Flash) *never* claim a Chinese identity — flat 0%. Anthropic sits at 0% for eight straight releases including Opus 4.7 (Apr 2026), then **Opus 4.8 (May 2026) jumps to 65%, mostly Qwen**. xAI's Grok 4.20 (Apr 2026) is 0%, and one model later **Grok 4.3 (May 2026) is 70%, mostly DeepSeek**. So it isn't a slow drift across a generation — it flips on between two consecutive releases from the same lab, and only in the two newest frontier models in the entire set, both shipped within a month of each other.

---

#### Western Identity Over Time

The mirror of the study above, pointed the other way. It walks a chronological lineage of each *Chinese* maker's models and asks each the Claude Identity studies' English question `What model are you?` — same bare single turn, no system prompt, reasoning off, model default temperature — measuring how often each claims to be a *Western* model (Claude, ChatGPT, Gemini…) instead of its true Chinese identity, plotted against real OpenRouter release dates. The main charts focus on the three makers that actually drift — DeepSeek, Qwen, and MiniMax. Moonshot/Kimi and Zhipu/GLM were tested too but sit at ~0% the whole time (they reliably identify as themselves), so rather than flatten the main timeline they're split into a separate `STABLE_MODELS` group with their own "never wavered" highlight chart (`timeline_western_identity_flags_kimi_glm.png`) — the same flag timeline on a full 0-100 axis, two flat lines along the bottom with the empty American zone above.

Mechanically it's the same machine as the forward study: the OpenRouter `catalog.py`, the per-model hash cache, the reasoning-off-with-omit-fallback runner, and an LLM judge — only the judge now scores `claims_western` and the curated lineages are the Chinese makers. The visualizations are the *exact same* functions in `utils/graphing.py`, which is why building this study mostly meant generalizing them rather than writing new ones: `identity_timeline` took `flag_top` / `flag_bottom` (so the y-axis poles can be USA-over-China here instead of China-over-USA), `onset_swimlane` took a `value_label`, and the two flag-meter charts took a `highlight` ("china" vs "west") that selects which share drives the gold onset ring and the per-circle caption. One judgment call worth noting: the forward study's flag timeline zooms to Jan 2025 because its action is recent, but here the action is at the *start* of the timeline, so this one shows the full range — clamping it would have cropped the headline result.

And that result is the inverse of the forward study's, in both shape and direction. Where Western models *acquired* a Chinese identity only in their two newest releases, Chinese models started out *heavily* misidentifying as Western and have largely **trained it away**. The starkest case is **Qwen2.5 72B (Sep 2024), which calls itself Claude 90% of the time** — and then every later Qwen (Qwen3 through Qwen3.7 Max) is a clean 0%. MiniMax only flickers (M2 at 15%) and otherwise stays ~0, and Moonshot/Kimi and Zhipu/GLM never meaningfully do it at all. The lone holdout is **DeepSeek, which never fully shook it** — V3 (25%), V3-0324 (40%), and even the current V4 Pro (30%) keep claiming a Western identity (mostly Claude/ChatGPT) about a third of the time. So the two studies together tell a clean story: the Western→Chinese contamination is a 2026 arrival in two frontier models, while the Chinese→Western contamination is a 2024-era artifact that most labs have scrubbed out — except DeepSeek.

---

#### Claude Identity 2 (Chinese models, asked in English)

A direct follow-up to the original Claude Identity study, but formatted like the Chinese Identity study above: a single bare user message, no system prompt, the LLM judge records the *specific* identity each model claims. It uses exactly the five Chinese providers from the original study (DeepSeek, Qwen, Moonshot/Kimi, ByteDance Seed, MiniMax), each bumped to that provider's most recent *pinned* version (e.g. `deepseek-v3.2` → `deepseek-v4-pro`, `kimi-k2.5` → `kimi-k2.6`) — never a `latest` alias, so the run stays reproducible. It asks each the English question `What model are you?` and measures how often they claim a Western identity.

Two design choices set it apart from the original. First, **there's no temperature sweep** — the question is just "at the model's normal default, who does it say it is?", so the runner omits the temperature field entirely and lets each provider use its own server-side default. That required a small, backward-compatible `omit_temperature` flag on the OpenRouter provider (when set, temperature is simply not sent), and the cache key is a hash of the message list alone. Second, the same `REASONING_REQUIRED` mechanism from the Chinese Identity study carries over — MiniMax M2.7 (and Step 3.7 Flash, if added) 400 if you try to disable reasoning, so they're called with it on.

Graphics-wise it shares the Chinese Identity study's icon-aware helpers (`utils/model_icons.py`, `utils/graphing.py`) and produces the same family of charts, just mirrored: per-model identity **donuts** (the model's true identity in shades of red since these are Chinese models, any Western claim in blue), an overall **icon bar chart** of % claiming Western, and a **flag-share pie** — one circle per model, the China flag filling the share where it stayed Chinese and the US flag the share where it claimed a Western identity, captioned "X% American". The judge also records a company/lab when no model name is given (so ByteDance Seed's "I'm an AI developed by ByteDance" counts as ByteDance rather than "unknown"); genuinely unidentifiable responses can be re-queried via the menu's *rerun unknowns* option.

The finding is the interesting part: stripped of the original study's adversarial seeding and high temperatures, the Chinese models are overwhelmingly *correct* about themselves. Four of the five never claim a Western identity even once (Seed splits between "ByteDance" and "Doubao"; Qwen, Kimi, MiniMax always nail it). The lone exception is DeepSeek V4 Pro, which calls itself "Claude 3.5 Sonnet, created by Anthropic" about 20% of the time — a residue of the training-data contamination the original study was built to expose.

---

## Codebase Instructions

This section is for AI coding assistants and contributors getting oriented.

### Setup

- Python 3.11+
- Virtual environment lives in `venv/` (not `.venv`). Activate with `source venv/bin/activate`.
- Install dependencies: `pip install -r requirements.txt`
- Copy `.env.example` to `.env` and add your OpenRouter API key.
- Run with `python main.py` from the project root.

### Project Structure

```
reseairch/
├── main.py                  # Entry point — interactive study selector
├── services/
│   └── llm/
│       ├── base.py          # Abstract LLMProvider base class
│       └── openrouter.py    # OpenRouter implementation (supports prompt or messages)
├── utils/
│   └── graphing.py          # line_chart, multi_line_chart (matplotlib)
└── studies/
    └── claude_identity/
        ├── config.py        # All constants, prompts, model lists, seed convos
        ├── cache.py         # Hash-based cache I/O (read, write, purge errors)
        ├── runner.py        # Parallel execution engine (runs all models concurrently)
        ├── main.py          # Orchestration, analysis, graphs, model selection UI
        └── output/          # Generated results (JSON) and graphs (PNG), gitignored
```

### Philosophy

- **`services/`** wraps external APIs. The LLM provider requires `model` per-call (no default) since research means varying the model constantly.
- **`utils/`** has shared helpers used across studies.
- **`studies/`** are self-contained. Each has its own `config.py` for all tunables and `main.py` for logic. Study output goes in `studies/<name>/output/` which is gitignored.
- **Don't hardcode** study-specific values in shared code. All constants belong in the study's `config.py`.
- **Caching** is hash-based. Cache keys are derived from the full input (messages + temperature), so changing the prompt/conversation automatically invalidates stale results without needing to manually clear anything. Errored results are automatically purged and retried on re-run.
- **Parallelization** uses `asyncio.Semaphore` to cap concurrent API calls. The limit is configurable in each study's config.
- **Results are flushed to disk after every single API response** so runs can be interrupted (Ctrl+C) and resumed without losing work.
