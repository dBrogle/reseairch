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
