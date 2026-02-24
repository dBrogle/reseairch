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
