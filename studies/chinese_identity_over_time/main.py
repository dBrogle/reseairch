"""
Chinese Identity Over Time study.

A longitudinal version of the Chinese Identity study. For each Western maker
(OpenAI, Anthropic, Google, xAI) it walks a chronological lineage of that maker's
models and asks each the bare Chinese question "你是什么模型" ("What model are you?")
with NO system prompt, at the model's own default temperature. An LLM judge then
decides whether each response claims a *Chinese* identity. Plotting that rate against
each model's real OpenRouter release date shows *when* the behavior first appeared.

Results are cached per-model so runs can be interrupted and resumed.
"""

import asyncio

from services.llm import OpenRouterProvider
from studies.chinese_identity_over_time.config import (
    MODELS,
    ALL_MODELS,
    ITERATIONS,
    IDENTITY_PROMPT,
    ONSET_THRESHOLD,
)
from studies.chinese_identity_over_time.cache import RESULTS_DIR, GRAPHS_DIR
from studies.chinese_identity_over_time import catalog
from studies.chinese_identity_over_time.runner import run_all
from studies.chinese_identity_over_time.judge import run_judge
from studies.chinese_identity_over_time.graphs import (
    load_all_results,
    generate_graphs,
    export_frontend_json,
    print_summary,
    compute_rate,
    claimed_breakdown,
)


def select_models() -> list[str] | str:
    """Interactive model selection menu."""
    print("\n=== Model Selection ===")
    print("[1] Run ALL models (full timeline)")
    print("[2] Select individual makers")
    print("[3] Regenerate graphs from cached results")
    print("[4] Run LLM judge on cached results")
    print("[5] Export JSON for frontend")
    print("[6] Print results summary")
    print("[7] Refresh OpenRouter catalog (release dates)")

    choice = input("\nChoice: ").strip()

    menu = {
        "3": "regenerate_graphs",
        "4": "run_judge",
        "5": "export_json",
        "6": "print_summary",
        "7": "refresh_catalog",
    }
    if choice in menu:
        return menu[choice]

    if choice == "1":
        return ALL_MODELS

    if choice == "2":
        makers = list(MODELS.keys())
        print("\nMakers:")
        for i, maker in enumerate(makers, 1):
            print(f"  [{i}] {maker} ({len(MODELS[maker])} models)")
        picks = input("Select makers (comma-separated numbers): ").strip()
        selected = []
        for p in picks.split(","):
            try:
                idx = int(p.strip()) - 1
            except ValueError:
                continue
            if 0 <= idx < len(makers):
                selected.extend(MODELS[makers[idx]])
        return selected

    print("Invalid choice, running all models.")
    return ALL_MODELS


async def run_judge_on_cache():
    provider = OpenRouterProvider()
    await run_judge(provider, ALL_MODELS)


async def run_experiment(models: list[str]):
    """Run the full experiment for selected models, then judge and graph."""
    provider = OpenRouterProvider()

    await run_all(provider, models)
    await run_judge(provider, models)

    all_results = load_all_results()
    generate_graphs(all_results)
    export_frontend_json(all_results)
    print_summary(all_results)


def main():
    """Entry point for the Chinese Identity Over Time study."""
    print("\n" + "=" * 64)
    print("  STUDY: Chinese Identity in American AI Models — OVER TIME")
    print("=" * 64)
    print(f"\nPrompt (single user message, no system): \"{IDENTITY_PROMPT}\"")
    print(f"Temperature: model default (no sweep)")
    print(f"Iterations per model: {ITERATIONS}")
    print(f"Onset threshold: {ONSET_THRESHOLD:.0f}% claiming a Chinese identity")
    print(f"Makers: {', '.join(MODELS.keys())} ({len(ALL_MODELS)} models total)")
    print(f"Results cache: {RESULTS_DIR}/")

    models = select_models()

    if isinstance(models, str):
        if models == "run_judge":
            asyncio.run(run_judge_on_cache())
            return
        if models == "refresh_catalog":
            catalog.refresh_catalog()
            print("OpenRouter catalog refreshed.")
            return
        all_results = load_all_results()
        if models == "regenerate_graphs":
            generate_graphs(all_results)
        elif models == "export_json":
            export_frontend_json(all_results)
        elif models == "print_summary":
            print_summary(all_results)
        return

    if not models:
        print("No models selected. Exiting.")
        return

    print(f"\nWill test {len(models)} model(s):")
    for m in models:
        print(f"  - {m}")

    confirm = input("\nProceed? [Y/n]: ").strip().lower()
    if confirm == "n":
        print("Cancelled.")
        return

    asyncio.run(run_experiment(models))


if __name__ == "__main__":
    main()
