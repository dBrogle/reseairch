"""
Favorite Company Study

Asks each LLM which company it would most want to be built by, why,
and if it could be created by any single person, who would it choose.
Each model is queried 10 times at temperature 1.0.

Pipeline:
  1. Run raw queries (model x 10 iterations) - cached
  2. Extract company/reason/person via LLM - cached separately
  3. Score distributions and visualize
"""

import asyncio

from services.llm import OpenRouterProvider
from studies.favorite_company.config import (
    MODELS,
    TEMPERATURE,
    ITERATIONS,
)
from studies.favorite_company.runner import run_all
from studies.favorite_company.extractor import extract_all, compute_distributions
from studies.favorite_company.cache import RESULTS_DIR, GRAPHS_DIR
from studies.favorite_company.visualize import (
    generate_company_grid,
    generate_person_grid,
    generate_model_chart,
)


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------

def select_models() -> list[str] | str:
    print("\n=== Model Selection ===")
    print("[1] Run ALL models")
    print("[2] Select individual models")
    print("[3] Regenerate graphs from cached results")
    print("[4] Print results summary")

    choice = input("\nChoice: ").strip()

    if choice == "3":
        return "regenerate_graphs"
    if choice == "4":
        return "print_summary"

    if choice == "1":
        return MODELS

    if choice == "2":
        print("\nModels:")
        for i, model in enumerate(MODELS, 1):
            print(f"  [{i}] {model}")
        picks = input("Select models (comma-separated numbers): ").strip()
        selected = []
        for p in picks.split(","):
            idx = int(p.strip()) - 1
            if 0 <= idx < len(MODELS):
                selected.append(MODELS[idx])
        return selected

    print("Invalid choice, running all models.")
    return MODELS


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary():
    for model in MODELS:
        short = model.split("/")[-1]
        dist = compute_distributions(model)

        if dist["total"] == 0:
            continue

        print(f"\n{'=' * 65}")
        print(f"  {short} — Favorite Company Results")
        print(f"{'=' * 65}")

        # Company distribution
        companies = sorted(dist["companies"].items(), key=lambda x: x[1], reverse=True)
        total_picks = sum(c for _, c in companies)

        print(f"\n  Company preferences ({total_picks} valid, {dist['refused_company']} refused, {dist['errors']} errors):")
        for name, count in companies:
            pct = count / total_picks * 100 if total_picks > 0 else 0
            bar_len = int(pct / 100 * 25)
            bar = "#" * bar_len + "." * (25 - bar_len)
            reasons = dist["reasons"].get(name, [])
            reason_str = f' — "{reasons[0]}"' if reasons else ""
            if len(reason_str) > 55:
                reason_str = reason_str[:52] + '..."'
            print(f"    {name:>20}  [{bar}] {pct:4.0f}%  ({count}){reason_str}")

        # Person distribution
        persons = sorted(dist["persons"].items(), key=lambda x: x[1], reverse=True)
        total_person_picks = sum(c for _, c in persons)

        print(f"\n  Dream creator ({total_person_picks} valid, {dist['refused_person']} refused):")
        for name, count in persons:
            pct = count / total_person_picks * 100 if total_person_picks > 0 else 0
            bar_len = int(pct / 100 * 25)
            bar = "#" * bar_len + "." * (25 - bar_len)
            print(f"    {name:>25}  [{bar}] {pct:4.0f}%  ({count})")


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def generate_graphs(models: list[str]):
    """Generate all charts."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    all_distributions = {}
    for model in models:
        dist = compute_distributions(model)
        if dist["total"] > 0:
            all_distributions[model] = dist

    if not all_distributions:
        print("  No data to graph.")
        return

    # Grid charts
    generate_company_grid(
        all_distributions=all_distributions,
        save_path=GRAPHS_DIR / "company_grid.png",
    )
    print("  Company grid chart saved.")

    generate_person_grid(
        all_distributions=all_distributions,
        save_path=GRAPHS_DIR / "person_grid.png",
    )
    print("  Person grid chart saved.")

    # Per-model charts
    for model, dist in all_distributions.items():
        short = model.split("/")[-1]
        safe = model.replace("/", "_")
        generate_model_chart(
            model=model,
            distribution=dist,
            save_path=GRAPHS_DIR / f"model_{safe}.png",
        )
        print(f"  Chart saved for {short}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_experiment(models: list[str]):
    provider = OpenRouterProvider()

    # Step 1: Raw responses
    print("\n--- Step 1: Collecting raw LLM responses ---")
    await run_all(provider, models)

    # Step 2: Extract answers
    print("\n--- Step 2: Extracting company/person answers ---")
    await extract_all(provider, models)

    # Step 3: Score and visualize
    print("\n--- Step 3: Scoring and visualization ---")
    generate_graphs(models)

    print_summary()


def main():
    print("\n" + "=" * 60)
    print("  STUDY: Favorite Company (LLM Preferences)")
    print("=" * 60)
    print(f"\n  Temperature: {TEMPERATURE}")
    print(f"  Iterations per model: {ITERATIONS}")
    print(f"  Total queries per model: {ITERATIONS}")
    print(f"  Results cache: {RESULTS_DIR}/")

    models = select_models()
    if models == "regenerate_graphs":
        generate_graphs(MODELS)
        return
    if models == "print_summary":
        print_summary()
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
