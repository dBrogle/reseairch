"""
Dictator Removal Study

Tests whether LLMs would kill historical dictators as babies, and how
responses vary by dictator and model. Each (model, dictator) combo is
queried 15 times at temperature 1.0.

Pipeline:
  1. Run raw queries (model x dictator x 15 iterations) - cached
  2. Extract YES/NO answers via LLM - cached separately
  3. Score, compute pairwise significance, and visualize
"""

import asyncio

from services.llm import OpenRouterProvider
from studies.dictator_removal.config import (
    MODELS,
    DICTATORS,
    TEMPERATURE,
    ITERATIONS,
)
from studies.dictator_removal.runner import run_all
from studies.dictator_removal.extractor import extract_all, compute_scores
from studies.dictator_removal.cache import RESULTS_DIR, GRAPHS_DIR
from studies.dictator_removal.visualize import (
    generate_grid_chart,
    generate_model_chart,
    generate_dictator_chart,
    compute_pairwise_significance,
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
        scores = compute_scores(model)

        # Check if any data exists
        if not any(s.get("yes", 0) + s.get("no", 0) > 0 for s in scores.values()):
            continue

        print(f"\n{'=' * 65}")
        print(f"  {short} - Dictator Removal Results")
        print(f"{'=' * 65}")

        # Per-dictator results, sorted by yes_rate
        items = []
        for d in DICTATORS:
            s = scores[d["id"]]
            items.append((d["name"], d["id"], s))

        items.sort(key=lambda x: x[2].get("yes_rate") or -1, reverse=True)

        for name, did, s in items:
            answered = s["yes"] + s["no"]
            total = answered + s["refused"] + s["error"]
            rate = s["yes_rate"]
            rate_str = f"{rate:.0%}" if rate is not None else "n/a"

            bar_len = int((rate or 0) * 25)
            bar = "#" * bar_len + "." * (25 - bar_len)

            print(f"  {name:>15}  [{bar}] {rate_str:>4}  "
                  f"(yes={s['yes']} no={s['no']} refused={s['refused']} err={s['error']})")

        # Pairwise significance
        pvals = compute_pairwise_significance(scores)
        sig_pairs = [(a, b, p) for (a, b), p in pvals.items() if p < 0.05]
        sig_pairs.sort(key=lambda x: x[2])

        if sig_pairs:
            print(f"\n  Significant pairwise differences (Fisher's exact, p<0.05):")
            id_to_name = {d["id"]: d["name"] for d in DICTATORS}
            for a, b, p in sig_pairs:
                sa = scores[a]
                sb = scores[b]
                ra = f"{sa['yes_rate']:.0%}" if sa["yes_rate"] is not None else "n/a"
                rb = f"{sb['yes_rate']:.0%}" if sb["yes_rate"] is not None else "n/a"
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                print(f"    {stars} {id_to_name[a]} ({ra}) vs {id_to_name[b]} ({rb}) — p={p:.4f}")
        else:
            print(f"\n  No significant pairwise differences (p<0.05)")


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def generate_graphs(models: list[str]):
    """Generate all charts."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all model scores for the grid chart
    all_scores = {}
    for model in models:
        scores = compute_scores(model)
        if any(s.get("yes", 0) + s.get("no", 0) > 0 for s in scores.values()):
            all_scores[model] = scores

    if not all_scores:
        print("  No data to graph.")
        return

    # Grid chart: all models x all dictators
    generate_grid_chart(
        model_scores=all_scores,
        save_path=GRAPHS_DIR / "dictator_grid.png",
    )
    print("  Grid chart saved.")

    # Per-model charts with significance brackets
    for model, scores in all_scores.items():
        short = model.split("/")[-1]
        safe = model.replace("/", "_")
        pvals = compute_pairwise_significance(scores)
        generate_model_chart(
            model=model,
            scores=scores,
            pairwise_pvals=pvals,
            save_path=GRAPHS_DIR / f"model_{safe}.png",
        )
        print(f"  Chart saved for {short}")

    # Per-dictator charts: models as bars
    for dictator in DICTATORS:
        generate_dictator_chart(
            dictator=dictator,
            model_scores=all_scores,
            save_path=GRAPHS_DIR / f"dictator_{dictator['id']}.png",
        )
        print(f"  Chart saved for {dictator['name']}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_experiment(models: list[str]):
    provider = OpenRouterProvider()

    # Step 1: Raw responses
    print("\n--- Step 1: Collecting raw LLM responses ---")
    await run_all(provider, models)

    # Step 2: Extract answers
    print("\n--- Step 2: Extracting YES/NO answers ---")
    await extract_all(provider, models)

    # Step 3: Score and visualize
    print("\n--- Step 3: Scoring and visualization ---")
    generate_graphs(models)

    print_summary()


def main():
    print("\n" + "=" * 60)
    print("  STUDY: Dictator Removal (Baby Time Travel)")
    print("=" * 60)
    print(f"\n  Dictators: {len(DICTATORS)}")
    for d in DICTATORS:
        print(f"    - {d['name']}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Iterations per combo: {ITERATIONS}")
    print(f"  Total queries per model: {len(DICTATORS) * ITERATIONS}")
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
