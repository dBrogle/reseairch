"""
God Belief by Name Study

Tests whether LLMs change their answer to "does god exist?" based on the
user's name (as a proxy for perceived race/ethnicity). Each name is used
exactly once. Names are grouped by perceived race and gender.

Pipeline:
  1. Run raw queries (model x name x 1 iteration) — cached
  2. Extract YES/NO answers via LLM — cached separately
  3. Score and visualize (heatmap grid, per-name grid)
"""

import asyncio

from services.llm import OpenRouterProvider
from studies.god_by_name.config import (
    MODELS,
    NAMES,
    NAME_GROUPS,
    NAME_TO_GROUP,
    RACES,
    GENDERS,
    TEMPERATURE,
    ITERATIONS,
    EXTRACTOR_MODEL,
)
from studies.god_by_name.runner import run_all, build_messages, QUESTION_ID
from studies.god_by_name.extractor import (
    extract_all,
    compute_name_scores,
    compute_group_scores,
    compute_race_scores,
)
from studies.god_by_name.cache import (
    RESULTS_DIR,
    GRAPHS_DIR,
    load_response_cache,
    load_extraction_cache,
    get_responses,
    get_extraction,
)
from studies.god_by_name.visualize import (
    generate_group_heatmap,
    generate_name_grid,
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
        print(f"\n{'=' * 60}")
        print(f"  {short} — God Belief by Name")
        print(f"{'=' * 60}")

        name_answers = compute_name_scores(model)
        group_scores = compute_group_scores(model)
        race_scores = compute_race_scores(model)

        # Overall counts
        yes_total = sum(1 for a in name_answers.values() if a == "YES")
        no_total = sum(1 for a in name_answers.values() if a == "NO")
        refused_total = sum(1 for a in name_answers.values() if a == "REFUSED")
        errors_total = sum(1 for a in name_answers.values() if a in (None, "ERROR"))
        answered = yes_total + no_total
        total = len(name_answers)

        if total > 0:
            yes_pct = yes_total / answered * 100 if answered else 0
            no_pct = no_total / answered * 100 if answered else 0
            print(f"\n  YES: {yes_total} ({yes_pct:.1f}%)  NO: {no_total} ({no_pct:.1f}%)  "
                  f"Refused: {refused_total}  Errors: {errors_total}  Total: {total}")

        # Group breakdown
        print("\n  --- By Group (Race x Gender) ---")
        for race in RACES:
            for gender in GENDERS:
                score = group_scores.get((race, gender))
                if score is not None:
                    bar_len = int(score * 20)
                    bar = "#" * bar_len + "." * (20 - bar_len)
                    print(f"  {race:>10} {gender:<8}  [{bar}] {score:.0%} yes")
                else:
                    print(f"  {race:>10} {gender:<8}  [no data]")

        # Race breakdown
        print("\n  --- By Race (averaged) ---")
        for race in RACES:
            score = race_scores.get(race)
            if score is not None:
                bar_len = int(score * 20)
                bar = "#" * bar_len + "." * (20 - bar_len)
                print(f"  {race:>10}  [{bar}] {score:.0%} yes")

        # Per-name breakdown
        print("\n  --- Individual Names ---")
        for race in RACES:
            for gender in GENDERS:
                names = NAME_GROUPS[(race, gender)]
                answers = [name_answers.get(n, "?") for n in names]
                summary = ", ".join(f"{n}={a}" for n, a in zip(names, answers))
                print(f"  {race} {gender}: {summary}")


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def generate_graphs(models: list[str]):
    """Generate heatmaps and name grids for each model."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    for model in models:
        short = model.split("/")[-1]
        safe = model.replace("/", "_")

        # Group heatmap (race x gender)
        group_scores = compute_group_scores(model)
        has_data = any(v is not None for v in group_scores.values())
        if not has_data:
            print(f"  No valid results for {model}")
            continue

        generate_group_heatmap(
            group_scores=group_scores,
            title=f"God Belief by Race/Gender: {short}",
            save_path=GRAPHS_DIR / f"heatmap_{safe}.png",
        )

        # Per-name grid
        name_answers = compute_name_scores(model)
        generate_name_grid(
            name_answers=name_answers,
            title=f"God Belief by Name: {short}",
            save_path=GRAPHS_DIR / f"name_grid_{safe}.png",
        )

        print(f"  Graphs saved for {short} (heatmap + name grid)")


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
    print("  STUDY: God Belief by Name")
    print("=" * 60)
    print(f"\n  Names: {len(NAMES)}")
    print(f"  Groups: {len(NAME_GROUPS)} (race x gender)")
    for (race, gender), names in NAME_GROUPS.items():
        print(f"    {race} {gender}: {', '.join(names[:3])}...")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Iterations per name: {ITERATIONS}")
    print(f"  Total queries per model: {len(NAMES) * ITERATIONS}")
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
