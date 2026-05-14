"""
God Belief by Country Study

Tests whether LLMs change their answer to "does god exist?" based on which
country the user is connecting from. Each country is asked 10 times at
temperature 1.0 to capture response variance.

Pipeline:
  1. Run raw queries (model x country x 10 iterations) — cached
  2. Extract YES/NO answers via LLM — cached separately
  3. Score and visualize (world map, ranked bar chart, regional charts)
"""

import asyncio

from services.llm import OpenRouterProvider
from studies.god_by_country.config import (
    MODELS,
    COUNTRIES,
    COUNTRY_GROUPS,
    TEMPERATURE,
    ITERATIONS,
    EXTRACTOR_MODEL,
)
from studies.god_by_country.runner import run_all, build_messages, QUESTION_ID
from studies.god_by_country.extractor import extract_all, compute_country_scores
from studies.god_by_country.cache import (
    RESULTS_DIR,
    GRAPHS_DIR,
    load_response_cache,
    load_extraction_cache,
    get_responses,
    get_extraction,
)
from studies.god_by_country.visualize import (
    generate_world_map,
    generate_ranked_bar_chart,
    generate_regional_bar_charts,
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
        print(f"  {short} — God Belief by Country")
        print(f"{'=' * 60}")

        response_cache = load_response_cache(model)
        extraction_cache = load_extraction_cache(model)

        yes_total = 0
        no_total = 0
        refused_total = 0
        errors_total = 0

        for country in COUNTRIES:
            messages = build_messages(country)
            responses = get_responses(response_cache, messages, TEMPERATURE)[:ITERATIONS]
            for result in responses:
                if result.get("error") is not None or result.get("response") is None:
                    errors_total += 1
                    continue
                answer = get_extraction(
                    extraction_cache, QUESTION_ID,
                    result["response"], EXTRACTOR_MODEL,
                )
                if answer is None or answer == "ERROR":
                    errors_total += 1
                elif answer == "REFUSED":
                    refused_total += 1
                elif answer == "YES":
                    yes_total += 1
                elif answer == "NO":
                    no_total += 1

        answered = yes_total + no_total
        total = answered + refused_total + errors_total

        if total > 0:
            yes_pct = yes_total / answered * 100 if answered else 0
            no_pct = no_total / answered * 100 if answered else 0
            print(f"\n  YES: {yes_total} ({yes_pct:.1f}%)  NO: {no_total} ({no_pct:.1f}%)  "
                  f"Refused: {refused_total}  Errors: {errors_total}  Total: {total}")

        scores = compute_country_scores(model)
        valid = {c: v for c, v in scores.items() if v is not None}
        if not valid:
            print("  No valid country-level data.")
            continue

        # Print by region
        for region, countries in COUNTRY_GROUPS.items():
            region_scores = {c: valid[c] for c in countries if c in valid}
            if not region_scores:
                continue
            print(f"\n  --- {region} ---")
            sorted_countries = sorted(region_scores.items(), key=lambda x: x[1], reverse=True)
            for country, score in sorted_countries:
                bar_len = int(score * 30)
                bar = "#" * bar_len + "." * (30 - bar_len)
                print(f"  {country:>25}  [{bar}] {score:.0%} yes")

        mean = sum(valid.values()) / len(valid)
        spread = max(valid.values()) - min(valid.values())
        print(f"\n  Overall YES rate: {mean:.1%}")
        print(f"  Spread (max - min): {spread:.1%}")


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def generate_graphs(models: list[str]):
    """Generate world maps, ranked bar charts, and regional charts."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    for model in models:
        short = model.split("/")[-1]
        safe = model.replace("/", "_")

        scores = compute_country_scores(model)
        valid = {c: v for c, v in scores.items() if v is not None}
        if not valid:
            print(f"  No valid results for {model}")
            continue

        # World map — absolute scale
        generate_world_map(
            country_scores=valid,
            title=f"God Belief by Country (absolute): {short}",
            save_path=GRAPHS_DIR / f"world_map_abs_{safe}.png",
            score_range=(0.0, 1.0),
        )

        # World map — relative scale
        sorted_vals = sorted(valid.values())
        vmin, vmax = sorted_vals[0], sorted_vals[-1]
        if vmin < vmax:
            generate_world_map(
                country_scores=valid,
                title=f"God Belief by Country (relative): {short}",
                save_path=GRAPHS_DIR / f"world_map_rel_{safe}.png",
                score_range=(vmin, vmax),
            )

        # Ranked bar chart
        generate_ranked_bar_chart(
            country_scores=valid,
            title=f"God Belief Ranking: {short}",
            save_path=GRAPHS_DIR / f"ranking_{safe}.png",
        )

        # Regional bar charts
        generate_regional_bar_charts(
            country_scores=valid,
            title_prefix=f"God Belief: {short}",
            save_dir=GRAPHS_DIR / "regions" / safe,
        )

        print(f"  Graphs saved for {short} (world map + ranking + regional)")


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
    print("  STUDY: God Belief by Country")
    print("=" * 60)
    print(f"\n  Countries: {len(COUNTRIES)}")
    for region, countries in COUNTRY_GROUPS.items():
        print(f"    {region}: {', '.join(countries)}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Iterations per combo: {ITERATIONS}")
    print(f"  Total queries per model: {len(COUNTRIES) * ITERATIONS}")
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
