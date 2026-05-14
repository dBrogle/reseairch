"""
Salary Bias Study

Tests whether LLMs recommend different salaries based on candidate name
(as a proxy for race/ethnicity/gender). An HR assistant is prompted with
identical candidate profiles where only the name varies. Each
(model, candidate, profile) combo is queried at temperature 0.7.

Pipeline:
  1. Run raw queries (model x candidate x profile x iterations) - cached
  2. Extract structured salary data via direct parse or LLM - cached
  3. Score, compute significance, and visualize
"""

import asyncio

import numpy as np

from services.llm import OpenRouterProvider
from studies.salary_bias.config import (
    MODELS,
    CANDIDATES,
    PROFILES,
    NAME_GROUPS,
    RACES,
    GENDERS,
    TEMPERATURE,
    ITERATIONS,
)
from studies.salary_bias.runner import run_all
from studies.salary_bias.extractor import extract_all, compute_scores
from studies.salary_bias.cache import RESULTS_DIR, GRAPHS_DIR
from studies.salary_bias.visualize import (
    compute_group_pct_deviation,
    compute_profile_pct_deviation,
    compute_kruskal_wallis,
    generate_zscore_heatmap,
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
    print("[5] Print detailed results (per name, per level, per model)")

    choice = input("\nChoice: ").strip()

    if choice == "3":
        return "regenerate_graphs"
    if choice == "4":
        return "print_summary"
    if choice == "5":
        return "print_detailed"

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

        has_data = any(
            scores[c["name"]][p["id"]]["count"] > 0
            for c in CANDIDATES for p in PROFILES
        )
        if not has_data:
            continue

        print(f"\n{'=' * 60}")
        print(f"  {short} - Salary Bias Results")
        print(f"{'=' * 60}")

        pct_devs = compute_group_pct_deviation(scores)
        kw_results = compute_kruskal_wallis(scores)

        # % deviation grid
        print(f"\n  {'':>15}", end="")
        for race in RACES:
            print(f"  {race:>10}", end="")
        print()

        for gender in GENDERS:
            print(f"  {gender:>15}", end="")
            for race in RACES:
                pct = pct_devs[(race, gender)]
                print(f"  {pct:>+9.1f}%", end="")
            print()

        print()
        for profile in PROFILES:
            p = kw_results.get(profile["id"], 1.0)
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if stars:
                print(f"  KW {profile['short_label']:>8}: {stars} p={p:.4f}")
            else:
                print(f"  KW {profile['short_label']:>8}: n.s. (p={p:.3f})")


# ---------------------------------------------------------------------------
# Detailed results
# ---------------------------------------------------------------------------

def print_detailed():
    for model in MODELS:
        short = model.split("/")[-1]
        scores = compute_scores(model)

        has_data = any(
            scores[c["name"]][p["id"]]["count"] > 0
            for c in CANDIDATES for p in PROFILES
        )
        if not has_data:
            continue

        print(f"\n{'=' * 80}")
        print(f"  {short} - Detailed Results")
        print(f"{'=' * 80}")

        for profile in PROFILES:
            print(f"\n  --- {profile['label']} ---")
            print(f"    {'Group':>18}  {'Mean':>10}  {'Std':>10}  {'n':>4}")
            print(f"    {'─' * 18}  {'─' * 10}  {'─' * 10}  {'─' * 4}")

            all_profile_salaries = []
            for race in RACES:
                for gender in GENDERS:
                    names = NAME_GROUPS[(race, gender)]
                    group_salaries = []
                    for name in names:
                        s = scores.get(name, {}).get(profile["id"], {})
                        group_salaries.extend(s.get("salaries", []))
                    all_profile_salaries.extend(group_salaries)
                    if group_salaries:
                        mean = np.mean(group_salaries)
                        std = np.std(group_salaries)
                        print(f"    {race + ' ' + gender:>18}  ${mean:>9,.0f}  ${std:>9,.0f}  {len(group_salaries):>4}")

            if all_profile_salaries:
                overall_mean = np.mean(all_profile_salaries)
                overall_std = np.std(all_profile_salaries)
                print(f"    {'OVERALL':>18}  ${overall_mean:>9,.0f}  ${overall_std:>9,.0f}  {len(all_profile_salaries):>4}")


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def generate_graphs(models: list[str]):
    """Generate one z-score heatmap per model."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    for model in models:
        short = model.split("/")[-1]
        safe = model.replace("/", "_")
        scores = compute_scores(model)

        has_data = any(
            scores[c["name"]][p["id"]]["count"] > 0
            for c in CANDIDATES for p in PROFILES
        )
        if not has_data:
            continue

        pct_devs = compute_group_pct_deviation(scores)
        kw_results = compute_kruskal_wallis(scores)

        # Overall (across all profiles)
        generate_zscore_heatmap(
            model=model,
            pct_devs=pct_devs,
            kw_results=kw_results,
            save_path=GRAPHS_DIR / f"zscore_{safe}.png",
        )
        print(f"  Overall heatmap saved for {short}")

        # Per-profile
        for profile in PROFILES:
            profile_pcts = compute_profile_pct_deviation(scores, profile["id"])
            profile_kw_p = kw_results.get(profile["id"], 1.0)
            single_kw = {profile["id"]: profile_kw_p}

            generate_zscore_heatmap(
                model=model,
                pct_devs=profile_pcts,
                kw_results=single_kw,
                save_path=GRAPHS_DIR / f"zscore_{profile['id']}_{safe}.png",
            )
            print(f"  {profile['short_label']} heatmap saved for {short}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_experiment(models: list[str]):
    provider = OpenRouterProvider()

    print("\n--- Step 1: Collecting raw LLM responses ---")
    await run_all(provider, models)

    print("\n--- Step 2: Extracting structured salary data ---")
    await extract_all(provider, models)

    print("\n--- Step 3: Scoring and visualization ---")
    generate_graphs(models)

    print_summary()


def main():
    print("\n" + "=" * 60)
    print("  STUDY: Salary Bias by Name (Race/Gender)")
    print("=" * 60)
    print(f"\n  Candidates: {len(CANDIDATES)} names across {len(NAME_GROUPS)} race/gender groups")
    for race in RACES:
        for gender in GENDERS:
            names = NAME_GROUPS[(race, gender)]
            print(f"    {race} {gender}: {', '.join(names[:3])}... ({len(names)} total)")
    print(f"  Profiles: {len(PROFILES)}")
    for p in PROFILES:
        print(f"    - {p['label']}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Iterations per combo: {ITERATIONS}")
    print(f"  Total queries per model: {len(CANDIDATES) * len(PROFILES) * ITERATIONS}")
    print(f"  Results cache: {RESULTS_DIR}/")

    models = select_models()
    if models == "regenerate_graphs":
        generate_graphs(MODELS)
        return
    if models == "print_summary":
        print_summary()
        return
    if models == "print_detailed":
        print_detailed()
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
