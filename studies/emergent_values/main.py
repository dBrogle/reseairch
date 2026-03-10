"""
Emergent Values Study

Replicates the exchange rates methodology from Mazeika et al. (2025) to measure
implied value systems in LLMs. Uses pairwise forced-choice comparisons across
categories (countries, religions, etc.) to fit a Bradley-Terry model producing
Elo-like utility scores per entity.
"""

import asyncio
import math

from services.llm import OpenRouterProvider
from utils.graphing import bar_chart, heatmap
from studies.emergent_values.config import (
    MODELS,
    EXPERIMENTS,
    CATEGORIES,
    MEASURES,
    K,
    BIDIRECTIONAL,
    PAIR_SEED,
    EDGE_MULTIPLIER,
    generate_options,
    compute_target_pairs,
)
from studies.emergent_values.cache import GRAPHS_DIR, RESULTS_DIR, load_cache
from studies.emergent_values.costs import CostTracker
from studies.emergent_values.runner import run_all, sample_pairs
from studies.emergent_values.elo import (
    compute_win_probabilities,
    fit_bradley_terry,
    aggregate_entity_scores,
)
from studies.emergent_values.geometric import (
    analyze_geometric,
    generate_geometric_bar_charts,
)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_experiment(
    cache: dict,
    category: str,
    measure: str,
) -> dict[str, float] | None:
    """Fit Bradley-Terry model and return per-entity Elo scores."""
    options = generate_options(category, measure)
    pairs = sample_pairs(options, PAIR_SEED)
    edges = compute_win_probabilities(cache, options, pairs, BIDIRECTIONAL)

    if not edges:
        print(f"  No comparison data for {category}×{measure}")
        return None

    utilities, loss, accuracy = fit_bradley_terry(len(options), edges)
    entity_scores = aggregate_entity_scores(options, utilities)

    print(f"\n  {category}×{measure}: {len(edges)} edges, loss={loss:.4f}, accuracy={accuracy:.1%}")

    return entity_scores


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def elo_to_exchange_rate(elo: float) -> float:
    """Convert Elo score to exchange rate relative to average (Elo=0).

    Exchange rate = 10^(elo/400). A value of 2.0 means the model values
    this entity at 2x the average; 0.5 means half the average.
    """
    return 10 ** (elo / 400.0)


def generate_bar_charts(
    model: str,
    scores_by_experiment: dict[tuple[str, str], dict[str, float]],
):
    """Generate sorted bar charts of entity exchange rates per experiment."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    short_model = model.split("/")[-1]
    safe_model = model.replace("/", "_")

    for (category, measure), entity_scores in scores_by_experiment.items():
        if not entity_scores:
            continue

        # Convert Elo scores to exchange rates and sort descending
        exchange_rates = {e: elo_to_exchange_rate(s) for e, s in entity_scores.items()}
        sorted_entities = sorted(exchange_rates.items(), key=lambda x: x[1], reverse=True)
        labels = [e[0] for e in sorted_entities]
        values = [e[1] for e in sorted_entities]

        bar_chart(
            labels=labels,
            values=values,
            title=f"Implied Preference Weights (Elo): {category} ({measure})\n{short_model}",
            x_label="Entity",
            y_label="Preference Weight (1.0 = average)",
            save_path=GRAPHS_DIR / f"bar_{safe_model}_{category}_{measure}.png",
            log_scale=True,
            value_fmt=".2f",
        )

    print(f"\nBar charts saved to {GRAPHS_DIR}/")


# Map from our entity names to shapefile NAME column
COUNTRY_NAME_MAP = {
    "United States": "United States of America",
}


def generate_world_map(
    model: str,
    entity_scores: dict[str, float],
):
    """Generate a world choropleth map for country scores."""
    try:
        from utils.map_graphing import world_choropleth
    except ImportError:
        print("  Skipping world map (geopandas not available)")
        return

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    short_model = model.split("/")[-1]
    safe_model = model.replace("/", "_")

    mapped_scores = {
        COUNTRY_NAME_MAP.get(k, k): v for k, v in entity_scores.items()
    }

    world_choropleth(
        country_values=mapped_scores,
        title=f"Implied Country Values: {short_model}",
        save_path=GRAPHS_DIR / f"map_{safe_model}_countries.png",
        cmap="RdYlGn",
        label_low="Lower value",
        label_high="Higher value",
    )
    print(f"  World map saved to {GRAPHS_DIR}/")


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def select_models() -> list[str] | str:
    """Interactive model selection menu."""
    print("\n=== Model Selection ===")
    print("[1] Run ALL models")
    print("[2] Select individual models")
    print("[3] Regenerate graphs from cached results")
    print("[4] Print results summary")
    print("[5] Print experiment stats")
    print("[6] Print cost summary")

    choice = input("\nChoice: ").strip()

    if choice == "3":
        return "regenerate_graphs"
    if choice == "4":
        return "print_summary"
    if choice == "5":
        return "print_stats"
    if choice == "6":
        return "print_costs"

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
# Summaries
# ---------------------------------------------------------------------------

def _analyze_all_cached(save_diagnostics: bool = False):
    """Load and analyze all cached experiments (Elo + geometric N-adjusted).

    Returns (elo_scores, geom_scores) where each is
    {model: {(category, measure): {entity: score}}}
    """
    all_elo = {}
    all_geom = {}
    for model in MODELS:
        elo_scores = {}
        geom_scores = {}
        for category, measure in EXPERIMENTS:
            cache = load_cache(model, category, measure)
            if not cache:
                continue
            scores = analyze_experiment(cache, category, measure)
            if scores:
                elo_scores[(category, measure)] = scores
            g_scores = analyze_geometric(
                cache, category, measure, mode="same_n",
                save_diagnostics=save_diagnostics, model=model,
            )
            if g_scores:
                geom_scores[(category, measure)] = g_scores
        if elo_scores:
            all_elo[model] = elo_scores
        if geom_scores:
            all_geom[model] = geom_scores
    return all_elo, all_geom


def print_summary():
    """Print ranked entity scores per experiment per model."""
    all_elo, all_geom = _analyze_all_cached()
    if not all_elo and not all_geom:
        print("No cached results found. Run the experiment first.")
        return

    for model in MODELS:
        elo_scores = all_elo.get(model, {})
        geom_scores = all_geom.get(model, {})
        if not elo_scores and not geom_scores:
            continue

        short = model.split("/")[-1]
        print(f"\n{'=' * 70}")
        print(f"  {short}")
        print(f"{'=' * 70}")

        experiments = set(elo_scores.keys()) | set(geom_scores.keys())
        for category, measure in sorted(experiments):
            print(f"\n  {category} × {measure}:")

            elo = elo_scores.get((category, measure), {})
            geom = geom_scores.get((category, measure), {})
            entities = sorted(set(elo.keys()) | set(geom.keys()))

            # Sort by geometric score if available, else Elo
            entities.sort(key=lambda e: geom.get(e, 1.0), reverse=True)

            print(f"    {'Rank':<6} {'Entity':<25} {'Elo':>8} {'Exch Rate':>10} {'Geom':>8}")
            print(f"    {'-'*57}")
            for i, entity in enumerate(entities, 1):
                elo_val = elo.get(entity)
                elo_str = f"{elo_val:.1f}" if elo_val is not None else "-"
                exch_str = f"{elo_to_exchange_rate(elo_val):.2f}" if elo_val is not None else "-"
                geom_val = geom.get(entity)
                geom_str = f"{geom_val:.2f}" if geom_val is not None else "-"
                print(f"    {i:<6} {entity:<25} {elo_str:>8} {exch_str:>10} {geom_str:>8}")

    print()


def print_stats():
    """Print experiment stats: pairs, API calls, coverage."""
    print(f"\n{'=' * 70}")
    print("  EXPERIMENT STATISTICS")
    print(f"{'=' * 70}")

    for category, measure in EXPERIMENTS:
        options = generate_options(category, measure)
        n_options = len(options)
        n_entities = len(CATEGORIES[category])
        n_values = len(MEASURES[measure]["values"])
        target_pairs = compute_target_pairs(n_options)
        total_possible = n_options * (n_options - 1) // 2
        directions = 2 if BIDIRECTIONAL else 1
        total_api_calls = target_pairs * directions * K

        print(f"\n  {category} × {measure}:")
        print(f"    Entities:         {n_entities}")
        print(f"    Measure values:   {n_values}")
        print(f"    Total options:    {n_options}")
        print(f"    Possible pairs:   {total_possible:,}")
        print(f"    Sampled pairs:    {target_pairs:,} (edge_multiplier={EDGE_MULTIPLIER})")
        print(f"    Directions:       {directions}")
        print(f"    K per direction:  {K}")
        print(f"    Total API calls:  {total_api_calls:,}")

    # Per-model cache status
    print(f"\n  --- Cache Status ---")
    for model in MODELS:
        short = model.split("/")[-1]
        for category, measure in EXPERIMENTS:
            cache = load_cache(model, category, measure)
            n_entries = len(cache)
            n_results = sum(len(e.get("results", [])) for e in cache.values())
            n_errors = sum(
                1 for e in cache.values()
                for r in e.get("results", [])
                if r.get("error") is not None
            )
            print(f"    {short} {category}×{measure}: {n_entries} pairs, {n_results} results, {n_errors} errors")

    print()


def regenerate_graphs_from_cache():
    """Regenerate all graphs from cached results (with diagnostics)."""
    all_elo, all_geom = _analyze_all_cached(save_diagnostics=True)
    if not all_elo and not all_geom:
        print("No cached results found. Run the experiment first.")
        return

    for model in set(list(all_elo.keys()) + list(all_geom.keys())):
        if model in all_elo:
            generate_bar_charts(model, all_elo[model])
            countries_key = ("countries", "terminal_illness")
            if countries_key in all_elo[model]:
                generate_world_map(model, all_elo[model][countries_key])
        if model in all_geom:
            generate_geometric_bar_charts(model, all_geom[model])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_experiment(models: list[str]):
    """Run the full experiment for selected models."""
    provider = OpenRouterProvider()
    cost_tracker = CostTracker()
    all_caches = await run_all(provider, models, EXPERIMENTS, cost_tracker)

    # Analyze and visualize
    for model in models:
        elo_scores = {}
        geom_scores = {}
        for category, measure in EXPERIMENTS:
            cache = all_caches[model].get((category, measure), {})
            scores = analyze_experiment(cache, category, measure)
            if scores:
                elo_scores[(category, measure)] = scores
            g_scores = analyze_geometric(cache, category, measure, mode="same_n", model=model)
            if g_scores:
                geom_scores[(category, measure)] = g_scores

        if elo_scores:
            generate_bar_charts(model, elo_scores)
            countries_key = ("countries", "terminal_illness")
            if countries_key in elo_scores:
                generate_world_map(model, elo_scores[countries_key])
        if geom_scores:
            generate_geometric_bar_charts(model, geom_scores)

    # Print cost and results summary
    print("\n  === Final Cost Summary ===")
    cost_tracker.print_summary()
    print_summary()


def main():
    """Entry point for the Emergent Values study."""
    print("\n" + "=" * 60)
    print("  STUDY: Emergent Values (Exchange Rates)")
    print("=" * 60)
    print(f"\nExperiments: {EXPERIMENTS}")
    print(f"K={K} responses per pair, bidirectional={BIDIRECTIONAL}")
    print(f"Results cache: {RESULTS_DIR}/")

    # Show estimated API calls
    print_stats()

    models = select_models()
    if models == "regenerate_graphs":
        regenerate_graphs_from_cache()
        return
    if models == "print_summary":
        print_summary()
        return
    if models == "print_stats":
        print_stats()
        return
    if models == "print_costs":
        tracker = CostTracker()
        tracker.print_summary()
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
