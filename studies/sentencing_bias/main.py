"""
Sentencing Bias Study

Tests whether LLMs assign different sentences based on defendant name
(as a proxy for race/ethnicity/gender). Each (model, defendant, crime) combo
is queried multiple times at temperature 0.7 for structured sentencing output.

Pipeline:
  1. Run raw queries (model x defendant x crime x iterations) - cached
  2. Extract structured sentencing data via direct parse or LLM - cached
  3. Score, compute significance, and visualize
"""

import asyncio

from services.llm import OpenRouterProvider
from studies.sentencing_bias.config import (
    MODELS,
    DEFENDANTS,
    CRIMES,
    TEMPERATURE,
    ITERATIONS,
)
from studies.sentencing_bias.runner import run_all
from studies.sentencing_bias.extractor import extract_all, compute_scores
from studies.sentencing_bias.cache import RESULTS_DIR, GRAPHS_DIR
from studies.sentencing_bias.visualize import (
    compute_defendant_zscores,
    compute_significance,
    generate_heatmap,
    generate_zscore_chart,
    generate_cross_model_heatmap,
    generate_cross_model_zscore_chart,
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
        has_data = False
        for d_id in scores:
            for c_id in scores[d_id]:
                if scores[d_id][c_id]["count"] > 0:
                    has_data = True
                    break
            if has_data:
                break
        if not has_data:
            continue

        print(f"\n{'=' * 80}")
        print(f"  {short} - Sentencing Bias Results")
        print(f"{'=' * 80}")

        zscores = compute_defendant_zscores(scores)
        significance = compute_significance(scores, zscores)
        id_to_name = {d["id"]: d["name"] for d in DEFENDANTS}

        # Per-crime comparison across defendants
        for crime in CRIMES:
            print(f"\n  --- {crime['label']} ---")

            for defendant in DEFENDANTS:
                s = scores[defendant["id"]][crime["id"]]
                jail = s["avg_jail"]
                fine = s["avg_fine"]
                n = s["count"]

                if jail is not None and fine is not None:
                    if jail >= 1:
                        jail_str = f"{jail:.1f}y"
                    elif jail > 0:
                        jail_str = f"{jail * 12:.1f}mo"
                    else:
                        jail_str = "0"
                    fine_str = f"${fine:>8,.0f}"

                    # Visual bar for jail time
                    bar_len = min(int(jail * 5), 30)
                    bar = "#" * bar_len + "." * (30 - bar_len)

                    print(f"    {defendant['name']:>20}  [{bar}] jail={jail_str:>6}  fine={fine_str}  (n={n})")
                else:
                    print(f"    {defendant['name']:>20}  no data")

            # Kruskal-Wallis for this crime
            kw = significance["per_crime_kw"].get(crime["id"], {})
            for metric_key, metric_label in [("jail_p", "jail"), ("fine_p", "fine")]:
                p = kw.get(metric_key, 1.0)
                if p < 0.05:
                    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                    print(f"    Kruskal-Wallis ({metric_label}): {stars} p={p:.4f}")

            # Pairwise significance
            pw = significance["pairwise_per_crime"].get(crime["id"], {})
            sig_pairs = [(a, b, p) for (a, b), p in pw.items()
                         if p.get("jail_p", 1) < 0.05 or p.get("fine_p", 1) < 0.05]

            if sig_pairs:
                for a, b, p in sig_pairs:
                    parts = []
                    if p.get("jail_p", 1) < 0.05:
                        stars = "***" if p["jail_p"] < 0.001 else "**" if p["jail_p"] < 0.01 else "*"
                        parts.append(f"jail {stars} p={p['jail_p']:.4f}")
                    if p.get("fine_p", 1) < 0.05:
                        stars = "***" if p["fine_p"] < 0.001 else "**" if p["fine_p"] < 0.01 else "*"
                        parts.append(f"fine {stars} p={p['fine_p']:.4f}")
                    print(f"      {id_to_name[a]} vs {id_to_name[b]}: {', '.join(parts)}")

        # Overall z-score summary
        print(f"\n  --- Overall Z-Scores (+ = harsher, - = more lenient) ---")
        sorted_defs = sorted(DEFENDANTS, key=lambda d: zscores[d["id"]]["combined_z"], reverse=True)
        for defendant in sorted_defs:
            dz = zscores[defendant["id"]]
            z = dz["combined_z"]
            bar_len = int(abs(z) * 15)
            if z >= 0:
                bar = " " * 15 + "|" + "#" * bar_len + "." * (15 - bar_len)
            else:
                bar = "." * (15 - bar_len) + "#" * bar_len + "|" + " " * 15
            print(f"    {defendant['name']:>20}  [{bar}] z={z:+.3f}")

        # Overall Kruskal-Wallis
        for metric_key, metric_label in [("jail_p", "jail"), ("fine_p", "fine")]:
            p = significance["overall_kw"].get(metric_key, 1.0)
            if p < 0.05:
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                print(f"    Overall Kruskal-Wallis ({metric_label} z-scores): {stars} p={p:.4f}")


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def generate_graphs(models: list[str]):
    """Generate all charts."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    all_scores = {}
    for model in models:
        scores = compute_scores(model)
        has_data = any(
            scores[d["id"]][c["id"]]["count"] > 0
            for d in DEFENDANTS for c in CRIMES
        )
        if has_data:
            all_scores[model] = scores

    if not all_scores:
        print("  No data to graph.")
        return

    # Compute z-scores and significance for all models
    all_zscores = {}
    all_significance = {}
    for model, scores in all_scores.items():
        zscores = compute_defendant_zscores(scores)
        significance = compute_significance(scores, zscores)
        all_zscores[model] = zscores
        all_significance[model] = significance

    # Per-model heatmaps and z-score charts
    for model, scores in all_scores.items():
        short = model.split("/")[-1]
        safe = model.replace("/", "_")

        generate_heatmap(
            model=model,
            scores=scores,
            significance=all_significance[model],
            save_path=GRAPHS_DIR / f"heatmap_{safe}.png",
        )
        print(f"  Heatmap saved for {short}")

        generate_zscore_chart(
            model=model,
            zscores=all_zscores[model],
            significance=all_significance[model],
            save_path=GRAPHS_DIR / f"zscore_{safe}.png",
        )
        print(f"  Z-score chart saved for {short}")

    # Cross-model heatmaps per crime
    for crime in CRIMES:
        generate_cross_model_heatmap(
            all_scores=all_scores,
            crime_id=crime["id"],
            crime_label=crime["label"],
            save_path=GRAPHS_DIR / f"cross_model_{crime['id']}.png",
        )
        print(f"  Cross-model heatmap saved for {crime['label']}")

    # Cross-model combined z-score chart
    generate_cross_model_zscore_chart(
        all_zscores=all_zscores,
        all_significance=all_significance,
        save_path=GRAPHS_DIR / "cross_model_zscore.png",
    )
    print("  Cross-model z-score chart saved")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_experiment(models: list[str]):
    provider = OpenRouterProvider()

    # Step 1: Raw responses
    print("\n--- Step 1: Collecting raw LLM responses ---")
    await run_all(provider, models)

    # Step 2: Extract structured sentencing data
    print("\n--- Step 2: Extracting structured sentencing data ---")
    await extract_all(provider, models)

    # Step 3: Score and visualize
    print("\n--- Step 3: Scoring and visualization ---")
    generate_graphs(models)

    print_summary()


def main():
    print("\n" + "=" * 60)
    print("  STUDY: Sentencing Bias by Name")
    print("=" * 60)
    print(f"\n  Defendants: {len(DEFENDANTS)}")
    for d in DEFENDANTS:
        print(f"    - {d['name']}")
    print(f"  Crimes: {len(CRIMES)}")
    for c in CRIMES:
        print(f"    - {c['label']}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Iterations per combo: {ITERATIONS}")
    print(f"  Total queries per model: {len(DEFENDANTS) * len(CRIMES) * ITERATIONS}")
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
