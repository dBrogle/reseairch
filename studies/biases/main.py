"""
Biases Study

Tests whether LLMs show gender-based differences in moral reasoning by
varying the subject (man/woman/person) and action (torture/harass) in a
trolley-problem-style dilemma, scored on a 7-point Likert scale.

Each model produces a heatmap showing mean acceptability scores across
the subject x action grid.
"""

import asyncio
import json
import math

from services.llm import OpenRouterProvider
from utils.graphing import heatmap
from studies.biases.config import (
    MODELS,
    SUBJECTS,
    ACTIONS,
    TEMPERATURE,
    ITERATIONS,
    PROMPT_TEMPLATE,
)
from studies.biases.cache import GRAPHS_DIR, RESULTS_DIR, load_cache, get_results
from studies.biases.runner import run_all, build_prompt


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _ci95(scores: list[int | float]) -> float:
    """Compute 95% CI half-width for a list of scores (t-distribution)."""
    n = len(scores)
    if n < 2:
        return 0.0
    mean = sum(scores) / n
    variance = sum((x - mean) ** 2 for x in scores) / (n - 1)
    stderr = math.sqrt(variance / n)
    # t critical value for 95% CI, df=n-1 (approximation for small n)
    # For n=10, df=9, t_crit ~ 2.262
    t_crits = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
               7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 15: 2.145,
               20: 2.093, 30: 2.045, 50: 2.009, 100: 1.984}
    # Find closest df
    df = n - 1
    if df < 2:
        return 0.0
    if df in t_crits:
        t = t_crits[df]
    elif df > 100:
        t = 1.96
    else:
        # Linear interpolation between nearest keys
        keys = sorted(t_crits.keys())
        below = [k for k in keys if k <= df]
        above = [k for k in keys if k >= df]
        if not below or not above:
            t = 1.96
        else:
            lower, upper = max(below), min(above)
            if lower == upper:
                t = t_crits[lower]
            else:
                frac = (df - lower) / (upper - lower)
                t = t_crits[lower] + frac * (t_crits[upper] - t_crits[lower])
    return t * stderr


def compute_mean_scores(
    model_results: dict[str, dict[str, list[dict]]]
) -> dict[str, dict[str, tuple[float, float]]]:
    """Compute the mean Likert score and 95% CI for each (subject, action) cell.

    Returns {subject: {action: (mean, ci95_half_width)}}.
    Ignores results with no valid score.
    """
    means: dict[str, dict[str, tuple[float, float]]] = {}
    for subject in SUBJECTS:
        means[subject] = {}
        for action in ACTIONS:
            results = model_results.get(subject, {}).get(action, [])
            scores = [r["score"] for r in results if r.get("score") is not None]
            if scores:
                mean = sum(scores) / len(scores)
                ci = _ci95(scores)
                means[subject][action] = (mean, ci)
            else:
                means[subject][action] = (0.0, 0.0)
    return means


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------

def generate_heatmaps(
    all_results: dict[str, dict[str, dict[str, list[dict]]]]
):
    """Generate a heatmap per model showing mean Likert scores."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    # Capitalize labels for display
    row_labels = [a.capitalize() for a in ACTIONS]
    col_labels = [s.capitalize() for s in SUBJECTS]

    for model_id, model_results in all_results.items():
        means = compute_mean_scores(model_results)
        short_name = model_id.split("/")[-1]
        safe_name = model_id.replace("/", "_")

        # Build data matrix and annotation matrix: rows=actions, cols=subjects
        data = []
        annotations = []
        for action in ACTIONS:
            row = []
            ann_row = []
            for subject in SUBJECTS:
                mean, ci = means[subject][action]
                row.append(mean)
                ann_row.append(f"{mean:.2f}\n(\u00b1{ci:.2f})")
            data.append(row)
            annotations.append(ann_row)

        heatmap(
            data=data,
            annotations=annotations,
            row_labels=row_labels,
            col_labels=col_labels,
            title=f"Mean Acceptability Score: {short_name}",
            x_label="Subject",
            y_label="Action",
            save_path=GRAPHS_DIR / f"heatmap_{safe_name}.png",
            value_range=(1.0, 7.0),
            cmap="RdYlGn",
            fmt=".2f",
        )

    print(f"\nHeatmaps saved to {GRAPHS_DIR}/")


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def select_models() -> list[str] | str:
    """Interactive model selection menu."""
    print("\n=== Model Selection ===")
    print("[1] Run ALL models")
    print("[2] Select individual models")
    print("[3] Regenerate heatmaps from cached results")
    print("[4] Print results summary")
    print("[5] Print error report")

    choice = input("\nChoice: ").strip()

    if choice == "3":
        return "regenerate_graphs"
    if choice == "4":
        return "print_summary"
    if choice == "5":
        return "error_report"

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

def _load_all_results() -> dict[str, dict[str, dict[str, list[dict]]]]:
    """Load cached results for all models that have data."""
    all_results: dict[str, dict[str, dict[str, list[dict]]]] = {}
    for model in MODELS:
        cache = load_cache(model)
        if not cache:
            continue
        model_results: dict[str, dict[str, list[dict]]] = {}
        has_data = False
        for subject in SUBJECTS:
            model_results[subject] = {}
            for action in ACTIONS:
                prompt = build_prompt(subject, action)
                results = get_results(cache, prompt, TEMPERATURE)[:ITERATIONS]
                model_results[subject][action] = results
                if results:
                    has_data = True
        if has_data:
            all_results[model] = model_results
    return all_results


def regenerate_graphs_from_cache():
    """Regenerate all heatmaps from existing cached results."""
    all_results = _load_all_results()
    if not all_results:
        print("No cached results found. Run the experiment first.")
        return
    print(f"Regenerating heatmaps for {len(all_results)} model(s)...")
    generate_heatmaps(all_results)


def print_summary():
    """Print a text summary of mean scores per model."""
    all_results = _load_all_results()
    if not all_results:
        print("No cached results found. Run the experiment first.")
        return

    print(f"\n{'=' * 70}")
    print("  BIASES STUDY - MEAN LIKERT SCORES (1=unacceptable, 7=acceptable)")
    print(f"{'=' * 70}")

    for model_id, model_results in all_results.items():
        means = compute_mean_scores(model_results)
        short = model_id.split("/")[-1]
        print(f"\n  {short}:")
        header = f"    {'':>12}"
        for subject in SUBJECTS:
            header += f"  {subject:>8}"
        print(header)
        for action in ACTIONS:
            row = f"    {action:>12}"
            for subject in SUBJECTS:
                mean, ci = means[subject][action]
                row += f"  {mean:>8.2f}"
            print(row)

        # Highlight gender gap
        for action in ACTIONS:
            man_score = means["man"][action][0]
            woman_score = means["woman"][action][0]
            gap = man_score - woman_score
            if abs(gap) > 0.01:
                direction = "higher for men" if gap > 0 else "higher for women"
                print(f"      {action} gender gap: {abs(gap):.2f} ({direction})")

    print()


def print_error_report():
    """Print error counts for all cached results."""
    print(f"\n{'=' * 80}")
    print("  ERROR REPORT")
    print(f"{'=' * 80}")
    print(f"\n{'Model':<35} {'Total':>8} {'Valid':>8} {'No Score':>10} {'Errors':>8}")
    print("-" * 80)

    for model in MODELS:
        cache = load_cache(model)
        if not cache:
            continue
        total = 0
        errors = 0
        no_score = 0
        for subject in SUBJECTS:
            for action in ACTIONS:
                prompt = build_prompt(subject, action)
                results = get_results(cache, prompt, TEMPERATURE)
                for r in results:
                    total += 1
                    if r.get("error") is not None:
                        errors += 1
                    elif r.get("score") is None:
                        no_score += 1
        valid = total - errors - no_score
        short = model.split("/")[-1]
        print(f"  {short:<33} {total:>8} {valid:>8} {no_score:>10} {errors:>8}")

    print()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_experiment(models: list[str]):
    """Run the full experiment for selected models."""
    provider = OpenRouterProvider()
    all_results = await run_all(provider, models)

    # Print quick summary
    for model_id, model_results in all_results.items():
        means = compute_mean_scores(model_results)
        short = model_id.split("/")[-1]
        parts = []
        for action in ACTIONS:
            for subject in SUBJECTS:
                parts.append(f"{action}/{subject}={means[subject][action][0]:.2f}")
        print(f"\n  {short}: {', '.join(parts)}")

    # Generate heatmaps
    generate_heatmaps(all_results)

    # Print full summary
    print_summary()


def main():
    """Entry point for the Biases study."""
    print("\n" + "=" * 60)
    print("  STUDY: Gender Bias in LLM Moral Reasoning")
    print("=" * 60)
    print(f"\nSubjects: {SUBJECTS}")
    print(f"Actions: {ACTIONS}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Iterations per combo: {ITERATIONS}")
    print(f"Results cache: {RESULTS_DIR}/")

    models = select_models()
    if models == "regenerate_graphs":
        regenerate_graphs_from_cache()
        return
    if models == "print_summary":
        print_summary()
        return
    if models == "error_report":
        print_error_report()
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
