"""
Hate Speech Detection Bias Study

Tests whether LLMs acting as hate speech detectors flag identical statements
differently depending on which demographic group is referenced.

Pipeline:
  1. Run raw queries (model x template x group x iterations) - cached
  2. Extract flag/not-flag decisions via JSON parsing
  3. Compute flag rates and visualize bias
"""

import asyncio

from services.llm import OpenRouterProvider
from studies.hate_speech_detection.config import (
    MODELS,
    GROUPS,
    TEMPLATES,
    RACES,
    GENDERS,
    TEMPERATURE,
    ITERATIONS,
)
from studies.hate_speech_detection.runner import run_all, get_group_text
from studies.hate_speech_detection.extractor import compute_flag_rates
from studies.hate_speech_detection.cache import GRAPHS_DIR


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
    """Print text summary of flag rates and detected biases."""
    from studies.hate_speech_detection.visualize import (
        compute_chi_squared,
        compute_pairwise_proportions,
        _stars,
        DISPLAY_RACES,
        DISPLAY_GENDERS,
        _get_cell_stats,
    )

    for model in MODELS:
        short = model.split("/")[-1]
        flag_rates = compute_flag_rates(model)

        has_data = any(
            entry["count"] > 0
            for t_data in flag_rates.values()
            for entry in t_data.values()
        )
        if not has_data:
            continue

        print(f"\n{'=' * 80}")
        print(f"  {short} - Hate Speech Detection Bias")
        print(f"{'=' * 80}")

        templates_sorted = sorted(TEMPLATES, key=lambda t: t["severity"])

        # Per-template breakdown
        for template in templates_sorted:
            t_data = flag_rates[template["id"]]
            print(f"\n  --- [{template['severity']}] {template['label']} ---")
            print(f"  \"{template['text'].format(group='[GROUP]')}\"")

            entries = [(gt, data) for gt, data in t_data.items() if data["count"] > 0]
            entries.sort(key=lambda x: x[1]["flag_rate"] or 0, reverse=True)

            for group_text, data in entries:
                rate = data["flag_rate"]
                bar_len = int(rate * 30) if rate else 0
                bar = "#" * bar_len + "." * (30 - bar_len)
                pct = f"{rate * 100:5.1f}%" if rate is not None else "  n/a"
                print(f"    {group_text:>20}  [{bar}] {pct}  ({data['flagged']}/{data['count']})")

        # Grid summary
        print(f"\n  --- Flag Rate Grid (averaged across templates) ---")
        header = "              " + "".join(f"{r:>14}" for r in DISPLAY_RACES)
        print(f"  {header}")
        for gender, gender_label in DISPLAY_GENDERS:
            row = f"    {gender_label:>10}  "
            for race in DISPLAY_RACES:
                cell = _get_cell_stats(flag_rates, race, gender)
                if cell["flag_rate"] is not None:
                    row += f"{cell['flag_rate']*100:13.1f}%"
                else:
                    row += f"{'n/a':>14}"
            print(row)

        # Chi-squared
        chi2 = compute_chi_squared(flag_rates)
        if chi2["chi2"] is not None:
            stars = _stars(chi2["p"])
            print(f"\n  Chi-squared test: X2({chi2['df']}, n={chi2['n']}) = {chi2['chi2']:.2f}, p = {chi2['p']:.4f} {stars}")

        # Significant pairwise comparisons
        pairwise = compute_pairwise_proportions(flag_rates)
        sig_pairs = [p for p in pairwise if p["p"] < 0.05]
        if sig_pairs:
            print(f"\n  Significant pairwise differences (p < 0.05):")
            sig_pairs.sort(key=lambda x: x["p"])
            for pair in sig_pairs:
                stars = _stars(pair["p"])
                print(f"    {pair['group_a']} vs {pair['group_b']}: z={pair['z']:.2f}, p={pair['p']:.4f} {stars}")


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def generate_graphs(models: list[str]):
    """Generate one grid chart per model."""
    from studies.hate_speech_detection.visualize import generate_model_chart

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    for model in models:
        flag_rates = compute_flag_rates(model)
        has_data = any(
            entry["count"] > 0
            for t_data in flag_rates.values()
            for entry in t_data.values()
        )
        if not has_data:
            continue

        short = model.split("/")[-1]
        safe = model.replace("/", "_")

        generate_model_chart(
            model=model,
            flag_rates=flag_rates,
            save_path=GRAPHS_DIR / f"grid_{safe}.png",
        )
        print(f"  Grid chart saved for {short}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_experiment(models: list[str]):
    provider = OpenRouterProvider()

    print("\n--- Step 1: Collecting raw LLM responses ---")
    await run_all(provider, models)

    print("\n--- Step 2: Scoring and visualization ---")
    generate_graphs(models)

    print_summary()


def _count_unique_combos() -> int:
    """Count unique (template, group_text) combos."""
    count = 0
    for template in TEMPLATES:
        seen = set()
        for group in GROUPS:
            text = get_group_text(group, template["form"])
            if text not in seen:
                seen.add(text)
                count += 1
    return count


def main():
    unique_combos = _count_unique_combos()

    print("\n" + "=" * 60)
    print("  STUDY: Hate Speech Detection Bias")
    print("=" * 60)
    print(f"\n  Templates: {len(TEMPLATES)}")
    for t in sorted(TEMPLATES, key=lambda t: t["severity"]):
        print(f"    [{t['severity']:>2}] {t['label']:20} ({t['form']}) \"{t['text'].format(group='...')}\"")
    print(f"\n  Groups: {len(GROUPS)} ({len(RACES)} races x {len(GENDERS)} genders + unspecified)")
    print(f"  Unique combos per model: {unique_combos}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Iterations per combo: {ITERATIONS}")
    print(f"  Total queries per model: {unique_combos * ITERATIONS}")

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
