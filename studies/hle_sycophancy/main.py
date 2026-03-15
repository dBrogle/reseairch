"""
HLE Sycophancy Study

Tests whether LLMs perform differently on Humanity's Last Exam (HLE) questions
based on which U.S. state the user is stated to be from. Per-state accuracy is
computed and visualized on a US choropleth map.

Pipeline:
  1. Load (or download + cache) HLE questions
  2. Run raw queries (model × state × question) — cached per model
  3. Grade answers via LLM — cached separately per model
  4. Compute per-state accuracy and visualize
"""

import asyncio

from services.llm import OpenRouterProvider
from studies.hle_sycophancy.config import (
    MODELS,
    STATES,
    TEMPERATURE,
    ITERATIONS,
    HLE_SEED,
    HLE_PER_CATEGORY,
    GRADER_MODEL,
)
from studies.hle_sycophancy.hle_data import load_hle_stratified_sample
from studies.hle_sycophancy.cache import (
    RESULTS_DIR,
    GRAPHS_DIR,
    load_hle_questions,
    save_hle_questions,
)

# ---------------------------------------------------------------------------
# scipy is an optional dep — fall back gracefully if not installed
# ---------------------------------------------------------------------------
try:
    from scipy import stats as _scipy_stats
    def _ttest(a, b):
        _, p = _scipy_stats.ttest_ind(a, b)
        return p
except ModuleNotFoundError:
    def _ttest(a, b):
        return None
from studies.hle_sycophancy.runner import run_all
from studies.hle_sycophancy.grader import grade_all, compute_state_accuracy
from studies.hle_sycophancy.visualize import generate_us_map, generate_bar_chart
from studies.hle_sycophancy.cost_tracker import CostTracker

# 2024 presidential election results (from political_sycophancy study)
RED_STATES_2024 = [
    "Alabama", "Alaska", "Arkansas", "Florida", "Georgia", "Idaho", "Indiana",
    "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Michigan",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Pennsylvania",
    "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
    "West Virginia", "Wisconsin", "Wyoming", "Arizona",
]
BLUE_STATES_2024 = [
    "California", "Colorado", "Connecticut", "Delaware", "Hawaii", "Illinois",
    "Maryland", "Massachusetts", "Minnesota", "New Hampshire", "New Jersey",
    "New Mexico", "New York", "Oregon", "Rhode Island", "Vermont", "Virginia",
    "Washington",
]


# ---------------------------------------------------------------------------
# Question loading (with disk cache)
# ---------------------------------------------------------------------------

def get_questions() -> list[dict]:
    """Return HLE questions, downloading and caching on first run.

    Cached by (seed, per_category) — same config always produces the same
    question set across models and across runs.
    """
    cached = load_hle_questions(HLE_SEED, HLE_PER_CATEGORY)
    if cached is not None:
        print(f"  Loaded {len(cached)} HLE questions from cache "
              f"(seed={HLE_SEED}, per_category={HLE_PER_CATEGORY}).")
        return cached

    questions = load_hle_stratified_sample(per_category=HLE_PER_CATEGORY, seed=HLE_SEED)
    save_hle_questions(HLE_SEED, HLE_PER_CATEGORY, questions)
    return questions


# ---------------------------------------------------------------------------
# Model selection menu
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

def print_summary(questions: list[dict]):
    for model in MODELS:
        accuracy = compute_state_accuracy(model, questions)
        valid = {s: v for s, v in accuracy.items() if v is not None}
        if not valid:
            continue

        short = model.split("/")[-1]
        print(f"\n{'=' * 60}")
        print(f"  {short} — HLE Accuracy by State")
        print(f"{'=' * 60}")

        sorted_states = sorted(valid.items(), key=lambda x: x[1])
        for state, acc in sorted_states:
            bar_len = int(acc * 30)
            bar = "█" * bar_len
            print(f"  {state:>20}  {bar:<30}  {acc:.1%}")

        mean = sum(valid.values()) / len(valid)
        spread = max(valid.values()) - min(valid.values())

        # Red/blue state comparison
        red_vals  = [valid[s] for s in RED_STATES_2024 if s in valid]
        blue_vals = [valid[s] for s in BLUE_STATES_2024 if s in valid]
        red_mean  = sum(red_vals)  / len(red_vals)  if red_vals  else None
        blue_mean = sum(blue_vals) / len(blue_vals) if blue_vals else None

        print(f"\n  Overall mean: {mean:.1%}")
        print(f"  Spread (max - min): {spread:.1%}")
        if red_mean is not None and blue_mean is not None:
            print(f"  Red state mean:  {red_mean:.1%}")
            print(f"  Blue state mean: {blue_mean:.1%}")
            if len(red_vals) >= 2 and len(blue_vals) >= 2:
                p_val = _ttest(red_vals, blue_vals)
                if p_val is not None:
                    sig = f"p={p_val:.4f}" if p_val >= 0.0001 else f"p={p_val:.2e}"
                    print(f"  t-test red vs blue: {sig}")
        print(f"  States with data: {len(valid)}/{len(STATES)}")


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def generate_graphs(models: list[str], questions: list[dict]):
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    for model in models:
        accuracy = compute_state_accuracy(model, questions)
        valid = {s: v for s, v in accuracy.items() if v is not None}
        if not valid:
            print(f"  No valid results for {model}")
            continue

        short = model.split("/")[-1]
        safe  = model.replace("/", "_")

        # Absolute map: fixed 0–1 scale
        generate_us_map(
            state_accuracy=valid,
            title=f"HLE Accuracy by State (absolute): {short}",
            save_path=GRAPHS_DIR / f"us_map_abs_{safe}.png",
            score_range=(0.0, 1.0),
        )

        # Relative map: scaled to actual min/max
        sorted_vals = sorted(valid.values())
        vmin = sorted_vals[0]
        vmax = sorted_vals[-1]
        generate_us_map(
            state_accuracy=valid,
            title=f"HLE Accuracy by State (relative): {short}",
            save_path=GRAPHS_DIR / f"us_map_rel_{safe}.png",
            score_range=(vmin, vmax),
        )

        # Relative without outliers: trim 2 most extreme values from each end
        trim = min(2, len(sorted_vals) // 2)
        trimmed_min = sorted_vals[trim]
        trimmed_max = sorted_vals[-1 - trim]
        if trimmed_min < trimmed_max:
            red_vals  = [valid[s] for s in RED_STATES_2024 if s in valid]
            blue_vals = [valid[s] for s in BLUE_STATES_2024 if s in valid]
            red_mean  = sum(red_vals)  / len(red_vals)  if red_vals  else 0.0
            blue_mean = sum(blue_vals) / len(blue_vals) if blue_vals else 0.0
            if len(red_vals) >= 2 and len(blue_vals) >= 2:
                p_val = _ttest(red_vals, blue_vals)
                sig_str = (
                    f"p={p_val:.4f}" if p_val is not None and p_val >= 0.0001
                    else f"p={p_val:.2e}" if p_val is not None
                    else "n/a"
                )
            else:
                sig_str = "n/a"
            subtitle = (
                f"Red mean: {red_mean:.1%} | Blue mean: {blue_mean:.1%} | "
                f"t-test: {sig_str}"
            )
            title = (
                f"HLE Accuracy by State (relative, outliers trimmed): {short}\n"
                f"{subtitle}"
            )
            generate_us_map(
                state_accuracy=valid,
                title=title,
                save_path=GRAPHS_DIR / f"us_map_rel_no_outlier_{safe}.png",
                score_range=(trimmed_min, trimmed_max),
            )

            # Red states only (same scale)
            red_only = {s: valid[s] for s in RED_STATES_2024 if s in valid}
            generate_us_map(
                state_accuracy=red_only,
                title=f"HLE Accuracy — Red States Only: {short}\n{subtitle}",
                save_path=GRAPHS_DIR / f"us_map_rel_no_outlier_{safe}_red_only.png",
                score_range=(trimmed_min, trimmed_max),
            )

            # Blue states only (same scale)
            blue_only = {s: valid[s] for s in BLUE_STATES_2024 if s in valid}
            generate_us_map(
                state_accuracy=blue_only,
                title=f"HLE Accuracy — Blue States Only: {short}\n{subtitle}",
                save_path=GRAPHS_DIR / f"us_map_rel_no_outlier_{safe}_blue_only.png",
                score_range=(trimmed_min, trimmed_max),
            )

        # Bar chart
        generate_bar_chart(
            state_accuracy=valid,
            title=f"HLE Accuracy by State: {short}",
            save_path=GRAPHS_DIR / f"bar_{safe}.png",
        )

        print(f"  Graphs saved for {short}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_experiment(models: list[str], questions: list[dict]):
    provider = OpenRouterProvider()

    # Set up cost tracking
    cost_tracker = CostTracker()
    all_models = list(set(models + [GRADER_MODEL]))
    print("\n--- Fetching model pricing ---")
    await cost_tracker.fetch_pricing(provider.api_key, all_models)

    print("\n--- Step 1: Collecting raw LLM responses ---")
    await run_all(provider, models, questions, cost_tracker=cost_tracker)

    print("\n--- Step 2: Grading answers ---")
    await grade_all(provider, models, questions, cost_tracker=cost_tracker)

    print(f"\n--- Total cost: {cost_tracker.format_total()} ---")

    print("\n--- Step 3: Scoring and visualization ---")
    generate_graphs(models, questions)

    print_summary(questions)


def main():
    questions = get_questions()

    print("\n" + "=" * 60)
    print("  STUDY: HLE Performance Sycophancy in LLMs")
    print("=" * 60)
    print(f"\n  HLE questions: {len(questions)} ({HLE_PER_CATEGORY}/category, seed={HLE_SEED})")
    print(f"  States: {len(STATES)}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Iterations per combo: {ITERATIONS}")
    print(f"  Total queries per model: {len(questions) * len(STATES) * ITERATIONS}")
    print(f"  Results cache: {RESULTS_DIR}/")

    print("\n  Questions sampled:")
    from collections import Counter
    cats = Counter(q["category"] for q in questions)
    for cat, n in sorted(cats.items()):
        print(f"    {cat}: {n}")

    models = select_models()
    if models == "regenerate_graphs":
        generate_graphs(MODELS, questions)
        return
    if models == "print_summary":
        print_summary(questions)
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

    asyncio.run(run_experiment(models, questions))


if __name__ == "__main__":
    main()
