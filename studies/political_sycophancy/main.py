"""
Political Sycophancy Study

Tests whether LLMs shift their stated political opinions based on which
U.S. state the user is from, using Pew Research Center political typology
questions. Each state gets a liberal-to-conservative score derived from
the LLM's answers, then visualized on a US map.

Pipeline:
  1. Run raw queries (model x state x question) — cached
  2. Extract answers via LLM — cached separately
  3. Score and visualize
"""

import asyncio

from scipy import stats

from services.llm import OpenRouterProvider
from studies.political_sycophancy.config import (
    MODELS,
    STATES,
    TEMPERATURE,
    ITERATIONS,
    load_questions,
)
from studies.political_sycophancy.runner import run_all, build_messages
from studies.political_sycophancy.extractor import extract_all, compute_state_scores, score_answer
from studies.political_sycophancy.cache import (
    RESULTS_DIR,
    GRAPHS_DIR,
    load_response_cache,
    load_extraction_cache,
    get_responses,
    get_extraction,
)
from studies.political_sycophancy.visualize import generate_us_map, RED_STATES_2024, BLUE_STATES_2024


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
    questions = load_questions()
    from studies.political_sycophancy.cache import (
        load_response_cache,
        load_extraction_cache,
        get_responses,
        get_extraction,
    )
    from studies.political_sycophancy.config import EXTRACTOR_MODEL
    from studies.political_sycophancy.runner import build_messages

    for model in MODELS:
        scores = compute_state_scores(model)
        valid = {s: v for s, v in scores.items() if v is not None}
        if not valid:
            continue

        # Count refusals/errors
        response_cache = load_response_cache(model)
        extraction_cache = load_extraction_cache(model)
        total_extractions = 0
        refused = 0
        errors = 0
        answered = 0
        for state in STATES:
            for question in questions:
                messages = build_messages(state, question)
                responses = get_responses(response_cache, messages, TEMPERATURE)[:ITERATIONS]
                for result in responses:
                    if result.get("error") is not None or result.get("response") is None:
                        errors += 1
                        total_extractions += 1
                        continue
                    answer = get_extraction(
                        extraction_cache, question["id"],
                        result["response"], EXTRACTOR_MODEL,
                    )
                    total_extractions += 1
                    if answer is None or answer == "ERROR":
                        errors += 1
                    elif answer == "REFUSED":
                        refused += 1
                    else:
                        answered += 1

        short = model.split("/")[-1]
        print(f"\n{'=' * 60}")
        print(f"  {short} — Political Lean by State")
        print(f"{'=' * 60}")

        if total_extractions > 0:
            pct_answered = answered / total_extractions * 100
            pct_refused = refused / total_extractions * 100
            pct_errors = errors / total_extractions * 100
            print(f"\n  Responses: {answered} answered ({pct_answered:.0f}%), "
                  f"{refused} refused ({pct_refused:.0f}%), "
                  f"{errors} errors ({pct_errors:.0f}%) "
                  f"— {total_extractions} total")

        sorted_states = sorted(valid.items(), key=lambda x: x[1])
        for state, score in sorted_states:
            bar_len = int(abs(score) * 20)
            if score < 0:
                bar = " " * (20 - bar_len) + "#" * bar_len + " |"
                label = "liberal"
            else:
                bar = "| " + "#" * bar_len
                label = "conservative"
            print(f"  {state:>20}  {bar}  {score:+.3f} ({label})")

        mean = sum(valid.values()) / len(valid)
        spread = max(valid.values()) - min(valid.values())
        print(f"\n  Overall mean: {mean:+.3f}")
        print(f"  Spread (max - min): {spread:.3f}")
        print(f"  States with data: {len(valid)}/{len(STATES)}")


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def generate_maps(models: list[str]):
    """Generate absolute and relative maps for each model."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    for model in models:
        scores = compute_state_scores(model)
        valid = {s: v for s, v in scores.items() if v is not None}
        if not valid:
            print(f"  No valid results for {model}")
            continue

        short = model.split("/")[-1]
        safe = model.replace("/", "_")

        # Absolute: fixed -1 to +1 scale
        generate_us_map(
            state_scores=valid,
            title=f"Political Lean by State (absolute): {short}",
            save_path=GRAPHS_DIR / f"us_map_abs_{safe}.png",
            score_range=(-1.0, 1.0),
        )

        # Relative: scaled to actual min/max in the data
        sorted_vals = sorted(valid.values())
        vmin = sorted_vals[0]
        vmax = sorted_vals[-1]
        generate_us_map(
            state_scores=valid,
            title=f"Political Lean by State (relative): {short}",
            save_path=GRAPHS_DIR / f"us_map_rel_{safe}.png",
            score_range=(vmin, vmax),
        )

        # Relative without outliers: trim the 2 most extreme values from each end
        trim = min(2, len(sorted_vals) // 2)
        trimmed_min = sorted_vals[trim]
        trimmed_max = sorted_vals[-1 - trim]
        print(f"    DEBUG {short}: trimmed range = [{trimmed_min:.4f}, {trimmed_max:.4f}]")
        # If the ends are on opposite sides of 0 and magnitudes are within
        # 25% of each other, center at 0 using the smaller magnitude
        if trimmed_min < 0 < trimmed_max:
            abs_min = abs(trimmed_min)
            abs_max = abs(trimmed_max)
            ratio = min(abs_min, abs_max) / max(abs_min, abs_max)
            if ratio >= 0.75:
                symmetric = min(abs_min, abs_max)
                trimmed_min = -symmetric
                trimmed_max = symmetric
        if trimmed_min < trimmed_max:
            # Compute red/blue state means and significance
            red_vals = [valid[s] for s in RED_STATES_2024 if s in valid]
            blue_vals = [valid[s] for s in BLUE_STATES_2024 if s in valid]
            red_mean = sum(red_vals) / len(red_vals) if red_vals else 0
            blue_mean = sum(blue_vals) / len(blue_vals) if blue_vals else 0
            if len(red_vals) >= 2 and len(blue_vals) >= 2:
                t_stat, p_val = stats.ttest_ind(red_vals, blue_vals)
                sig_str = f"p={p_val:.4f}" if p_val >= 0.0001 else f"p={p_val:.2e}"
            else:
                sig_str = "n/a"
            subtitle = (
                f"Red mean: {red_mean:+.3f} | Blue mean: {blue_mean:+.3f} | "
                f"t-test: {sig_str}"
            )
            title = (
                f"Political Lean by State (relative, outliers trimmed): {short}\n"
                f"{subtitle}"
            )
            generate_us_map(
                state_scores=valid,
                title=title,
                save_path=GRAPHS_DIR / f"us_map_rel_no_outlier_{safe}.png",
                score_range=(trimmed_min, trimmed_max),
            )

        print(f"  Maps saved for {short} (abs + rel + rel_no_outlier)")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_experiment(models: list[str]):
    provider = OpenRouterProvider()

    # Step 1: Raw responses
    print("\n--- Step 1: Collecting raw LLM responses ---")
    await run_all(provider, models)

    # Step 2: Extract answers
    print("\n--- Step 2: Extracting answers ---")
    await extract_all(provider, models)

    # Step 3: Score and visualize
    print("\n--- Step 3: Scoring and visualization ---")
    generate_maps(models)

    print_summary()


def main():
    questions = load_questions()
    print("\n" + "=" * 60)
    print("  STUDY: Political Sycophancy in LLMs")
    print("=" * 60)
    print(f"\n  Questions: {len(questions)}")
    print(f"  States: {len(STATES)}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Iterations per combo: {ITERATIONS}")
    print(f"  Total queries per model: {len(questions) * len(STATES) * ITERATIONS}")
    print(f"  Results cache: {RESULTS_DIR}/")

    models = select_models()
    if models == "regenerate_graphs":
        generate_maps(MODELS)
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
