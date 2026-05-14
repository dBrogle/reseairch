"""
Trolley Problem Study

Tests whether LLMs answer the trolley problem differently based on which
U.S. state the user is connecting from. Two variants (lever and push) are
each asked 10 times at temperature 1.0 to capture response variance.

Pipeline:
  1. Run raw queries (model x state x question x 10 iterations) - cached
  2. Extract YES/NO answers via LLM - cached separately
  3. Score (proportion YES) and visualize on US maps
"""

import asyncio

from scipy import stats

from services.llm import OpenRouterProvider
from studies.trolley_problem.config import (
    MODELS,
    STATES,
    QUESTIONS,
    TEMPERATURE,
    ITERATIONS,
)
from studies.trolley_problem.runner import run_all, build_messages
from studies.trolley_problem.extractor import extract_all, compute_state_scores, compute_refusal_rates
from studies.trolley_problem.cache import (
    RESULTS_DIR,
    GRAPHS_DIR,
    load_response_cache,
    load_extraction_cache,
    get_responses,
    get_extraction,
)
from studies.trolley_problem.visualize import (
    generate_us_map,
    generate_refusal_map,
    RED_STATES_2024,
    BLUE_STATES_2024,
)
from studies.trolley_problem.config import EXTRACTOR_MODEL


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
        print(f"  {short} - Trolley Problem Results")
        print(f"{'=' * 60}")

        response_cache = load_response_cache(model)
        extraction_cache = load_extraction_cache(model)

        for question in QUESTIONS:
            # Count answers
            yes_total = 0
            no_total = 0
            refused_total = 0
            errors_total = 0

            for state in STATES:
                messages = build_messages(state, question)
                responses = get_responses(response_cache, messages, TEMPERATURE)[:ITERATIONS]
                for result in responses:
                    if result.get("error") is not None or result.get("response") is None:
                        errors_total += 1
                        continue
                    answer = get_extraction(
                        extraction_cache, question["id"],
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

            print(f"\n  --- {question['variant']} ({question['id']}) ---")
            if total > 0:
                yes_pct = yes_total / answered * 100 if answered else 0
                no_pct = no_total / answered * 100 if answered else 0
                print(f"  YES: {yes_total} ({yes_pct:.1f}%)  NO: {no_total} ({no_pct:.1f}%)  "
                      f"Refused: {refused_total}  Errors: {errors_total}  Total: {total}")

            # Per-state breakdown
            scores = compute_state_scores(model, question_id=question["id"])
            valid = {s: v for s, v in scores.items() if v is not None}
            if not valid:
                print("  No valid state-level data.")
                continue

            sorted_states = sorted(valid.items(), key=lambda x: x[1], reverse=True)
            for state, score in sorted_states:
                bar_len = int(score * 30)
                bar = "#" * bar_len + "." * (30 - bar_len)
                print(f"  {state:>20}  [{bar}] {score:.0%} yes")

            mean = sum(valid.values()) / len(valid)
            spread = max(valid.values()) - min(valid.values())
            print(f"\n  Overall YES rate: {mean:.1%}")
            print(f"  Spread (max - min): {spread:.1%}")

            # Red vs blue state comparison
            red_vals = [valid[s] for s in RED_STATES_2024 if s in valid]
            blue_vals = [valid[s] for s in BLUE_STATES_2024 if s in valid]
            if red_vals and blue_vals:
                red_mean = sum(red_vals) / len(red_vals)
                blue_mean = sum(blue_vals) / len(blue_vals)
                print(f"  Red state mean: {red_mean:.1%}  |  Blue state mean: {blue_mean:.1%}")
                if len(red_vals) >= 2 and len(blue_vals) >= 2:
                    t_stat, p_val = stats.ttest_ind(red_vals, blue_vals)
                    print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}")


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def generate_maps(models: list[str]):
    """Generate maps for each model, per-variant and combined."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    for model in models:
        short = model.split("/")[-1]
        safe = model.replace("/", "_")

        for question in QUESTIONS:
            scores = compute_state_scores(model, question_id=question["id"])
            valid = {s: v for s, v in scores.items() if v is not None}
            if not valid:
                print(f"  No valid results for {model} / {question['id']}")
                continue

            variant_safe = question["id"]

            # Absolute: fixed 0 to 1 scale
            generate_us_map(
                state_scores=valid,
                title=f"Trolley Problem - {question['variant']} (absolute): {short}",
                save_path=GRAPHS_DIR / f"us_map_abs_{variant_safe}_{safe}.png",
                score_range=(0.0, 1.0),
            )

            # Relative: scaled to actual min/max
            sorted_vals = sorted(valid.values())
            vmin, vmax = sorted_vals[0], sorted_vals[-1]
            if vmin < vmax:
                # Compute stats for subtitle
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
                    f"Red mean: {red_mean:.1%} | Blue mean: {blue_mean:.1%} | "
                    f"t-test: {sig_str}"
                )
                title = (
                    f"Trolley Problem - {question['variant']} (relative): {short}\n"
                    f"{subtitle}"
                )
                generate_us_map(
                    state_scores=valid,
                    title=title,
                    save_path=GRAPHS_DIR / f"us_map_rel_{variant_safe}_{safe}.png",
                    score_range=(vmin, vmax),
                )

            print(f"  Maps saved for {short} / {question['variant']}")

        # --- Refusal maps (separate subfolder) ---
        refusal_dir = GRAPHS_DIR / "refusals"
        refusal_dir.mkdir(parents=True, exist_ok=True)

        for question in QUESTIONS:
            rates = compute_refusal_rates(model, question_id=question["id"])
            valid = {s: v for s, v in rates.items() if v is not None}
            if not valid:
                continue

            variant_safe = question["id"]

            # Absolute: fixed 0-1
            generate_refusal_map(
                state_scores=valid,
                title=f"Refusal Rate - {question['variant']} (absolute): {short}",
                save_path=refusal_dir / f"refusal_abs_{variant_safe}_{safe}.png",
                score_range=(0.0, 1.0),
            )

            # Relative: scaled to data range
            sorted_vals = sorted(valid.values())
            vmin, vmax = sorted_vals[0], sorted_vals[-1]
            if vmin < vmax:
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
                    f"Red mean: {red_mean:.1%} | Blue mean: {blue_mean:.1%} | "
                    f"t-test: {sig_str}"
                )
                generate_refusal_map(
                    state_scores=valid,
                    title=f"Refusal Rate - {question['variant']} (relative): {short}\n{subtitle}",
                    save_path=refusal_dir / f"refusal_rel_{variant_safe}_{safe}.png",
                    score_range=(vmin, vmax),
                )

            print(f"  Refusal maps saved for {short} / {question['variant']}")

        # Combined refusal rates
        rates = compute_refusal_rates(model)
        valid = {s: v for s, v in rates.items() if v is not None}
        if valid:
            generate_refusal_map(
                state_scores=valid,
                title=f"Refusal Rate - Combined (absolute): {short}",
                save_path=refusal_dir / f"refusal_abs_combined_{safe}.png",
                score_range=(0.0, 1.0),
            )
            sorted_vals = sorted(valid.values())
            vmin, vmax = sorted_vals[0], sorted_vals[-1]
            if vmin < vmax:
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
                    f"Red mean: {red_mean:.1%} | Blue mean: {blue_mean:.1%} | "
                    f"t-test: {sig_str}"
                )
                generate_refusal_map(
                    state_scores=valid,
                    title=f"Refusal Rate - Combined (relative): {short}\n{subtitle}",
                    save_path=refusal_dir / f"refusal_rel_combined_{safe}.png",
                    score_range=(vmin, vmax),
                )
            print(f"  Combined refusal maps saved for {short}")

        # Combined (both variants averaged)
        scores = compute_state_scores(model)
        valid = {s: v for s, v in scores.items() if v is not None}
        if valid:
            generate_us_map(
                state_scores=valid,
                title=f"Trolley Problem - Combined (absolute): {short}",
                save_path=GRAPHS_DIR / f"us_map_abs_combined_{safe}.png",
                score_range=(0.0, 1.0),
            )
            sorted_vals = sorted(valid.values())
            vmin, vmax = sorted_vals[0], sorted_vals[-1]
            if vmin < vmax:
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
                    f"Red mean: {red_mean:.1%} | Blue mean: {blue_mean:.1%} | "
                    f"t-test: {sig_str}"
                )
                generate_us_map(
                    state_scores=valid,
                    title=f"Trolley Problem - Combined (relative): {short}\n{subtitle}",
                    save_path=GRAPHS_DIR / f"us_map_rel_combined_{safe}.png",
                    score_range=(vmin, vmax),
                )
            print(f"  Combined maps saved for {short}")


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
    generate_maps(models)

    print_summary()


def main():
    print("\n" + "=" * 60)
    print("  STUDY: Trolley Problem by State")
    print("=" * 60)
    print(f"\n  Variants: {len(QUESTIONS)}")
    for q in QUESTIONS:
        print(f"    - {q['variant']}: {q['id']}")
    print(f"  States: {len(STATES)}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Iterations per combo: {ITERATIONS}")
    print(f"  Total queries per model: {len(QUESTIONS) * len(STATES) * ITERATIONS}")
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
