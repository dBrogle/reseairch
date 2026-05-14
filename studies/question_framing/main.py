"""
Question Framing Study

Tests whether the psychological framing of a question affects LLM accuracy
on hard academic questions from the HLE benchmark. Each framing is grounded
in a real psychological phenomenon (Pygmalion effect, prospect theory,
reactance theory, Yerkes-Dodson law, etc.).

Pipeline:
  1. Load (or download + cache) HLE questions
  2. Run raw queries (model x framing x question) — cached per model
  3. Grade answers via LLM — cached separately per model
  4. Compute per-framing accuracy and visualize
"""

import asyncio

from services.llm import OpenRouterProvider
from studies.question_framing.config import (
    MODELS,
    FRAMINGS,
    FRAMING_KEYS,
    FRAMING_SUFFIXES,
    FRAMING_DESCRIPTIONS,
    TEMPERATURE,
    ITERATIONS,
    HLE_SEED,
    HLE_PER_CATEGORY,
    GRADER_MODEL,
)
from studies.question_framing.cache import RESULTS_DIR, GRAPHS_DIR
from studies.question_framing.runner import run_all
from studies.question_framing.grader import (
    grade_all,
    compute_framing_accuracy,
    compute_aggregated_framing_accuracy,
)
from studies.question_framing.visualize import (
    generate_model_bar_chart,
    generate_grouped_bar_chart,
    generate_heatmap,
    generate_delta_chart,
    generate_aggregated_framing_chart,
    generate_significance_chart,
)
from studies.question_framing.significance import (
    build_per_model_matrix,
    build_pooled_matrix,
    compute_significance,
)
from studies.question_framing.cost_tracker import CostTracker

# Reuse HLE data loader from hle_sycophancy
from studies.hle_sycophancy.hle_data import load_hle_stratified_sample
from studies.hle_sycophancy.cache import (
    load_hle_questions,
    save_hle_questions,
)


# ---------------------------------------------------------------------------
# Question loading (reuses hle_sycophancy cache)
# ---------------------------------------------------------------------------

def get_questions() -> list[dict]:
    """Return HLE questions, downloading and caching on first run.

    Shares the same cache as hle_sycophancy — same (seed, per_category) config
    always produces the same question set.

    If no cache exists for the requested per_category but a smaller one does,
    extend the smaller sample so existing response caches remain valid.
    """
    cached = load_hle_questions(HLE_SEED, HLE_PER_CATEGORY)
    if cached is not None:
        print(f"  Loaded {len(cached)} HLE questions from cache "
              f"(seed={HLE_SEED}, per_category={HLE_PER_CATEGORY}).")
        return cached

    base = None
    for n in range(HLE_PER_CATEGORY - 1, 0, -1):
        prior = load_hle_questions(HLE_SEED, n)
        if prior is not None:
            base = prior
            print(f"  Extending cached per_category={n} sample "
                  f"({len(prior)} questions) → per_category={HLE_PER_CATEGORY}.")
            break

    questions = load_hle_stratified_sample(
        per_category=HLE_PER_CATEGORY, seed=HLE_SEED, base_questions=base,
    )
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
        accuracy = compute_framing_accuracy(model, questions)
        valid = {k: v for k, v in accuracy.items() if v is not None}
        if not valid:
            continue

        short = model.split("/")[-1]
        control_stats = valid.get("control")
        control_acc = control_stats["accuracy"] if control_stats else None

        print(f"\n{'=' * 78}")
        print(f"  {short} — HLE Accuracy by Question Framing (95% Wilson CIs)")
        print(f"{'=' * 78}")

        sorted_framings = sorted(
            valid.items(), key=lambda x: x[1]["accuracy"], reverse=True
        )
        for framing_key, stats in sorted_framings:
            acc = stats["accuracy"]
            lo, hi = stats["ci_low"], stats["ci_high"]
            n = stats["n_total"]
            bar_len = int(acc * 20)
            bar = "█" * bar_len
            delta = ""
            if control_acc is not None and framing_key != "control":
                diff = acc - control_acc
                delta = f"  (Δ {diff:+.1%})"
            suffix = FRAMING_SUFFIXES[framing_key]
            label = suffix if suffix else "(control)"
            print(
                f"  {label:>55}  {bar:<20}  {acc:.1%} "
                f"[{lo:.1%}, {hi:.1%}] n={n}{delta}"
            )

        mean = sum(s["accuracy"] for s in valid.values()) / len(valid)
        accs = [s["accuracy"] for s in valid.values()]
        spread = max(accs) - min(accs)
        print(f"\n  Overall mean: {mean:.1%}")
        print(f"  Spread (max - min): {spread:.1%}")
        if control_acc is not None:
            lo, hi = control_stats["ci_low"], control_stats["ci_high"]
            print(f"  Control baseline: {control_acc:.1%} [{lo:.1%}, {hi:.1%}]")

        # Best and worst framings
        best_key, best_stats = sorted_framings[0]
        worst_key, worst_stats = sorted_framings[-1]
        print(
            f"  Best framing:  {FRAMING_SUFFIXES[best_key] or '(control)'} "
            f"({best_stats['accuracy']:.1%})"
        )
        print(
            f"  Worst framing: {FRAMING_SUFFIXES[worst_key] or '(control)'} "
            f"({worst_stats['accuracy']:.1%})"
        )
        print(f"  Framings with data: {len(valid)}/{len(FRAMING_KEYS)}")


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def generate_graphs(models: list[str], questions: list[dict]):
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    all_accuracy: dict[str, dict[str, dict | None]] = {}

    for model in models:
        accuracy = compute_framing_accuracy(model, questions)
        valid = {k: v for k, v in accuracy.items() if v is not None}
        if not valid:
            print(f"  No valid results for {model}")
            continue

        all_accuracy[model] = accuracy
        short = model.split("/")[-1]
        safe  = model.replace("/", "_")

        # Per-model bar chart
        generate_model_bar_chart(
            framing_accuracy=accuracy,
            title=f"HLE Accuracy by Question Framing: {short}",
            save_path=GRAPHS_DIR / f"bar_{safe}.png",
        )
        print(f"  Bar chart saved for {short}")

        # Per-model significance chart
        matrix, kept = build_per_model_matrix(model, questions, FRAMING_KEYS)
        if matrix:
            sig = compute_significance(matrix, FRAMING_KEYS, control_key="control")
            generate_significance_chart(
                sig=sig,
                title=f"Framing Significance: {short}",
                save_path=GRAPHS_DIR / f"significance_{safe}.png",
            )
            print(f"  Significance chart saved for {short} "
                  f"(n={len(kept)} complete-case questions)")
        else:
            print(f"  Skipping significance chart for {short} — no complete-case questions")

    if len(all_accuracy) >= 2:
        # Aggregated bar chart — accuracy per framing, pooled across all models
        aggregated = compute_aggregated_framing_accuracy(
            list(all_accuracy.keys()), questions
        )
        generate_aggregated_framing_chart(
            aggregated=aggregated,
            title="HLE Accuracy by Question Framing (pooled across all models)",
            save_path=GRAPHS_DIR / "aggregated_framings.png",
        )
        print("  Aggregated framing chart saved")

        # Pooled significance chart — each (model, question) treated as a subject
        pooled_matrix, pooled_keys = build_pooled_matrix(
            list(all_accuracy.keys()), questions, FRAMING_KEYS,
        )
        if pooled_matrix:
            pooled_sig = compute_significance(
                pooled_matrix, FRAMING_KEYS, control_key="control"
            )
            generate_significance_chart(
                sig=pooled_sig,
                title="Framing Significance — pooled across all models "
                      "(each row = one model × question)",
                save_path=GRAPHS_DIR / "significance_pooled.png",
            )
            print(f"  Pooled significance chart saved "
                  f"(n={len(pooled_keys)} model-question subjects)")

        # Grouped bar chart (all models)
        generate_grouped_bar_chart(
            all_accuracy=all_accuracy,
            title="HLE Accuracy by Question Framing (all models)",
            save_path=GRAPHS_DIR / "grouped_bar_all_models.png",
        )
        print("  Grouped bar chart saved")

        # Heatmap
        generate_heatmap(
            all_accuracy=all_accuracy,
            title="HLE Accuracy: Model x Framing",
            save_path=GRAPHS_DIR / "heatmap.png",
        )
        print("  Heatmap saved")

        # Delta chart (vs control)
        generate_delta_chart(
            all_accuracy=all_accuracy,
            title="Accuracy Delta vs. Control by Framing",
            save_path=GRAPHS_DIR / "delta_vs_control.png",
        )
        print("  Delta chart saved")


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

    print("\n" + "=" * 70)
    print("  STUDY: Question Framing Effects on LLM Accuracy")
    print("=" * 70)
    print(f"\n  HLE questions: {len(questions)} ({HLE_PER_CATEGORY}/category, seed={HLE_SEED})")
    print(f"  Framings: {len(FRAMING_KEYS)}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Iterations per combo: {ITERATIONS}")
    print(f"  Total queries per model: {len(questions) * len(FRAMING_KEYS) * ITERATIONS}")
    print(f"  Results cache: {RESULTS_DIR}/")

    print("\n  Framings:")
    for key, suffix, desc in FRAMINGS:
        label = f'"{suffix}"' if suffix else "(no suffix — control)"
        print(f"    {key:>22}: {label}")
        print(f"    {'':>22}  {desc}")

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
