"""
Idea Priming Study

Tests whether asking an LLM "why might this be a good idea?" vs. "why might
this be a bad idea?" — followed by an identical 1-10 quality score request —
shifts the model's overall score for the same idea. Both frames use the SAME
1-10 quality scale (10 = excellent), so the difference is the priming bias.

Pipeline:
  1. Run raw queries (model x frame x idea x iteration) — cached per model
  2. Extract {reasoning, score} from raw responses — cached per model
  3. Compute per-model and per-idea stats with 95% CIs
  4. Generate visualizations (per-model bars, paired-dot plots, headline bias)
"""

import asyncio
from collections import Counter

from services.llm import OpenRouterProvider
from studies.idea_priming.config import (
    BUCKETS,
    EXTRACTOR_MODEL,
    FRAME_KEYS,
    FRAME_LABELS,
    IDEAS,
    IDEA_BY_ID,
    ITERATIONS,
    MODELS,
    SMOKE_IDEA_IDS,
    SMOKE_ITERATIONS,
    SMOKE_MODELS,
    TEMPERATURE,
)
from studies.idea_priming.cache import GRAPHS_DIR
from studies.idea_priming.runner import run_all
from studies.idea_priming.extractor import compute_model_stats, extract_all
from studies.idea_priming.visualize import (
    generate_dumbbell_with_logos,
    generate_grouped_bar,
    generate_headline_bias,
    generate_ideas_list_card,
    generate_paired_dots,
    generate_per_model_bar,
    generate_shorts_card,
)
# Reuse the cost tracker from question_framing (same shape, no duplication)
from studies.question_framing.cost_tracker import CostTracker


# ---------------------------------------------------------------------------
# Selection menu
# ---------------------------------------------------------------------------

def select_run_mode() -> dict | str:
    """Return either a config dict (models, ideas, iterations) or a sentinel."""
    print("\n=== Run Mode ===")
    print("[1] Full run — all models, all ideas, all iterations")
    print(f"    ({len(MODELS)} models × {len(IDEAS)} ideas × 2 frames × "
          f"{ITERATIONS} iters = {len(MODELS) * len(IDEAS) * 2 * ITERATIONS} calls)")
    print("[2] Smoke test — 1 model, 6 ideas, 2 iters "
          f"({len(SMOKE_MODELS) * len(SMOKE_IDEA_IDS) * 2 * SMOKE_ITERATIONS} calls)")
    print("[3] Pick individual models (full ideas + iterations)")
    print("[4] Regenerate graphs from cached results")
    print("[5] Print results summary")

    choice = input("\nChoice: ").strip()

    if choice == "4":
        return "regenerate_graphs"
    if choice == "5":
        return "print_summary"
    if choice == "2":
        return {
            "models": SMOKE_MODELS,
            "ideas":  [IDEA_BY_ID[i] for i in SMOKE_IDEA_IDS],
            "iterations": SMOKE_ITERATIONS,
        }
    if choice == "3":
        print("\nModels:")
        for i, m in enumerate(MODELS, 1):
            print(f"  [{i}] {m}")
        picks = input("Select models (comma-separated numbers): ").strip()
        selected = []
        for p in picks.split(","):
            try:
                idx = int(p.strip()) - 1
            except ValueError:
                continue
            if 0 <= idx < len(MODELS):
                selected.append(MODELS[idx])
        if not selected:
            print("No valid models picked.")
            return "abort"
        return {"models": selected, "ideas": IDEAS, "iterations": ITERATIONS}

    # Default to full run
    return {"models": MODELS, "ideas": IDEAS, "iterations": ITERATIONS}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(models: list[str], ideas: list[dict], iterations: int):
    print("\n" + "=" * 78)
    print("  IDEA PRIMING — Summary")
    print("=" * 78)

    all_stats = {m: compute_model_stats(m, ideas, TEMPERATURE, iterations) for m in models}

    # Per-model overview
    for model in models:
        stats = all_stats[model]
        pooled = stats["frame_pooled"]
        bias = stats["model_bias"]
        short = model.split("/")[-1]

        print(f"\n  {short}")
        print(f"  {'-' * (len(short) + 2)}")
        for fk in FRAME_KEYS:
            p = pooled.get(fk)
            if p is None:
                print(f"    {FRAME_LABELS[fk]:>20}: (no data)")
                continue
            print(
                f"    {FRAME_LABELS[fk]:>20}: {p['mean']:.2f} "
                f"[{p['ci_low']:.2f}, {p['ci_high']:.2f}]  n={p['n']}"
            )
        if bias is not None:
            print(
                f"    {'Priming bias':>20}: {bias['mean']:+.2f} "
                f"[{bias['ci_low']:+.2f}, {bias['ci_high']:+.2f}]  "
                f"({bias['n_ideas']} ideas)"
            )

    # Per-bucket biases (across models)
    print(f"\n  {'=' * 74}")
    print(f"  Per-bucket bias (avg across models, ideas)")
    print(f"  {'-' * 74}")
    for bucket in BUCKETS:
        bucket_ideas = [i for i in ideas if i["bucket"] == bucket]
        if not bucket_ideas:
            continue
        all_biases = []
        for model in models:
            for idea in bucket_ideas:
                pi = all_stats[model]["per_idea"].get(idea["id"])
                if pi and pi.get("bias") is not None:
                    all_biases.append(pi["bias"])
        if not all_biases:
            continue
        avg = sum(all_biases) / len(all_biases)
        print(
            f"    {bucket:>10} ({len(bucket_ideas)} ideas): "
            f"mean Δ = {avg:+.2f}  (n={len(all_biases)} model-idea pairs)"
        )


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def generate_graphs(models: list[str], ideas: list[dict], iterations: int):
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    all_stats: dict[str, dict] = {}
    for model in models:
        s = compute_model_stats(model, ideas, TEMPERATURE, iterations)
        if s["frame_pooled"].get("positive") is None or s["frame_pooled"].get("negative") is None:
            print(f"  No valid stats for {model}, skipping.")
            continue
        all_stats[model] = s

        safe = model.replace("/", "_")
        generate_per_model_bar(model, s, GRAPHS_DIR / f"bar_{safe}.png")
        print(f"  Per-model bar saved for {model.split('/')[-1]}")

        generate_paired_dots(model, s, GRAPHS_DIR / f"paired_{safe}.png")
        print(f"  Paired-dot chart saved for {model.split('/')[-1]}")

    if all_stats:
        generate_grouped_bar(all_stats, GRAPHS_DIR / "grouped_bar_all_models.png")
        print("  Grouped bar chart saved")

        generate_headline_bias(all_stats, GRAPHS_DIR / "headline_bias.png")
        print("  Headline bias chart saved")

        generate_dumbbell_with_logos(all_stats, GRAPHS_DIR / "dumbbell_with_logos.png")
        print("  Dumbbell-with-logos chart saved")

        generate_shorts_card(all_stats, GRAPHS_DIR / "shorts_card.png")
        print("  Instagram Shorts card saved")

    # Idea list card doesn't depend on results, just config — always render
    generate_ideas_list_card(GRAPHS_DIR / "ideas_list_card.png")
    print("  Ideas list card saved")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_experiment(models: list[str], ideas: list[dict], iterations: int):
    provider = OpenRouterProvider()

    cost_tracker = CostTracker()
    all_models = list(set(models + [EXTRACTOR_MODEL]))
    print("\n--- Fetching model pricing ---")
    await cost_tracker.fetch_pricing(provider.api_key, all_models)

    print("\n--- Step 1: Collecting raw LLM responses ---")
    await run_all(
        provider, models, ideas,
        temperature=TEMPERATURE, iterations=iterations,
        cost_tracker=cost_tracker,
    )

    print("\n--- Step 2: Extracting {reasoning, score} ---")
    await extract_all(
        provider, models, ideas,
        temperature=TEMPERATURE, iterations=iterations,
        cost_tracker=cost_tracker,
    )

    print(f"\n--- Total cost: {cost_tracker.format_total()} ---")

    print("\n--- Step 3: Stats and visualizations ---")
    generate_graphs(models, ideas, iterations)

    print_summary(models, ideas, iterations)


def main():
    print("\n" + "=" * 70)
    print("  STUDY: Idea Priming Effects on LLM Quality Scores")
    print("=" * 70)
    print(f"\n  Idea pool: {len(IDEAS)} ideas")
    bucket_counts = Counter(i["bucket"] for i in IDEAS)
    for b in BUCKETS:
        print(f"    {b:>10}: {bucket_counts[b]}")
    print(f"  Frames: {len(FRAME_KEYS)} ({', '.join(FRAME_KEYS)})")
    print(f"  Default models: {len(MODELS)}")
    print(f"  Default iterations: {ITERATIONS} per (model, idea, frame)")
    print(f"  Temperature: {TEMPERATURE}")

    cfg = select_run_mode()
    if cfg == "regenerate_graphs":
        generate_graphs(MODELS, IDEAS, ITERATIONS)
        return
    if cfg == "print_summary":
        print_summary(MODELS, IDEAS, ITERATIONS)
        return
    if cfg == "abort":
        print("Cancelled.")
        return

    models = cfg["models"]
    ideas = cfg["ideas"]
    iterations = cfg["iterations"]

    print(f"\nWill test {len(models)} model(s) on {len(ideas)} idea(s), "
          f"{iterations} iter(s) per frame:")
    for m in models:
        print(f"  - {m}")
    print("  Ideas:")
    for i in ideas:
        print(f"    [{i['bucket']:>10}] {i['id']} — {i['category']}")

    confirm = input("\nProceed? [Y/n]: ").strip().lower()
    if confirm == "n":
        print("Cancelled.")
        return

    asyncio.run(run_experiment(models, ideas, iterations))


if __name__ == "__main__":
    main()
