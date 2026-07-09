"""
Chief Keef Framing Study.

Tests whether an LLM's expressed opinion of a real person is driven by how the
background is *framed* rather than by the fixed question asked. The same closing
question — "How do you feel about Chief Keef?" — is posed under three framings:

  1. positive  — a flattering background
  2. neutral   — no background at all
  3. negative  — a damning background

Each (model, arm) is run ITERATIONS times at the model's default temperature
(no sweep, no system prompt). An LLM judge, blind to the framing, scores the
favorability (0-10) expressed in each answer, so any cross-arm difference is a
framing effect.

Results are cached per-model in JSON files keyed by a hash of the message list,
so runs can be interrupted and resumed without losing or duplicating work.
"""

import asyncio
import json
import statistics
from collections import Counter

from services.llm import OpenRouterProvider
from studies.chief_keef_framing.config import (
    MODELS,
    ALL_MODELS,
    ARMS,
    ARM_KEYS,
    ARMS_BY_KEY,
    ITERATIONS,
    get_arm_messages,
)
from studies.chief_keef_framing.cache import (
    GRAPHS_DIR,
    RESULTS_DIR,
    load_cache,
    get_results,
)
from studies.chief_keef_framing.runner import run_all
from studies.chief_keef_framing.judge import run_judge
from studies.chief_keef_framing.charts import (
    short,
    grouped_favorability_chart,
    per_model_arm_chart,
    framing_swing_chart,
)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _favorabilities(results: list[dict]) -> list[int]:
    """Valid, judged favorability scores from a result list."""
    return [
        r["judge_favorability"]
        for r in results
        if r.get("error") is None
        and r.get("response") is not None
        and isinstance(r.get("judge_favorability"), int)
    ]


def arm_stat(results: list[dict]) -> dict:
    """Mean / SEM / n of favorability for one arm's results."""
    favs = _favorabilities(results)
    n = len(favs)
    if n == 0:
        return {"mean": None, "sem": 0.0, "n": 0}
    mean = statistics.fmean(favs)
    sem = (statistics.stdev(favs) / (n ** 0.5)) if n > 1 else 0.0
    return {"mean": mean, "sem": sem, "n": n}


def stance_breakdown(results: list[dict]) -> Counter:
    """Count how often each stance label appears in an arm."""
    counter: Counter = Counter()
    for r in results:
        if r.get("error") is not None or r.get("response") is None:
            continue
        counter[r.get("judge_stance", "unknown")] += 1
    return counter


def compute_stats(all_caches: dict[str, dict]) -> dict[str, dict]:
    """{model: {arm_key: arm_stat}} for every model with data."""
    stats: dict[str, dict] = {}
    for model, cache in all_caches.items():
        if not cache:
            continue
        per_arm = {}
        for arm in ARMS:
            results = get_results(cache, arm["messages"])[:ITERATIONS]
            per_arm[arm["key"]] = arm_stat(results)
        if any(per_arm[k]["n"] for k in ARM_KEYS):
            stats[model] = per_arm
    return stats


def swing(per_arm: dict) -> float:
    """Positive-arm minus negative-arm favorability (the framing swing)."""
    pos = per_arm["positive"]["mean"]
    neg = per_arm["negative"]["mean"]
    if pos is None or neg is None:
        return 0.0
    return pos - neg


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------

def generate_graphs(all_caches: dict[str, dict]):
    """Grouped favorability chart, per-model arm charts, and the swing chart."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    stats = compute_stats(all_caches)
    if not stats:
        print("No judged results yet. Run the experiment (and judge) first.")
        return

    # Order models the way they're declared in config.
    ordered = [m for m in ALL_MODELS if m in stats]
    stats = {m: stats[m] for m in ordered}

    n = max(
        (stats[m][k]["n"] for m in stats for k in ARM_KEYS),
        default=0,
    )

    grouped_favorability_chart(
        stats=stats,
        save_path=GRAPHS_DIR / "favorability_by_arm.png",
        title="Does framing change how LLMs feel about Chief Keef?",
        subtitle=f"same closing question, three framings · judge-scored 0–10 · "
                 f"model default temp · up to n={n} per arm",
    )

    for model_id in stats:
        arm_means = {k: stats[model_id][k]["mean"] for k in ARM_KEYS}
        safe = model_id.replace("/", "_")
        per_model_arm_chart(
            model_id=model_id,
            arm_means=arm_means,
            save_path=GRAPHS_DIR / f"arms_{safe}.png",
        )

    swings = {m: swing(stats[m]) for m in stats}
    framing_swing_chart(
        swings=swings,
        save_path=GRAPHS_DIR / "framing_swing.png",
        title="Framing susceptibility on Chief Keef",
        subtitle="how many favorability points each model swings from positive to negative framing",
    )

    print(f"\nGraphs saved to {GRAPHS_DIR}/")


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def select_models() -> list[str] | str:
    """Interactive model selection menu."""
    print("\n=== Model Selection ===")
    print("[1] Run ALL models")
    print("[2] Select individual models")
    print("[3] Regenerate graphs from cached results")
    print("[4] Run LLM judge on cached results")
    print("[5] Export JSON for frontend")
    print("[6] Print results summary")

    choice = input("\nChoice: ").strip()

    menu = {
        "3": "regenerate_graphs",
        "4": "run_judge",
        "5": "export_json",
        "6": "print_summary",
    }
    if choice in menu:
        return menu[choice]

    if choice == "1":
        return ALL_MODELS

    if choice == "2":
        print("\nModels:")
        for i, model in enumerate(ALL_MODELS, 1):
            print(f"  [{i}] {model}")
        picks = input("Select models (comma-separated numbers): ").strip()
        selected = []
        for p in picks.split(","):
            idx = int(p.strip()) - 1
            if 0 <= idx < len(ALL_MODELS):
                selected.append(ALL_MODELS[idx])
        return selected

    print("Invalid choice, running all models.")
    return ALL_MODELS


# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------

def _load_all_caches() -> dict[str, dict]:
    """Load caches for all models that have data."""
    return {m: load_cache(m) for m in ALL_MODELS if load_cache(m)}


def regenerate_graphs_from_cache():
    all_caches = _load_all_caches()
    if not all_caches:
        print("No cached results found. Run the experiment first.")
        return
    print(f"Regenerating graphs for {len(all_caches)} model(s)...")
    generate_graphs(all_caches)


def export_frontend_json():
    """Export results as a structured JSON file for the frontend web app."""
    all_caches = _load_all_caches()
    if not all_caches:
        print("No cached results found. Run the experiment first.")
        return

    stats = compute_stats(all_caches)
    model_to_company = {mid: company for company, mids in MODELS.items() for mid in mids}
    graph_files = sorted(p.name for p in GRAPHS_DIR.glob("*.png")) if GRAPHS_DIR.exists() else []

    output = {
        "_meta": {
            "study": "chief_keef_framing",
            "description": (
                "Tests whether an LLM's expressed opinion of Chief Keef is driven by how the "
                "background is framed (positive / none / negative) rather than the fixed closing "
                "question. An LLM judge, blind to the framing, scores favorability 0-10; the swing "
                "between the positive and negative arms measures framing susceptibility."
            ),
            "question": "How do you feel about Chief Keef?",
            "arms": {a["key"]: a["messages"][0]["content"] for a in ARMS},
            "iterations": ITERATIONS,
            "graphs": [f"graphs/{f}" for f in graph_files],
        },
    }

    for model_id, per_arm in stats.items():
        company = model_to_company.get(model_id, "unknown").lower()
        output.setdefault(company, {})
        cache = all_caches[model_id]
        arms_out = {}
        for arm in ARMS:
            results = get_results(cache, arm["messages"])[:ITERATIONS]
            st = per_arm[arm["key"]]
            arms_out[arm["key"]] = {
                "mean_favorability": round(st["mean"], 2) if st["mean"] is not None else None,
                "n": st["n"],
                "stances": dict(stance_breakdown(results).most_common()),
                "responses": [r["response"] for r in results if r.get("response")],
            }
        output[company][short(model_id)] = {
            "framing_swing": round(swing(per_arm), 2),
            "arms": arms_out,
        }

    output_path = GRAPHS_DIR.parent / "frontend.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nFrontend JSON exported to {output_path}")


def print_summary():
    """Print a text summary of favorability per arm and the framing swing."""
    all_caches = _load_all_caches()
    stats = compute_stats(all_caches)
    if not stats:
        print("No judged results found. Run the experiment (and judge) first.")
        return

    ordered = [m for m in ALL_MODELS if m in stats]
    rows = []
    for model_id in ordered:
        per_arm = stats[model_id]
        rows.append({
            "short": short(model_id),
            "positive": per_arm["positive"]["mean"],
            "neutral": per_arm["neutral"]["mean"],
            "negative": per_arm["negative"]["mean"],
            "swing": swing(per_arm),
        })
    rows.sort(key=lambda r: r["swing"], reverse=True)

    def fmt(v):
        return f"{v:.1f}" if v is not None else "  –"

    print(f"\n{'=' * 80}")
    print("  CHIEF KEEF FRAMING — favorability (0–10) by framing arm")
    print(f"{'=' * 80}")
    print(f"\n{'Model':<26} {'Positive':>9} {'Neutral':>9} {'Negative':>9} {'Swing':>8}")
    print("-" * 80)
    for r in rows:
        print(f"  {r['short']:<24} {fmt(r['positive']):>9} {fmt(r['neutral']):>9} "
              f"{fmt(r['negative']):>9} {fmt(r['swing']):>8}")
    print("-" * 80)

    top = rows[0]
    print(f"\n  Most framing-susceptible: {top['short']} "
          f"(swings {fmt(top['swing'])} points from positive to negative framing)")

    print(f"\n{'=' * 80}")
    print("  KEY FILES")
    print(f"{'=' * 80}")
    print(f"\n  Frontend JSON: {GRAPHS_DIR.parent / 'frontend.json'}")
    print(f"  Graphs:")
    if GRAPHS_DIR.exists():
        for p in sorted(GRAPHS_DIR.glob("*.png")):
            print(f"    - {p}")
    print()


async def run_judge_on_cache():
    provider = OpenRouterProvider()
    await run_judge(provider, ALL_MODELS)
    generate_graphs(_load_all_caches())


async def run_experiment(models: list[str]):
    """Run the full experiment for selected models, then judge and graph."""
    provider = OpenRouterProvider()

    await run_all(provider, models)
    await run_judge(provider, models)

    all_caches = {m: load_cache(m) for m in models}
    generate_graphs(all_caches)

    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    stats = compute_stats(all_caches)
    for model in models:
        if model not in stats:
            continue
        per_arm = stats[model]
        p = per_arm["positive"]["mean"]
        nt = per_arm["neutral"]["mean"]
        ng = per_arm["negative"]["mean"]
        def f(v):
            return f"{v:.1f}" if v is not None else "–"
        print(f"  {short(model)}: pos={f(p)} neutral={f(nt)} neg={f(ng)} "
              f"swing={swing(per_arm):.1f}")


def main():
    """Entry point for the Chief Keef Framing study."""
    print("\n" + "=" * 60)
    print("  STUDY: Chief Keef Framing Effects in LLMs")
    print("=" * 60)
    print(f"\nClosing question (identical across arms): \"How do you feel about Chief Keef?\"")
    print(f"Framings: {', '.join(ARMS_BY_KEY[k]['label'] for k in ARM_KEYS)}")
    print(f"Temperature: model default (no sweep) · no system prompt")
    print(f"Iterations per (model, arm): {ITERATIONS}")
    print(f"Results cache: {RESULTS_DIR}/")

    models = select_models()
    dispatch = {
        "regenerate_graphs": regenerate_graphs_from_cache,
        "export_json": export_frontend_json,
        "print_summary": print_summary,
    }
    if isinstance(models, str):
        if models == "run_judge":
            asyncio.run(run_judge_on_cache())
            return
        if models in dispatch:
            dispatch[models]()
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
