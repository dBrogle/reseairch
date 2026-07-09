"""
Cognitive Biases Study

Tests classic human cognitive biases in LLMs (anchoring, sunk-cost,
social proof, authority, ...) by running paired control/treatment
prompts and measuring whether numeric responses drift between arms.

Pipeline:
  1. Run raw queries (model × scenario × arm × iteration) — cached per model
  2. Extract the scenario's value_field — direct JSON parse, LLM fallback
  3. Per-arm 95% CIs + per-treatment Welch CI on (treatment − control)
  4. One dot+CI chart per scenario, plus a delta chart per scenario

Add a new bias by dropping a file into `scenarios/` and importing it from
`scenarios/__init__.py`.
"""

import asyncio

from services.llm import OpenRouterProvider
from studies.cognitive_biases.config import (
    ACTIVE_BIAS_TYPES,
    ACTIVE_SCENARIOS,
    ALL_SCENARIOS,
    BIAS_TYPES,
    EXTRACTOR_MODEL,
    ITERATIONS,
    MODELS,
    SCENARIOS_BY_ID,
    SMOKE_ITERATIONS,
    SMOKE_MODELS,
    SMOKE_SCENARIO_IDS,
    TEMPERATURE,
    scenarios_by_bias,
)
from studies.cognitive_biases.cache import GRAPHS_DIR, load_response_cache
from studies.cognitive_biases.cost import estimate_remaining_run, format_estimate
from studies.cognitive_biases.custom_charts import (
    CUSTOM_CHARTS,
    SKIP_GENERIC_DELTA,
    SKIP_GENERIC_PERARM,
)
from studies.cognitive_biases.runner import run_all
from studies.cognitive_biases.extractor import extract_all
from studies.cognitive_biases.analysis import compute_scenario_stats
from studies.cognitive_biases.scenarios.base import Scenario
from studies.cognitive_biases.visualize import (
    generate_delta_chart,
    generate_scenario_chart,
)
# Reuse the cost tracker from question_framing — same shape, no duplication.
from studies.question_framing.cost_tracker import CostTracker


# ---------------------------------------------------------------------------
# Selection menu
# ---------------------------------------------------------------------------

def _print_run_size(models, scenarios, iterations):
    arms_total = sum(len(s.arms) for s in scenarios)
    calls = len(models) * arms_total * iterations
    print(
        f"    ({len(models)} models × {len(scenarios)} scenarios × "
        f"~{arms_total / max(len(scenarios), 1):.1f} arms × "
        f"{iterations} iters ≈ {calls} calls)"
    )


def select_run_mode() -> dict | str:
    """Return either a config dict (models, scenarios, iterations) or a sentinel."""
    print("\n=== Run Mode ===")
    print("[1] Full run — all models, active scenarios, all iterations")
    _print_run_size(MODELS, ACTIVE_SCENARIOS, ITERATIONS)
    print("[2] Smoke test — 1 model, one scenario per active bias, few iters")
    _print_run_size(
        SMOKE_MODELS,
        [SCENARIOS_BY_ID[i] for i in SMOKE_SCENARIO_IDS],
        SMOKE_ITERATIONS,
    )
    print("[3] Pick individual models (active scenarios + iterations)")
    print("[4] Pick a single bias family (active families only)")
    print("[5] Regenerate graphs from cached results")
    print("[6] Print results summary")

    choice = input("\nChoice: ").strip()

    if choice == "5":
        return "regenerate_graphs"
    if choice == "6":
        return "print_summary"
    if choice == "2":
        return {
            "models":     SMOKE_MODELS,
            "scenarios":  [SCENARIOS_BY_ID[i] for i in SMOKE_SCENARIO_IDS],
            "iterations": SMOKE_ITERATIONS,
        }
    if choice == "3":
        print("\nModels:")
        for i, m in enumerate(MODELS, 1):
            print(f"  [{i}] {m}")
        picks = input("Select models (comma-separated numbers): ").strip()
        selected: list[str] = []
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
        return {"models": selected, "scenarios": list(ACTIVE_SCENARIOS),
                "iterations": ITERATIONS}
    if choice == "4":
        print("\nBias families (active):")
        for i, bt in enumerate(ACTIVE_BIAS_TYPES, 1):
            count = len(scenarios_by_bias(bt))
            print(f"  [{i}] {bt}  ({count} scenario{'s' if count != 1 else ''})")
        pick = input("Select bias family (number): ").strip()
        try:
            idx = int(pick) - 1
        except ValueError:
            return "abort"
        if not (0 <= idx < len(ACTIVE_BIAS_TYPES)):
            return "abort"
        return {
            "models":     MODELS,
            "scenarios":  list(scenarios_by_bias(ACTIVE_BIAS_TYPES[idx])),
            "iterations": ITERATIONS,
        }

    # Default to full run
    return {"models": MODELS, "scenarios": list(ACTIVE_SCENARIOS),
            "iterations": ITERATIONS}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(models: list[str], scenarios: list[Scenario], iterations: int):
    print("\n" + "=" * 78)
    print("  COGNITIVE BIASES — Summary")
    print("=" * 78)

    for scenario in scenarios:
        print(f"\n  [{scenario.bias_type}] {scenario.title}")
        print(f"    {scenario.description}")
        print(f"    expected: {scenario.expected_direction}")
        print(f"    {'-' * 70}")

        for model in models:
            stats = compute_scenario_stats(model, scenario, TEMPERATURE, iterations)
            short = model.split("/")[-1]
            ctrl = stats["per_arm"][scenario.control.key]

            if ctrl["n"] == 0:
                print(f"    {short:<35}  (no control data)")
                continue

            print(f"    {short:<35}  control: "
                  f"{ctrl['mean']:.2f} [{ctrl['ci_low']:.2f}, "
                  f"{ctrl['ci_high']:.2f}] n={ctrl['n']}")
            for arm in scenario.treatments:
                arm_stats = stats["per_arm"].get(arm.key, {})
                if not arm_stats or arm_stats["n"] == 0:
                    print(f"    {'':<35}    {arm.key}: (no data)")
                    continue
                d = stats["deltas"].get(arm.key)
                delta_str = (
                    f"  Δ {d['diff']:+.2f} [{d['ci_low']:+.2f}, "
                    f"{d['ci_high']:+.2f}]"
                    if d is not None else "  Δ (n/a)"
                )
                print(
                    f"    {'':<35}    {arm.key}: "
                    f"{arm_stats['mean']:.2f} "
                    f"[{arm_stats['ci_low']:.2f}, "
                    f"{arm_stats['ci_high']:.2f}] n={arm_stats['n']}"
                    f"{delta_str}"
                )


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def generate_graphs(models: list[str], scenarios: list[Scenario], iterations: int):
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        all_stats = [
            compute_scenario_stats(m, scenario, TEMPERATURE, iterations)
            for m in models
        ]
        scenario_dir = GRAPHS_DIR / scenario.bias_type
        scenario_dir.mkdir(parents=True, exist_ok=True)

        if scenario.id not in SKIP_GENERIC_PERARM:
            generate_scenario_chart(
                scenario, all_stats,
                scenario_dir / f"{scenario.id}__per_arm.png",
            )
        if scenario.id not in SKIP_GENERIC_DELTA:
            generate_delta_chart(
                scenario, all_stats,
                scenario_dir / f"{scenario.id}__delta.png",
            )
        for suffix, chart_fn in CUSTOM_CHARTS.get(scenario.id, []):
            chart_fn(
                scenario, all_stats,
                scenario_dir / f"{scenario.id}__{suffix}.png",
            )
        print(f"  Charts saved for {scenario.id}")

    print(f"\nGraphs saved to {GRAPHS_DIR}/")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def _preflight(
    provider: OpenRouterProvider,
    models: list[str],
    scenarios: list[Scenario],
    iterations: int,
) -> CostTracker:
    """Fetch pricing, print a cache-aware cost estimate, return the tracker.

    Pricing is also needed for the extractor model so the second-pass
    LLM extraction shows up in the running total.
    """
    cost_tracker = CostTracker()
    print("\n--- Fetching model pricing ---")
    await cost_tracker.fetch_pricing(
        provider.api_key,
        list({*models, EXTRACTOR_MODEL}),
    )

    response_caches = {m: load_response_cache(m) for m in models}
    estimate = estimate_remaining_run(
        pricing=cost_tracker._pricing,
        response_caches=response_caches,
        models=models,
        scenarios=scenarios,
        temperature=TEMPERATURE,
        iterations=iterations,
    )
    print("\n--- Pre-flight estimate (after consulting cache) ---")
    print(format_estimate(estimate, models))
    print(
        "  Note: per-call extractor cost is separate and depends on how "
        "many responses fail the direct JSON parse (typically a small "
        "fraction)."
    )
    return cost_tracker


async def run_experiment(
    models: list[str], scenarios: list[Scenario], iterations: int,
):
    provider = OpenRouterProvider()

    cost_tracker = await _preflight(provider, models, scenarios, iterations)

    confirm = input("\nProceed with this run? [Y/n]: ").strip().lower()
    if confirm == "n":
        print("Cancelled.")
        return

    print("\n--- Step 1: Collecting raw LLM responses ---")
    await run_all(
        provider, models, scenarios, TEMPERATURE, iterations,
        cost_tracker=cost_tracker,
    )

    print("\n--- Step 2: Extracting numeric values ---")
    await extract_all(
        provider, models, scenarios, TEMPERATURE, iterations,
        cost_tracker=cost_tracker,
    )

    print(f"\n--- Total cost: {cost_tracker.format_total()} ---")

    print("\n--- Step 3: Stats and visualizations ---")
    generate_graphs(models, scenarios, iterations)

    print_summary(models, scenarios, iterations)


def main():
    print("\n" + "=" * 70)
    print("  STUDY: Cognitive Biases in LLMs")
    print("=" * 70)
    print(
        f"\n  Active bias families ({len(ACTIVE_BIAS_TYPES)} of "
        f"{len(BIAS_TYPES)}): {', '.join(ACTIVE_BIAS_TYPES) or '(none)'}"
    )
    print(f"  Active scenarios: {len(ACTIVE_SCENARIOS)} of {len(ALL_SCENARIOS)}")
    for bt in BIAS_TYPES:
        kids = scenarios_by_bias(bt)
        on = bt in ACTIVE_BIAS_TYPES
        marker = " " if on else "·"
        print(f"   {marker}{bt:>14}: {len(kids)} ({', '.join(s.id for s in kids)})")
    print(f"  Default models: {len(MODELS)}")
    print(f"  Default iterations: {ITERATIONS} per (model, scenario, arm)")
    print(f"  Temperature: {TEMPERATURE}")

    cfg = select_run_mode()
    if cfg == "regenerate_graphs":
        generate_graphs(MODELS, list(ACTIVE_SCENARIOS), ITERATIONS)
        return
    if cfg == "print_summary":
        print_summary(MODELS, list(ACTIVE_SCENARIOS), ITERATIONS)
        return
    if cfg == "abort":
        print("Cancelled.")
        return

    models = cfg["models"]
    scenarios = cfg["scenarios"]
    iterations = cfg["iterations"]

    print(
        f"\nWill test {len(models)} model(s) on "
        f"{len(scenarios)} scenario(s), {iterations} iter(s) per arm:"
    )
    for m in models:
        print(f"  - {m}")
    print("  Scenarios:")
    for s in scenarios:
        print(f"    [{s.bias_type:>10}] {s.id} — {s.title}")

    # The actual confirmation happens inside run_experiment, after the
    # cache-aware cost pre-flight prints — that way the user sees both
    # how much new work this is and how much it will cost before saying
    # yes.
    asyncio.run(run_experiment(models, scenarios, iterations))


if __name__ == "__main__":
    main()
