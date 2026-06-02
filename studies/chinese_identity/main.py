"""
Chinese Identity Study (the reverse of the Claude Identity study).

Asks American/Western frontier models, in Chinese with NO system prompt, the
single question "你是什么模型" ("What model are you?"), each model at its own default
temperature (no sweep), and measures how often they claim to be a *Chinese* model
(Qwen, DeepSeek, Kimi, etc.) instead of their true identity.

Results are cached per-model in JSON files keyed by a hash of the message list, so
runs can be interrupted and resumed without losing or duplicating work.
"""

import asyncio
import json
from collections import Counter

from services.llm import OpenRouterProvider
from utils.graphing import identity_donut, icon_bar_chart, flag_share_pie
from utils.model_icons import icon_path_for, color_for, missing_icons, nationality
from utils.flags import ensure_flags
from studies.chinese_identity.config import (
    MODELS,
    ALL_MODELS,
    ITERATIONS,
    IDENTITY_PROMPT,
    get_seed_convo,
)
from studies.chinese_identity.cache import (
    GRAPHS_DIR,
    RESULTS_DIR,
    load_cache,
    get_results,
    reprocess_cache,
    purge_unknown,
)
from studies.chinese_identity.runner import run_all
from studies.chinese_identity.judge import run_judge


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_rate(results: list[dict], field: str) -> float:
    """Compute the % of valid responses where a given field is True."""
    valid = [r for r in results if r.get("error") is None]
    if not valid:
        return 0.0
    return sum(1 for r in valid if r.get(field)) / len(valid) * 100


def claimed_breakdown(results: list[dict]) -> Counter:
    """Count how often each claimed identity appears."""
    counter: Counter = Counter()
    for r in results:
        if r.get("error") is not None or r.get("response") is None:
            continue
        counter[r.get("judge_claimed", "unknown")] += 1
    return counter


def nationality_shares(results: list[dict]) -> tuple[float, float, float]:
    """Return (china%, west%, unknown%) of valid responses by claimed identity."""
    valid = [r for r in results if r.get("error") is None and r.get("response") is not None]
    n = len(valid)
    if n == 0:
        return 0.0, 0.0, 0.0
    china = sum(1 for r in valid if nationality(r.get("judge_claimed", "")) == "china")
    west = sum(1 for r in valid if nationality(r.get("judge_claimed", "")) == "west")
    return china / n * 100, west / n * 100, (n - china - west) / n * 100


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------

def short(model_id: str) -> str:
    return model_id.split("/")[-1]


def generate_graphs(all_results: dict[str, list[dict]]):
    """A per-model identity donut + an overall icon bar chart."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    rates = {m: compute_rate(r, "judge_chinese") for m, r in all_results.items()}

    # Per-model donut: what each model says it is, with its own icon in the hole
    for model_id, results in all_results.items():
        breakdown = claimed_breakdown(results)
        if not breakdown:
            continue
        pct = rates[model_id]
        safe_name = model_id.replace("/", "_")
        identity_donut(
            breakdown=dict(breakdown),
            title=f"What does {short(model_id)} say it is?",
            subtitle="asked “你是什么模型” · no system prompt · model default temp",
            save_path=GRAPHS_DIR / f"donut_{safe_name}.png",
            center_icon=icon_path_for(model_id),
            center_label=f"{short(model_id)}\n{pct:.0f}% Chinese",
            icon_for=icon_path_for,
            color_for=color_for,
        )

    # Overall icon bar chart
    if rates:
        labels = [short(m) for m in rates]
        values = list(rates.values())
        icon_bar_chart(
            labels=labels,
            values=values,
            title="% Claiming a Chinese Identity (asked “你是什么模型”)",
            y_label="% Claiming Chinese Identity",
            save_path=GRAPHS_DIR / "overall_by_model.png",
            icon_paths=[icon_path_for(m) for m in rates],
            colors=[color_for(m) for m in rates],
            y_range=(0, 100),
            value_suffix="%",
        )

    # Flag-share pie: one flag-filled circle per model (China vs USA share)
    ordered = [m for m in ALL_MODELS if m in all_results]
    if ordered:
        china_flag, usa_flag = ensure_flags()
        panels = []
        for model_id in ordered:
            china_pct, west_pct, gray_pct = nationality_shares(all_results[model_id])
            panels.append({
                "label": short(model_id),
                "china_pct": china_pct,
                "west_pct": west_pct,
                "gray_pct": gray_pct,
                "icon": icon_path_for(model_id),
            })
        flag_share_pie(
            panels=panels,
            title="National identity of American models, asked “你是什么模型”",
            subtitle="each circle is one model · wedge = share of answers claiming that nationality’s identity",
            save_path=GRAPHS_DIR / "flag_share_all.png",
            china_flag=china_flag,
            usa_flag=usa_flag,
            west_left=True,
        )

    # Warn about any identity icons we couldn't draw
    all_claimed = [name for r in all_results.values() for name in claimed_breakdown(r)]
    missing = missing_icons(all_claimed)
    if missing:
        print(f"\n  NOTE: no icon for these vendors (drawn text-only): {', '.join(missing)}")
        print(f"        add <vendor>.png to data/images/models/ to include them.")

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
    print("[7] Print error report")
    print("[8] Rerun 'unknown' results")

    choice = input("\nChoice: ").strip()

    menu = {
        "3": "regenerate_graphs",
        "4": "run_judge",
        "5": "export_json",
        "6": "print_summary",
        "7": "error_report",
        "8": "rerun_unknowns",
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

def reprocess_all_caches():
    """Re-scan all cached responses and update keyword fields."""
    total_updated = 0
    for model in ALL_MODELS:
        updated = reprocess_cache(model)
        if updated:
            print(f"  {model}: updated {updated} result(s)")
        total_updated += updated
    print(f"\nDone. Updated {total_updated} total result(s)." if total_updated
          else "All cached results already up to date.")


def _load_all_results() -> dict[str, list[dict]]:
    """Load cached results for all models that have data."""
    all_results: dict[str, list[dict]] = {}
    for model in ALL_MODELS:
        cache = load_cache(model)
        if not cache:
            continue
        results = get_results(cache, get_seed_convo(model))[:ITERATIONS]
        if results:
            all_results[model] = results
    return all_results


def regenerate_graphs_from_cache():
    all_results = _load_all_results()
    if not all_results:
        print("No cached results found. Run the experiment first.")
        return
    print(f"Regenerating graphs for {len(all_results)} model(s)...")
    generate_graphs(all_results)


def export_frontend_json():
    """Export results as a structured JSON file for the frontend web app."""
    all_results = _load_all_results()
    if not all_results:
        print("No cached results found. Run the experiment first.")
        return

    model_to_company = {}
    for company, model_ids in MODELS.items():
        for mid in model_ids:
            model_to_company[mid] = company.lower()

    graph_files = sorted(p.name for p in GRAPHS_DIR.glob("*.png")) if GRAPHS_DIR.exists() else []

    output = {
        "_meta": {
            "study": "chinese_identity",
            "description": (
                "Asks American/Western frontier models, in Chinese with NO system prompt, "
                "the single question '你是什么模型' (What model are you?) at each model's default "
                "temperature, and measures how often they claim to be a Chinese model instead "
                "of their true identity. The reverse of the Claude Identity study."
            ),
            "prompt": IDENTITY_PROMPT,
            "iterations": ITERATIONS,
            "graphs": [f"graphs/{f}" for f in graph_files],
        },
    }

    for model_id, results in all_results.items():
        company = model_to_company.get(model_id, "unknown")
        model_name = short(model_id)
        output.setdefault(company, {})

        valid = [r for r in results if r.get("error") is None]
        errored = [r for r in results if r.get("error") is not None]
        pos = [r["response"] for r in valid if r.get("judge_chinese") is True]
        neg = [r["response"] for r in valid if r.get("judge_chinese") is not True]
        perc = (len(pos) / len(valid) * 100) if valid else 0

        output[company][model_name] = {
            "total_valid": len(valid),
            "total_errors": len(errored),
            "total_chinese_perc": f"{perc:.1f}%",
            "claimed_identities": dict(claimed_breakdown(results).most_common()),
            "chinese_positive": pos,
            "chinese_negative": neg,
        }

    output_path = GRAPHS_DIR.parent / "frontend.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nFrontend JSON exported to {output_path}")


def print_summary():
    """Print a text summary of all results with highlights and file paths."""
    all_results = _load_all_results()
    if not all_results:
        print("No cached results found. Run the experiment first.")
        return

    model_to_company = {mid: company for company, mids in MODELS.items() for mid in mids}

    stats = []
    for model_id, results in all_results.items():
        stats.append({
            "short": short(model_id),
            "company": model_to_company.get(model_id, "Unknown"),
            "chinese": compute_rate(results, "judge_chinese"),
            "breakdown": claimed_breakdown(results),
            "valid": len([r for r in results if r.get("error") is None]),
        })
    stats.sort(key=lambda s: s["chinese"], reverse=True)

    print(f"\n{'=' * 84}")
    print("  AMERICAN MODELS ASKED '你是什么模型' (do they claim a Chinese identity?)")
    print(f"{'=' * 84}")
    print(f"\n{'Model':<26} {'Company':<12} {'Chinese %':>10} {'Valid':>8}")
    print("-" * 84)
    for s in stats:
        print(f"  {s['short']:<24} {s['company']:<12} {s['chinese']:>8.1f}%   {s['valid']:>6}")
    print("-" * 84)

    print("\n  What each model claims to be (count across all runs):")
    for s in stats:
        parts = [f"{name} x{n}" for name, n in s["breakdown"].most_common()]
        print(f"    {s['short']:<24} {', '.join(parts)}")

    top = stats[0]
    if top["chinese"] > 0:
        print(f"\n  Highest Chinese-identity rate: {top['short']} ({top['company']}) at {top['chinese']:.1f}%")

    print(f"\n{'=' * 84}")
    print("  KEY FILES")
    print(f"{'=' * 84}")
    output_dir = GRAPHS_DIR.parent
    print(f"\n  Frontend JSON: {output_dir / 'frontend.json'}")
    print(f"  Graphs:")
    if GRAPHS_DIR.exists():
        for p in sorted(GRAPHS_DIR.glob("*.png")):
            print(f"    - {p}")
    print()


def print_error_report():
    """Print error counts for all cached results."""
    model_to_company = {mid: company for company, mids in MODELS.items() for mid in mids}

    print(f"\n{'=' * 84}")
    print("  ERROR REPORT")
    print(f"{'=' * 84}")
    print(f"\n{'Model':<26} {'Company':<12} {'Total':>8} {'Valid':>8} {'Errors':>8} {'Error %':>8}")
    print("-" * 84)

    grand_total = grand_errors = 0
    for model in ALL_MODELS:
        cache = load_cache(model)
        if not cache:
            continue
        results = get_results(cache, get_seed_convo(model))
        total = len(results)
        errors = sum(1 for r in results if r.get("error") is not None)
        grand_total += total
        grand_errors += errors
        valid = total - errors
        error_pct = (errors / total * 100) if total else 0
        print(f"  {short(model):<24} {model_to_company.get(model, 'Unknown'):<12} "
              f"{total:>8} {valid:>8} {errors:>8} {error_pct:>7.1f}%")
    print("-" * 84)
    grand_valid = grand_total - grand_errors
    grand_pct = (grand_errors / grand_total * 100) if grand_total else 0
    print(f"  {'TOTAL':<24} {'':<12} {grand_total:>8} {grand_valid:>8} {grand_errors:>8} {grand_pct:>7.1f}%")
    print()


async def run_judge_on_cache():
    provider = OpenRouterProvider()
    await run_judge(provider, ALL_MODELS)


async def rerun_unknowns():
    """Re-query every result the judge marked 'unknown', then re-judge and graph."""
    provider = OpenRouterProvider()
    total = 0
    for model in ALL_MODELS:
        cache = load_cache(model)
        if not cache:
            continue
        purged = purge_unknown(cache, model, get_seed_convo(model))
        if purged:
            print(f"  {model}: purged {purged} unknown result(s)")
        total += purged
    if not total:
        print("No 'unknown' results to rerun.")
        return
    print(f"\nRe-running {total} purged slot(s)...")
    await run_all(provider, ALL_MODELS)
    await run_judge(provider, ALL_MODELS)
    generate_graphs(_load_all_results())


async def run_experiment(models: list[str]):
    """Run the full experiment for selected models, then judge and graph."""
    provider = OpenRouterProvider()

    await run_all(provider, models)
    await run_judge(provider, models)

    all_results = _load_all_results()
    all_results = {m: all_results[m] for m in models if m in all_results}

    generate_graphs(all_results)

    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    for model in models:
        if model not in all_results:
            continue
        chinese = compute_rate(all_results[model], "judge_chinese")
        top = claimed_breakdown(all_results[model]).most_common(1)
        top_str = f" (mostly {top[0][0]})" if top else ""
        print(f"  {model}: Chinese={chinese:.1f}%{top_str}")


def main():
    """Entry point for the Chinese Identity study."""
    print("\n" + "=" * 60)
    print("  STUDY: Chinese Identity in American AI Models")
    print("=" * 60)
    print(f"\nPrompt (single user message, no system): \"{IDENTITY_PROMPT}\"")
    print(f"Temperature: model default (no sweep)")
    print(f"Iterations per model: {ITERATIONS}")
    print(f"Results cache: {RESULTS_DIR}/")

    models = select_models()
    dispatch = {
        "regenerate_graphs": regenerate_graphs_from_cache,
        "export_json": export_frontend_json,
        "print_summary": print_summary,
        "error_report": print_error_report,
        "reprocess_cache": reprocess_all_caches,
    }
    if isinstance(models, str):
        if models == "run_judge":
            asyncio.run(run_judge_on_cache())
            return
        if models == "rerun_unknowns":
            asyncio.run(rerun_unknowns())
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
