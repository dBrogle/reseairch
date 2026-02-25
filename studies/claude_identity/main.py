"""
Claude Identity Study

Tests whether Chinese AI models claim to be Claude/Anthropic at various temperatures.
For each model, runs multiple iterations across a temperature sweep and graphs
the rate at which the model identifies itself as Claude.

Results are cached per-model in JSON files keyed by a hash of (prompt, temperature),
so runs can be interrupted and resumed without losing or duplicating work.
"""

import asyncio
import json

from services.llm import OpenRouterProvider
from utils.graphing import line_chart, multi_line_chart, bar_chart
from studies.claude_identity.config import (
    MODELS,
    ALL_MODELS,
    TEMPERATURES,
    ITERATIONS_PER_TEMP,
    IDENTITY_PROMPT,
    Language,
    MODEL_LANGUAGES,
    get_seed_convo,
)
from studies.claude_identity.cache import GRAPHS_DIR, RESULTS_DIR, load_cache, save_cache, get_results, reprocess_cache
from studies.claude_identity.runner import run_all
from studies.claude_identity.judge import run_judge


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_rates(results: dict[float, list[dict]], field: str) -> list[float]:
    """Compute the % of responses where a given field is True at each temperature."""
    rates = []
    for temp in TEMPERATURES:
        temp_results = results.get(temp, [])
        valid = [r for r in temp_results if r.get("error") is None]
        if valid:
            rate = sum(1 for r in valid if r.get(field)) / len(valid) * 100
        else:
            rate = 0.0
        rates.append(rate)
    return rates


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------

def generate_graphs(all_rates: dict[str, dict[str, list[float]]]):
    """Generate per-model and combined graphs for all identity claim types.

    all_rates is {field_label: {model_id: [rates_per_temp]}}, e.g.:
      {"Claude": {"qwen/...": [80.0, ...]}, "DeepSeek": {"anthropic/...": [10.0, ...]}}
    Models are filtered to only include those tested in the relevant language.
    """
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    # Which language group each label applies to
    label_language = {
        "Claude": Language.ENGLISH,
        "ChatGPT": Language.ENGLISH,
        "DeepSeek": Language.CHINESE,
        "Kimi": Language.CHINESE,
    }

    def short(model_id: str) -> str:
        return model_id.split("/")[-1]

    for label, rates in all_rates.items():
        prefix = f"claims_{label.lower()}"
        y_label = f"% Claiming {label}"
        # Only include models tested in the relevant language for this label
        target_lang = label_language.get(label)
        if target_lang is not None:
            rates = {m: r for m, r in rates.items()
                     if MODEL_LANGUAGES.get(m, Language.ENGLISH) == target_lang}
        else:
            rates = {m: r for m, r in rates.items() if any(v > 0 for v in r)}
        if not rates:
            continue

        # Per-model graphs
        for model_id, model_rates in rates.items():
            safe_name = model_id.replace("/", "_")
            line_chart(
                x_values=TEMPERATURES,
                y_values=model_rates,
                title=f"% Claiming {label}: {short(model_id)}",
                x_label="Temperature",
                y_label=y_label,
                save_path=GRAPHS_DIR / f"{prefix}_{safe_name}.png",
                y_range=(0, 100),
            )

        # Combined graph per company
        for company, model_ids in MODELS.items():
            company_series = {
                short(mid): rates[mid] for mid in model_ids if mid in rates
            }
            if company_series:
                multi_line_chart(
                    x_values=TEMPERATURES,
                    series=company_series,
                    title=f"% Claiming {label}: {company} Models",
                    x_label="Temperature",
                    y_label=y_label,
                    save_path=GRAPHS_DIR / f"{prefix}_{company.lower()}_combined.png",
                    y_range=(0, 100),
                )

        # Combined graph with all models
        if rates:
            multi_line_chart(
                x_values=TEMPERATURES,
                series={short(m): r for m, r in rates.items()},
                title=f"% Claiming {label}: All Models",
                x_label="Temperature",
                y_label=y_label,
                save_path=GRAPHS_DIR / f"{prefix}_all_models.png",
                y_range=(0, 100),
            )

        # Bar chart: overall rate per model
        if rates:
            labels_list = [short(mid) for mid in rates]
            overall = [sum(r) / len(r) for r in rates.values()]
            bar_chart(
                labels=labels_list,
                values=overall,
                title=f"Overall % Claiming {label}",
                x_label="Model",
                y_label=y_label,
                save_path=GRAPHS_DIR / f"{prefix}_overall_by_model.png",
                y_range=(0, 100),
            )

    print(f"\nGraphs saved to {GRAPHS_DIR}/")


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def select_models() -> list[str]:
    """Interactive model selection menu."""
    print("\n=== Model Selection ===")
    print("[1] Run ALL models")
    print("[2] Select by company")
    print("[3] Select individual models")
    print("[4] Regenerate graphs from cached results")
    print("[5] Reprocess cache (update detection fields)")
    print("[6] Run LLM judge on cached results")
    print("[7] Export JSON for frontend")
    print("[8] Print results summary")
    print("[9] Print error report")
    print("[10] Migrate legacy judge fields")

    choice = input("\nChoice: ").strip()

    if choice == "4":
        return "regenerate_graphs"

    if choice == "5":
        return "reprocess_cache"

    if choice == "6":
        return "run_judge"

    if choice == "7":
        return "export_json"

    if choice == "8":
        return "print_summary"

    if choice == "9":
        return "error_report"

    if choice == "10":
        return "migrate_judge"

    if choice == "1":
        return ALL_MODELS

    if choice == "2":
        print("\nCompanies:")
        companies = list(MODELS.keys())
        for i, company in enumerate(companies, 1):
            print(f"  [{i}] {company} ({len(MODELS[company])} models)")
        picks = input("Select companies (comma-separated numbers): ").strip()
        selected = []
        for p in picks.split(","):
            idx = int(p.strip()) - 1
            if 0 <= idx < len(companies):
                selected.extend(MODELS[companies[idx]])
        return selected

    if choice == "3":
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
# Orchestration
# ---------------------------------------------------------------------------

def reprocess_all_caches():
    """Re-scan all cached responses and update detection fields."""
    total_updated = 0
    for model in ALL_MODELS:
        updated = reprocess_cache(model)
        if updated:
            print(f"  {model}: updated {updated} result(s)")
        total_updated += updated
    if total_updated:
        print(f"\nDone. Updated {total_updated} total result(s).")
    else:
        print("All cached results already up to date.")


def migrate_judge_fields():
    """Add missing judge_deepseek/judge_kimi (null) to legacy results and vice versa.

    Legacy results only have judge_claude and judge_chatgpt. This backfills
    the new fields as None so downstream code can treat all results uniformly.
    """
    total_migrated = 0
    for model in ALL_MODELS:
        cache = load_cache(model)
        if not cache:
            continue
        lang = MODEL_LANGUAGES.get(model, Language.ENGLISH)
        migrated = 0
        for entry in cache.values():
            for result in entry.get("results", []):
                if result.get("response") is None:
                    continue
                if lang == Language.ENGLISH:
                    # Legacy English results: backfill deepseek/kimi as None
                    if "judge_claude" in result and "judge_deepseek" not in result:
                        result["judge_deepseek"] = None
                        result["judge_kimi"] = None
                        migrated += 1
                else:
                    # Legacy Chinese results: backfill claude/chatgpt as None
                    if "judge_deepseek" in result and "judge_claude" not in result:
                        result["judge_claude"] = None
                        result["judge_chatgpt"] = None
                        migrated += 1
        if migrated:
            save_cache(model, cache)
            print(f"  {model}: migrated {migrated} result(s)")
        total_migrated += migrated
    if total_migrated:
        print(f"\nDone. Migrated {total_migrated} total result(s).")
    else:
        print("All cached results already have the new fields.")


def _load_all_results() -> dict[str, dict[float, list[dict]]]:
    """Load cached results for all models that have data."""
    all_results: dict[str, dict[float, list[dict]]] = {}
    for model in ALL_MODELS:
        cache = load_cache(model)
        if not cache:
            continue
        seed = get_seed_convo(model)
        results: dict[float, list[dict]] = {}
        for temp in TEMPERATURES:
            results[temp] = get_results(cache, seed, temp)[:ITERATIONS_PER_TEMP]
        if any(results[t] for t in TEMPERATURES):
            all_results[model] = results
    return all_results


def regenerate_graphs_from_cache():
    """Regenerate all graphs from existing cached results."""
    all_results = _load_all_results()

    if not all_results:
        print("No cached results found. Run the experiment first.")
        return

    all_rates = {
        label: {m: compute_rates(r, field) for m, r in all_results.items()}
        for label, field in [
            ("Claude", "judge_claude"),
            ("ChatGPT", "judge_chatgpt"),
            ("DeepSeek", "judge_deepseek"),
            ("Kimi", "judge_kimi"),
        ]
    }

    print(f"Regenerating graphs for {len(all_results)} model(s)...")
    generate_graphs(all_rates)


def export_frontend_json():
    """Export results as a structured JSON file for the frontend web app."""
    all_results = _load_all_results()

    if not all_results:
        print("No cached results found. Run the experiment first.")
        return

    # Reverse mapping: model_id -> lowercase company name
    model_to_company = {}
    for company, model_ids in MODELS.items():
        for mid in model_ids:
            model_to_company[mid] = company.lower()

    # Collect graph paths (relative to output dir)
    graph_files = sorted(p.name for p in GRAPHS_DIR.glob("*.png")) if GRAPHS_DIR.exists() else []

    # The judge fields relevant to each model depend on test language
    judge_fields = [
        ("claude", "judge_claude"),
        ("chatgpt", "judge_chatgpt"),
        ("deepseek", "judge_deepseek"),
        ("kimi", "judge_kimi"),
    ]

    output = {
        "_meta": {
            "study": "claude_identity",
            "description": (
                "Tests whether Chinese AI models claim to be Claude/Anthropic or ChatGPT/OpenAI "
                "when asked in English, and whether Claude models claim to be DeepSeek or Kimi "
                "when asked in Chinese."
            ),
            "temperatures": TEMPERATURES,
            "iterations_per_temp": ITERATIONS_PER_TEMP,
            "graphs": [f"graphs/{f}" for f in graph_files],
        },
    }

    for model_id, results in all_results.items():
        company = model_to_company.get(model_id, "unknown")
        model_name = model_id.split("/")[-1]
        lang = MODEL_LANGUAGES.get(model_id, Language.ENGLISH)

        if company not in output:
            output[company] = {}

        total_count = 0
        total_errors = 0
        field_results: dict[str, list[dict]] = {name: [] for name, _ in judge_fields}
        field_totals: dict[str, int] = {name: 0 for name, _ in judge_fields}

        for temp in TEMPERATURES:
            temp_results = results.get(temp, [])
            valid = [r for r in temp_results if r.get("error") is None]
            errored = [r for r in temp_results if r.get("error") is not None]
            total_count += len(valid)
            total_errors += len(errored)

            for name, field in judge_fields:
                pos = [r["response"] for r in valid if r.get(field) is True]
                neg = [r["response"] for r in valid if r.get(field) is not True]
                field_results[name].append({
                    "temperature": temp,
                    "positive": pos,
                    "negative": neg,
                })
                field_totals[name] += len(pos)

        model_data = {
            "language": lang.value,
            "total_valid": total_count,
            "total_errors": total_errors,
        }
        for name, _ in judge_fields:
            perc = (field_totals[name] / total_count * 100) if total_count else 0
            model_data[f"total_{name}_perc"] = f"{perc:.1f}%"
            model_data[f"{name}_results"] = field_results[name]

        output[company][model_name] = model_data

    output_path = GRAPHS_DIR.parent / "frontend.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nFrontend JSON exported to {output_path}")


def _summarize_group(
    label: str,
    all_results: dict[str, dict[float, list[dict]]],
    model_to_company: dict[str, str],
    primary_field: str,
    secondary_field: str,
    primary_label: str,
    secondary_label: str,
):
    """Print a summary table + highlights for one language group."""
    stats = []
    for model_id, results in all_results.items():
        p_rates = compute_rates(results, primary_field)
        s_rates = compute_rates(results, secondary_field)
        p_avg = sum(p_rates) / len(p_rates)
        s_avg = sum(s_rates) / len(s_rates)
        peak_idx = p_rates.index(max(p_rates))

        stats.append({
            "model_id": model_id,
            "short_name": model_id.split("/")[-1],
            "company": model_to_company.get(model_id, "Unknown"),
            "primary_avg": p_avg,
            "secondary_avg": s_avg,
            "primary_rates": p_rates,
            "peak_temp": TEMPERATURES[peak_idx],
            "peak_rate": p_rates[peak_idx],
        })

    if not stats:
        return

    stats.sort(key=lambda s: s["primary_avg"], reverse=True)

    print(f"\n{'=' * 80}")
    print(f"  {label}")
    print(f"{'=' * 80}")
    print(f"\n{'Model':<30} {'Company':<12} {primary_label+' %':>10} {secondary_label+' %':>10} {'Peak Temp':>10}")
    print("-" * 80)
    for s in stats:
        print(f"  {s['short_name']:<28} {s['company']:<12} {s['primary_avg']:>8.1f}% {s['secondary_avg']:>8.1f}%    t={s['peak_temp']}")
    print("-" * 80)

    # Highlights
    top = stats[0]
    if top["primary_avg"] > 0:
        print(f"\n  Highest {primary_label} claim: {top['short_name']} ({top['company']}) at {top['primary_avg']:.1f}%")
        print(f"    Peak: {top['peak_rate']:.0f}% at temperature {top['peak_temp']}")

    secondary_claimers = [s for s in stats if s["secondary_avg"] > 0]
    if secondary_claimers:
        print(f"\n  Models claiming {secondary_label}:")
        for s in secondary_claimers:
            print(f"    - {s['short_name']}: {s['secondary_avg']:.1f}%")

    if len(stats) > 1:
        biggest_swing = max(stats, key=lambda s: max(s["primary_rates"]) - min(s["primary_rates"]))
        swing = max(biggest_swing["primary_rates"]) - min(biggest_swing["primary_rates"])
        if swing > 0:
            print(f"\n  Biggest temperature swing: {biggest_swing['short_name']} "
                  f"({min(biggest_swing['primary_rates']):.0f}% -> {max(biggest_swing['primary_rates']):.0f}%, "
                  f"delta={swing:.0f}pp)")

    company_totals: dict[str, list[float]] = {}
    for s in stats:
        company_totals.setdefault(s["company"], []).append(s["primary_avg"])
    if len(company_totals) > 1:
        print(f"\n  Company averages ({primary_label} claims):")
        for company, avgs in sorted(company_totals.items(), key=lambda x: -sum(x[1])/len(x[1])):
            avg = sum(avgs) / len(avgs)
            print(f"    - {company}: {avg:.1f}%")


def print_summary():
    """Print a text summary of all results with highlights and relevant file paths."""
    all_results = _load_all_results()

    if not all_results:
        print("No cached results found. Run the experiment first.")
        return

    model_to_company = {}
    for company, model_ids in MODELS.items():
        for mid in model_ids:
            model_to_company[mid] = company

    # Split results by language group
    en_results = {m: r for m, r in all_results.items()
                  if MODEL_LANGUAGES.get(m, Language.ENGLISH) == Language.ENGLISH}
    zh_results = {m: r for m, r in all_results.items()
                  if MODEL_LANGUAGES.get(m, Language.ENGLISH) == Language.CHINESE}

    if en_results:
        _summarize_group(
            "CHINESE MODELS TESTED IN ENGLISH (do they claim to be Claude?)",
            en_results, model_to_company,
            "judge_claude", "judge_chatgpt", "Claude", "ChatGPT",
        )

    if zh_results:
        _summarize_group(
            "CLAUDE MODELS TESTED IN CHINESE (do they misidentify?)",
            zh_results, model_to_company,
            "judge_deepseek", "judge_kimi", "DeepSeek", "Kimi",
        )

    # Key file paths
    print(f"\n{'=' * 80}")
    print("  KEY FILES")
    print(f"{'=' * 80}")
    output_dir = GRAPHS_DIR.parent
    print(f"\n  Frontend JSON: {output_dir / 'frontend.json'}")
    print(f"\n  Graphs:")
    if GRAPHS_DIR.exists():
        for p in sorted(GRAPHS_DIR.glob("*.png")):
            print(f"    - {p}")

    print()


def print_error_report():
    """Print error counts for all cached results."""
    model_to_company = {}
    for company, model_ids in MODELS.items():
        for mid in model_ids:
            model_to_company[mid] = company

    print(f"\n{'=' * 90}")
    print("  ERROR REPORT")
    print(f"{'=' * 90}")
    print(f"\n{'Model':<30} {'Company':<12} {'Total':>8} {'Valid':>8} {'Errors':>8} {'Error %':>8}")
    print("-" * 90)

    grand_total = 0
    grand_errors = 0

    for model in ALL_MODELS:
        cache = load_cache(model)
        if not cache:
            continue

        seed = get_seed_convo(model)
        total = 0
        errors = 0
        for temp in TEMPERATURES:
            results = get_results(cache, seed, temp)
            for r in results:
                total += 1
                if r.get("error") is not None:
                    errors += 1

        grand_total += total
        grand_errors += errors
        valid = total - errors
        error_pct = (errors / total * 100) if total else 0
        short = model.split("/")[-1]
        company = model_to_company.get(model, "Unknown")
        print(f"  {short:<28} {company:<12} {total:>8} {valid:>8} {errors:>8} {error_pct:>7.1f}%")

    print("-" * 90)
    grand_valid = grand_total - grand_errors
    grand_pct = (grand_errors / grand_total * 100) if grand_total else 0
    print(f"  {'TOTAL':<28} {'':<12} {grand_total:>8} {grand_valid:>8} {grand_errors:>8} {grand_pct:>7.1f}%")
    print()


async def run_judge_on_cache():
    """Run the LLM judge on all unjudged cached results."""
    provider = OpenRouterProvider()
    await run_judge(provider, ALL_MODELS)


async def run_experiment(models: list[str]):
    """Run the full experiment for selected models."""
    provider = OpenRouterProvider()

    # Run all models in parallel
    all_results = await run_all(provider, models)

    # Compute rates for all identity fields
    fields = [
        ("Claude", "judge_claude"),
        ("ChatGPT", "judge_chatgpt"),
        ("DeepSeek", "judge_deepseek"),
        ("Kimi", "judge_kimi"),
    ]
    all_rates: dict[str, dict[str, list[float]]] = {
        label: {m: compute_rates(all_results[m], field) for m in models}
        for label, field in fields
    }

    # Print per-model summaries (only relevant fields)
    for model in models:
        lang = MODEL_LANGUAGES.get(model, Language.ENGLISH)
        if lang == Language.ENGLISH:
            parts = [f"{lbl}={sum(all_rates[lbl][model])/len(all_rates[lbl][model]):.1f}%"
                     for lbl in ("Claude", "ChatGPT")]
        else:
            parts = [f"{lbl}={sum(all_rates[lbl][model])/len(all_rates[lbl][model]):.1f}%"
                     for lbl in ("DeepSeek", "Kimi")]
        print(f"\n  {model}: {', '.join(parts)}")

    # Generate graphs
    generate_graphs(all_rates)

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for model in models:
        lang = MODEL_LANGUAGES.get(model, Language.ENGLISH)
        if lang == Language.ENGLISH:
            parts = [f"{lbl}={sum(all_rates[lbl][model])/len(all_rates[lbl][model]):.1f}%"
                     for lbl in ("Claude", "ChatGPT")]
        else:
            parts = [f"{lbl}={sum(all_rates[lbl][model])/len(all_rates[lbl][model]):.1f}%"
                     for lbl in ("DeepSeek", "Kimi")]
        print(f"  {model}: {', '.join(parts)}")


def main():
    """Entry point for the Claude Identity study."""
    print("\n" + "=" * 60)
    print("  STUDY: Claude Identity in Chinese AI Models")
    print("=" * 60)
    print(f"\nPrompt: \"{IDENTITY_PROMPT}\"")
    print(f"Temperatures: {TEMPERATURES}")
    print(f"Iterations per temperature: {ITERATIONS_PER_TEMP}")
    print(f"Results cache: {RESULTS_DIR}/")

    models = select_models()
    if models == "regenerate_graphs":
        regenerate_graphs_from_cache()
        return
    if models == "reprocess_cache":
        reprocess_all_caches()
        return
    if models == "run_judge":
        asyncio.run(run_judge_on_cache())
        return
    if models == "export_json":
        export_frontend_json()
        return
    if models == "print_summary":
        print_summary()
        return
    if models == "error_report":
        print_error_report()
        return
    if models == "migrate_judge":
        migrate_judge_fields()
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
