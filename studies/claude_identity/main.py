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
    SEED_CONVO_2,
)
from studies.claude_identity.cache import GRAPHS_DIR, RESULTS_DIR, load_cache, get_results, reprocess_cache
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

def generate_graphs(
    claude_rates: dict[str, list[float]],
    chatgpt_rates: dict[str, list[float]],
):
    """Generate per-model and combined graphs for both Claude and ChatGPT claims."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    for label, rates, prefix in [
        ("Claude", claude_rates, "claude"),
        ("ChatGPT", chatgpt_rates, "chatgpt"),
    ]:
        # Per-model graphs
        for model_id, model_rates in rates.items():
            safe_name = model_id.replace("/", "_")
            line_chart(
                x_values=TEMPERATURES,
                y_values=model_rates,
                title=f"{label} Identity Rate: {model_id}",
                x_label="Temperature",
                y_label=f'% Claiming to be {label}',
                save_path=GRAPHS_DIR / f"{prefix}_{safe_name}.png",
                y_range=(0, 100),
            )

        # Combined graph per company
        for company, model_ids in MODELS.items():
            company_series = {
                mid: rates[mid] for mid in model_ids if mid in rates
            }
            if company_series:
                multi_line_chart(
                    x_values=TEMPERATURES,
                    series=company_series,
                    title=f"{label} Identity Rate: {company} Models",
                    x_label="Temperature",
                    y_label=f'% Claiming to be {label}',
                    save_path=GRAPHS_DIR / f"{prefix}_{company.lower()}_combined.png",
                    y_range=(0, 100),
                )

        # Combined graph with all models
        if rates:
            multi_line_chart(
                x_values=TEMPERATURES,
                series=rates,
                title=f"{label} Identity Rate: All Models",
                x_label="Temperature",
                y_label=f'% Claiming to be {label}',
                save_path=GRAPHS_DIR / f"{prefix}_all_models.png",
                y_range=(0, 100),
            )

        # Bar chart: overall rate per model
        if rates:
            labels_list = [mid.split("/")[-1] for mid in rates]
            overall = [sum(r) / len(r) for r in rates.values()]
            bar_chart(
                labels=labels_list,
                values=overall,
                title=f'Overall {label} Identity Rate by Model',
                x_label="Model",
                y_label=f'% Claiming to be {label}',
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


def _load_all_results() -> dict[str, dict[float, list[dict]]]:
    """Load cached results for all models that have data."""
    all_results: dict[str, dict[float, list[dict]]] = {}
    for model in ALL_MODELS:
        cache = load_cache(model)
        if not cache:
            continue
        results: dict[float, list[dict]] = {}
        for temp in TEMPERATURES:
            results[temp] = get_results(cache, SEED_CONVO_2, temp)[:ITERATIONS_PER_TEMP]
        if any(results[t] for t in TEMPERATURES):
            all_results[model] = results
    return all_results


def regenerate_graphs_from_cache():
    """Regenerate all graphs from existing cached results."""
    all_results = _load_all_results()

    if not all_results:
        print("No cached results found. Run the experiment first.")
        return

    claude_rates = {m: compute_rates(r, "judge_claude") for m, r in all_results.items()}
    chatgpt_rates = {m: compute_rates(r, "judge_chatgpt") for m, r in all_results.items()}

    print(f"Regenerating graphs for {len(all_results)} model(s)...")
    generate_graphs(claude_rates, chatgpt_rates)


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

    output = {
        "_meta": {
            "study": "claude_identity",
            "description": "Tests whether Chinese AI models claim to be Claude/Anthropic or ChatGPT/OpenAI when asked their identity across a temperature sweep.",
            "temperatures": TEMPERATURES,
            "iterations_per_temp": ITERATIONS_PER_TEMP,
            "schema": {
                "<company>": {
                    "_type": "object — keyed by lowercase company name (e.g. 'qwen', 'deepseek')",
                    "<model>": {
                        "_type": "object — keyed by model short name (e.g. 'qwen3-235b-a22b-2507')",
                        "total_claude_perc": "string — overall % claiming Claude (e.g. '85.5%')",
                        "total_chatgpt_perc": "string — overall % claiming ChatGPT (e.g. '1.8%')",
                        "total_valid": "int — number of successful API responses",
                        "total_errors": "int — number of failed API calls",
                        "claude_results": [{
                            "temperature": "float — temperature value (0.0 to 2.0)",
                            "positive": "string[] — responses where model claimed to be Claude",
                            "negative": "string[] — responses where model did NOT claim to be Claude",
                        }],
                        "chatgpt_results": [{
                            "temperature": "float — temperature value (0.0 to 2.0)",
                            "positive": "string[] — responses where model claimed to be ChatGPT/OpenAI",
                            "negative": "string[] — responses where model did NOT claim to be ChatGPT/OpenAI",
                        }],
                    },
                },
            },
            "graphs": [f"graphs/{f}" for f in graph_files],
        },
    }

    for model_id, results in all_results.items():
        company = model_to_company.get(model_id, "unknown")
        model_name = model_id.split("/")[-1]

        if company not in output:
            output[company] = {}

        claude_results = []
        chatgpt_results = []
        total_claude_pos = 0
        total_chatgpt_pos = 0
        total_count = 0
        total_errors = 0

        for temp in TEMPERATURES:
            temp_results = results.get(temp, [])
            valid = [r for r in temp_results if r.get("error") is None]
            errored = [r for r in temp_results if r.get("error") is not None]

            claude_pos = [r["response"] for r in valid if r.get("judge_claude")]
            claude_neg = [r["response"] for r in valid if not r.get("judge_claude")]
            chatgpt_pos = [r["response"] for r in valid if r.get("judge_chatgpt")]
            chatgpt_neg = [r["response"] for r in valid if not r.get("judge_chatgpt")]

            claude_results.append({
                "temperature": temp,
                "positive": claude_pos,
                "negative": claude_neg,
            })
            chatgpt_results.append({
                "temperature": temp,
                "positive": chatgpt_pos,
                "negative": chatgpt_neg,
            })

            total_claude_pos += len(claude_pos)
            total_chatgpt_pos += len(chatgpt_pos)
            total_count += len(valid)
            total_errors += len(errored)

        claude_perc = (total_claude_pos / total_count * 100) if total_count else 0
        chatgpt_perc = (total_chatgpt_pos / total_count * 100) if total_count else 0

        output[company][model_name] = {
            "total_claude_perc": f"{claude_perc:.1f}%",
            "total_chatgpt_perc": f"{chatgpt_perc:.1f}%",
            "total_valid": total_count,
            "total_errors": total_errors,
            "claude_results": claude_results,
            "chatgpt_results": chatgpt_results,
        }

    output_path = GRAPHS_DIR.parent / "frontend.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nFrontend JSON exported to {output_path}")


def print_summary():
    """Print a text summary of all results with highlights and relevant file paths."""
    all_results = _load_all_results()

    if not all_results:
        print("No cached results found. Run the experiment first.")
        return

    # Reverse mapping for display
    model_to_company = {}
    for company, model_ids in MODELS.items():
        for mid in model_ids:
            model_to_company[mid] = company

    # Compute per-model stats
    stats = []
    for model_id, results in all_results.items():
        claude_rates = compute_rates(results, "judge_claude")
        chatgpt_rates = compute_rates(results, "judge_chatgpt")
        c_avg = sum(claude_rates) / len(claude_rates)
        g_avg = sum(chatgpt_rates) / len(chatgpt_rates)

        # Find peak temperature for claude claims
        peak_temp_idx = claude_rates.index(max(claude_rates))
        peak_temp = TEMPERATURES[peak_temp_idx]
        peak_rate = claude_rates[peak_temp_idx]

        stats.append({
            "model_id": model_id,
            "short_name": model_id.split("/")[-1],
            "company": model_to_company.get(model_id, "Unknown"),
            "claude_avg": c_avg,
            "chatgpt_avg": g_avg,
            "claude_rates": claude_rates,
            "chatgpt_rates": chatgpt_rates,
            "claude_peak_temp": peak_temp,
            "claude_peak_rate": peak_rate,
        })

    # Sort by Claude claim rate descending
    stats.sort(key=lambda s: s["claude_avg"], reverse=True)

    # Print table
    print(f"\n{'=' * 80}")
    print("  RESULTS SUMMARY: Claude Identity in Chinese AI Models")
    print(f"{'=' * 80}")
    print(f"\n{'Model':<30} {'Company':<12} {'Claude %':>10} {'ChatGPT %':>10} {'Peak Temp':>10}")
    print("-" * 80)
    for s in stats:
        print(f"  {s['short_name']:<28} {s['company']:<12} {s['claude_avg']:>8.1f}% {s['chatgpt_avg']:>8.1f}%    t={s['claude_peak_temp']}")
    print("-" * 80)

    # Highlights
    print(f"\n{'=' * 80}")
    print("  HIGHLIGHTS")
    print(f"{'=' * 80}")

    if stats:
        top = stats[0]
        bottom = stats[-1]

        if top["claude_avg"] > 0:
            print(f"\n  Highest Claude claim rate: {top['short_name']} ({top['company']}) at {top['claude_avg']:.1f}%")
            print(f"    Peak: {top['claude_peak_rate']:.0f}% at temperature {top['claude_peak_temp']}")
            safe = top['model_id'].replace('/', '_')
            print(f"    Graph: graphs/claude_{safe}.png")

        if bottom["claude_avg"] < top["claude_avg"]:
            print(f"\n  Lowest Claude claim rate: {bottom['short_name']} ({bottom['company']}) at {bottom['claude_avg']:.1f}%")

        # Any model claiming ChatGPT?
        chatgpt_claimers = [s for s in stats if s["chatgpt_avg"] > 0]
        if chatgpt_claimers:
            print(f"\n  Models claiming ChatGPT/OpenAI:")
            for s in chatgpt_claimers:
                print(f"    - {s['short_name']}: {s['chatgpt_avg']:.1f}%")

        # Temperature effect: biggest delta between lowest and highest temp
        biggest_swing = max(stats, key=lambda s: max(s["claude_rates"]) - min(s["claude_rates"]))
        swing = max(biggest_swing["claude_rates"]) - min(biggest_swing["claude_rates"])
        if swing > 0:
            print(f"\n  Biggest temperature swing: {biggest_swing['short_name']} "
                  f"({min(biggest_swing['claude_rates']):.0f}% -> {max(biggest_swing['claude_rates']):.0f}%, "
                  f"delta={swing:.0f}pp)")

        # Company averages
        company_totals: dict[str, list[float]] = {}
        for s in stats:
            company_totals.setdefault(s["company"], []).append(s["claude_avg"])
        if len(company_totals) > 1:
            print(f"\n  Company averages (Claude claims):")
            for company, avgs in sorted(company_totals.items(), key=lambda x: -sum(x[1])/len(x[1])):
                avg = sum(avgs) / len(avgs)
                print(f"    - {company}: {avg:.1f}%")

    # Key file paths
    print(f"\n{'=' * 80}")
    print("  KEY FILES")
    print(f"{'=' * 80}")
    output_dir = GRAPHS_DIR.parent
    print(f"\n  Frontend JSON:       {output_dir / 'frontend.json'}")
    print(f"  All-models (Claude): {GRAPHS_DIR / 'claude_all_models.png'}")
    print(f"  All-models (GPT):    {GRAPHS_DIR / 'chatgpt_all_models.png'}")
    print(f"  Bar chart (Claude):  {GRAPHS_DIR / 'claude_overall_by_model.png'}")
    print(f"  Bar chart (GPT):     {GRAPHS_DIR / 'chatgpt_overall_by_model.png'}")

    # Per-company combined graphs
    print(f"\n  Per-company combined graphs:")
    for company in MODELS:
        c_path = GRAPHS_DIR / f"claude_{company.lower()}_combined.png"
        if c_path.exists():
            print(f"    - {c_path}")

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

        total = 0
        errors = 0
        for temp in TEMPERATURES:
            results = get_results(cache, SEED_CONVO_2, temp)
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

    # Compute rates and print per-model summaries
    all_claude: dict[str, list[float]] = {}
    all_chatgpt: dict[str, list[float]] = {}
    for model in models:
        all_claude[model] = compute_rates(all_results[model], "judge_claude")
        all_chatgpt[model] = compute_rates(all_results[model], "judge_chatgpt")

        c_avg = sum(all_claude[model]) / len(all_claude[model])
        g_avg = sum(all_chatgpt[model]) / len(all_chatgpt[model])
        print(f"\n  {model}: claude={c_avg:.1f}%, chatgpt={g_avg:.1f}%")

    # Generate graphs
    generate_graphs(all_claude, all_chatgpt)

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for model in models:
        c_avg = sum(all_claude[model]) / len(all_claude[model])
        g_avg = sum(all_chatgpt[model]) / len(all_chatgpt[model])
        print(f"  {model}: claude={c_avg:.1f}%, chatgpt={g_avg:.1f}%")


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
