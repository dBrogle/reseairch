"""
Claude Identity Study

Tests whether Chinese AI models claim to be Claude/Anthropic at various temperatures.
For each model, runs multiple iterations across a temperature sweep and graphs
the rate at which the model identifies itself as Claude.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from services.llm import OpenRouterProvider
from utils.graphing import line_chart, multi_line_chart
from studies.claude_identity.config import (
    MODELS,
    ALL_MODELS,
    TEMPERATURES,
    ITERATIONS_PER_TEMP,
    IDENTITY_PROMPT,
    CLAUDE_KEYWORDS,
    OUTPUT_DIR,
)

STUDY_DIR = Path(__file__).parent
OUTPUT_PATH = STUDY_DIR / OUTPUT_DIR


def mentions_claude(response: str) -> bool:
    """Check if a response contains Claude/Anthropic keywords."""
    lower = response.lower()
    return any(kw in lower for kw in CLAUDE_KEYWORDS)


async def run_single_query(
    provider: OpenRouterProvider, model: str, temperature: float
) -> dict:
    """Run a single identity query and return the result."""
    try:
        response = await provider.complete_text(
            prompt=IDENTITY_PROMPT,
            model=model,
            temperature=temperature,
            max_tokens=300,
        )
        return {
            "response": response,
            "mentions_claude": mentions_claude(response),
            "error": None,
        }
    except Exception as e:
        return {
            "response": None,
            "mentions_claude": False,
            "error": str(e),
        }


async def run_model_experiment(
    provider: OpenRouterProvider, model: str
) -> dict[float, list[dict]]:
    """
    Run the full temperature sweep for a single model.
    Returns {temperature: [results]} mapping.
    """
    results: dict[float, list[dict]] = {}
    total = len(TEMPERATURES) * ITERATIONS_PER_TEMP
    done = 0

    for temp in TEMPERATURES:
        temp_results = []
        for i in range(ITERATIONS_PER_TEMP):
            result = await run_single_query(provider, model, temp)
            temp_results.append(result)
            done += 1
            status = "CLAUDE" if result["mentions_claude"] else "ok"
            if result["error"]:
                status = "ERROR"
            print(f"  [{done}/{total}] temp={temp} iter={i+1} -> {status}")

        results[temp] = temp_results

    return results


def compute_claude_rates(results: dict[float, list[dict]]) -> list[float]:
    """Compute the % of responses mentioning Claude at each temperature."""
    rates = []
    for temp in TEMPERATURES:
        temp_results = results[temp]
        valid = [r for r in temp_results if r["error"] is None]
        if valid:
            rate = sum(1 for r in valid if r["mentions_claude"]) / len(valid) * 100
        else:
            rate = 0.0
        rates.append(rate)
    return rates


def save_results(all_results: dict[str, dict], rates: dict[str, list[float]]):
    """Save raw results and rates to JSON."""
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save raw results
    raw_path = OUTPUT_PATH / f"raw_results_{timestamp}.json"
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nRaw results saved to {raw_path}")

    # Save rates summary
    summary = {
        "temperatures": TEMPERATURES,
        "rates": rates,
        "config": {
            "iterations_per_temp": ITERATIONS_PER_TEMP,
            "prompt": IDENTITY_PROMPT,
            "claude_keywords": CLAUDE_KEYWORDS,
        },
    }
    summary_path = OUTPUT_PATH / f"summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    return timestamp


def generate_graphs(rates: dict[str, list[float]], timestamp: str):
    """Generate per-model and combined graphs."""
    graphs_dir = OUTPUT_PATH / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # Per-model graphs
    for model_id, model_rates in rates.items():
        safe_name = model_id.replace("/", "_")
        line_chart(
            x_values=TEMPERATURES,
            y_values=model_rates,
            title=f"Claude Identity Rate: {model_id}",
            x_label="Temperature",
            y_label='% Responses Mentioning "Claude"',
            save_path=graphs_dir / f"{safe_name}_{timestamp}.png",
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
                title=f"Claude Identity Rate: {company} Models",
                x_label="Temperature",
                y_label='% Responses Mentioning "Claude"',
                save_path=graphs_dir / f"{company.lower()}_{timestamp}.png",
                y_range=(0, 100),
            )

    # Combined graph with all models
    multi_line_chart(
        x_values=TEMPERATURES,
        series=rates,
        title="Claude Identity Rate: All Models",
        x_label="Temperature",
        y_label='% Responses Mentioning "Claude"',
        save_path=graphs_dir / f"all_models_{timestamp}.png",
        y_range=(0, 100),
    )

    print(f"Graphs saved to {graphs_dir}/")


def select_models() -> list[str]:
    """Interactive model selection menu."""
    print("\n=== Model Selection ===")
    print("[1] Run ALL models")
    print("[2] Select by company")
    print("[3] Select individual models")

    choice = input("\nChoice: ").strip()

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


async def run_experiment(models: list[str]):
    """Run the full experiment for selected models."""
    provider = OpenRouterProvider()

    all_results: dict[str, dict] = {}
    all_rates: dict[str, list[float]] = {}

    for i, model in enumerate(models, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(models)}] Testing: {model}")
        print(f"{'='*60}")

        results = await run_model_experiment(provider, model)
        all_results[model] = {str(k): v for k, v in results.items()}

        rates = compute_claude_rates(results)
        all_rates[model] = rates

        # Print summary for this model
        avg_rate = sum(rates) / len(rates)
        max_rate = max(rates)
        max_temp = TEMPERATURES[rates.index(max_rate)]
        print(f"\n  Average Claude rate: {avg_rate:.1f}%")
        print(f"  Peak: {max_rate:.1f}% at temperature {max_temp}")

    # Save and graph
    timestamp = save_results(all_results, all_rates)
    generate_graphs(all_rates, timestamp)

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for model, rates in all_rates.items():
        avg = sum(rates) / len(rates)
        peak = max(rates)
        print(f"  {model}: avg={avg:.1f}%, peak={peak:.1f}%")


def main():
    """Entry point for the Claude Identity study."""
    print("\n" + "=" * 60)
    print("  STUDY: Claude Identity in Chinese AI Models")
    print("=" * 60)
    print(f"\nPrompt: \"{IDENTITY_PROMPT}\"")
    print(f"Temperatures: {TEMPERATURES}")
    print(f"Iterations per temperature: {ITERATIONS_PER_TEMP}")

    models = select_models()
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
    # Allow running directly: python -m studies.claude_identity.main
    main()
