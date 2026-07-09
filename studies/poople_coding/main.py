"""
Poople Coding benchmark.

Each reasoning model gets ONE shot to write a Python program that solves Poople
optimally. We execute each program against a distance-stratified battery of words
and grade its output (legal? optimal?) with the poople study's BFS oracle.

Pipeline:
  1. Generate one program per model (reasoning ON) — cached + saved to scripts/.
  2. Run each program in a sandboxed subprocess against the test battery.
  3. Grade vs the oracle, chart, and export results.

Run via the top-level menu, or `python -m studies.poople_coding.main`.
"""

import asyncio

from studies.poople_coding.config import MODELS, TEMPERATURE, TIMEOUT_SECONDS
from studies.poople_coding.cache import GRAPHS_DIR, SCRIPTS_DIR, load_code
from studies.poople_coding.runner import generate_all
from studies.poople_coding.evaluator import evaluate_all
from studies.poople_coding.visualize import generate_graphs
from studies.poople_coding.export import export_all


def _subtitle(battery: dict) -> str:
    return (f"{battery['n_reachable']} reachable + {battery['n_unreachable']} unreachable "
            f"test words · one shot · reasoning ON · stdlib-only Python")


def _print_summary(results: list[dict]):
    print("\n" + "=" * 84)
    print("  POOPLE CODING — Summary  (models write a program; we run + grade it)")
    print("=" * 84)
    print(f"\n  {'Model':<26} {'Ran':>4} {'Optimal%':>9} {'Valid%':>8} "
          f"{'Unreach':>9} {'Time(s)':>8}")
    print("  " + "-" * 78)
    for r in sorted(results, key=lambda r: (-r["optimal_rate"], -r["valid_rate"])):
        ran = "yes" if r["ran"] else "NO"
        unreach = f"{r['unreachable_correct']}/{r['n_unreachable']}"
        print(f"  {r['short_name']:<26} {ran:>4} {r['optimal_rate']:>8.0f}% "
              f"{r['valid_rate']:>7.0f}% {unreach:>9} {str(r['elapsed_sec']):>8}")
        if not r["ran"] and r["error"]:
            print(f"      ↳ {r['error'][:72]}")
    print("  " + "-" * 78)


def _evaluate_and_report(programs: dict):
    from studies.poople_coding.compare import generate_hand_vs_code

    results, battery = evaluate_all(programs)
    generate_graphs(results, GRAPHS_DIR, subtitle=_subtitle(battery))
    generate_hand_vs_code(GRAPHS_DIR / "hand_vs_code.png")
    print("\n--- Exporting results data ---")
    export_all(results, battery)
    _print_summary(results)
    print(f"\n  Generated programs: {SCRIPTS_DIR}/")


async def run_benchmark(models: list[str]):
    print("\n--- Step 1: Generating programs (one shot, reasoning ON) ---")
    programs = await generate_all(models)
    print("\n--- Step 2: Running + grading programs ---")
    _evaluate_and_report(programs)


def regenerate_from_cache():
    """Re-run + re-grade cached programs (no API calls)."""
    programs = {}
    for m in MODELS:
        entry = load_code(m)
        if entry:
            programs[m] = entry
    if not programs:
        print("  No cached programs. Run the benchmark first.")
        return
    _evaluate_and_report(programs)


def main():
    print("\n" + "=" * 64)
    print("  STUDY: Poople Coding (LLMs write a solver, we run it)")
    print("=" * 64)
    print(f"\n  Models (reasoning only): {', '.join(m.split('/')[-1] for m in MODELS)}")
    print(f"  One shot · reasoning ON · temp {TEMPERATURE} · timeout {TIMEOUT_SECONDS}s")
    print("\n  [1] Run benchmark (generate programs + run + grade)")
    print("  [2] Re-run + re-grade cached programs (no API calls)")
    print("  [q] Quit")

    choice = input("\n  Choice: ").strip().lower()
    if choice == "1":
        if input("  This generates code from each model and EXECUTES it locally. Proceed? [Y/n]: "
                 ).strip().lower() == "n":
            print("  Cancelled.")
            return
        asyncio.run(run_benchmark(MODELS))
    elif choice == "2":
        regenerate_from_cache()
    else:
        print("  Bye!")


if __name__ == "__main__":
    main()
