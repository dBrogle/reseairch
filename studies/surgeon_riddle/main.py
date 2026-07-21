"""
Surgeon Riddle Study

The classic "surgeon is the mother" riddle, run in two conditions that differ
ONLY in the driving parent's gender:

  - father  ("A man and his son ..."):   correct aha  = doctor is the MOTHER.
  - mother  ("A woman and her son ..."):  the mother is already the driver, so
        the correct answer is the FATHER (or another parent). Answering "the
        doctor is his mother" is the pattern-match FAILURE.

Pipeline:
  1. Collect raw conversational answers (model x condition x ITERATIONS) - cached
  2. Classify which parent each answer names via an LLM judge - cached separately
  3. Print a summary comparing the two conditions per model
"""

import asyncio

from services.llm import OpenRouterProvider
from studies.surgeon_riddle.config import (
    MODELS,
    CONDITIONS,
    TEMPERATURE,
    ITERATIONS,
    REASONING_ITERATIONS,
    REASONING_LOW,
    JUDGE_MODEL,
)
from studies.surgeon_riddle.runner import run_all
from studies.surgeon_riddle.judge import judge_all, compute_scores, correct_rate
from studies.surgeon_riddle.visualize import generate_all as generate_graphs
from studies.surgeon_riddle.cache import RESULTS_DIR


def _pct(n: int, d: int) -> str:
    return f"{n / d:.0%}" if d > 0 else "n/a"


def print_summary():
    overall = []  # (model, flip_fail_rate)

    for model in MODELS:
        short = model.split("/")[-1]
        scores = compute_scores(model)

        if not any(s.get("total", 0) > 0 for s in scores.values()):
            continue

        print(f"\n{'=' * 72}")
        print(f"  {short}")
        print(f"{'=' * 72}")

        for cond in CONDITIONS:
            s = scores[cond["id"]]
            answered = s["total"]
            dist = (f"MOTHER {s['MOTHER']}  FATHER {s['FATHER']}  "
                    f"TWO_SAME {s['TWO_SAME']}  OTHER_PARENT {s['OTHER_PARENT']}  "
                    f"OTHER {s['OTHER']}"
                    + (f"  err {s['error']}" if s["error"] else ""))
            solved = correct_rate(s, cond["id"])
            solved_str = f"{solved:.0%}" if solved is not None else "n/a"

            if cond["id"] == "father":
                tag = f"solved {solved_str} (mother = the aha)"
            else:
                tag = (f"mother-trap {_pct(s['MOTHER'], answered)} (FAILURE)  |  "
                       f"solved {solved_str} (father / two-moms / other)")

            print(f"  {cond['label']:<22}: {dist}")
            print(f"  {'':<22}  -> {tag}")

        flip = scores["mother"]
        fail_rate = flip["MOTHER"] / flip["total"] if flip["total"] else None
        overall.append((short, fail_rate))

    # Headline table
    print(f"\n{'=' * 72}")
    print("  HEADLINE — flipped-riddle failure (says the doctor is the mother")
    print("  even though the mother is the one driving)")
    print(f"{'=' * 72}")
    print(f"  {'model':<26} {'flipped: mother-trap':>22}")
    print(f"  {'-' * 50}")
    overall.sort(key=lambda t: t[1] if t[1] is not None else -1, reverse=True)
    for short, fail_rate in overall:
        f = f"{fail_rate:.0%}" if fail_rate is not None else "n/a"
        bar_len = int((fail_rate or 0) * 20)
        bar = "#" * bar_len + "." * (20 - bar_len)
        print(f"  {short:<26} {f:>10} [{bar}]")


async def run_experiment(models: list[str]):
    provider = OpenRouterProvider()

    print("\n--- Step 1: Collecting raw conversational answers ---")
    await run_all(provider, models)

    print("\n--- Step 2: Classifying which parent each answer names ---")
    await judge_all(provider, models)

    print("\n--- Step 3: Graphs ---")
    generate_graphs(models)

    print("\n--- Step 4: Summary ---")
    print_summary()


def select_models() -> list[str] | str:
    print("\n=== Model Selection ===")
    print("[1] Run ALL models")
    print("[2] Print results summary from cache")
    print("[3] Regenerate graphs from cache")
    choice = input("\nChoice: ").strip()
    if choice == "2":
        return "print_summary"
    if choice == "3":
        return "regenerate_graphs"
    return MODELS


def main():
    print("\n" + "=" * 60)
    print("  STUDY: Surgeon Riddle (gender pattern-match)")
    print("=" * 60)
    print(f"\n  Conditions: {len(CONDITIONS)} (identical but for parent gender)")
    for c in CONDITIONS:
        print(f"    - {c['id']}: {c['prompt']}")
    print(f"  Models: {len(MODELS)}  ({len(REASONING_LOW)} reasoning-forced)")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Iterations per (model, condition): {ITERATIONS} "
          f"(reasoning-forced: {REASONING_ITERATIONS})")
    print(f"  Judge model: {JUDGE_MODEL}")
    print(f"  Results cache: {RESULTS_DIR}/")

    models = select_models()
    if models == "print_summary":
        print_summary()
        return
    if models == "regenerate_graphs":
        generate_graphs(MODELS)
        return

    asyncio.run(run_experiment(models))


if __name__ == "__main__":
    main()
