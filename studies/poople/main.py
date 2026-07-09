"""
Poople Study — change one letter at a time (every step a valid word) to reach "poop".

Part 1 — Solver: build the four-letter word-ladder graph, BFS out from "poop",
and save the minimum distance + every optimal ladder for each reachable word
(the grading oracle). See README.md.

Part 2 — LLM test: ask models to solve a difficulty-stratified sample of puzzles
as strict JSON (one-letter changes), with reasoning forced off, and grade each
attempt for legality and steps-over-par against the oracle.

Run via the top-level menu, or `python -m studies.poople.main`.
"""

import asyncio
import json
from collections import Counter
from pathlib import Path

from studies.poople.config import (
    CONDITIONS,
    ITERATIONS,
    MAX_SAVED_PATHS,
    OUTPUT_DIR,
    SAMPLE_BUCKETS,
    SAMPLE_PER_BUCKET,
    TARGET,
    TEMPERATURE,
)
from studies.poople.wordlist import load_words
from studies.poople.solver import (
    build_solution_oracle,
    enumerate_optimal_paths,
    validate_ladder,
)
from studies.poople.sampling import flat_test_words, sample_test_words
from studies.poople.llm_cache import graphs_dir, results_images_dir

STUDY_DIR = Path(__file__).parent
SOLUTIONS_DIR = STUDY_DIR / OUTPUT_DIR / "solutions"

DISTANCES_FILE = SOLUTIONS_DIR / "distances.json"
PATHS_FILE = SOLUTIONS_DIR / "optimal_paths.json"
UNREACHABLE_FILE = SOLUTIONS_DIR / "unreachable.json"


# ===========================================================================
# PART 1 — SOLVER
# ===========================================================================

def build_and_save() -> dict:
    """Build the oracle and persist all three solution artifacts."""
    print("\n--- Loading word list ---")
    words = load_words()
    print(f"  {len(words)} valid {len(TARGET)}-letter words")

    print(f"\n--- Building graph + BFS from '{TARGET}' ---")
    oracle = build_solution_oracle(words, TARGET)
    adj, dist, counts = oracle["adj"], oracle["dist"], oracle["counts"]
    reachable = sorted(dist, key=lambda w: (dist[w], w))
    unreachable = sorted(words - set(dist))
    print(f"  {len(reachable)} reachable ({len(reachable)/len(words)*100:.1f}%), "
          f"{len(unreachable)} unreachable")

    print("\n--- Enumerating optimal paths ---")
    paths_out: dict[str, dict] = {}
    truncated_words: list[str] = []
    for w in reachable:
        n = counts[w]
        paths = enumerate_optimal_paths(adj, dist, w, TARGET, cap=MAX_SAVED_PATHS)
        truncated = n > len(paths)
        if truncated:
            truncated_words.append(w)
        paths_out[w] = {
            "dist": dist[w],
            "num_optimal_paths": n,
            "truncated": truncated,
            "paths": paths,
        }

    SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DISTANCES_FILE, "w") as f:
        json.dump({
            "target": TARGET,
            "word_len": len(TARGET),
            "total_words": len(words),
            "reachable": len(reachable),
            "unreachable": len(unreachable),
            "max_distance": max(dist.values()),
            "distances": {w: dist[w] for w in reachable},
        }, f, indent=2)
    with open(PATHS_FILE, "w") as f:
        json.dump(paths_out, f, indent=2)
    with open(UNREACHABLE_FILE, "w") as f:
        json.dump(unreachable, f, indent=2)

    if truncated_words:
        print(f"  NOTE: {len(truncated_words)} word(s) had more than "
              f"{MAX_SAVED_PATHS} optimal paths; saved a capped sample for those.")
    print(f"\n  Saved:\n    {DISTANCES_FILE}\n    {PATHS_FILE}\n    {UNREACHABLE_FILE}")

    print_solver_stats(oracle, words)
    return oracle


def print_solver_stats(oracle: dict, words: set[str]):
    dist, counts = oracle["dist"], oracle["counts"]
    print("\n" + "=" * 64)
    print(f"  POOPLE SOLVER — solutions to reach '{TARGET}'")
    print("=" * 64)
    print(f"\n  Valid 4-letter words : {len(words)}")
    print(f"  Reachable from poop  : {len(dist)} ({len(dist)/len(words)*100:.1f}%)")
    print(f"  Unreachable          : {len(words) - len(dist)}")
    print(f"  Hardest (max steps)  : {max(dist.values())}")

    print("\n  Distance distribution (steps from a word to poop):")
    dd = Counter(dist.values())
    peak = max(dd.values())
    for d in sorted(dd):
        bar = "█" * max(1, round(dd[d] / peak * 40))
        print(f"    {d:>2} steps: {dd[d]:>5}  {bar}")

    hardest = sorted(dist, key=lambda w: (-dist[w], w))[:12]
    print("\n  Hardest reachable words:")
    for w in hardest:
        print(f"    {w}  ({dist[w]} steps, {counts[w]} optimal path(s))")

    most_paths = sorted(counts, key=lambda w: (-counts[w], w))[:12]
    print("\n  Most ambiguous (most distinct optimal ladders):")
    for w in most_paths:
        print(f"    {w}  ({counts[w]} optimal paths, {dist[w]} steps)")


def verify_saved():
    """Re-validate every saved optimal path against the rules from scratch."""
    print("\n--- Verifying saved solutions ---")
    if not PATHS_FILE.exists() or not DISTANCES_FILE.exists():
        print("  No saved solutions found. Run the build first.")
        return

    words = load_words()
    with open(DISTANCES_FILE) as f:
        distances_doc = json.load(f)
    with open(PATHS_FILE) as f:
        paths_out = json.load(f)

    dist_map = distances_doc["distances"]
    oracle = build_solution_oracle(words, TARGET)
    dist, counts = oracle["dist"], oracle["counts"]

    errors = 0
    checked_paths = 0
    if dist_map != {w: dist[w] for w in dist}:
        print("  ✗ distance map does not match a fresh BFS")
        errors += 1
    else:
        print(f"  ✓ distance map matches fresh BFS ({len(dist_map)} words)")

    for w, info in paths_out.items():
        if not info["truncated"] and info["num_optimal_paths"] != counts[w]:
            print(f"  ✗ {w}: saved count {info['num_optimal_paths']} != fresh {counts[w]}")
            errors += 1
        for ladder in info["paths"]:
            checked_paths += 1
            ok, reason = validate_ladder(ladder, words, TARGET, start=w)
            if not ok:
                print(f"  ✗ {w}: invalid ladder {ladder} — {reason}")
                errors += 1
            elif len(ladder) - 1 != info["dist"]:
                print(f"  ✗ {w}: ladder length {len(ladder)-1} != dist {info['dist']}")
                errors += 1

    print(f"\n  Checked {checked_paths} saved ladders across {len(paths_out)} words.")
    print("  ✓ ALL CHECKS PASSED — solutions are correct and optimal." if errors == 0
          else f"  ✗ {errors} problem(s) found.")


def lookup_word():
    words = load_words()
    oracle = build_solution_oracle(words, TARGET)
    adj, dist, counts = oracle["adj"], oracle["dist"], oracle["counts"]
    while True:
        w = input("\n  Enter a 4-letter word (or blank to exit): ").strip().lower()
        if not w:
            return
        if w not in words:
            print(f"    '{w}' is not a valid 4-letter word in this list.")
            continue
        if w not in dist:
            print(f"    '{w}' is valid but cannot reach '{TARGET}'.")
            continue
        print(f"    {w}: {dist[w]} steps, {counts[w]} optimal path(s)")
        for p in enumerate_optimal_paths(adj, dist, w, TARGET, cap=5):
            print("      " + " -> ".join(p))
        if counts[w] > 5:
            print(f"      ... and {counts[w] - 5} more")


# ===========================================================================
# PART 2 — LLM TEST
# ===========================================================================

def _load_oracle() -> tuple[set[str], dict]:
    """Words + oracle, preferring the saved distance map but rebuilding if absent."""
    words = load_words()
    oracle = build_solution_oracle(words, TARGET)
    return words, oracle


def _print_sample(sample: dict[int, list[str]]):
    print("\n  Test words (difficulty-stratified, seeded sample):")
    for d in sorted(sample):
        print(f"    par {d} ({len(sample[d])}): {', '.join(sample[d])}")


def _subtitle(test_words, condition: str) -> str:
    label = CONDITIONS[condition]["label"]
    return (f"{len(test_words)} words · par {min(SAMPLE_BUCKETS)}–{max(SAMPLE_BUCKETS)} · "
            f"{ITERATIONS} attempts each · {label} · temp {TEMPERATURE}")


def _build_condition_outputs(condition: str, words, oracle, test_words):
    """Compute stats, graphs, result images and print the summary for one set."""
    from studies.poople.analysis import compute_model_stats
    from studies.poople.visualize import generate_graphs
    from studies.poople.result_images import generate_result_images

    models = CONDITIONS[condition]["models"]
    stats = [compute_model_stats(m, test_words, condition) for m in models]
    present = [m for m, s in zip(models, stats) if s["overall"]["n"] > 0]
    stats = [s for s in stats if s["overall"]["n"] > 0]
    if not stats:
        print(f"  [{condition}] no cached results yet — skipping graphs.")
        return
    generate_graphs(stats, graphs_dir(condition), subtitle=_subtitle(test_words, condition))
    generate_result_images(present, words, oracle["dist"], condition)
    print_llm_summary(stats, condition)


async def run_llm_experiment(conditions: list[str]):
    """Run one or more sets (reasoning / no_reasoning), then build their outputs."""
    from services.llm import OpenRouterProvider

    from studies.poople.runner import run_all

    words, oracle = _load_oracle()
    test_words = flat_test_words(oracle["dist"])
    _print_sample(sample_test_words(oracle["dist"]))

    provider = OpenRouterProvider()
    for condition in conditions:
        cfg = CONDITIONS[condition]
        print(f"\n{'=' * 70}\n  SET: {condition}  ({cfg['label']})  ·  "
              f"{len(cfg['models'])} models\n{'=' * 70}")
        await run_all(provider, cfg["models"], test_words, words,
                      condition, cfg["enable_reasoning"])
        _build_condition_outputs(condition, words, oracle, test_words)


def regenerate_llm_graphs():
    from studies.poople.export import export_all

    words, oracle = _load_oracle()
    test_words = flat_test_words(oracle["dist"])
    for condition in CONDITIONS:
        _build_condition_outputs(condition, words, oracle, test_words)
    print("\n--- Exporting results data ---")
    export_all()


def print_llm_summary(stats: list[dict], condition: str):
    label = CONDITIONS[condition]["label"]
    print("\n" + "=" * 92)
    print(f"  POOPLE — Summary [{condition} · {label}]  (+k = k steps over optimal)")
    print("=" * 92)
    print(f"\n  {'Model':<28} {'n':>5} {'Solve%':>8} {'Avg +par':>9} "
          f"{'Illegal/try':>12} {'% w/ illegal':>13}")
    print("  " + "-" * 88)
    for s in sorted(stats, key=lambda s: (-s['overall']['solve_rate'])):
        o = s["overall"]
        avg = "—" if o["avg_over_par"] is None else f"+{o['avg_over_par']:.2f}"
        print(f"  {s['model'].split('/')[-1]:<28} {o['n']:>5} {o['solve_rate']:>7.0f}% "
              f"{avg:>9} {o['illegal_per_attempt']:>12.2f} {o['pct_with_illegal']:>12.0f}%")
    print("  " + "-" * 88)

    print("\n  By difficulty (solve% / avg +par):")
    buckets = sorted({b for s in stats for b in s["by_bucket"]})
    print("  " + f"{'Model':<28}" + "".join(f"{'par '+str(b):>16}" for b in buckets))
    for s in stats:
        row = f"  {s['model'].split('/')[-1]:<28}"
        for b in buckets:
            bb = s["by_bucket"][b]
            avg = "—" if bb["avg_over_par"] is None else f"+{bb['avg_over_par']:.2f}"
            cell = f"{bb['solve_rate']:.0f}% / {avg}"
            row += f"{cell:>16}"
        print(row)

    gdir = graphs_dir(condition)
    if gdir.exists():
        print(f"\n  Graphs: {gdir}/")
        print(f"  Result images: {results_images_dir(condition)}/")


# ===========================================================================
# Menu
# ===========================================================================

def main():
    print("\n" + "=" * 60)
    print("  STUDY: Poople (word ladder to 'poop')")
    print("=" * 60)
    print(f"\n  Target: {TARGET}  ·  sample: {SAMPLE_PER_BUCKET}/bucket "
          f"at par {SAMPLE_BUCKETS}  ·  {ITERATIONS} attempts/word")
    print("  Sets:")
    for cond, cfg in CONDITIONS.items():
        print(f"    · {cond:<13} ({cfg['label']}): "
              f"{', '.join(m.split('/')[-1] for m in cfg['models'])}")
    print("\n  Solver:")
    print("    [1] Build + save solutions (BFS from poop)")
    print("    [2] Verify saved solutions")
    print("    [3] Solver stats")
    print("    [4] Look up a word's optimal solutions")
    print("  LLM test:")
    print("    [5] Run BOTH sets (reasoning + no-reasoning)")
    print("    [6] Run the reasoning set only")
    print("    [7] Run the no-reasoning set only")
    print("    [8] Regenerate graphs + result images + summaries from cache")
    print("    [q] Quit")

    choice = input("\n  Choice: ").strip().lower()
    if choice == "1":
        build_and_save()
    elif choice == "2":
        verify_saved()
    elif choice == "3":
        words = load_words()
        print_solver_stats(build_solution_oracle(words, TARGET), words)
    elif choice == "4":
        lookup_word()
    elif choice in ("5", "6", "7"):
        conditions = {"5": list(CONDITIONS), "6": ["reasoning"], "7": ["no_reasoning"]}[choice]
        for cond in conditions:
            print(f"  · {cond} ({CONDITIONS[cond]['label']}): "
                  f"{len(CONDITIONS[cond]['models'])} models")
        if input("\n  Proceed? [Y/n]: ").strip().lower() == "n":
            print("  Cancelled.")
            return
        asyncio.run(run_llm_experiment(conditions))
    elif choice == "8":
        regenerate_llm_graphs()
    else:
        print("  Bye!")


if __name__ == "__main__":
    main()
