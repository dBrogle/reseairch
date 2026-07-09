"""Export the Poople results as a self-contained data file for downstream use.

Writes two siblings into output/:
  * results_export.json — exhaustive, machine-readable: methodology, solver
    oracle facts, the sampled words, per-model stats for both conditions, the
    reasoning-vs-one-shot gap, concrete example ladders, asset paths, findings.
  * RESULTS.md          — a readable digest of the same, so a writer/agent can
    lift copy directly.

Nothing here is webpage code; it's just the data and prose summary.
"""

import datetime
import json
from collections import Counter
from pathlib import Path

from studies.poople.config import (
    CONDITIONS,
    ITERATIONS,
    MAX_TOKENS,
    MODEL_REASONING,
    PROMPT_VERSION,
    SAMPLE_BUCKETS,
    SAMPLE_PER_BUCKET,
    SAMPLE_SEED,
    TARGET,
    TEMPERATURE,
    WORDLIST_PATH,
)
from studies.poople.wordlist import load_words
from studies.poople.solver import build_solution_oracle, enumerate_optimal_paths
from studies.poople.sampling import flat_test_words, sample_test_words
from studies.poople.analysis import compute_model_stats
from studies.poople.prompt import build_prompt
from studies.poople.llm_cache import (
    OUTPUT_ROOT,
    get_results,
    graphs_dir,
    load_cache,
    results_images_dir,
)
from studies.poople.result_images import (
    attempt_score,
    caption,
    pick_example_words,
    safe_model,
)
from studies.poople.prompt import build_messages

EXPORT_JSON = OUTPUT_ROOT / "results_export.json"
EXPORT_MD = OUTPUT_ROOT / "RESULTS.md"


def _short(model: str) -> str:
    return model.split("/")[-1]


def _round(v, n=2):
    return None if v is None else round(v, n)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _model_block(model: str, test_words, condition: str) -> dict:
    s = compute_model_stats(model, test_words, condition)
    o = s["overall"]
    if o["n"] == 0:
        return None
    return {
        "model": model,
        "short_name": _short(model),
        "n_attempts": o["n"],
        "solve_rate_pct": _round(o["solve_rate"], 1),
        "avg_over_par": _round(o["avg_over_par"]),
        "illegal_moves_per_attempt": _round(o["illegal_per_attempt"]),
        "pct_attempts_with_illegal_move": _round(o["pct_with_illegal"], 1),
        "outcomes_count": o["outcomes"],
        "pie_count": o["pie"],
        "over_par_distribution": {str(k): v for k, v in sorted(o["over_par_dist"].items())},
        "by_difficulty": {
            f"par_{b}": {
                "solve_rate_pct": _round(bs["solve_rate"], 1),
                "avg_over_par": _round(bs["avg_over_par"]),
                "illegal_moves_per_attempt": _round(bs["illegal_per_attempt"]),
                "n_attempts": bs["n"],
            }
            for b, bs in s["by_bucket"].items()
        },
    }


def _example_attempts(words: set[str], dist: dict, test_words) -> list[dict]:
    """For each example-board word, each model's BEST attempt in each condition."""
    out = []
    for word, par in pick_example_words(dist):
        entry = {
            "word": word,
            "par": par,
            "optimal_example": enumerate_optimal_paths(
                build_solution_oracle(words)["adj"], dist, word, TARGET, cap=1
            )[0] if word in dist else None,
            "by_condition": {},
        }
        for condition, cfg in CONDITIONS.items():
            attempts = []
            for model in cfg["models"]:
                res = get_results(load_cache(model, condition), build_messages(word))[:ITERATIONS]
                if not res:
                    continue
                best = min(res, key=attempt_score)
                attempts.append({
                    "model": _short(model),
                    "ladder": best.get("ladder"),
                    "num_moves": best.get("num_moves"),
                    "illegal_moves": best.get("illegal_moves"),
                    "solved": best.get("solved"),
                    "over_par": best.get("over_par"),
                    "summary": caption(best),
                    "board_image": f"results_images_{condition}/{word}/{safe_model(model)}.png",
                })
            entry["by_condition"][condition] = attempts
        out.append(entry)
    return out


def build_export() -> dict:
    words = load_words()
    oracle = build_solution_oracle(words, TARGET)
    dist, counts = oracle["dist"], oracle["counts"]
    test_words = flat_test_words(dist)
    sample = sample_test_words(dist)

    # Per-condition model stats.
    conditions_out = {}
    for condition, cfg in CONDITIONS.items():
        blocks = [_model_block(m, test_words, condition) for m in cfg["models"]]
        blocks = [b for b in blocks if b]
        blocks.sort(key=lambda b: -(b["solve_rate_pct"] or 0))
        conditions_out[condition] = {
            "label": cfg["label"],
            "reasoning_enabled": cfg["enable_reasoning"],
            "models_tested": [_short(m) for m in cfg["models"]],
            "models": blocks,
        }

    # Reasoning-vs-no-reasoning gap for models that ran in both sets.
    def _rate(cond, model):
        for b in conditions_out[cond]["models"]:
            if b["model"] == model:
                return b["solve_rate_pct"]
        return None
    gap = []
    for model, cap in MODEL_REASONING.items():
        if cap == "both":
            r, nr = _rate("reasoning", model), _rate("no_reasoning", model)
            gap.append({
                "model": _short(model),
                "no_reasoning_solve_pct": nr,
                "reasoning_solve_pct": r,
                "lift_pct_points": (None if (r is None or nr is None) else _round(r - nr, 1)),
            })
    gap.sort(key=lambda g: -(g["reasoning_solve_pct"] or 0))

    dd = Counter(dist.values())

    return {
        "meta": {
            "study": "poople",
            "title": "Poople — can LLMs solve a word ladder to 'poop'?",
            "generated": datetime.date.today().isoformat(),
            "description": (
                "Poople is a Wordle-adjacent word-ladder puzzle: start from a four-letter "
                "word and change exactly one letter at a time — every intermediate must be a "
                "real four-letter word — until you reach 'poop'. We BFS out from 'poop' over "
                "all four-letter English words to get the optimal solution length (par) and "
                "every optimal ladder, then ask LLMs to solve a difficulty-stratified sample "
                "of puzzles as strict JSON and grade each attempt for legality and steps over "
                "par. Models are tested in two sets: one-shot with reasoning OFF, and with "
                "reasoning ON."
            ),
            "rules": [
                "Start from the given four-letter word.",
                "Each step changes exactly one letter (no adding/removing/reordering).",
                "Every word, including the last, must be a valid four-letter word.",
                "Continue until you reach 'poop'.",
                "Use as few steps as possible.",
            ],
            "target_word": TARGET,
            "dictionary": {
                "name": "ENABLE2k (enable1.txt)",
                "note": "Open, common-English word-game lexicon (a standard Scrabble-style list).",
                "file": str(WORDLIST_PATH),
            },
            "prompt_version": PROMPT_VERSION,
            "prompt_example": build_prompt("cods"),
            "temperature": TEMPERATURE,
            "attempts_per_word": ITERATIONS,
            "max_tokens": MAX_TOKENS,
            "sample_design": {
                "difficulty_buckets_par": list(SAMPLE_BUCKETS),
                "words_per_bucket": SAMPLE_PER_BUCKET,
                "seed": SAMPLE_SEED,
                "note": "Each bucket is shuffled once with the seed and a prefix taken, so "
                        "growing the sample keeps the original words (and their cached results).",
            },
            "conditions_explained": {
                "no_reasoning": "Reasoning disabled — a true one-shot answer with no thinking.",
                "reasoning": "Reasoning enabled — the model thinks before answering.",
                "model_assignment": MODEL_REASONING,
            },
        },
        "metric_definitions": {
            "par": "Optimal (minimum) number of one-letter steps from the start word to 'poop'.",
            "solved": "Ladder ends exactly on 'poop' with zero illegal moves.",
            "over_par": "For solved attempts: moves used minus par. +0 = optimal.",
            "illegal_move": "A step whose stated 'from' doesn't match the current word, OR "
                            "whose target isn't a valid word, OR that changes != 1 letter.",
            "solve_rate_pct": "Percent of attempts that were solved legally (errors count as non-solves).",
            "pie_categories": {
                "par": "Solved at par (+0).",
                "over_par": "Solved but used more than par steps.",
                "illegal": "Used at least one illegal move (even if it still reached poop).",
                "failed": "Did not reach poop (also covers unparseable / API errors).",
            },
        },
        "solver_oracle": {
            "total_four_letter_words": len(words),
            "reachable_to_poop": len(dist),
            "reachable_pct": _round(len(dist) / len(words) * 100, 1),
            "unreachable": len(words) - len(dist),
            "max_distance_steps": max(dist.values()),
            "distance_distribution": {str(d): dd[d] for d in sorted(dd)},
            "total_optimal_ladders_saved": sum(counts.values()),
            "example_optimal_paths": {
                w: enumerate_optimal_paths(oracle["adj"], dist, w, TARGET, cap=1)[0]
                for w in ["love", "word", "quiz"] if w in dist
            },
        },
        "sample_words": {f"par_{d}": sample[d] for d in sorted(sample)},
        "results": conditions_out,
        "reasoning_vs_no_reasoning": gap,
        "key_findings": [
            "One-shot (no reasoning) Poople is hard: solve rates ran ~9–21%, and most "
            "attempts that reached 'poop' did so via an illegal move (often a sneaky "
            "two-letter change like 'pope'->'poop').",
            "Turning reasoning ON roughly triples solve rates for every model.",
            "Difficulty matters sharply: one-shot solve rates collapse to ~0% on par-5 "
            "words; even with reasoning, par-5 stays the hardest tier.",
            "Models almost never make 'over par' solves — when they solve, they tend to "
            "solve optimally; the dominant failure is illegal moves, not inefficiency.",
            "claude-opus-4.8 only improved once reasoning was actually engaged "
            "(13%->17% looked flat until a provider fix sent reasoning:{enabled:true}, "
            "after which it jumped to 57%).",
        ],
        "caveats": [
            "gemini-3.1-pro-preview cannot disable reasoning (mandatory), so it only "
            "appears in the reasoning set; its one-shot ability is not measured.",
            "kimi-k2.6 returns empty content ~1/3 of the time with reasoning ON via this "
            "endpoint, so it was dropped from the reasoning set (it remains in no-reasoning).",
            "A few reasoning attempts (deepseek ~13/90, opus ~5/90) returned empty content "
            "(API errors); these count as non-solves and slightly understate those models.",
            "Validity is judged against the ENABLE word list. Some valid plays route through "
            "obscure words (e.g. 'pood', 'holp'); conversely 'poos' is NOT in ENABLE, so "
            "going there is scored as a fail.",
            "Reasoning was requested via OpenRouter's unified reasoning:{enabled:true}; exact "
            "thinking budgets are provider-defined and not normalized across models.",
        ],
        "assets": {
            "note": "PNG paths are relative to studies/poople/output/.",
            "graphs": {
                cond: sorted(p.name for p in graphs_dir(cond).glob("*.png"))
                for cond in CONDITIONS if graphs_dir(cond).exists()
            },
            "graph_descriptions": {
                "solve_rate.png": "Headline horizontal bar: legal solve rate per model (logos).",
                "outcomes_pie.png": "Donut per model: par / over-par / illegal / failed split.",
                "illegal_per_attempt.png": "Mean illegal moves per attempt per model.",
                "avg_over_par.png": "Mean steps over par (solved attempts only).",
                "outcomes_stacked.png": "100%-stacked outcome breakdown per model (6-way).",
                "heatmap_solve_rate.png": "Solve rate by model × difficulty (par 3/4/5).",
                "heatmap_over_par.png": "Avg steps over par by model × difficulty.",
            },
            "result_image_dirs": {
                cond: f"results_images_{cond}/<word>/<model>.png"
                for cond in CONDITIONS
            },
            "result_image_words": [w for w, _ in pick_example_words(dist)],
        },
        "example_attempts": _example_attempts(words, dist, test_words),
    }


# ---------------------------------------------------------------------------
# Markdown digest
# ---------------------------------------------------------------------------

def _md(data: dict) -> str:
    m = data["meta"]
    L = []
    L.append(f"# {m['title']}\n")
    L.append(f"_Generated {m['generated']}._\n")
    L.append(m["description"] + "\n")

    L.append("## Rules\n")
    for r in m["rules"]:
        L.append(f"- {r}")
    L.append("")

    L.append("## Setup\n")
    L.append(f"- **Dictionary:** {m['dictionary']['name']} — {m['dictionary']['note']}")
    L.append(f"- **Sample:** {m['sample_design']['words_per_bucket']} words at each of "
             f"par {m['sample_design']['difficulty_buckets_par']}; "
             f"{m['attempts_per_word']} attempts per word; temperature {m['temperature']}.")
    so = data["solver_oracle"]
    L.append(f"- **Solver:** {so['total_four_letter_words']} four-letter words; "
             f"{so['reachable_to_poop']} ({so['reachable_pct']}%) can reach poop; "
             f"hardest is {so['max_distance_steps']} steps; "
             f"{so['total_optimal_ladders_saved']} optimal ladders catalogued.")
    L.append("")

    for cond in data["results"]:
        c = data["results"][cond]
        L.append(f"## Results — {cond} ({c['label']})\n")
        L.append("| Model | Solve % | Avg +par | Illegal/attempt | % w/ illegal |")
        L.append("|---|---|---|---|---|")
        for b in c["models"]:
            avg = "—" if b["avg_over_par"] is None else f"+{b['avg_over_par']}"
            L.append(f"| {b['short_name']} | {b['solve_rate_pct']}% | {avg} | "
                     f"{b['illegal_moves_per_attempt']} | {b['pct_attempts_with_illegal_move']}% |")
        L.append("")

    L.append("## Reasoning vs one-shot (solve %)\n")
    L.append("| Model | No reasoning | Reasoning | Lift |")
    L.append("|---|---|---|---|")
    for g in data["reasoning_vs_no_reasoning"]:
        lift = "—" if g["lift_pct_points"] is None else f"+{g['lift_pct_points']} pts"
        L.append(f"| {g['model']} | {g['no_reasoning_solve_pct']}% | "
                 f"{g['reasoning_solve_pct']}% | {lift} |")
    L.append("")

    L.append("## Key findings\n")
    for f in data["key_findings"]:
        L.append(f"- {f}")
    L.append("")
    L.append("## Caveats\n")
    for c in data["caveats"]:
        L.append(f"- {c}")
    L.append("")

    L.append("## Example attempts (best of each model)\n")
    for ex in data["example_attempts"][:4]:
        opt = " → ".join(ex["optimal_example"]) if ex["optimal_example"] else "n/a"
        L.append(f"**{ex['word']}** (par {ex['par']}) · one optimal: `{opt}`\n")
        for cond, attempts in ex["by_condition"].items():
            for a in attempts:
                ladder = " → ".join(a["ladder"]) if a["ladder"] else "(no answer)"
                L.append(f"- [{cond}] {a['model']}: `{ladder}` — {a['summary']}")
        L.append("")

    L.append("## Assets\n")
    L.append("All under `studies/poople/output/`. Graphs per condition in "
             "`graphs_<cond>/`; Wordle boards in `results_images_<cond>/<word>/<model>.png`.")
    L.append(f"\nFull structured data: `results_export.json`.")
    return "\n".join(L)


def export_all():
    data = build_export()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(EXPORT_JSON, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    with open(EXPORT_MD, "w") as f:
        f.write(_md(data))
    print(f"  Wrote {EXPORT_JSON}")
    print(f"  Wrote {EXPORT_MD}")
    return data


if __name__ == "__main__":
    export_all()
