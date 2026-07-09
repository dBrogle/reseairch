"""Hand-vs-code comparison chart.

A grouped (double) bar per model, with its brand icon: how often it solves
Poople optimally *by hand* (the poople study's reasoning set) versus how often
its *written program* solves optimally (this study). Makes the jump obvious.
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from studies.poople.visualize import _place_icon, BG
from studies.poople.solver import build_solution_oracle
from studies.poople.sampling import flat_test_words
from studies.poople.wordlist import load_words
from studies.poople.analysis import compute_model_stats
from studies.poople_coding.config import MODELS
from studies.poople_coding.cache import load_code
from studies.poople_coding.evaluator import evaluate_all

HAND_COLOR = "#c47d4a"   # solving by hand (reasoning)
CODE_COLOR = "#2e9b57"   # solving via written code


def _short(m: str) -> str:
    return m.split("/")[-1]


def _hand_optimal_pct() -> dict[str, float]:
    """% of attempts solved OPTIMALLY (par) by each model in the reasoning set."""
    dist = build_solution_oracle(load_words())["dist"]
    test_words = flat_test_words(dist)
    out = {}
    for m in MODELS:
        s = compute_model_stats(m, test_words, "reasoning")["overall"]
        if s["n"]:
            out[m] = s["pie"]["par"] / s["n"] * 100
    return out


def _code_optimal_pct() -> dict[str, float]:
    """% of test-battery words each model's program solves optimally."""
    programs = {m: load_code(m) for m in MODELS}
    programs = {m: p for m, p in programs.items() if p}
    if not programs:
        return {}
    results, _battery = evaluate_all(programs)
    return {r["model"]: r["optimal_rate"] for r in results}


def generate_hand_vs_code(save_path: Path):
    hand = _hand_optimal_pct()
    code = _code_optimal_pct()
    models = [m for m in MODELS if m in hand and m in code]
    if not models:
        print("  hand-vs-code: missing data in one of the studies.")
        return
    models.sort(key=lambda m: -hand[m])

    n = len(models)
    x = np.arange(n)
    w = 0.38

    fig, ax = plt.subplots(figsize=(max(9, 2.0 * n), 7.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    hb = ax.bar(x - w / 2, [hand[m] for m in models], w, color=HAND_COLOR,
                edgecolor="white", linewidth=1.2, label="By hand (reasoning)", zorder=3)
    cb = ax.bar(x + w / 2, [code[m] for m in models], w, color=CODE_COLOR,
                edgecolor="white", linewidth=1.2, label="Via written code", zorder=3)

    for bars in (hb, cb):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                    f"{b.get_height():.0f}%", ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color="#2b2f36", zorder=5)

    # Brand icon above each model group.
    for xi, m in zip(x, models):
        _place_icon(ax, m, (xi, 112), zoom=0.07, xycoords="data")

    ax.set_xticks(x)
    ax.set_xticklabels([_short(m) for m in models], fontsize=12, fontweight="bold",
                       color="#2b2f36")
    ax.set_ylim(0, 124)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("% solved optimally", fontsize=13, color="#4a505a", labelpad=10)
    ax.tick_params(axis="y", labelsize=11, colors="#8a909a", length=0)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="#e6e9ed", linewidth=1.0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d2d6dc")

    fig.suptitle("Poople — solving by hand vs. writing code",
                 fontsize=20, fontweight="bold", color="#15171a", y=0.99)
    ax.set_title("% solved optimally  ·  by hand = reasoning models on the 30-word sample  ·  "
                 "via code = their program over all 342 reachable words",
                 fontsize=11, color="#8a909a", pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False,
              fontsize=12)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Wrote {save_path}")
