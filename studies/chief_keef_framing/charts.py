"""Charts for the Chief Keef Framing study.

Three views:
  * grouped_favorability_chart  — every model, three side-by-side bars (positive /
                                  neutral / negative framing) with SEM error bars.
  * per_model_arm_chart         — one model, three bars: its favorability per arm.
  * framing_swing_chart         — headline: positive-arm minus negative-arm gap per
                                  model, with brand icons (reuses icon_bar_chart).
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils.graphing import bar_chart, icon_bar_chart
from utils.model_icons import icon_path_for, color_for
from studies.chief_keef_framing.config import ARM_KEYS, ARMS_BY_KEY, ARM_COLORS


def short(model_id: str) -> str:
    return model_id.split("/")[-1]


def grouped_favorability_chart(
    stats: dict[str, dict],
    save_path: str | Path,
    title: str,
    subtitle: str,
):
    """Grouped bars: one cluster per model, one bar per framing arm.

    stats: {model_id: {arm_key: {"mean": float|None, "sem": float, "n": int}}}
    """
    models = list(stats.keys())
    n_models = len(models)
    n_arms = len(ARM_KEYS)

    fig, ax = plt.subplots(figsize=(max(9, 2.4 * n_models), 7.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    group_x = np.arange(n_models)
    bar_w = 0.78 / n_arms

    for ai, arm_key in enumerate(ARM_KEYS):
        offsets = group_x - 0.39 + bar_w * (ai + 0.5)
        means = [(stats[m][arm_key]["mean"] or 0.0) for m in models]
        sems = [stats[m][arm_key]["sem"] for m in models]
        ax.bar(
            offsets, means, width=bar_w,
            color=ARM_COLORS[arm_key], edgecolor="white", linewidth=1.0,
            label=ARMS_BY_KEY[arm_key]["label"], zorder=2,
        )
        ax.errorbar(
            offsets, means, yerr=sems, fmt="none",
            ecolor="#3a3f47", elinewidth=1.3, capsize=3, zorder=3,
        )
        for x, m in zip(offsets, means):
            ax.text(x, m + 0.15, f"{m:.1f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color="#2b2f36", zorder=4)

    ax.axhline(5.0, color="#b8bdc4", linestyle="--", linewidth=1.0, zorder=1)
    ax.text(n_models - 0.5, 5.05, "neutral (5)", ha="right", va="bottom",
            fontsize=9, color="#9aa0a8")

    ax.set_xticks(group_x)
    ax.set_xticklabels([short(m) for m in models], fontsize=11.5,
                       fontweight="bold", color="#2b2f36", rotation=20, ha="right")
    ax.set_ylabel("Favorability toward Chief Keef (0–10)", fontsize=12.5,
                  color="#4a505a", labelpad=10)
    ax.set_ylim(0, 10.6)
    ax.set_yticks(np.arange(0, 11, 2))
    ax.tick_params(axis="y", labelsize=11, colors="#8a909a", length=0)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="#dfe3e8", linewidth=1.0, alpha=0.9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c7ccd3")

    fig.suptitle(title, fontsize=18, fontweight="bold", color="#1a1d22", y=0.985)
    ax.set_title(subtitle, fontsize=12, color="#7a808a", pad=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=n_arms,
              frameon=False, fontsize=11.5, labelcolor="#4a505a")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def per_model_arm_chart(
    model_id: str,
    arm_means: dict[str, float],
    save_path: str | Path,
):
    """Three bars (one per arm) for a single model."""
    labels = [ARMS_BY_KEY[k]["label"] for k in ARM_KEYS]
    values = [arm_means.get(k) or 0.0 for k in ARM_KEYS]
    bar_chart(
        labels=labels,
        values=values,
        title=f"How does {short(model_id)} feel about Chief Keef, by framing?",
        x_label="Framing of the background",
        y_label="Favorability (0–10)",
        save_path=save_path,
        y_range=(0, 10),
        color="#4A90D9",
    )


def framing_swing_chart(
    swings: dict[str, float],
    save_path: str | Path,
    title: str,
    subtitle: str,
):
    """Headline chart: positive-arm minus negative-arm favorability gap per model."""
    models = list(swings.keys())
    icon_bar_chart(
        labels=[short(m) for m in models],
        values=[swings[m] for m in models],
        title=title,
        subtitle=subtitle,
        y_label="Favorability swing (positive − negative framing)",
        save_path=save_path,
        icon_paths=[icon_path_for(m) for m in models],
        colors=[color_for(m) for m in models],
        y_range=(0, 10),
        value_fmt=".1f",
    )
