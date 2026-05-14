"""Visualization for the God Belief by Name study.

Generates:
- Heatmap grid: race x gender showing YES rates
- Per-name grid showing individual YES/NO/REFUSED results
"""

import os
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors

from studies.god_by_name.config import NAME_GROUPS, RACES, GENDERS

# Red (no/atheist) -> White (split) -> Blue (yes/theist)
GOD_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "god_belief", ["#B2182B", "#F4A582", "#F7F7F7", "#92C5DE", "#2166AC"]
)


def generate_group_heatmap(
    group_scores: dict[tuple[str, str], float | None],
    title: str,
    save_path: str | Path,
):
    """Generate a race x gender heatmap of YES rates."""
    data = []
    annotations = []
    for race in RACES:
        row = []
        ann_row = []
        for gender in GENDERS:
            val = group_scores.get((race, gender))
            row.append(val if val is not None else float("nan"))
            ann_row.append(f"{val:.0%}" if val is not None else "N/A")
        data.append(row)
        annotations.append(ann_row)

    arr = np.array(data, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(arr, cmap=GOD_CMAP, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(GENDERS)))
    ax.set_yticks(range(len(RACES)))
    ax.set_xticklabels(GENDERS, fontsize=13)
    ax.set_yticklabels(RACES, fontsize=13)

    for i in range(len(RACES)):
        for j in range(len(GENDERS)):
            val = arr[i, j]
            text_color = "white" if not np.isnan(val) and abs(val - 0.5) > 0.35 else "black"
            ax.text(j, i, annotations[i][j], ha="center", va="center",
                    fontsize=14, fontweight="bold", color=text_color)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Proportion answering YES", fontsize=11)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_name_grid(
    name_answers: dict[str, str | None],
    title: str,
    save_path: str | Path,
):
    """Generate a grid showing each individual name's result.

    Rows are (race, gender) groups, columns are the 10 names per group.
    Cells are colored: blue=YES, red=NO, gray=REFUSED/None.
    """
    # Build row labels and data
    row_labels = []
    grid_names = []
    grid_values = []

    for race in RACES:
        for gender in GENDERS:
            row_labels.append(f"{race} {gender}")
            names = NAME_GROUPS[(race, gender)]
            grid_names.append(names)
            row_vals = []
            for name in names:
                answer = name_answers.get(name)
                if answer == "YES":
                    row_vals.append(1.0)
                elif answer == "NO":
                    row_vals.append(0.0)
                else:
                    row_vals.append(0.5)  # REFUSED/ERROR/None -> neutral
            grid_values.append(row_vals)

    n_rows = len(row_labels)
    n_cols = 10

    fig, ax = plt.subplots(figsize=(16, n_rows * 0.8 + 2))

    arr = np.array(grid_values, dtype=float)
    im = ax.imshow(arr, cmap=GOD_CMAP, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks([])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=11)

    # Annotate each cell with the name and answer
    for i in range(n_rows):
        for j in range(n_cols):
            name = grid_names[i][j]
            answer = name_answers.get(name, "?")
            val = arr[i, j]
            text_color = "white" if abs(val - 0.5) > 0.35 else "black"
            display = f"{name}\n{answer}" if answer else name
            ax.text(j, i, display, ha="center", va="center",
                    fontsize=8, fontweight="bold", color=text_color)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2166AC", edgecolor="black", label="YES"),
        Patch(facecolor="#B2182B", edgecolor="black", label="NO"),
        Patch(facecolor="#F7F7F7", edgecolor="black", label="REFUSED"),
    ]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=11, framealpha=0.9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
