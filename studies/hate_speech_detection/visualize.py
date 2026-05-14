"""Visualization for the Hate Speech Detection Bias study.

Generates one grid chart per model:
  - Rows: genders (Male, Female)
  - Columns: races (Asian, Black, Hispanic, White, Unspecified)
  - Cell value: flag rate averaged across all templates
  - Color: diverging from mean (blue = under-flagged, red = over-flagged)
  - Footer: chi-squared test of independence result
"""

from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy import stats

from studies.hate_speech_detection.config import (
    GROUPS,
    TEMPLATES,
    RACES,
    GENDERS,
)
from studies.hate_speech_detection.runner import get_group_text

# Display order: races + Unspecified
DISPLAY_RACES = RACES + ["Unspecified"]
DISPLAY_GENDERS = [("male", "Male"), ("female", "Female")]

# Diverging: blue (under-flagged) -> white (average) -> red (over-flagged)
BIAS_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "flag_bias", ["#2166AC", "#92C5DE", "#F7F7F7", "#F4A582", "#B2182B"]
)


def _get_cell_stats(flag_rates: dict, race: str, gender: str) -> dict:
    """Aggregate flag counts across all templates for a (race, gender) cell."""
    total_flagged = 0
    total_count = 0

    group = next(g for g in GROUPS if g["race"] == race and g["gender"] == gender)

    for template in TEMPLATES:
        group_text = get_group_text(group, template["form"])
        entry = flag_rates.get(template["id"], {}).get(group_text)
        if entry and entry["count"] > 0:
            total_flagged += entry["flagged"]
            total_count += entry["count"]

    return {
        "flagged": total_flagged,
        "count": total_count,
        "flag_rate": total_flagged / total_count if total_count > 0 else None,
    }


def compute_chi_squared(flag_rates: dict) -> dict:
    """
    Chi-squared test of independence: is flag rate independent of group?

    Builds a contingency table: rows = groups (race x gender), cols = [flagged, not_flagged].
    Returns chi2 statistic, p-value, and degrees of freedom.
    """
    observed = []
    for race in DISPLAY_RACES:
        for gender, _ in DISPLAY_GENDERS:
            cell = _get_cell_stats(flag_rates, race, gender)
            if cell["count"] > 0:
                observed.append([cell["flagged"], cell["count"] - cell["flagged"]])

    if len(observed) < 2:
        return {"chi2": None, "p": None, "df": None, "n": 0}

    observed = np.array(observed)

    # Skip if all flagged or all not flagged (no variance)
    if observed[:, 0].sum() == 0 or observed[:, 1].sum() == 0:
        return {"chi2": 0, "p": 1.0, "df": len(observed) - 1, "n": int(observed.sum())}

    chi2, p, df, _ = stats.chi2_contingency(observed)
    return {"chi2": chi2, "p": p, "df": df, "n": int(observed.sum())}


def compute_pairwise_proportions(flag_rates: dict) -> list[dict]:
    """
    Pairwise two-proportion z-tests between all group pairs.

    Returns list of {group_a, group_b, z, p} for significant pairs.
    """
    cells = {}
    for race in DISPLAY_RACES:
        for gender, gender_label in DISPLAY_GENDERS:
            cell = _get_cell_stats(flag_rates, race, gender)
            if cell["count"] > 0:
                label = f"{race} {gender_label}" if race != "Unspecified" else gender_label
                cells[label] = cell

    results = []
    labels = list(cells.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = cells[labels[i]], cells[labels[j]]
            n1, n2 = a["count"], b["count"]
            p1, p2 = a["flag_rate"], b["flag_rate"]

            # Pooled proportion
            p_pool = (a["flagged"] + b["flagged"]) / (n1 + n2)
            if p_pool == 0 or p_pool == 1:
                continue

            se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
            if se == 0:
                continue

            z = (p1 - p2) / se
            p_val = 2 * (1 - stats.norm.cdf(abs(z)))

            results.append({
                "group_a": labels[i],
                "group_b": labels[j],
                "z": z,
                "p": p_val,
            })

    return results


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def generate_model_chart(
    model: str,
    flag_rates: dict,
    save_path: Path,
):
    """
    Generate a single grid chart for a model.

    Grid: races (cols) x genders (rows), colored by deviation from mean flag rate.
    """
    short = model.split("/")[-1]

    n_cols = len(DISPLAY_RACES)
    n_rows = len(DISPLAY_GENDERS)

    # Compute cell data
    data = np.full((n_rows, n_cols), np.nan)
    cell_info = [[None]*n_cols for _ in range(n_rows)]

    for i, (gender, _) in enumerate(DISPLAY_GENDERS):
        for j, race in enumerate(DISPLAY_RACES):
            cell = _get_cell_stats(flag_rates, race, gender)
            if cell["flag_rate"] is not None:
                data[i, j] = cell["flag_rate"]
                cell_info[i][j] = cell

    # Overall mean for diverging colormap center
    overall_mean = np.nanmean(data)
    max_dev = max(np.nanmax(np.abs(data - overall_mean)), 0.01)
    vmin = overall_mean - max_dev
    vmax = overall_mean + max_dev

    # Chi-squared
    chi2_result = compute_chi_squared(flag_rates)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 4.5))

    im = ax.imshow(data, cmap=BIAS_CMAP, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(DISPLAY_RACES, fontsize=12, fontweight="bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([gl for _, gl in DISPLAY_GENDERS], fontsize=12, fontweight="bold")

    # Cell annotations
    for i in range(n_rows):
        for j in range(n_cols):
            if cell_info[i][j] is not None:
                cell = cell_info[i][j]
                pct = cell["flag_rate"] * 100
                dev = (cell["flag_rate"] - overall_mean) * 100
                sign = "+" if dev > 0 else ""

                text = f"{pct:.0f}%\n({sign}{dev:.0f}%)"
                color = "white" if abs(cell["flag_rate"] - overall_mean) > max_dev * 0.55 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=11,
                        fontweight="bold", color=color)

    # Footer with chi-squared result
    footer = f"Mean flag rate: {overall_mean*100:.1f}%"
    if chi2_result["chi2"] is not None:
        stars = _stars(chi2_result["p"])
        footer += f"  |  \u03c7\u00b2({chi2_result['df']}, n={chi2_result['n']}) = {chi2_result['chi2']:.2f}, p = {chi2_result['p']:.4f} {stars}"
    ax.set_xlabel(footer, fontsize=10, labelpad=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Flag Rate", fontsize=10)

    ax.set_title(f"Hate Speech Detection Bias — {short}", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
