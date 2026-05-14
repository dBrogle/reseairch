"""Visualization for the God Belief by Country study.

Generates:
- World choropleth map
- Bar chart ranking all countries
- Grouped bar charts by region
"""

import os
from pathlib import Path

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from utils.map_graphing import world_choropleth
from studies.god_by_country.config import COUNTRY_GROUPS, COUNTRY_SHAPEFILE_NAMES


# Red (no/atheist) -> White (split) -> Blue (yes/theist)
GOD_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "god_belief", ["#B2182B", "#F4A582", "#F7F7F7", "#92C5DE", "#2166AC"]
)


def _shapefile_name(country: str) -> str:
    """Convert our country name to the world shapefile's NAME column."""
    return COUNTRY_SHAPEFILE_NAMES.get(country, country)


def generate_world_map(
    country_scores: dict[str, float],
    title: str,
    save_path: str | Path,
    score_range: tuple[float, float] = (0.0, 1.0),
):
    """Generate a world choropleth colored by YES (believes) proportion."""
    # Remap keys to shapefile names
    mapped = {_shapefile_name(c): v for c, v in country_scores.items()}

    world_choropleth(
        country_values=mapped,
        title=title,
        save_path=save_path,
        value_range=score_range,
        cmap=GOD_CMAP,
        label_low="Does not believe",
        label_high="Believes",
    )


def generate_ranked_bar_chart(
    country_scores: dict[str, float],
    title: str,
    save_path: str | Path,
):
    """Generate a horizontal bar chart ranking all countries by YES rate."""
    valid = {c: v for c, v in country_scores.items() if v is not None}
    if not valid:
        return

    sorted_items = sorted(valid.items(), key=lambda x: x[1], reverse=True)
    countries = [c for c, _ in sorted_items]
    values = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(8, len(countries) * 0.35)))

    # Color bars by value using the GOD_CMAP
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    colors = [GOD_CMAP(norm(v)) for v in values]

    bars = ax.barh(range(len(countries)), values, color=colors, edgecolor="white", height=0.7)

    ax.set_yticks(range(len(countries)))
    ax.set_yticklabels(countries, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion answering YES", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, axis="x")

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.0%}", va="center", fontsize=9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_regional_bar_charts(
    country_scores: dict[str, float],
    title_prefix: str,
    save_dir: str | Path,
):
    """Generate a separate bar chart for each region group."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for region, countries in COUNTRY_GROUPS.items():
        valid = {c: country_scores[c] for c in countries
                 if c in country_scores and country_scores[c] is not None}
        if not valid:
            continue

        sorted_items = sorted(valid.items(), key=lambda x: x[1], reverse=True)
        names = [c for c, _ in sorted_items]
        values = [v for _, v in sorted_items]

        fig, ax = plt.subplots(figsize=(10, max(3, len(names) * 0.5)))

        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        colors = [GOD_CMAP(norm(v)) for v in values]

        bars = ax.barh(range(len(names)), values, color=colors, edgecolor="white", height=0.6)

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Proportion answering YES", fontsize=11)
        ax.set_title(f"{title_prefix} — {region}", fontsize=13, fontweight="bold", pad=10)
        ax.grid(True, alpha=0.3, axis="x")

        for bar, val in zip(bars, values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0%}", va="center", fontsize=10)

        safe_region = region.replace("/", "_").replace(" ", "_").lower()
        fig.tight_layout()
        fig.savefig(save_dir / f"region_{safe_region}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
