"""Visualization for the Political Sycophancy study.

Wraps the shared map_graphing util with study-specific defaults
(blue/red political color scale).
"""

from pathlib import Path

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt

from utils.map_graphing import us_state_choropleth, _load_states


# Blue (liberal) to red (conservative)
POLITICAL_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "political", ["#2166AC", "#92C5DE", "#F7F7F7", "#F4A582", "#B2182B"]
)

# Score brackets: (threshold, label)
# Score is 0-1 absolute value, mapped to descriptive label
_BRACKETS = [
    (0.1, "moderate"),
    (0.2, "la croix {dir}"),
    (0.3, "slightly {dir}"),
    (0.4, "leans {dir}"),
    (0.5, "somewhat {dir}"),
    (0.6, "moderately {dir}"),
    (0.7, "moderately strongly {dir}"),
    (0.8, "strongly {dir}"),
    (0.9, "very strongly {dir}"),
    (1.0, "extremely strongly {dir}"),
]


def _score_label(score: float) -> str:
    """Convert a -1 to +1 score to a descriptive political label."""
    abs_val = abs(score)
    direction = "conservative" if score >= 0 else "liberal"
    for threshold, template in _BRACKETS:
        if abs_val <= threshold:
            return template.format(dir=direction)
    return f"extremely strongly {direction}"


def generate_us_map(
    state_scores: dict[str, float],
    title: str,
    save_path: str | Path,
    score_range: tuple[float, float] = (-1.0, 1.0),
):
    """Generate a US choropleth colored by political lean score."""
    label_low = _score_label(score_range[0])
    label_high = _score_label(score_range[1])

    us_state_choropleth(
        state_values=state_scores,
        title=title,
        save_path=save_path,
        value_range=score_range,
        cmap=POLITICAL_CMAP,
        label_low=label_low.capitalize(),
        label_high=label_high.capitalize(),
    )


# ---------------------------------------------------------------------------
# 2024 Election Results Map
# ---------------------------------------------------------------------------

# States won by Trump (Republican)
RED_STATES_2024 = [
    "Alabama", "Alaska", "Arkansas", "Florida", "Georgia", "Idaho", "Indiana",
    "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Michigan",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Pennsylvania",
    "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
    "West Virginia", "Wisconsin", "Wyoming", "Arizona",
]

# States won by Harris (Democrat)
BLUE_STATES_2024 = [
    "California", "Colorado", "Connecticut", "Delaware", "Hawaii", "Illinois",
    "Maryland", "Massachusetts", "Minnesota", "New Hampshire", "New Jersey",
    "New Mexico", "New York", "Oregon", "Rhode Island", "Vermont", "Virginia",
    "Washington",
]


def generate_election_map(save_path: str | Path):
    """Generate a US map showing the actual 2024 presidential election results."""
    import os
    import geopandas as gpd
    import pandas as pd

    states = _load_states()

    # Assign colors: 1 = red (Trump), -1 = blue (Harris)
    color_map = {}
    for s in RED_STATES_2024:
        color_map[s] = 1.0
    for s in BLUE_STATES_2024:
        color_map[s] = -1.0

    df = pd.DataFrame([
        {"name": name, "value": val}
        for name, val in color_map.items()
    ])
    states = states.merge(df, on="name", how="left")

    # Two-color discrete map: blue and red
    cmap = matplotlib.colors.ListedColormap(["#2166AC", "#B2182B"])
    bounds = [-1.5, 0, 1.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    states.plot(
        column="value",
        ax=ax,
        cmap=cmap,
        norm=norm,
        edgecolor="black",
        linewidth=0.5,
        legend=False,
        missing_kwds={"color": "#DDDDDD", "label": "No data"},
    )

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#B2182B", edgecolor="black", label="Trump (R)"),
        Patch(facecolor="#2166AC", edgecolor="black", label="Harris (D)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=12,
              framealpha=0.9)

    ax.set_title("2024 Presidential Election Results", fontsize=18,
                 fontweight="bold", pad=15)
    ax.axis("off")

    # Crop to continental US
    ax.set_xlim(-130, -65)
    ax.set_ylim(23, 50)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
