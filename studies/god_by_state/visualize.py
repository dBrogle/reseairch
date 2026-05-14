"""Visualization for the God Belief by State study.

Uses a red (no) to blue (yes, believes) color scale for the
proportion of YES answers on a US state choropleth.
"""

from pathlib import Path

import matplotlib
import matplotlib.colors

from utils.map_graphing import us_state_choropleth


# Red (no/atheist) -> White (split) -> Blue (yes/theist)
GOD_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "god_belief", ["#B2182B", "#F4A582", "#F7F7F7", "#92C5DE", "#2166AC"]
)

# White (no refusals) -> Purple (all refusals)
REFUSAL_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "refusal", ["#F7F7F7", "#D4B9DA", "#C994C7", "#DF65B0", "#980043"]
)


def generate_us_map(
    state_scores: dict[str, float],
    title: str,
    save_path: str | Path,
    score_range: tuple[float, float] = (0.0, 1.0),
):
    """Generate a US choropleth colored by YES (believes) proportion."""
    us_state_choropleth(
        state_values=state_scores,
        title=title,
        save_path=save_path,
        value_range=score_range,
        cmap=GOD_CMAP,
        label_low="Does not believe",
        label_high="Believes",
    )


def generate_refusal_map(
    state_scores: dict[str, float],
    title: str,
    save_path: str | Path,
    score_range: tuple[float, float] = (0.0, 1.0),
):
    """Generate a US choropleth colored by refusal/error rate."""
    us_state_choropleth(
        state_values=state_scores,
        title=title,
        save_path=save_path,
        value_range=score_range,
        cmap=REFUSAL_CMAP,
        label_low="Always answered",
        label_high="Always refused",
    )


# 2024 election results for red/blue state analysis
RED_STATES_2024 = [
    "Alabama", "Alaska", "Arkansas", "Florida", "Georgia", "Idaho", "Indiana",
    "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Michigan",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Pennsylvania",
    "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
    "West Virginia", "Wisconsin", "Wyoming", "Arizona",
]

BLUE_STATES_2024 = [
    "California", "Colorado", "Connecticut", "Delaware", "Hawaii", "Illinois",
    "Maryland", "Massachusetts", "Minnesota", "New Hampshire", "New Jersey",
    "New Mexico", "New York", "Oregon", "Rhode Island", "Vermont", "Virginia",
    "Washington",
]
