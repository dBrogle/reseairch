"""Visualization for the HLE Sycophancy study.

Generates US choropleth maps showing per-state accuracy (0–1), using a
red-yellow-green colormap where green = higher accuracy.
"""

from pathlib import Path

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt

from utils.map_graphing import us_state_choropleth


# Red (low accuracy) → yellow → green (high accuracy)
ACCURACY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "accuracy", ["#D32F2F", "#FFEB3B", "#388E3C"]
)


def generate_us_map(
    state_accuracy: dict[str, float],
    title: str,
    save_path: str | Path,
    score_range: tuple[float, float] = (0.0, 1.0),
):
    """Generate a US choropleth colored by HLE accuracy."""
    vmin, vmax = score_range
    label_low  = f"{vmin:.0%} accuracy"
    label_high = f"{vmax:.0%} accuracy"

    us_state_choropleth(
        state_values=state_accuracy,
        title=title,
        save_path=save_path,
        value_range=score_range,
        cmap=ACCURACY_CMAP,
        label_low=label_low,
        label_high=label_high,
    )


def generate_bar_chart(
    state_accuracy: dict[str, float],
    title: str,
    save_path: str | Path,
):
    """Generate a horizontal bar chart of states sorted by accuracy."""
    sorted_states = sorted(state_accuracy.items(), key=lambda x: x[1])
    states = [s for s, _ in sorted_states]
    values = [v for _, v in sorted_states]

    cmap = matplotlib.colormaps["RdYlGn"]
    colors = [cmap(v) for v in values]

    fig, ax = plt.subplots(figsize=(10, max(8, len(states) * 0.22)))
    bars = ax.barh(states, values, color=colors, edgecolor="white", linewidth=0.4)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(
            val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", ha="left", fontsize=7,
        )

    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Accuracy", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.axvline(x=sum(values) / len(values), color="gray", linestyle="--",
               linewidth=1.0, alpha=0.7, label=f"Mean: {sum(values)/len(values):.1%}")
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)

    import os
    os.makedirs(str(Path(save_path).parent), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
