"""Graphing and visualization utilities"""

import os
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib


def line_chart(
    x_values: Sequence[float],
    y_values: Sequence[float],
    title: str,
    x_label: str,
    y_label: str,
    save_path: str | Path,
    y_range: tuple[float, float] | None = None,
    marker: str = "o",
    color: str = "#4A90D9",
):
    """
    Create and save a simple line chart.

    Args:
        x_values: Data for x-axis
        y_values: Data for y-axis
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        save_path: File path to save the chart image
        y_range: Optional (min, max) for y-axis
        marker: Marker style
        color: Line color
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_values, y_values, marker=marker, color=color, linewidth=2, markersize=6)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    if y_range is not None:
        ax.set_ylim(y_range)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def multi_line_chart(
    x_values: Sequence[float],
    series: dict[str, Sequence[float]],
    title: str,
    x_label: str,
    y_label: str,
    save_path: str | Path,
    y_range: tuple[float, float] | None = None,
):
    """
    Create and save a multi-line chart.

    Args:
        x_values: Shared x-axis data
        series: Dict mapping series name -> y-values
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        save_path: File path to save the chart image
        y_range: Optional (min, max) for y-axis
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, y_vals in series.items():
        ax.plot(x_values, y_vals, marker="o", linewidth=2, markersize=5, label=name)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    if y_range is not None:
        ax.set_ylim(y_range)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
