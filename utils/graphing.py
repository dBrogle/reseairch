"""Graphing and visualization utilities"""

import os
from pathlib import Path
from typing import Sequence

import numpy as np
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


def bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    x_label: str,
    y_label: str,
    save_path: str | Path,
    y_range: tuple[float, float] | None = None,
    color: str = "#4A90D9",
    log_scale: bool = False,
    value_fmt: str = ".1f",
):
    """
    Create and save a bar chart.

    Args:
        labels: Category labels for each bar
        values: Height of each bar
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        save_path: File path to save the chart image
        y_range: Optional (min, max) for y-axis
        color: Bar color
        log_scale: Use logarithmic y-axis
        value_fmt: Format string for value labels on bars
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    if log_scale:
        # Diverging bars from baseline=1.0 on log scale
        ax.set_yscale("log")
        colors = [color if v >= 1.0 else "#D94A4A" for v in values]
        # Bar height is the ratio from 1.0; bottom is always 1.0
        # For values > 1: bar goes up from 1.0 to value
        # For values < 1: bar goes down from 1.0 to value
        heights = [v - 1.0 if v >= 1.0 else 1.0 - v for v in values]
        bottoms = [1.0 if v >= 1.0 else v for v in values]
        bars = ax.bar(labels, heights, bottom=bottoms, color=colors,
                      edgecolor="white", width=0.6)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)

        # Value labels: centered in the bar (geometric mean between 1.0 and value)
        for bar, val in zip(bars, values):
            x = bar.get_x() + bar.get_width() / 2
            y = (val * 1.0) ** 0.5  # geometric mean of val and 1.0
            ax.text(x, y, f"{val:{value_fmt}}", ha="center", va="center",
                    fontsize=10, fontweight="bold")
    else:
        bars = ax.bar(labels, values, color=color, edgecolor="white", width=0.6)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:{value_fmt}}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    if y_range is not None:
        ax.set_ylim(y_range)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right", fontsize=9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def heatmap(
    data: list[list[float]],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    x_label: str,
    y_label: str,
    save_path: str | Path,
    value_range: tuple[float, float] | None = None,
    cmap: str = "RdYlGn",
    fmt: str = ".2f",
    annotations: list[list[str]] | None = None,
):
    """
    Create and save a heatmap.

    Args:
        data: 2D list of values [rows][cols]
        row_labels: Labels for each row (y-axis)
        col_labels: Labels for each column (x-axis)
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        save_path: File path to save the chart image
        value_range: Optional (vmin, vmax) for color scale
        cmap: Matplotlib colormap name
        fmt: Format string for cell annotations (ignored if annotations provided)
        annotations: Optional 2D list of pre-formatted annotation strings per cell
    """
    arr = np.array(data)
    fig, ax = plt.subplots(figsize=(8, 5))

    kwargs = {"cmap": cmap, "aspect": "auto"}
    if value_range is not None:
        kwargs["vmin"] = value_range[0]
        kwargs["vmax"] = value_range[1]

    im = ax.imshow(arr, **kwargs)

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=12)
    ax.set_yticklabels(row_labels, fontsize=12)

    # Annotate each cell
    vmin = kwargs.get("vmin", arr.min())
    vmax = kwargs.get("vmax", arr.max())
    mid = (vmin + vmax) / 2
    span = vmax - vmin
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = arr[i, j]
            text_color = "white" if abs(val - mid) > span * 0.35 else "black"
            label = annotations[i][j] if annotations else f"{val:{fmt}}"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=10)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
