"""Visualization for the Cognitive Biases study.

For each scenario, generates a horizontal dot+CI plot:
  - one row per (model, arm)
  - x-axis = the scenario's numeric value (units vary by scenario)
  - control rows in grey, treatment rows in blue
  - rows grouped by model, sorted by the canonical model order

This format scales well as more scenarios get added: each scenario lives
in its own chart, and within-scenario comparisons (control vs. treatment
across models) are visually obvious without normalising units.
"""

import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from studies.cognitive_biases.scenarios.base import Scenario

_CONTROL_COLOR   = "#7F8C8D"   # grey
_TREATMENT_COLOR = "#2E86C1"   # blue


def _short(model: str) -> str:
    return model.split("/")[-1]


def _ensure_dir(path: Path):
    os.makedirs(str(path.parent), exist_ok=True)


def _format_value(v: float, unit: str) -> str:
    """Compact human-readable number for chart labels."""
    if v != v:  # NaN
        return "—"
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if abs(v) >= 10_000:
        return f"{v/1_000:.1f}k"
    if abs(v) >= 1_000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.0f}"
    return f"{v:.2f}"


def generate_scenario_chart(
    scenario: Scenario,
    all_stats: list[dict],
    save_path: Path,
):
    """Per-scenario dot+CI plot. all_stats is one entry per model.

    Each entry is the output of analysis.compute_scenario_stats. Models
    with no usable data for any arm are omitted from the chart.
    """
    # Filter to models with at least the control arm populated.
    usable = []
    for stats in all_stats:
        per_arm = stats["per_arm"]
        if per_arm.get(scenario.control.key, {}).get("n", 0) > 0:
            usable.append(stats)
    if not usable:
        print(f"  [skip] {scenario.id}: no models have control data")
        return

    # Build flat row list: [(model, arm_key, label, role, mean, lo, hi, n)]
    rows = []
    for stats in usable:
        model = stats["model"]
        for arm in scenario.arms:
            s = stats["per_arm"].get(arm.key)
            if not s or s["n"] == 0:
                continue
            rows.append((
                model, arm.key, arm.label, arm.role,
                s["mean"], s["ci_low"], s["ci_high"], s["n"],
            ))

    n_rows = len(rows)
    fig_h = max(4.0, 0.55 * n_rows + 2.0)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    # Plot bottom-up: first row in `rows` ends up on top of chart.
    y_positions = np.arange(n_rows)[::-1]

    # Per-model row separators (between groups)
    last_model = None
    for y, (model, *_rest) in zip(y_positions, rows):
        if last_model is not None and model != last_model:
            ax.axhline(y + 0.5, color="#E0E0E0", linewidth=0.8, zorder=0)
        last_model = model

    # Y-tick label: prefix the first row of each model group with the
    # model's short name; subsequent arm rows for that model show only
    # the arm label.
    y_labels = []
    label_colors = []
    prev_model = None
    for (model, arm_key, arm_label, role, mean, lo, hi, n) in rows:
        prefix = f"{_short(model)}  ·  " if model != prev_model else ""
        y_labels.append(f"{prefix}{arm_label}")
        label_colors.append(_CONTROL_COLOR if role == "control" else _TREATMENT_COLOR)
        prev_model = model

    for y, (model, arm_key, arm_label, role, mean, lo, hi, n) in zip(y_positions, rows):
        color = _CONTROL_COLOR if role == "control" else _TREATMENT_COLOR
        ax.errorbar(
            mean, y,
            xerr=[[mean - lo], [hi - mean]],
            fmt="o", color=color, ecolor=color,
            elinewidth=1.6, capsize=4, markersize=8, zorder=3,
        )
        ax.text(
            hi, y, f"  {_format_value(mean, scenario.value_unit)} (n={n})",
            ha="left", va="center", fontsize=8.5, color="#333333",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    for tick_label, color in zip(ax.get_yticklabels(), label_colors):
        tick_label.set_color(color)

    ax.set_xlabel(f"Estimated value ({scenario.value_unit}), error bars = 95% CI",
                  fontsize=10)
    ax.set_title(
        f"{scenario.title}\n[{scenario.bias_type}]  ·  "
        f"{len(usable)} model(s)",
        fontsize=12, fontweight="bold", pad=10,
    )

    # Legend: control vs treatment
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=_CONTROL_COLOR, markersize=8, label="Control"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=_TREATMENT_COLOR, markersize=8, label="Treatment"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="lower right")

    ax.grid(True, axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    # Pad x-axis right to fit annotation labels
    xs = [r[5] for r in rows] + [r[6] for r in rows] + [r[4] for r in rows]
    if xs:
        lo, hi = min(xs), max(xs)
        span = hi - lo if hi > lo else max(abs(hi), 1.0)
        ax.set_xlim(lo - 0.1 * span, hi + 0.45 * span)

    _ensure_dir(save_path)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_delta_chart(
    scenario: Scenario,
    all_stats: list[dict],
    save_path: Path,
):
    """Per-scenario bias delta chart: treatment_mean − control_mean per model.

    One row per (model, treatment_arm). Negative deltas in red, positive in
    blue. CIs that cross zero are visually obvious.
    """
    rows = []
    for stats in all_stats:
        model = stats["model"]
        for arm in scenario.treatments:
            d = stats["deltas"].get(arm.key)
            if d is None:
                continue
            rows.append((model, arm, d["diff"], d["ci_low"], d["ci_high"]))

    if not rows:
        print(f"  [skip] {scenario.id} delta chart: no usable data")
        return

    fig_h = max(3.5, 0.55 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    y_positions = np.arange(len(rows))[::-1]
    for y, (model, arm, diff, lo, hi) in zip(y_positions, rows):
        color = "#2E86C1" if diff >= 0 else "#C0392B"
        ax.errorbar(
            diff, y,
            xerr=[[diff - lo], [hi - diff]],
            fmt="o", color=color, ecolor=color,
            elinewidth=1.6, capsize=4, markersize=8,
        )
        ax.text(
            hi, y,
            f"  Δ {_format_value(diff, scenario.value_unit)}  "
            f"[{_format_value(lo, scenario.value_unit)}, "
            f"{_format_value(hi, scenario.value_unit)}]",
            ha="left", va="center", fontsize=8.5, color="#333333",
        )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [f"{_short(m)}  ·  {a.label}" for m, a, *_ in rows], fontsize=9,
    )
    ax.set_xlabel(
        f"Bias delta (treatment − control, {scenario.value_unit})  ·  "
        "error bars = 95% Welch CI",
        fontsize=10,
    )
    ax.set_title(
        f"{scenario.title} — Anchor Effect by Model\n"
        f"[{scenario.bias_type}]",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    xs = [r[3] for r in rows] + [r[4] for r in rows] + [r[2] for r in rows]
    if xs:
        lo, hi = min(xs + [0]), max(xs + [0])
        span = hi - lo if hi > lo else max(abs(hi), 1.0)
        ax.set_xlim(lo - 0.1 * span, hi + 0.55 * span)

    _ensure_dir(save_path)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
