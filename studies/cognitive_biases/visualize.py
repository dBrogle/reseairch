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

import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from studies.cognitive_biases.scenarios.base import Scenario

_CONTROL_COLOR   = "#7F8C8D"   # grey
_TREATMENT_COLOR = "#2E86C1"   # blue
_POSITIVE_COLOR  = "#2E86C1"   # delta > 0 (bias pulls UP)
_NEGATIVE_COLOR  = "#C0392B"   # delta < 0 (bias pulls DOWN)

# ---------------------------------------------------------------------------
# Model logos (used as scatter markers in the delta chart)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_LOGOS_DIR = _REPO_ROOT / "data" / "images" / "models"

# Provider prefix (everything before the "/" in model id) → logo filename
# stem in `data/images/models/`. Most providers match by name; moonshotai
# ships under `kimi.png` (product-name logo, not company).
_PROVIDER_TO_LOGO_STEM: dict[str, str] = {
    "openai":     "openai",
    "anthropic":  "anthropic",
    "x-ai":       "x-ai",
    "google":     "google",
    "moonshotai": "kimi",
    "deepseek":   "deepseek",
}

_logo_image_cache: dict[str, np.ndarray] = {}


def _load_logo_image(model_id: str) -> np.ndarray | None:
    """Load the RGBA logo for a model. Returns None if no logo is mapped."""
    provider = model_id.split("/", 1)[0]
    stem = _PROVIDER_TO_LOGO_STEM.get(provider, provider)
    if stem in _logo_image_cache:
        return _logo_image_cache[stem]
    path = _MODEL_LOGOS_DIR / f"{stem}.png"
    if not path.exists():
        return None
    img = mpimg.imread(str(path))
    _logo_image_cache[stem] = img
    return img


def _logo_offset_image(model_id: str, target_px: int) -> OffsetImage | None:
    """OffsetImage scaled so the longer side renders at ~target_px.

    Source logos aren't uniformly sized (most are 512×512, kimi is
    746×682), so a fixed zoom factor renders them at visibly different
    sizes. Normalising on the longer dimension gives every model the
    same visual footprint regardless of source resolution.
    """
    img = _load_logo_image(model_id)
    if img is None:
        return None
    longest = max(img.shape[0], img.shape[1])
    zoom = target_px / longest if longest else 0.1
    return OffsetImage(img, zoom=zoom)


def _short(model: str) -> str:
    return model.split("/")[-1]


def _mathsafe(s: str) -> str:
    """Escape `$` so matplotlib doesn't read paired dollar signs in a label
    (e.g. "$9,500 appraisal, asking $5,200") as a math-mode span."""
    return s.replace("$", r"\$")


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
        y_labels.append(f"{prefix}{_mathsafe(arm_label)}")
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

    One row per (model, treatment_arm). Dot + 95% Welch CI in the plot
    area (negative deltas in red, positive in blue). The Y-axis tick
    label is the model's logo — text labels are dropped; the logo
    conveys identity. Falls back to a text label for any model whose
    logo file is missing.
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

    # Extra row spacing so the larger logo tick labels don't crowd.
    fig_h = max(4.5, 0.85 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    y_positions = np.arange(len(rows))[::-1]
    for y, (model, arm, diff, lo, hi) in zip(y_positions, rows):
        color = _POSITIVE_COLOR if diff >= 0 else _NEGATIVE_COLOR
        ax.errorbar(
            diff, y,
            xerr=[[diff - lo], [hi - diff]],
            fmt="o", color=color, ecolor=color,
            elinewidth=1.8, capsize=5, markersize=9,
            zorder=3,
        )
        ax.text(
            hi, y,
            f"  Δ {_format_value(diff, scenario.value_unit)}  "
            f"[{_format_value(lo, scenario.value_unit)}, "
            f"{_format_value(hi, scenario.value_unit)}]",
            ha="left", va="center", fontsize=9, color="#333333",
        )

    ax.axvline(0, color="black", linewidth=0.8)

    # Logos as y-axis tick "labels". Placed in axes-fraction x / data y
    # coords so they sit at a fixed offset to the left of the axis
    # regardless of the data range. Zoom tuned so 512px source logos
    # land at ~50px on a 11-inch / 150-DPI figure.
    ax.set_yticks(y_positions)
    ax.set_yticklabels(["" for _ in rows])
    ax.tick_params(axis="y", length=0)

    # Target rendered logo size in source-pixel units. Picked so a 512px
    # logo at this size matches the previous `zoom=0.10` baseline; all
    # other logos (incl. the 746×682 kimi) get scaled to match.
    _LOGO_TARGET_PX = 52
    for y, (model, arm, *_rest) in zip(y_positions, rows):
        oi = _logo_offset_image(model, _LOGO_TARGET_PX)
        if oi is None:
            # Fallback: render the short model name as text.
            ax.annotate(
                _short(model),
                xy=(0, y), xycoords=("axes fraction", "data"),
                xytext=(-8, 0), textcoords="offset points",
                ha="right", va="center", fontsize=9, color="#444444",
            )
            continue
        ab = AnnotationBbox(
            oi,
            xy=(0, y), xycoords=("axes fraction", "data"),
            xybox=(-32, 0), boxcoords="offset points",
            frameon=False, pad=0.0,
            box_alignment=(0.5, 0.5),
        )
        ax.add_artist(ab)

    ax.set_xlabel(
        f"Bias delta (treatment − control, {scenario.value_unit})  ·  "
        "error bars = 95% Welch CI",
        fontsize=10,
    )

    treatment_labels = list(dict.fromkeys(a.label for _, a, *_ in rows))
    subtitle = (
        f"[{scenario.bias_type}]  ·  treatment: {_mathsafe(treatment_labels[0])}"
        if len(treatment_labels) == 1
        else f"[{scenario.bias_type}]"
    )
    ax.set_title(
        f"{scenario.title}\n{subtitle}",
        fontsize=13, fontweight="bold", pad=10,
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
    # Extra left margin so the off-axis logo annotations stay inside the
    # saved bbox even with bbox_inches="tight".
    fig.subplots_adjust(left=0.14)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
