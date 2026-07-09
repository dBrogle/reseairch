"""Graphing and visualization utilities"""

import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Patch, Wedge, Circle
from matplotlib.colors import to_rgb

# Use a CJK-capable font first so Chinese glyphs (e.g. 你是什么模型) render instead
# of showing as missing-glyph boxes. This matplotlib build does not reliably do
# per-glyph fallback through the list, so the primary font must itself cover both
# Latin and CJK (Arial Unicode MS does); DejaVu Sans is the non-CJK fallback.
for _cjk_font in ("Arial Unicode MS", "Hiragino Sans GB", "Songti SC", "PingFang SC"):
    try:
        matplotlib.font_manager.findfont(_cjk_font, fallback_to_default=False)
        matplotlib.rcParams["font.sans-serif"] = [_cjk_font, "DejaVu Sans"]
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False


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


# ---------------------------------------------------------------------------
# Icon-aware identity charts
# ---------------------------------------------------------------------------

# Icons are normalized against this reference height (px), so the `zoom` values
# below stay meaningful no matter the source PNG resolution (our icons range from
# 512px to 3840px tall). A 512px icon renders at exactly `zoom`.
_ICON_REF_PX = 512.0


def _place_image(ax, img_path, xy, zoom: float, xycoords="data", alpha: float = 1.0,
                 zorder: float | None = None):
    """Drop a PNG onto the axes at data-coords xy. Silently no-ops if unreadable.

    `zoom` is normalized to the image's pixel height so icons of different source
    resolutions all render at a consistent on-screen size.
    """
    try:
        img = mpimg.imread(str(img_path))
    except Exception:
        return
    eff_zoom = zoom * (_ICON_REF_PX / img.shape[0])
    im = OffsetImage(img, zoom=eff_zoom, alpha=alpha)
    ab = AnnotationBbox(im, xy, xycoords=xycoords, frameon=False,
                        box_alignment=(0.5, 0.5), pad=0)
    if zorder is not None:
        ab.set_zorder(zorder)
    ax.add_artist(ab)


def identity_donut(
    breakdown: dict[str, int],
    title: str,
    save_path: str | Path,
    center_icon: str | Path | None = None,
    center_label: str | None = None,
    subtitle: str | None = None,
    icon_for: Callable[[str], Path | None] | None = None,
    color_for: Callable[[str], str] | None = None,
    max_slices: int = 6,
    slice_zoom: float = 0.085,
    center_zoom: float = 0.20,
):
    """Donut chart of an identity breakdown, with brand icons on each slice.

    Args:
        breakdown: {identity_label: count}. Zero/negative counts are dropped.
        title: Chart title.
        save_path: Output PNG path.
        center_icon: Icon for the model being asked, drawn in the donut hole.
        center_label: Text under the center icon (e.g. the model name + headline stat).
        subtitle: Small line under the title.
        icon_for: label -> icon Path (or None). "Other" never gets an icon.
        color_for: label -> hex color. Falls back to a gray for "Other"/None.
        max_slices: Slices beyond this are aggregated into a single "Other" wedge.
        slice_zoom: Zoom for the per-slice icons.
        center_zoom: Zoom for the center icon.
    """
    items = sorted(
        ((k, v) for k, v in breakdown.items() if v > 0),
        key=lambda kv: kv[1], reverse=True,
    )
    if len(items) > max_slices:
        head, tail = items[:max_slices], items[max_slices:]
        items = head + [("Other", sum(v for _, v in tail))]

    if not items:
        return

    labels = [k for k, _ in items]
    values = [v for _, v in items]
    total = sum(values)
    gray = "#B6B6BE"
    colors = [
        (gray if (label == "Other" or color_for is None) else color_for(label))
        for label in labels
    ]

    fig, ax = plt.subplots(figsize=(9, 9))
    wedges, _ = ax.pie(
        values, colors=colors, startangle=90, counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2.5),
    )
    ax.set(aspect="equal")

    for wedge, label, val in zip(wedges, labels, values):
        ang = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        x, y = np.cos(ang), np.sin(ang)
        pct = val / total * 100

        ipath = icon_for(label) if (icon_for and label != "Other") else None
        if ipath is not None:
            _place_image(ax, ipath, (x * 1.12, y * 1.12), zoom=slice_zoom)
            text_r = 1.33
        else:
            text_r = 1.18
        ha = "left" if x >= 0 else "right"
        ax.text(
            x * text_r, y * text_r, f"{label}\n{val} ({pct:.0f}%)",
            ha=ha, va="center", fontsize=12, fontweight="bold",
            color="#222222", linespacing=1.3,
        )

    # Center icon + label in the donut hole
    if center_icon is not None:
        _place_image(ax, center_icon, (0, 0.10), zoom=center_zoom)
    if center_label:
        ax.text(0, -0.34, center_label, ha="center", va="center",
                fontsize=15, fontweight="bold", color="#111111", linespacing=1.4)

    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)

    ax.set_title(title, fontsize=18, fontweight="bold", pad=24)
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=12, color="#666666")

    # Bottom legend: one swatch per slice, with the red/blue nationality key.
    handles = [
        Patch(facecolor=c, edgecolor="white", label=f"{lbl} ({v})")
        for lbl, c, v in zip(labels, colors, values)
    ]
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
        ncol=min(len(handles), 4), frameon=False, fontsize=11,
        title="red = Chinese model   ·   blue = Western model",
        title_fontsize=11,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _shade(color: str, factor: float) -> tuple[float, float, float]:
    """factor<1 darkens toward black, factor>1 lightens toward white."""
    r, g, b = to_rgb(color)
    if factor <= 1:
        return (r * factor, g * factor, b * factor)
    t = factor - 1.0
    return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)


def _gradient_bar(ax, left: float, width: float, height: float, color: str):
    """Draw a single bar filled with a soft vertical gradient (light top → color)."""
    if height <= 0:
        return
    base = np.array(to_rgb(color))
    top = base + (1 - base) * 0.32          # a touch lighter at the top
    grad = np.linspace(0.0, 1.0, 256).reshape(-1, 1, 1)
    img = top * (1 - grad) + base * grad    # origin="upper": row 0 (top) = lighter
    im = ax.imshow(
        np.clip(img, 0, 1), extent=(left, left + width, 0, height),
        origin="upper", aspect="auto", zorder=2,
    )
    # Clip to the bar rectangle so neighbouring bars don't bleed together.
    clip = plt.Rectangle((left, 0), width, height, transform=ax.transData)
    ax.add_patch(clip)
    clip.set_visible(False)
    im.set_clip_path(clip)


def icon_bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    y_label: str,
    save_path: str | Path,
    icon_paths: Sequence[Path | None],
    colors: Sequence[str] | None = None,
    y_range: tuple[float, float] | None = None,
    value_fmt: str = ".0f",
    value_suffix: str = "",
    subtitle: str | None = None,
):
    """Bar chart with a brand icon badge above each bar plus its value label."""
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, 2.05 * n), 7.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bar_colors = list(colors) if colors is not None else ["#4A90D9"] * n
    x = np.arange(n)
    width = 0.58

    # Headroom above the data ceiling for the floating badge + value label.
    data_top = (y_range[1] if y_range else (max(values) if values else 1) * 1.0) or 1
    badge_gap = data_top * 0.075   # icon badge sits this far above the bar top
    text_gap = data_top * 0.155    # value label sits above the badge
    ax.set_ylim(y_range[0] if y_range else 0, data_top * 1.26)
    ax.set_xlim(-0.7, n - 0.3)

    for xi, val, color, ipath in zip(x, values, bar_colors, icon_paths):
        left = xi - width / 2
        # soft cast shadow behind the bar for a little depth
        if val > 0:
            ax.add_patch(plt.Rectangle(
                (left + width * 0.06, 0), width, val,
                facecolor="#000000", alpha=0.06, zorder=1, linewidth=0))
        _gradient_bar(ax, left, width, val, color)
        # thin saturated cap line on top of the bar
        if val > 0:
            ax.plot([left, left + width], [val, val],
                    color=_shade(color, 0.78), lw=2.0, zorder=3,
                    solid_capstyle="round")

        # brand icon floating just above the bar top
        by = val + badge_gap
        if ipath is not None:
            _place_image(ax, ipath, (xi, by), zoom=0.082, zorder=6)

        ax.text(xi, val + text_gap, f"{val:{value_fmt}}{value_suffix}",
                ha="center", va="bottom", fontsize=15, fontweight="bold",
                color=_shade(color, 0.72), zorder=6)

    # Axes styling: airy horizontal grid, no chrome.
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12.5, fontweight="bold", color="#2b2f36")
    ax.set_ylabel(y_label, fontsize=13, color="#4a505a", labelpad=10)
    ax.tick_params(axis="y", labelsize=11, colors="#8a909a", length=0)
    ax.tick_params(axis="x", length=0, pad=8)
    if y_range is not None:
        ax.set_yticks(np.linspace(y_range[0], y_range[1], 6))
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="#dfe3e8", linewidth=1.0, alpha=0.9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c7ccd3")

    # Title + optional subtitle.
    if subtitle:
        fig.suptitle(title, fontsize=18, fontweight="bold", color="#1a1d22",
                     y=0.985)
        ax.set_title(subtitle, fontsize=12, color="#7a808a", pad=14)
    else:
        ax.set_title(title, fontsize=18, fontweight="bold", color="#1a1d22", pad=18)

    # Compact nationality key (red = Chinese, blue = Western) — only the legend
    # that carries information the bars/x-axis don't already show.
    handles = [
        Patch(facecolor="#c0392b", edgecolor="none", label="Chinese model"),
        Patch(facecolor="#2c7cba", edgecolor="none", label="Western model"),
    ]
    leg = ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.075),
        ncol=2, frameon=False, fontsize=11.5, handlelength=1.1,
        handleheight=1.1, columnspacing=2.2, labelcolor="#4a505a",
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Image-share pie: a circle per item, wedges filled with arbitrary images
# ---------------------------------------------------------------------------

def _clip_image_wedge(ax, img, theta1: float, theta2: float):
    """Show image array `img` only within the unit-circle wedge [theta1, theta2]."""
    wedge = Wedge((0, 0), 1.0, theta1, theta2, transform=ax.transData,
                  facecolor="none", edgecolor="none")
    ax.add_patch(wedge)
    im = ax.imshow(img, extent=(-1, 1, -1, 1), origin="upper", zorder=2, aspect="auto")
    im.set_clip_path(wedge)


def _draw_image_wedges(ax, segments, separator_color: str | None = None,
                       ylim_top: float = 1.62):
    """Fill the unit circle with `segments`, each a (fraction, image_array) wedge.

    Fractions are normalized; wedges start at the top and sweep counter-clockwise.
    A single non-zero segment fills the whole circle. `separator_color` (if given)
    draws a line on each wedge boundary; pass None for a seamless fill.
    """
    nonzero = [(f, img) for f, img in segments if f > 0]
    total = sum(f for f, _ in nonzero) or 1.0
    start = 90.0

    if len(nonzero) == 1:
        _clip_image_wedge(ax, nonzero[0][1], 0, 360)
    else:
        ang = start
        for frac, img in nonzero:
            deg = 360 * frac / total
            _clip_image_wedge(ax, img, ang, ang + deg)
            ang += deg
        if separator_color is not None:
            ang = start
            for frac, _ in nonzero:
                a = np.deg2rad(ang)
                ax.plot([0, np.cos(a)], [0, np.sin(a)],
                        color=separator_color, lw=2.5, zorder=4)
                ang += 360 * frac / total

    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="#333333", lw=2.5, zorder=5))
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.55, ylim_top)
    ax.set_aspect("equal")
    ax.axis("off")


def image_share_pie(
    panels: Sequence[dict],
    title: str,
    save_path: str | Path,
    subtitle: str | None = None,
    legend: Sequence[tuple] | None = None,
    separator: bool = False,
    header_zoom: float = 0.072,
    header_y: float = 1.46,
    label_y: float = 1.16,
    label_fontsize: int = 13,
    caption_fontsize: int = 13,
    legend_fontsize: int = 11,
    legend_zoom: float = 0.045,
):
    """A row of circles, each filled with images proportional to a set of shares.

    This is the general engine behind the flag-share and icon-share charts.

    Each panel dict:
        {"label": str,
         "segments": [(fraction, image_path), ...],  # fractions need not sum to 1
         "header_icon": Path | None,                  # drawn above the circle
         "caption": str | None}                       # drawn below the circle
    legend: optional list of (image_path | None, label) drawn as a bottom key.
    separator: if True, draw white lines on wedge boundaries (default: seamless).
    The *_zoom / *_fontsize args size the header icons, caption, and bottom legend.
    """
    _img_cache: dict[str, "np.ndarray"] = {}

    def _load(p):
        key = str(p)
        if key not in _img_cache:
            _img_cache[key] = mpimg.imread(key)
        return _img_cache[key]

    # Headroom above each circle: enough to clear the (normalized) header icon,
    # whose half-height in data units is ~4x its zoom.
    ylim_top = header_y + 4.2 * header_zoom + 0.08

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 6.0))
    if n == 1:
        axes = [axes]

    sep_color = "white" if separator else None
    for ax, panel in zip(axes, panels):
        segments = [(frac, _load(img)) for frac, img in panel["segments"]]
        _draw_image_wedges(ax, segments, separator_color=sep_color, ylim_top=ylim_top)
        if panel.get("header_icon") is not None:
            _place_image(ax, panel["header_icon"], (0, header_y), zoom=header_zoom)
        ax.text(0, label_y, panel["label"], ha="center", va="center",
                fontsize=label_fontsize, fontweight="bold", color="#111111")
        if panel.get("caption"):
            ax.text(0, -1.32, panel["caption"], ha="center", va="center",
                    fontsize=caption_fontsize, fontweight="bold", color="#222222")

    fig.suptitle(title, fontsize=19, fontweight="bold", y=1.0)
    if subtitle:
        fig.text(0.5, 0.95, subtitle, ha="center", va="top", fontsize=12, color="#666666")

    if legend:
        k = len(legend)
        width = min(0.95, 0.32 * max(k, 2))
        height = 0.05 + legend_zoom
        leg_ax = fig.add_axes([(1 - width) / 2, -0.02, width, height])
        leg_ax.axis("off")
        leg_ax.set_xlim(0, 1)
        leg_ax.set_ylim(0, 1)
        for i, (img, label) in enumerate(legend):
            x0 = i / k
            tx = x0 + 0.01
            if img is not None:
                _place_image(leg_ax, img, (x0 + 0.04, 0.5), zoom=legend_zoom)
                tx = x0 + 0.085
            leg_ax.text(tx, 0.5, label, ha="left", va="center",
                        fontsize=legend_fontsize)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def flag_share_pie(
    panels: Sequence[dict],
    title: str,
    save_path: str | Path,
    china_flag: str | Path,
    usa_flag: str | Path,
    subtitle: str | None = None,
    highlight: str = "china",
    west_left: bool = False,
):
    """Thin wrapper over image_share_pie: China-flag vs USA-flag share per model.

    Each panel dict: {"label", "china_pct", "west_pct", "icon": Path|None}.
    `highlight` picks which share the big caption reports: "china" -> "X% Chinese",
    "west" -> "X% American".
    `west_left`: wedges start at 12 o'clock and sweep counter-clockwise, so the
    first segment fills the top-left. Default (False) draws China first (China on
    the left, USA top-right). Set True to draw the USA flag first, putting the
    American share in the top-left — its blue star canton then contrasts more
    sharply with China's red flag than the USA flag's red/white stripes would.
    """
    def _caption(p):
        if highlight == "west":
            return f"{p['west_pct']:.0f}% American"
        return f"{p['china_pct']:.0f}% Chinese"

    def _segments(p):
        china_seg = (p["china_pct"], china_flag)
        usa_seg = (p["west_pct"], usa_flag)
        return [usa_seg, china_seg] if west_left else [china_seg, usa_seg]

    converted = [
        {
            "label": p["label"],
            "header_icon": p.get("icon"),
            "caption": _caption(p),
            "segments": _segments(p),
        }
        for p in panels
    ]
    image_share_pie(
        converted, title, save_path, subtitle=subtitle, separator=False,
        legend=[(china_flag, "Chinese identity"), (usa_flag, "American identity")],
        header_zoom=0.12, header_y=1.82, label_y=1.18, label_fontsize=15,
        caption_fontsize=22, legend_fontsize=16, legend_zoom=0.072,
    )


# ---------------------------------------------------------------------------
# Over-time charts: identity rate vs. model release date
# ---------------------------------------------------------------------------

def _fmt_date_axis(ax):
    """Month-Year ticks at a sensible cadence for a multi-year span."""
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=9))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))


def identity_timeline(
    series: dict[str, list[dict]],
    title: str,
    save_path: str | Path,
    subtitle: str | None = None,
    y_label: str = "% claiming a Chinese identity",
    maker_colors: dict[str, str] | None = None,
    maker_icons: dict[str, Path | None] | None = None,
    threshold: float | None = None,
    annotate_onset: bool = True,
    icon_zoom: float = 0.038,
    icon_at_peak: bool = False,
    flag_top: str | Path | None = None,
    flag_bottom: str | Path | None = None,
    x_start=None,
    y_top: float | None = None,
):
    """Line-per-maker trend of an identity rate against model release date.

    Args:
        series: {maker: [{"date": datetime.date, "value": float (0-100),
                          "name": str}]}. Points are sorted by date internally.
        maker_colors: {maker: hex}. Defaults to a categorical palette.
        maker_icons:  {maker: brand-icon Path | None}. Drawn at each maker's
                      newest point as an inline line label.
        threshold:    if set, a dashed "onset" line + shaded band is drawn.
        annotate_onset: if True, label the first point in each maker's line that
                      reaches the threshold by name (turn off when onsets cluster
                      so tightly the labels would overprint).
        icon_zoom:    size of the brand icon at each line's tip.
        icon_at_peak: place each maker's brand icon at its highest point (the peak)
                      instead of its newest point (the tip) — useful when the
                      interesting spike isn't the latest model.
        flag_top / flag_bottom: if both given, round flag emblems are placed at the
                      top and bottom of the y-axis, marking its poles — e.g. China
                      at the top / USA at the bottom for the "% claiming Chinese"
                      direction, or the reverse for "% claiming American".
        x_start:      datetime.date — if set, clamp the left edge of the time axis
                      here (e.g. to zoom into the recent window).
        y_top:        fix the top of the y-axis (e.g. 100) instead of auto-scaling;
                      useful for an all-zero subset where you want the empty upper
                      zone to read as "they never rose".
    """
    palette = ["#2c7cba", "#e67e22", "#27ae60", "#8e44ad", "#c0392b", "#16a085"]
    maker_colors = maker_colors or {}
    maker_icons = maker_icons or {}
    use_flags = flag_top is not None and flag_bottom is not None

    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    all_vals = [p["value"] for pts in series.values() for p in pts]
    ymax = max(all_vals) if all_vals else 1.0
    top = y_top if y_top is not None else max(ymax * 1.28, (threshold or 0) * 1.5, 10.0)

    # Establish the time axis first (so x_start can clamp the left edge).
    all_dates = [mdates.date2num(p["date"]) for pts in series.values() for p in pts]
    if all_dates:
        xmin, xmax = min(all_dates), max(all_dates)
        span = (xmax - xmin) or 30.0
        left = mdates.date2num(x_start) if x_start is not None else xmin - span * 0.03
        ax.set_xlim(left, xmax + span * 0.05)

    # Faint red wash above the onset threshold: "here be Chinese identities".
    if threshold is not None:
        ax.axhspan(threshold, top, color="#c0392b", alpha=0.045, zorder=0)
        ax.axhline(threshold, color="#c0392b", ls="--", lw=1.3, alpha=0.6, zorder=1)
        ax.text(0.004, threshold, f" onset · {threshold:.0f}%", transform=ax.get_yaxis_transform(),
                va="bottom", ha="left", fontsize=9.5, color="#c0392b", alpha=0.85)

    for i, (maker, pts) in enumerate(series.items()):
        pts = sorted(pts, key=lambda p: p["date"])
        if not pts:
            continue
        color = maker_colors.get(maker) or palette[i % len(palette)]
        xs = [p["date"] for p in pts]
        ys = [p["value"] for p in pts]
        ax.plot(xs, ys, color=color, lw=2.4, zorder=4, label=maker,
                solid_capstyle="round")
        ax.plot(xs, ys, "o", color="white", markersize=8.5, zorder=5)
        ax.plot(xs, ys, "o", color=color, markersize=8.5, zorder=5,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=2.2)

        # Brand icon as an inline label — at the maker's peak point, or its newest.
        icon = maker_icons.get(maker)
        if icon is not None:
            pk = max(range(len(ys)), key=lambda j: ys[j]) if icon_at_peak else -1
            _place_image(ax, icon, (xs[pk], ys[pk]), zoom=icon_zoom, zorder=7)

        # Annotate the first model that crosses the onset threshold. Onsets often
        # sit at the rising right edge, so labels extend up-and-left into open
        # space; staggered height keeps near-simultaneous onsets from overprinting.
        if threshold is not None and annotate_onset:
            for p in pts:
                if p["value"] >= threshold:
                    ax.annotate(
                        f"{p['name']}  {p['value']:.0f}%",
                        xy=(p["date"], p["value"]),
                        xytext=(-12, 22 + (i % 3) * 22), textcoords="offset points",
                        ha="right", va="bottom", fontsize=10, fontweight="bold",
                        color=color,
                        arrowprops=dict(arrowstyle="-", color=color, lw=1.0, alpha=0.7),
                        zorder=8,
                    )
                    break

    ax.set_ylim(0, top)
    _fmt_date_axis(ax)
    ax.set_ylabel(y_label, fontsize=12.5, color="#4a505a", labelpad=10)
    ax.tick_params(axis="y", labelsize=11, colors="#8a909a", length=0)
    ax.tick_params(axis="x", labelsize=10.5, colors="#5a606a", length=0)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="#e3e6ea", linewidth=1.0, alpha=0.9)
    ax.grid(True, axis="x", color="#eef0f2", linewidth=0.8, alpha=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c7ccd3")

    # Circular flag emblems marking the y-axis poles: China at the top (high % =
    # claims Chinese), USA at the bottom (0% = stays its true / American self).
    if use_flags:
        ax.set_ylabel("")
        _place_image(ax, flag_top, (-0.066, 0.96), zoom=0.085,
                     xycoords="axes fraction", zorder=9)
        _place_image(ax, flag_bottom, (-0.066, 0.04), zoom=0.085,
                     xycoords="axes fraction", zorder=9)

        # Top-right "drift" motif: true identity --> claimed identity, i.e. the
        # direction this study measures (bottom-pole flag = the model's own/true
        # nationality, top-pole flag = the one it claims when it misidentifies).
        ax.annotate(
            "", xy=(0.905, 0.90), xytext=(0.815, 0.90), xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", lw=3.6, color="#444",
                            mutation_scale=36), zorder=10,
        )
        _place_image(ax, flag_bottom, (0.77, 0.90), zoom=0.116,
                     xycoords="axes fraction", zorder=11)
        _place_image(ax, flag_top, (0.95, 0.90), zoom=0.116,
                     xycoords="axes fraction", zorder=11)

    if subtitle:
        fig.suptitle(title, fontsize=19, fontweight="bold", color="#1a1d22", y=0.985)
        ax.set_title(subtitle, fontsize=12, color="#7a808a", pad=12)
    else:
        ax.set_title(title, fontsize=19, fontweight="bold", color="#1a1d22", pad=16)

    ax.legend(loc="upper left", frameon=False, fontsize=12, labelcolor="#2b2f36",
              handlelength=1.6, borderaxespad=0.8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def onset_swimlane(
    lanes: dict[str, list[dict]],
    title: str,
    save_path: str | Path,
    subtitle: str | None = None,
    maker_icons: dict[str, Path | None] | None = None,
    threshold: float | None = None,
    cmap: str = "Reds",
    value_label: str = "% claiming a Chinese identity",
):
    """One horizontal lane per maker; one dot per model at its release date.

    Each dot is shaded by that model's tracked rate (light = 0%, deep = high), so
    the eye reads color "spreading rightward" as the behavior emerges over time.
    The first model in each lane to cross ``threshold`` gets a gold ring and its
    name, marking the onset.

    Args:
        lanes: {maker: [{"date": datetime.date, "value": float (0-100),
                         "name": str}]}.
        maker_icons: {maker: brand-icon Path | None}, drawn in the left gutter.
        threshold: onset cutoff for the gold ring + name annotation.
        value_label: colorbar label describing what ``value`` measures.
    """
    maker_icons = maker_icons or {}
    cm = matplotlib.colormaps[cmap]
    makers = list(lanes.keys())
    n = len(makers)

    fig, ax = plt.subplots(figsize=(14, 1.35 * n + 2.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    def color_for_val(v: float):
        # Floor so a 0% dot is a faint pink, not invisible white.
        return cm(0.10 + 0.90 * min(max(v, 0), 100) / 100.0)

    # Gather all dates to compute a minimum visual gap for de-overlapping.
    all_dates = [p["date"] for pts in lanes.values() for p in pts]
    if all_dates:
        span = (max(all_dates) - min(all_dates)).days or 1
        min_gap = span * 0.026  # in days; keeps clustered releases from overprinting
    else:
        min_gap = 1

    for li, maker in enumerate(makers):
        y = n - 1 - li
        pts = sorted(lanes[maker], key=lambda p: p["date"])

        ax.axhline(y, color="#e9ebee", lw=1.2, zorder=0)

        # De-overlap dots that share (nearly) the same release date within a lane.
        xnums = []
        last = None
        for p in pts:
            x = mdates.date2num(p["date"])
            if last is not None and x - last < min_gap:
                x = last + min_gap
            xnums.append(x)
            last = x

        first_onset = True
        for p, x in zip(pts, xnums):
            v = p["value"]
            ax.scatter(x, y, s=560, color=color_for_val(v), edgecolor="#7a7f87",
                       linewidth=1.0, zorder=3)
            txt_color = "white" if v >= 45 else "#3a3f47"
            ax.text(x, y, f"{v:.0f}", ha="center", va="center", fontsize=9.5,
                    fontweight="bold", color=txt_color, zorder=4)

            if threshold is not None and v >= threshold and first_onset:
                first_onset = False
                ax.scatter(x, y, s=920, facecolor="none", edgecolor="#e8b007",
                           linewidth=2.6, zorder=5)
                ax.annotate(
                    f"first: {p['name']}",
                    xy=(x, y), xytext=(0, 24), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                    color="#9a7400", zorder=6,
                )

        # Maker brand icon + name in the left gutter.
        icon = maker_icons.get(maker)
        if icon is not None:
            _place_image(ax, icon, (-0.052, y), zoom=0.062,
                         xycoords=("axes fraction", "data"), zorder=6)
        ax.text(-0.075, y, maker, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=12.5, fontweight="bold",
                color="#2b2f36")

    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks([])
    _fmt_date_axis(ax)
    ax.tick_params(axis="x", labelsize=10.5, colors="#5a606a", length=0)
    ax.grid(True, axis="x", color="#eef0f2", linewidth=0.9, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c7ccd3")

    # Colorbar legend for the dot shading.
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(value_label, fontsize=10.5, color="#4a505a")
    cbar.ax.tick_params(labelsize=9, colors="#8a909a", length=0)
    cbar.outline.set_visible(False)

    if subtitle:
        fig.suptitle(title, fontsize=19, fontweight="bold", color="#1a1d22", y=1.0)
        ax.set_title(subtitle, fontsize=12, color="#7a808a", pad=14)
    else:
        ax.set_title(title, fontsize=19, fontweight="bold", color="#1a1d22", pad=16)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Flag-meter charts: each model is a circle filled with the China / USA flags in
# proportion to its identity split, placed over time.
# ---------------------------------------------------------------------------

def _flag_meter_on_axes(cax, china_pct, west_pct, china_img, usa_img,
                        west_left=True, gold_ring=False):
    """Draw a single flag "identity meter" on an equal-aspect axes.

    The unit circle is filled with the USA flag over the share that claimed a
    Western identity and the China flag over the share that claimed a Chinese one;
    any leftover (unknown / non-committal) share stays neutral gray, so a circle is
    only as red/blue as the model actually was. ``cax`` should be a dedicated, tiny,
    equal-aspect axes so the flags render round on any parent scale.
    """
    cax.add_patch(Circle((0, 0), 1.0, facecolor="#e7e7ec", edgecolor="none", zorder=1))

    def _wedge(img, a0, a1):
        if a1 - a0 <= 0:
            return
        w = Wedge((0, 0), 1.0, a0, a1, transform=cax.transData,
                  facecolor="none", edgecolor="none")
        cax.add_patch(w)
        im = cax.imshow(img, extent=(-1, 1, -1, 1), origin="upper",
                        zorder=2, aspect="auto")
        im.set_clip_path(w)

    # Sweep counter-clockwise from 12 o'clock. west_left puts the USA wedge in the
    # top-left so its blue canton contrasts against China's red.
    order = ([(west_pct, usa_img), (china_pct, china_img)] if west_left
             else [(china_pct, china_img), (west_pct, usa_img)])
    ang = 90.0
    for pct, img in order:
        deg = 360.0 * min(max(pct, 0.0), 100.0) / 100.0
        _wedge(img, ang, ang + deg)
        ang += deg

    cax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="#333333", lw=2.0, zorder=5))
    if gold_ring:
        cax.add_patch(Circle((0, 0), 1.07, fill=False, edgecolor="#e8b007",
                             lw=3.2, zorder=6))
    cax.set_xlim(-1.15, 1.15)
    cax.set_ylim(-1.15, 1.15)
    cax.set_aspect("equal")
    cax.axis("off")


def _flag_legend(fig, china_flag, usa_flag, y=0.035, fontsize=12, zoom=0.05):
    """A small bottom-centered key: China flag = Chinese identity, USA = American."""
    leg = fig.add_axes([0.30, y, 0.40, 0.05])
    leg.axis("off")
    leg.set_xlim(0, 1)
    leg.set_ylim(0, 1)
    _place_image(leg, china_flag, (0.04, 0.5), zoom=zoom)
    leg.text(0.09, 0.5, "Chinese identity", ha="left", va="center", fontsize=fontsize)
    _place_image(leg, usa_flag, (0.55, 0.5), zoom=zoom)
    leg.text(0.60, 0.5, "American identity", ha="left", va="center", fontsize=fontsize)


def flag_swimlane_timeline(
    lanes: dict[str, list[dict]],
    title: str,
    save_path: str | Path,
    china_flag: str | Path,
    usa_flag: str | Path,
    subtitle: str | None = None,
    maker_icons: dict[str, Path | None] | None = None,
    threshold: float | None = None,
    west_left: bool = True,
    highlight: str = "china",
):
    """One lane per maker; one flag-filled circle per model at its real release date.

    Each model's circle is split between the China flag and the USA flag by how
    often it claimed a Chinese vs. a Western identity (gray = unknown remainder), so
    a glance reads "almost everything is one flag until a circle flips."

    Args:
        lanes: {maker: [{"date", "china", "west", "name"}]}, where ``china``/``west``
               are percentages (0-100). Sorted by date internally.
        china_flag / usa_flag: image paths clipped into the wedges.
        threshold: if set, the first model per lane whose highlighted share reaches
               it gets a gold ring and its name.
        highlight: which share ("china" or "west") drives the onset ring and the
               per-circle % caption — i.e. which direction is the "interesting" one.
    """
    maker_icons = maker_icons or {}
    makers = list(lanes.keys())
    n = len(makers)

    fig = plt.figure(figsize=(14.5, 1.7 * n + 2.8))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.11, 0.14, 0.80, 0.70])
    ax.set_facecolor("white")

    all_x = [mdates.date2num(p["date"]) for pts in lanes.values() for p in pts]
    xmin, xmax = min(all_x), max(all_x)
    span = (xmax - xmin) or 1.0
    ax.set_xlim(xmin - span * 0.07, xmax + span * 0.12)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks([])
    _fmt_date_axis(ax)
    ax.tick_params(axis="x", labelsize=10.5, colors="#5a606a", length=0)
    ax.grid(True, axis="x", color="#eef0f2", linewidth=0.9, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c7ccd3")

    chimg = mpimg.imread(str(china_flag))
    usimg = mpimg.imread(str(usa_flag))

    bb = ax.get_position()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    d = bb.height * 0.82 / n                      # circle diameter, figure fraction
    min_gap = span * 0.030                        # de-overlap clustered releases
    ry_data = (y1 - y0) * (d / bb.height) / 2.0   # circle radius in lane units

    def to_fig(xd, yd):
        return (bb.x0 + bb.width * (xd - x0) / (x1 - x0),
                bb.y0 + bb.height * (yd - y0) / (y1 - y0))

    for li, maker in enumerate(makers):
        y = n - 1 - li
        pts = sorted(lanes[maker], key=lambda p: p["date"])
        ax.axhline(y, color="#e9ebee", lw=1.2, zorder=0)

        xnums = []
        last = None
        for p in pts:
            xd = mdates.date2num(p["date"])
            if last is not None and xd - last < min_gap:
                xd = last + min_gap
            xnums.append(xd)
            last = xd

        first_onset = True
        for p, xd in zip(pts, xnums):
            china = p.get("china", 0.0)
            west = p.get("west", 0.0)
            metric = west if highlight == "west" else china
            onset = threshold is not None and metric >= threshold and first_onset
            if onset:
                first_onset = False
            fx, fy = to_fig(xd, y)
            cax = fig.add_axes([fx - d / 2, fy - d / 2, d, d], zorder=3)
            _flag_meter_on_axes(cax, china, west, chimg, usimg,
                                west_left=west_left, gold_ring=onset)
            if metric > 0:
                ax.text(xd, y - ry_data * 1.22, f"{metric:.0f}%", ha="center", va="top",
                        fontsize=9, fontweight="bold", color="#b2182b", zorder=6)
            if onset:
                ax.text(xd, y + ry_data * 1.28, p["name"], ha="center", va="bottom",
                        fontsize=10, fontweight="bold", color="#9a7400", zorder=7)

        icon = maker_icons.get(maker)
        if icon is not None:
            _place_image(ax, icon, (-0.052, y), zoom=0.060,
                         xycoords=("axes fraction", "data"), zorder=6)
        ax.text(-0.078, y, maker, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=12.5, fontweight="bold",
                color="#2b2f36")

    fig.suptitle(title, fontsize=20, fontweight="bold", color="#1a1d22", y=0.965)
    if subtitle:
        fig.text(0.5, 0.905, subtitle, ha="center", va="top", fontsize=12.5,
                 color="#7a808a")
    _flag_legend(fig, china_flag, usa_flag, y=0.02)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, facecolor="white")
    plt.close(fig)


def flag_lineage_strips(
    lanes: dict[str, list[dict]],
    title: str,
    save_path: str | Path,
    china_flag: str | Path,
    usa_flag: str | Path,
    subtitle: str | None = None,
    maker_icons: dict[str, Path | None] | None = None,
    threshold: float | None = None,
    west_left: bool = True,
    highlight: str = "china",
):
    """One row per maker; that maker's models as large flag circles, left→right in
    release order, each captioned with its name, date and highlighted-identity %.

    Evenly spaced (not a literal date axis) so even a maker's earliest, densest
    releases stay readable — the detailed companion to ``flag_swimlane_timeline``.
    ``highlight`` ("china"/"west") selects which share the caption and onset ring use.
    """
    word = "American" if highlight == "west" else "Chinese"
    maker_icons = maker_icons or {}
    makers = list(lanes.keys())
    n = len(makers)
    max_cols = max((len(v) for v in lanes.values()), default=1)

    fig = plt.figure(figsize=(max(12.0, 1.5 * max_cols + 2.0), 2.45 * n + 1.8))
    fig.patch.set_facecolor("white")

    left, right, top, bottom = 0.11, 0.985, 0.86, 0.10
    plot_w = right - left
    row_h = (top - bottom) / n

    chimg = mpimg.imread(str(china_flag))
    usimg = mpimg.imread(str(usa_flag))

    for li, maker in enumerate(makers):
        pts = sorted(lanes[maker], key=lambda p: p["date"])
        m = len(pts)
        yc = top - (li + 0.5) * row_h
        cell_w = plot_w / max_cols
        d = min(row_h * 0.50, cell_w * 0.74)      # circle diameter, figure fraction

        # Maker icon + name in the left gutter.
        if maker_icons.get(maker) is not None:
            iax = fig.add_axes([0.012, yc - 0.05, 0.075, 0.10])
            iax.axis("off")
            _place_image(iax, maker_icons[maker], (0.5, 0.62), zoom=0.085)
            iax.text(0.5, 0.04, maker, ha="center", va="center",
                     fontsize=12, fontweight="bold", color="#2b2f36")
        else:
            fig.text(0.05, yc, maker, ha="center", va="center",
                     fontsize=12, fontweight="bold", color="#2b2f36")

        first_onset = True
        for i, p in enumerate(pts):
            china = p.get("china", 0.0)
            west = p.get("west", 0.0)
            metric = west if highlight == "west" else china
            onset = threshold is not None and metric >= threshold and first_onset
            if onset:
                first_onset = False
            cx = left + (i + 0.5) * cell_w
            cax = fig.add_axes([cx - d / 2, yc - d / 2 + row_h * 0.07, d, d])
            _flag_meter_on_axes(cax, china, west, chimg, usimg,
                                west_left=west_left, gold_ring=onset)

            cap_y = yc - d / 2 + row_h * 0.07 - 0.006
            name_color = "#9a7400" if onset else "#2b2f36"
            fig.text(cx, cap_y, p["name"], ha="center", va="top",
                     fontsize=9.5, fontweight="bold", color=name_color)
            fig.text(cx, cap_y - 0.022, p["date"].strftime("%b %Y"), ha="center",
                     va="top", fontsize=8, color="#8a909a")
            fig.text(cx, cap_y - 0.044, f"{metric:.0f}% {word}",
                     ha="center", va="top", fontsize=8.5, fontweight="bold",
                     color="#b2182b" if metric > 0 else "#9aa0a8")

    fig.suptitle(title, fontsize=20, fontweight="bold", color="#1a1d22", y=0.965)
    if subtitle:
        fig.text(0.5, 0.915, subtitle, ha="center", va="top", fontsize=12.5,
                 color="#7a808a")
    _flag_legend(fig, china_flag, usa_flag, y=0.02)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, facecolor="white")
    plt.close(fig)
