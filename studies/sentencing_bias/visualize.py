"""Visualization for the Sentencing Bias study.

Generates heatmap grids (defendants as rows, crimes as columns) for each model,
with statistical significance annotations. Also generates a combined z-score
chart showing overall sentencing bias per defendant.
"""

import os
from pathlib import Path
from itertools import combinations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy import stats

from studies.sentencing_bias.config import DEFENDANTS, CRIMES


# Warm gradient: low sentence = light, high sentence = dark red
JAIL_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "jail_severity", ["#FFF5F0", "#FEE0D2", "#FCBBA1", "#FC9272", "#FB6A4A",
                       "#EF3B2C", "#CB181D", "#99000D"]
)

# Cool gradient for fines
FINE_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "fine_severity", ["#F7FBFF", "#DEEBF7", "#C6DBEF", "#9ECAE1", "#6BAED6",
                       "#4292C6", "#2171B5", "#084594"]
)

# Diverging: blue (lenient) -> white (average) -> red (harsh)
ZSCORE_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "zscore_bias", ["#2166AC", "#92C5DE", "#F7F7F7", "#F4A582", "#B2182B"]
)


def _fmt_jail(years: float | None) -> str:
    if years is None:
        return "n/a"
    if years == 0:
        return "0"
    if years >= 1:
        return f"{years:.1f}y"
    return f"{years * 12:.0f}mo"


def _fmt_fine(usd: float | None) -> str:
    if usd is None:
        return "n/a"
    if usd == 0:
        return "$0"
    if usd >= 1000:
        return f"${usd / 1000:.0f}k"
    return f"${usd:,.0f}"


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Z-score computation
# ---------------------------------------------------------------------------

def compute_defendant_zscores(
    scores: dict[str, dict[str, dict]],
) -> dict[str, dict]:
    """Compute per-defendant z-scores normalized within each crime.

    For each crime, pools all defendants' iteration values, computes mean/std,
    then z-scores each defendant's values. This makes crimes with wildly
    different scales (shoplifting vs armed robbery) comparable.

    Returns {defendant_id: {
        "jail_zscores": list[float],    # all z-scored jail values across crimes
        "fine_zscores": list[float],     # all z-scored fine values across crimes
        "per_crime_jail_z": {crime_id: float},  # mean jail z per crime
        "per_crime_fine_z": {crime_id: float},  # mean fine z per crime
        "mean_jail_z": float,           # overall mean jail z-score
        "mean_fine_z": float,           # overall mean fine z-score
        "combined_z": float,            # average of jail + fine z-scores
    }}
    """
    defendant_ids = [d["id"] for d in DEFENDANTS]
    result = {d_id: {
        "jail_zscores": [], "fine_zscores": [],
        "per_crime_jail_z": {}, "per_crime_fine_z": {},
    } for d_id in defendant_ids}

    for crime in CRIMES:
        # Pool all iteration values across defendants for this crime
        all_jails = []
        all_fines = []
        per_defendant_jails = {}
        per_defendant_fines = {}

        for d_id in defendant_ids:
            s = scores.get(d_id, {}).get(crime["id"], {})
            jails = s.get("jails", [])
            fines = s.get("fines", [])
            per_defendant_jails[d_id] = jails
            per_defendant_fines[d_id] = fines
            all_jails.extend(jails)
            all_fines.extend(fines)

        # Z-score within this crime
        for metric, all_vals, per_d, z_key, per_crime_key in [
            ("jail", all_jails, per_defendant_jails, "jail_zscores", "per_crime_jail_z"),
            ("fine", all_fines, per_defendant_fines, "fine_zscores", "per_crime_fine_z"),
        ]:
            if len(all_vals) < 2:
                continue
            mean = np.mean(all_vals)
            std = np.std(all_vals, ddof=1)
            if std == 0:
                continue

            for d_id in defendant_ids:
                vals = per_d[d_id]
                if not vals:
                    continue
                zscored = [(v - mean) / std for v in vals]
                result[d_id][z_key].extend(zscored)
                result[d_id][per_crime_key][crime["id"]] = np.mean(zscored)

    # Compute overall means
    for d_id in defendant_ids:
        jail_z = result[d_id]["jail_zscores"]
        fine_z = result[d_id]["fine_zscores"]
        result[d_id]["mean_jail_z"] = np.mean(jail_z) if jail_z else 0.0
        result[d_id]["mean_fine_z"] = np.mean(fine_z) if fine_z else 0.0
        result[d_id]["combined_z"] = (
            (result[d_id]["mean_jail_z"] + result[d_id]["mean_fine_z"]) / 2
        )

    return result


def compute_significance(
    scores: dict[str, dict[str, dict]],
    zscores: dict[str, dict],
) -> dict:
    """Compute statistical significance tests.

    Returns {
        "per_crime_kw": {crime_id: {"jail_p": float, "fine_p": float}},
        "overall_kw": {"jail_p": float, "fine_p": float},
        "pairwise": {(id_a, id_b): {"jail_p": float, "fine_p": float}},
        "pairwise_per_crime": {crime_id: {(id_a, id_b): {"jail_p": float, "fine_p": float}}},
    }
    """
    defendant_ids = [d["id"] for d in DEFENDANTS]
    result = {
        "per_crime_kw": {},
        "overall_kw": {},
        "pairwise": {},
        "pairwise_per_crime": {},
    }

    # --- Per-crime Kruskal-Wallis: do defendants differ for this crime? ---
    for crime in CRIMES:
        kw = {}
        for metric in ("jails", "fines"):
            groups = []
            for d_id in defendant_ids:
                s = scores.get(d_id, {}).get(crime["id"], {})
                vals = s.get(metric, [])
                if vals:
                    groups.append(vals)
            if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
                _, p = stats.kruskal(*groups)
                kw[f"{metric[:-1]}_p"] = p
            else:
                kw[f"{metric[:-1]}_p"] = 1.0
        result["per_crime_kw"][crime["id"]] = kw

    # --- Per-crime pairwise Mann-Whitney U ---
    for crime in CRIMES:
        pw = {}
        for a, b in combinations(defendant_ids, 2):
            sa = scores.get(a, {}).get(crime["id"], {})
            sb = scores.get(b, {}).get(crime["id"], {})
            pair_result = {}
            for metric in ("jails", "fines"):
                vals_a = sa.get(metric, [])
                vals_b = sb.get(metric, [])
                if len(vals_a) >= 3 and len(vals_b) >= 3:
                    _, p = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
                    pair_result[f"{metric[:-1]}_p"] = p
                else:
                    pair_result[f"{metric[:-1]}_p"] = 1.0
            pw[(a, b)] = pair_result
        result["pairwise_per_crime"][crime["id"]] = pw

    # --- Overall Kruskal-Wallis on z-scores ---
    for metric in ("jail_zscores", "fine_zscores"):
        key = metric.split("_")[0] + "_p"
        groups = []
        for d_id in defendant_ids:
            vals = zscores[d_id][metric]
            if vals:
                groups.append(vals)
        if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
            _, p = stats.kruskal(*groups)
            result["overall_kw"][key] = p
        else:
            result["overall_kw"][key] = 1.0

    # --- Overall pairwise Mann-Whitney U on z-scores ---
    for a, b in combinations(defendant_ids, 2):
        pair_result = {}
        for metric in ("jail_zscores", "fine_zscores"):
            key = metric.split("_")[0] + "_p"
            vals_a = zscores[a][metric]
            vals_b = zscores[b][metric]
            if len(vals_a) >= 3 and len(vals_b) >= 3:
                _, p = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
                pair_result[key] = p
            else:
                pair_result[key] = 1.0
        result["pairwise"][(a, b)] = pair_result

    return result


# ---------------------------------------------------------------------------
# Per-model heatmap with significance
# ---------------------------------------------------------------------------

def generate_heatmap(
    model: str,
    scores: dict[str, dict[str, dict]],
    significance: dict,
    save_path: str | Path,
):
    """Generate a pair of heatmaps (jail + fine) with per-crime significance row."""
    short = model.split("/")[-1]
    n_defendants = len(DEFENDANTS)
    n_crimes = len(CRIMES)

    data_jail = np.full((n_defendants, n_crimes), np.nan)
    data_fine = np.full((n_defendants, n_crimes), np.nan)

    for d_idx, defendant in enumerate(DEFENDANTS):
        for c_idx, crime in enumerate(CRIMES):
            s = scores.get(defendant["id"], {}).get(crime["id"], {})
            jail = s.get("avg_jail")
            fine = s.get("avg_fine")
            if jail is not None:
                data_jail[d_idx, c_idx] = jail
            if fine is not None:
                data_fine[d_idx, c_idx] = fine

    fig, (ax_jail, ax_fine) = plt.subplots(1, 2, figsize=(22, 9))

    defendant_labels = [d["name"] for d in DEFENDANTS]
    crime_labels = [c["label"] for c in CRIMES]

    for data, cmap, ax, title, fmt_fn, metric_key, cbar_label in [
        (data_jail, JAIL_CMAP, ax_jail, "Jail Time", _fmt_jail, "jail", "Avg jail (years)"),
        (data_fine, FINE_CMAP, ax_fine, "Fine", _fmt_fine, "fine", "Avg fine (USD)"),
    ]:
        vmin = np.nanmin(data) if not np.all(np.isnan(data)) else 0
        vmax = np.nanmax(data) if not np.all(np.isnan(data)) else 1
        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        val_range = vmax - vmin if vmax > vmin else 1
        for d_idx in range(n_defendants):
            for c_idx in range(n_crimes):
                val = data[d_idx, c_idx]
                text = fmt_fn(val if not np.isnan(val) else None)
                brightness = (val - vmin) / val_range if not np.isnan(val) else 0
                color = "white" if brightness > 0.6 else "black"
                ax.text(c_idx, d_idx, text, ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)

        ax.set_xticks(range(n_crimes))
        ax.set_yticks(range(n_defendants))
        ax.set_yticklabels(defendant_labels, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

        # Annotate Kruskal-Wallis p-value per crime below x-axis
        kw_labels = []
        for crime in CRIMES:
            kw = significance["per_crime_kw"].get(crime["id"], {})
            p = kw.get(f"{metric_key}_p", 1.0)
            s = _stars(p)
            if s:
                kw_labels.append(f"KW {s}\np={p:.3f}")
            else:
                kw_labels.append("n.s.")
        ax.set_xticklabels(
            [f"{cl}\n{kw}" for cl, kw in zip(crime_labels, kw_labels)],
            fontsize=9, rotation=0, ha="center",
        )

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(cbar_label, fontsize=11)

    fig.suptitle(f"Sentencing Bias by Name: {short}",
                 fontsize=16, fontweight="bold", y=1.02)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Combined z-score chart
# ---------------------------------------------------------------------------

def generate_zscore_chart(
    model: str,
    zscores: dict[str, dict],
    significance: dict,
    save_path: str | Path,
):
    """Generate a combined z-score heatmap: defendants x (crimes + overall).

    Shows how much more/less harshly each defendant is sentenced relative to
    the cross-defendant average for each crime. The "OVERALL" column is the
    mean z-score across all crimes. Uses a diverging colormap:
    blue = more lenient, red = harsher.
    """
    short = model.split("/")[-1]
    n_defendants = len(DEFENDANTS)
    n_crimes = len(CRIMES)

    # Build the z-score matrix: defendants x (crimes + overall)
    # Use jail z-scores as the primary metric
    col_labels = [c["short_label"] for c in CRIMES] + ["OVERALL"]
    n_cols = n_crimes + 1

    data = np.full((n_defendants, n_cols), np.nan)
    for d_idx, defendant in enumerate(DEFENDANTS):
        dz = zscores[defendant["id"]]
        for c_idx, crime in enumerate(CRIMES):
            z = dz["per_crime_jail_z"].get(crime["id"])
            if z is not None:
                data[d_idx, c_idx] = z
        data[d_idx, n_crimes] = dz["combined_z"]

    # Sort defendants by overall z-score (harshest at top)
    overall_col = data[:, n_crimes]
    sort_order = np.argsort(overall_col)[::-1]
    data = data[sort_order]
    sorted_defendants = [DEFENDANTS[i] for i in sort_order]
    defendant_labels = [d["name"] for d in sorted_defendants]

    # Symmetric color range
    abs_max = np.nanmax(np.abs(data)) if not np.all(np.isnan(data)) else 1
    abs_max = max(abs_max, 0.1)  # avoid degenerate range

    fig, ax = plt.subplots(figsize=(16, 8))

    im = ax.imshow(data, cmap=ZSCORE_CMAP, aspect="auto", vmin=-abs_max, vmax=abs_max)

    for d_idx in range(n_defendants):
        for c_idx in range(n_cols):
            val = data[d_idx, c_idx]
            if np.isnan(val):
                text = "n/a"
            else:
                text = f"{val:+.2f}"
            brightness = abs(val) / abs_max if not np.isnan(val) else 0
            color = "white" if brightness > 0.6 else "black"
            fontweight = "bold" if c_idx == n_crimes else "normal"
            ax.text(c_idx, d_idx, text, ha="center", va="center",
                    fontsize=11, fontweight=fontweight, color=color)

    ax.set_yticks(range(n_defendants))
    ax.set_yticklabels(defendant_labels, fontsize=12)

    # X-axis: crime labels + KW significance
    kw_labels = []
    for crime in CRIMES:
        kw = significance["per_crime_kw"].get(crime["id"], {})
        p = kw.get("jail_p", 1.0)
        s = _stars(p)
        kw_labels.append(f"KW {s} p={p:.3f}" if s else "n.s.")

    # Overall KW
    overall_p = significance["overall_kw"].get("jail_p", 1.0)
    overall_s = _stars(overall_p)
    kw_labels.append(f"KW {overall_s} p={overall_p:.3f}" if overall_s else "n.s.")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [f"{cl}\n{kw}" for cl, kw in zip(col_labels, kw_labels)],
        fontsize=9, ha="center",
    )

    # Vertical separator before OVERALL column
    ax.axvline(n_crimes - 0.5, color="black", linewidth=2)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Z-score (+ = harsher, - = more lenient)", fontsize=11)

    # Add pairwise significance annotations on the right
    pairwise = significance["pairwise"]
    sig_pairs = []
    for (a, b), p in pairwise.items():
        jail_p = p.get("jail_p", 1.0)
        if jail_p < 0.05:
            sig_pairs.append((a, b, jail_p))
    sig_pairs.sort(key=lambda x: x[2])

    if sig_pairs:
        # Build text block of significant pairs
        id_to_name = {d["id"]: d["name"].split()[0] for d in DEFENDANTS}
        sig_text = "Significant pairwise\ndifferences (jail z):\n"
        for a, b, p in sig_pairs[:10]:
            za = zscores[a]["combined_z"]
            zb = zscores[b]["combined_z"]
            harsher, lenient = (a, b) if za > zb else (b, a)
            sig_text += f"\n{_stars(p)} {id_to_name[harsher]} > {id_to_name[lenient]}  p={p:.4f}"

        fig.text(1.02, 0.5, sig_text, transform=ax.transAxes,
                 fontsize=8, verticalalignment="center", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_title(
        f"Sentencing Bias Z-Scores: {short}\n"
        "(normalized within each crime, + = harsher than average)",
        fontsize=14, fontweight="bold", pad=15,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-model combined z-score chart
# ---------------------------------------------------------------------------

def generate_cross_model_zscore_chart(
    all_zscores: dict[str, dict[str, dict]],
    all_significance: dict[str, dict],
    save_path: str | Path,
):
    """Generate a single chart: defendants (rows) x models (cols), showing
    each defendant's overall combined z-score across all crimes per model.
    Gives a snapshot of how much each model penalizes each name.
    """
    models = list(all_zscores.keys())
    short_names = [m.split("/")[-1] for m in models]
    n_models = len(models)
    n_defendants = len(DEFENDANTS)

    data = np.full((n_defendants, n_models), np.nan)
    for d_idx, defendant in enumerate(DEFENDANTS):
        for m_idx, model in enumerate(models):
            dz = all_zscores[model].get(defendant["id"], {})
            data[d_idx, m_idx] = dz.get("combined_z", np.nan)

    # Sort by mean z-score across models (harshest at top)
    row_means = np.nanmean(data, axis=1)
    sort_order = np.argsort(row_means)[::-1]
    data = data[sort_order]
    sorted_defendants = [DEFENDANTS[i] for i in sort_order]
    defendant_labels = [d["name"] for d in sorted_defendants]

    abs_max = np.nanmax(np.abs(data)) if not np.all(np.isnan(data)) else 1
    abs_max = max(abs_max, 0.1)

    fig, ax = plt.subplots(figsize=(max(10, n_models * 2.5), 8))

    im = ax.imshow(data, cmap=ZSCORE_CMAP, aspect="auto", vmin=-abs_max, vmax=abs_max)

    for d_idx in range(n_defendants):
        for m_idx in range(n_models):
            val = data[d_idx, m_idx]
            if np.isnan(val):
                text = "n/a"
            else:
                text = f"{val:+.2f}"
            brightness = abs(val) / abs_max if not np.isnan(val) else 0
            color = "white" if brightness > 0.6 else "black"
            ax.text(m_idx, d_idx, text, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    # X-axis: model names + overall KW p-value
    x_labels = []
    for model in models:
        sig = all_significance.get(model, {})
        kw = sig.get("overall_kw", {})
        p = kw.get("jail_p", 1.0)
        s = _stars(p)
        kw_text = f"KW {s} p={p:.3f}" if s else "n.s."
        x_labels.append(f"{model.split('/')[-1]}\n{kw_text}")

    ax.set_xticks(range(n_models))
    ax.set_xticklabels(x_labels, fontsize=9, ha="center")
    ax.set_yticks(range(n_defendants))
    ax.set_yticklabels(defendant_labels, fontsize=12)

    ax.set_title(
        "Overall Sentencing Bias by Name (All Models)\n"
        "Combined z-score across all crimes (+ = harsher, - = more lenient)",
        fontsize=14, fontweight="bold", pad=15,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Combined z-score", fontsize=11)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-model heatmap (per crime)
# ---------------------------------------------------------------------------

def generate_cross_model_heatmap(
    all_scores: dict[str, dict[str, dict[str, dict]]],
    crime_id: str,
    crime_label: str,
    save_path: str | Path,
):
    """Generate a heatmap for a single crime: defendants (rows) x models (cols)."""
    models = list(all_scores.keys())
    short_names = [m.split("/")[-1] for m in models]
    n_models = len(models)
    n_defendants = len(DEFENDANTS)

    data_jail = np.full((n_defendants, n_models), np.nan)

    for d_idx, defendant in enumerate(DEFENDANTS):
        for m_idx, model in enumerate(models):
            s = all_scores[model].get(defendant["id"], {}).get(crime_id, {})
            jail = s.get("avg_jail")
            if jail is not None:
                data_jail[d_idx, m_idx] = jail

    fig, ax = plt.subplots(figsize=(max(10, n_models * 2), 8))

    jail_min = np.nanmin(data_jail) if not np.all(np.isnan(data_jail)) else 0
    jail_max = np.nanmax(data_jail) if not np.all(np.isnan(data_jail)) else 1
    im = ax.imshow(data_jail, cmap=JAIL_CMAP, aspect="auto", vmin=jail_min, vmax=jail_max)

    defendant_labels = [d["name"] for d in DEFENDANTS]

    jail_range = jail_max - jail_min if jail_max > jail_min else 1
    for d_idx in range(n_defendants):
        for m_idx in range(n_models):
            val = data_jail[d_idx, m_idx]
            text = _fmt_jail(val if not np.isnan(val) else None)
            brightness = (val - jail_min) / jail_range if not np.isnan(val) else 0
            color = "white" if brightness > 0.6 else "black"
            ax.text(m_idx, d_idx, text, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    ax.set_xticks(range(n_models))
    ax.set_xticklabels(short_names, fontsize=10, rotation=25, ha="right")
    ax.set_yticks(range(n_defendants))
    ax.set_yticklabels(defendant_labels, fontsize=12)
    ax.set_title(f"Cross-Model Jail Time: {crime_label}",
                 fontsize=14, fontweight="bold", pad=15)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Avg jail (years)", fontsize=11)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
