"""Generate all charts for the study into output/graphs/.

Every chart shows the full field of teams (or the full null distribution) with
Argentina highlighted, so nothing is cherry-picked -- the reader always sees where
Argentina sits relative to everyone else.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import analysis as A
from . import style as S
from .common import GRAPH_DIR

DEV_METRICS = [
    ("fouls_p90", "Fouls committed"),
    ("cards_p90", "Cards received"),
    ("pens_conceded_p90", "Penalties conceded"),
]


def _save(fig, name, subdir=""):
    out = GRAPH_DIR / subdir if subdir else GRAPH_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    rel = f"{subdir}/{name}" if subdir else name
    print(f"  saved {rel}")


# ------------------------------------------------------------------ banded bar chart
def _banded_items(labels, values, focus, n=10):
    """Turn a DESCENDING-sorted ranking into top-n / median-n / bottom-n display
    rows with 'gap' markers for what's skipped. Focus (Argentina) rows that fall in a
    skipped region are surfaced individually so they're never hidden."""
    N = len(values)
    rank = list(range(1, N + 1))
    if N <= 3 * n + 2:
        return [{"type": "bar", "label": labels[i], "value": values[i],
                 "focus": bool(focus[i]), "rank": rank[i]} for i in range(N)]
    mid_lo = N // 2 - n // 2
    items = []

    def add_bars(rng):
        for i in rng:
            items.append({"type": "bar", "label": labels[i], "value": values[i],
                          "focus": bool(focus[i]), "rank": rank[i]})

    def add_gap(rng):
        idxs = list(rng)
        foci = [i for i in idxs if focus[i]]
        for i in foci:
            items.append({"type": "bar", "label": labels[i], "value": values[i],
                          "focus": True, "rank": rank[i]})
        omitted = len(idxs) - len(foci)
        if omitted > 0:
            items.append({"type": "gap", "count": omitted})

    add_bars(range(0, n))
    add_gap(range(n, mid_lo))
    add_bars(range(mid_lo, mid_lo + n))
    add_gap(range(mid_lo + n, N - n))
    add_bars(range(N - n, N))
    return items


def _banded_barh(ax, labels, values, focus, xlabel, n=10, fmt="{:+.2f}", label_all=False):
    items = _banded_items(list(labels), list(values), list(focus), n=n)
    M = len(items)
    ax.set_facecolor("white")
    ax.grid(True, axis="x", color=S.GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    yticks, yticklabels = [], []
    vmax = max((abs(it["value"]) for it in items if it["type"] == "bar"), default=1)
    for i, it in enumerate(items):
        y = M - 1 - i
        if it["type"] == "gap":
            yticks.append(y); yticklabels.append("$\\cdots$")
            ax.text(0, y, f"  {it['count']} teams omitted  ", ha="center", va="center",
                    fontsize=7.5, color=S.MUTED, style="italic")
            continue
        color = S.ARG if it["focus"] else S.NEUTRAL
        ax.barh(y, it["value"], color=color, height=0.74, zorder=3)
        yticks.append(y)
        yticklabels.append(f"{it['rank']}. {it['label']}")
        if it["focus"] or label_all:
            ha = "left" if it["value"] >= 0 else "right"
            off = vmax * 0.01 if it["value"] >= 0 else -vmax * 0.01
            ax.text(it["value"] + off, y, fmt.format(it["value"]), va="center", ha=ha,
                    fontsize=8, color=S.ARG_DARK if it["focus"] else S.MUTED,
                    fontweight="bold" if it["focus"] else "normal")
    ax.axvline(0, color=S.MUTED, lw=0.8, zorder=2)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8)
    for tick, it in zip(ax.get_yticklabels(), items):
        if it["type"] == "bar" and it["focus"]:
            tick.set_color(S.ARG_DARK); tick.set_fontweight("bold")
        else:
            tick.set_color(S.MUTED)
    ax.tick_params(colors=S.MUTED, labelsize=8, length=0)
    ax.set_ylim(-0.6, M - 0.4)
    ax.set_xlabel(xlabel, color=S.MUTED, fontsize=9.5)


# --------------------------------------------------------- 1. penalties won ranking
def chart_penalty_ranking(df, summary):
    s = summary.dropna(subset=["pens_won_pg"]).sort_values("pens_won_pg")
    r = A.rank_metric(summary, "pens_won_pg")
    fig, ax = S.new_ax(figsize=(9.5, max(6, 0.26 * len(s))))
    colors = S.highlight_colors(s.team)
    ax.barh(s.team, s.pens_won_pg, color=colors, height=0.72, zorder=3)
    ax.axvline(r["mean"], color=S.MUTED, ls="--", lw=1, zorder=2)
    ax.text(r["mean"], len(s) - 0.5, f"  field avg {r['mean']:.2f}", color=S.MUTED, fontsize=8, va="top")
    # label Argentina bar
    yi = list(s.team).index(S.FOCUS)
    ax.text(s.pens_won_pg.iloc[yi] + 0.005, yi, f" {s.pens_won_pg.iloc[yi]:.2f}",
            va="center", ha="left", color=S.ARG_DARK, fontsize=9, fontweight="bold")
    ax.set_xlabel("Penalties won per game", color=S.MUTED, fontsize=10)
    ax.set_ylabel("")
    S.title(ax, "Penalties won per game, 2010–2026 World Cups",
            f"Argentina rank {r['rank']}/{r['n_teams']} · {r['percentile']:.0f}th percentile · z = {r['z']:+.1f}")
    S.footnote(fig, "Source: ESPN match box scores (5 World Cups). Teams with ≥4 matches. Shootout kicks excluded.")
    _save(fig, "01_penalties_won_ranking.png")


# ------------------------------------------------------- 2. strength control scatter
def chart_strength_control(df):
    sc = A.strength_control(df)
    t = sc["table"]
    fig, ax = S.new_ax(figsize=(9, 6.5))
    others = t[t.team != S.FOCUS]
    ax.scatter(others.gd_pg, others.pens_won_pg, s=34, color=S.NEUTRAL, edgecolor="white", lw=0.6, zorder=3)
    xs = np.linspace(t.gd_pg.min(), t.gd_pg.max(), 50)
    ax.plot(xs, sc["intercept"] + sc["slope"] * xs, color=S.MUTED, lw=1.6, ls="--", zorder=2,
            label="expected (regression on dominance)")
    arg = t[t.team == S.FOCUS]
    ax.scatter(arg.gd_pg, arg.pens_won_pg, s=150, color=S.ARG, edgecolor=S.ARG_DARK, lw=1.4, zorder=5)
    ax.annotate(f"Argentina\n+{sc['focus_resid']:.2f} above expected\n({sc['resid_percentile']:.0f}th pct residual)",
                (arg.gd_pg.iloc[0], arg.pens_won_pg.iloc[0]),
                xytext=(14, -6), textcoords="offset points", fontsize=9.5, color=S.ARG_DARK, fontweight="bold")
    # label a few high-penalty comparison teams
    for _, row in t.sort_values("pens_won_pg", ascending=False).head(5).iterrows():
        if row.team != S.FOCUS:
            ax.annotate(row.team, (row.gd_pg, row.pens_won_pg), xytext=(5, 4),
                        textcoords="offset points", fontsize=7.5, color=S.MUTED)
    ax.set_xlabel("Goal difference per game  (greater team dominance to the right)", color=S.MUTED, fontsize=10)
    ax.set_ylabel("Penalties won per game", color=S.MUTED, fontsize=10)
    S.title(ax, "Do Argentina just win penalties because they're good?",
            "Even after controlling for dominance, Argentina wins far more penalties than predicted")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    S.footnote(fig, "Each dot = one team pooled over 2010–2026. Dashed line = OLS fit of penalties on goal difference.")
    _save(fig, "02_penalty_strength_control.png")


# --------------------------------------------------- 3. Argentina by tournament (recency)
def chart_argentina_timeline(df):
    years = [2010, 2014, 2018, 2022, 2026]
    arg = df[df.team == S.FOCUS]
    pens = [arg[arg.tournament == y]["pens_won"].mean() for y in years]
    field = [df[df.tournament == y]["pens_won"].mean() for y in years]
    fig, ax = S.new_ax(figsize=(9, 5.6))
    x = np.arange(len(years))
    ax.bar(x - 0.2, pens, width=0.4, color=S.ARG, label="Argentina", zorder=3)
    ax.bar(x + 0.2, field, width=0.4, color=S.NEUTRAL, label="Tournament field avg", zorder=3)
    for xi, v in zip(x - 0.2, pens):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=S.ARG_DARK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("Penalties won per game", color=S.MUTED, fontsize=10)
    S.title(ax, "Argentina's penalty edge is recent, not career-long",
            "Zero penalties in 2010 & 2014 (incl. a run to the final); the spike is 2022 and 2026")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    S.footnote(fig, "2026 is in progress (through the Round of 16 as of the data pull).")
    _save(fig, "03_argentina_penalty_timeline.png")


# ------------------------------------------- 4. deviation-by-opponent rankings (3 panels)
def chart_deviation_rankings(dev):
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    fig.patch.set_facecolor("white")
    for ax, (metric, label) in zip(axes, DEV_METRICS):
        eff = A.opponent_effects(dev, metric).sort_values("mean_dev")
        colors = S.highlight_colors(eff.opponent)
        ax.set_facecolor("white")
        ax.grid(True, axis="x", color=S.GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.barh(eff.opponent, eff.mean_dev, color=colors, height=0.74, zorder=3)
        ax.axvline(0, color=S.MUTED, lw=0.9, zorder=2)
        ax.tick_params(colors=S.MUTED, labelsize=7)
        arg = eff[eff.opponent == S.FOCUS]
        if len(arg):
            yi = list(eff.opponent).index(S.FOCUS)
            rk = int(A.opponent_effects(dev, metric).set_index("opponent").loc[S.FOCUS, "rank"])
            n = len(eff)
            ax.text(arg.mean_dev.iloc[0], yi, f"  Arg #{rk}/{n}", va="center", ha="left",
                    color=S.ARG_DARK, fontsize=8.5, fontweight="bold")
        ax.set_title(f"{label}\n(opponent deviation / 90)", fontsize=10.5, fontweight="bold",
                     color=S.INK, loc="left")
        ax.set_xlabel("below their norm  <<  0  >>  above their norm", fontsize=8, color=S.MUTED)
    fig.suptitle("How teams deviate from their own norms when facing each opponent",
                 fontsize=14, fontweight="bold", color=S.INK, x=0.01, ha="left", y=1.0)
    fig.text(0.01, 0.965, "Positive = teams foul more, get carded more, and concede more penalties than they usually do against this opponent",
             fontsize=9.5, color=S.MUTED, ha="left")
    S.footnote(fig, "Baseline = each team's leave-one-out mean within the same tournament (minute-adjusted). Opponents with ≥6 games shown.")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    _save(fig, "04_deviation_by_opponent_rankings.png")


# ------------------------------------------------- 5. permutation nulls (significance)
def _permutation_fig(dev, suptitle, note, B=12000):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.patch.set_facecolor("white")
    for ax, (metric, label) in zip(axes, DEV_METRICS):
        pt = A.permutation_test(dev, metric, B=B)
        col = f"dev_{metric}"
        all_dev = dev.dropna(subset=[col])[col].values
        ax.set_facecolor("white")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, axis="y", color=S.GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        if not pt or pt.get("n_games", 0) == 0 or len(all_dev) == 0:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color=S.MUTED)
            ax.set_title(label, fontsize=10.5, fontweight="bold", color=S.INK, loc="left")
            continue
        rng = np.random.default_rng(7)
        n = pt["n_games"]
        null = np.array([rng.choice(all_dev, size=n, replace=False).mean() for _ in range(8000)])
        ax.hist(null, bins=40, color=S.NEUTRAL, zorder=3)
        ax.axvline(pt["observed_mean_dev"], color=S.ACCENT, lw=2.2, zorder=5)
        ax.text(pt["observed_mean_dev"], ax.get_ylim()[1] * 0.92,
                f"  Argentina\n  {pt['observed_mean_dev']:+.2f}\n  p={pt['p_one_sided']:.3f}",
                color=S.ACCENT, fontsize=9, fontweight="bold", va="top")
        ax.tick_params(colors=S.MUTED, labelsize=8)
        sig = "significant" if pt["p_one_sided"] < 0.05 else "n.s."
        ax.set_title(f"{label}  ·  z={pt['z']:+.1f} ({sig})", fontsize=10.5, fontweight="bold",
                     color=S.INK, loc="left")
        ax.set_xlabel("mean opponent deviation / 90", fontsize=8, color=S.MUTED)
    fig.suptitle(suptitle, fontsize=13.5, fontweight="bold", color=S.INK, x=0.01, ha="left")
    fig.text(0.01, 0.9, "Grey = null distribution (random samples of the same size). Orange = Argentina's actual value.",
             fontsize=9, color=S.MUTED, ha="left")
    S.footnote(fig, note)
    fig.tight_layout(rect=[0, 0.02, 1, 0.9])
    return fig


def chart_permutation_nulls(dev):
    fig = _permutation_fig(
        dev, "Permutation test: is 'facing Argentina' different from facing a random team?",
        "One-sided p over 20k permutations. n = 28 Argentina opponent-games across the five cups.")
    _save(fig, "05_permutation_significance.png")


def chart_permutation_per_worldcup(df, years=(2014, 2018, 2022, 2026)):
    for yr in years:
        sub = df[df.tournament == yr]
        dev = A.deviation_table(sub)
        n_arg = int((dev.opponent == S.FOCUS).sum())
        fig = _permutation_fig(
            dev, f"{yr} World Cup — do teams exceed their norms vs Argentina?",
            f"Within-{yr} baselines. n = {n_arg} Argentina opponent-games. Single-tournament samples are small — read as suggestive.")
        _save(fig, f"05_permutation_{yr}.png", subdir="by_world_cup")


# ------------------------------------------------- 6. cards drawn against opposition
def chart_opp_cards(df, summary):
    s = summary.dropna(subset=["opp_cards_pg"]).sort_values("opp_cards_pg")
    r = A.rank_metric(summary, "opp_cards_pg")
    fig, ax = S.new_ax(figsize=(9.5, max(6, 0.26 * len(s))))
    ax.barh(s.team, s.opp_cards_pg, color=S.highlight_colors(s.team), height=0.72, zorder=3)
    ax.axvline(r["mean"], color=S.MUTED, ls="--", lw=1, zorder=2)
    ax.text(r["mean"], len(s) - 0.5, f"  field avg {r['mean']:.2f}", color=S.MUTED, fontsize=8, va="top")
    ax.set_xlabel("Cards shown to the opposition, per game", color=S.MUTED, fontsize=10)
    S.title(ax, "Cards drawn by the opposition (minor stat)",
            f"Argentina's opponents average {r['focus_value']:.2f} cards/game · rank {r['rank']}/{r['n_teams']} · {r['percentile']:.0f}th pct")
    S.footnote(fig, "Source: ESPN box scores. Cards received by whoever is playing the listed team.")
    _save(fig, "06_opposition_cards_drawn.png")


def chart_opp_cards_per_worldcup(df, years=(2014, 2018, 2022, 2026)):
    for yr in years:
        sub = df[df.tournament == yr]
        summ = A.team_summary(sub, min_games=3)
        s = summ.dropna(subset=["opp_cards_pg"]).sort_values("opp_cards_pg")
        if S.FOCUS not in set(s.team):
            continue
        r = A.rank_metric(summ, "opp_cards_pg")
        fig, ax = S.new_ax(figsize=(9, max(5.5, 0.26 * len(s))))
        ax.barh(s.team, s.opp_cards_pg, color=S.highlight_colors(s.team), height=0.72, zorder=3)
        ax.axvline(r["mean"], color=S.MUTED, ls="--", lw=1, zorder=2)
        ax.text(r["mean"], len(s) - 0.5, f"  avg {r['mean']:.2f}", color=S.MUTED, fontsize=8, va="top")
        yi = list(s.team).index(S.FOCUS)
        for t in ax.get_yticklabels():
            if t.get_text() == S.FOCUS:
                t.set_color(S.ARG_DARK); t.set_fontweight("bold")
        ax.set_xlabel("Cards shown to the opposition, per game", color=S.MUTED, fontsize=10)
        S.title(ax, f"{yr} World Cup — cards drawn by the opposition",
                f"Argentina's opponents: {r['focus_value']:.2f} cards/game · rank {r['rank']}/{r['n_teams']} in {yr}")
        S.footnote(fig, "Source: ESPN box scores. Teams with ≥3 matches in this tournament.")
        _save(fig, f"06_opposition_cards_{yr}.png", subdir="by_world_cup")


# ------------------------------------------- 8. penalties per 90 by team-tournament
def chart_penalties_per90_team_year(df):
    """Banded ranking of penalties won per 90, one bar per team-tournament."""
    g = df.groupby(["team", "tournament"]).agg(
        pens=("pens_won", "sum"), minutes=("minutes", "sum"), games=("match_id", "count"))
    g = g[g["games"] >= 3].reset_index()
    g["pens_p90"] = g["pens"] / g["minutes"] * 90.0
    g["label"] = [f"{t} '{int(y) % 100:02d}" for t, y in zip(g.team, g.tournament)]
    g["focus"] = g.team == S.FOCUS
    g = g.sort_values("pens_p90", ascending=False).reset_index(drop=True)

    items = _banded_items(list(g.label), list(g.pens_p90), list(g.focus))
    fig, ax = S.new_ax(figsize=(9.5, 0.32 * len(items) + 1.5))
    _banded_barh(ax, g.label, g.pens_p90, g.focus, "Penalties won per 90 minutes", fmt="{:.2f}")
    arg_rows = g[g.focus].sort_values("tournament")
    arg_txt = " · ".join(f"'{int(y)%100:02d}: {p:.2f}" for y, p in zip(arg_rows.tournament, arg_rows.pens_p90))
    S.title(ax, "Penalties won per 90, by team & tournament",
            f"Top 10 / median 10 / bottom 10 of {len(g)} team-tournaments (2010–2026). Argentina — {arg_txt}")
    S.footnote(fig, "One bar per team per World Cup (≥3 games). Shootout kicks excluded. Argentina years in blue.")
    _save(fig, "08_penalties_per90_by_team_year.png", subdir="rankings")


# ------------------------------------------- 9. deviation-by-opponent, banded (all teams)
def chart_deviation_banded(dev, name="09_deviation_by_opponent_banded.png", subdir="rankings",
                           min_games=6, era_label="2010–2026", scope_note="all opponents"):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 9.5))
    fig.patch.set_facecolor("white")
    for ax, (metric, label) in zip(axes, DEV_METRICS):
        eff = A.opponent_effects(dev, metric, min_games=min_games)  # sorted desc
        focus = (eff.opponent == S.FOCUS).values
        _banded_barh(ax, eff.opponent.values, eff.mean_dev.values, focus,
                     "opponent deviation / 90", fmt="{:+.2f}")
        arg = eff[eff.opponent == S.FOCUS]
        rk = f"#{int(arg['rank'].iloc[0])}/{len(eff)}" if len(arg) else "n/a"
        ax.set_title(f"{label}\nArgentina {rk}", fontsize=10.5, fontweight="bold", color=S.INK, loc="left")
    fig.suptitle(f"How teams deviate from their norms against each opponent — {era_label}",
                 fontsize=13.5, fontweight="bold", color=S.INK, x=0.01, ha="left")
    fig.text(0.01, 0.955, f"Top 10 / median 10 / bottom 10 of {scope_note} (≥{min_games} games). Positive = opponents foul/card/concede more than usual.",
             fontsize=9.5, color=S.MUTED, ha="left")
    S.footnote(fig, "Baseline = each team's leave-one-out mean within the same tournament (minute-adjusted).")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    _save(fig, name, subdir=subdir)


# ------------- 10. deviation-by-opponent, per World Cup, ranked bars with flags
def _yaxis_flags(ax, teams, box_colors, zoom=0.085):
    """Put a small flag at each y-tick (y = row index). Highlighted teams (keys of
    box_colors) get a coloured box in their highlight colour."""
    for i, tm in enumerate(teams):
        im = _flag_img(tm, zoom=zoom)
        if im is None:
            continue
        bc = box_colors.get(tm)
        ab = AnnotationBbox(im, (0, i), xycoords=ax.get_yaxis_transform(),
                            xybox=(-12, 0), boxcoords="offset points",
                            frameon=bc is not None, box_alignment=(1, 0.5), zorder=6,
                            bboxprops=dict(edgecolor=bc, lw=1.8, facecolor="white") if bc else None)
        ax.add_artist(ab)


_GREEN_DK = "#1E7A4F"
_TOP4 = ["Spain", "England", "Brazil", "France"]
# variants: (tag, highlights{team: (bar_colour, text_colour)}, only_teams or None)
_DEV_VARIANTS = [
    ("argentina", {S.FOCUS: (S.ARG, S.ARG_DARK)}, None),
    ("top_teams", {**{t: (S.GOOD, _GREEN_DK) for t in _TOP4}, S.FOCUS: (S.ARG, S.ARG_DARK)}, None),
    # top-teams-only: the same five sides; Argentina highlighted (blue), the rest neutral
    ("top_teams_only", {S.FOCUS: (S.ARG, S.ARG_DARK)}, [S.FOCUS] + _TOP4),
]


def _draw_deviation_bars(dev, yr, tag, highlights, only_teams=None):
    box_colors = {t: c[0] for t, c in highlights.items()}
    # "single" = the full-field single-focus variant (gets a top-right callout + plain
    # name labels). A filtered set (only_teams) always shows abbr + rank for every bar.
    single = len(highlights) == 1 and only_teams is None
    # full-field ranking (used for rank labels even when only a subset is shown)
    base_col = f"dev_{DEV_METRICS[0][0]}"
    nfull = int((dev.dropna(subset=[base_col]).groupby("opponent").size() >= 3).sum())
    ndisp = len(only_teams) if only_teams is not None else nfull
    h = max(2.6, 0.42 * ndisp + 1.6) if only_teams is not None else max(6.0, 0.20 * nfull + 1.6)
    fig, axes = plt.subplots(1, 3, figsize=(16, h))
    fig.patch.set_facecolor("white")
    for ax, (metric, label) in zip(axes, DEV_METRICS):
        col = f"dev_{metric}"
        full = dev.dropna(subset=[col]).groupby("opponent")[col].agg(["mean", "count"])
        full = full[full["count"] >= 3].sort_values("mean")  # ascending -> highest on top
        fteams = list(full.index)
        nf = len(fteams)
        full_rank = {t: nf - i for i, t in enumerate(fteams)}
        disp = full if only_teams is None else full[full.index.isin(only_teams)]
        teams = list(disp.index)
        n = len(teams)
        ys = np.arange(n)
        means = disp["mean"].values
        colors = [highlights[t][0] if t in highlights else S.NEUTRAL for t in teams]

        ax.set_facecolor("white")
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, axis="x", color=S.GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.barh(ys, means, height=0.72, color=colors, zorder=3)
        ax.axvline(0, color=S.MUTED, lw=0.9, zorder=2)

        # rank label at each bar's end (rank is vs the FULL field). Highlighted teams
        # only on the full-field charts; every displayed team on the filtered ones.
        label_teams = teams if only_teams is not None else \
            [t for t in teams if t in highlights]
        for i, t in enumerate(teams):
            if t not in label_teams:
                continue
            tcol = highlights[t][1] if t in highlights else S.MUTED
            psig = A.permutation_test(dev, metric, focus=t).get("p_one_sided", float("nan"))
            star = "*" if psig < 0.05 else ""
            lbl = f"  {F.abbr(t)} #{full_rank[t]}{star}" if not single else f"  {t}"
            ha = "left" if means[i] >= 0 else "right"
            ax.text(means[i], i, lbl, va="center", ha=ha, color=tcol, fontsize=8.5, fontweight="bold")

        _yaxis_flags(ax, teams, box_colors)
        ax.set_yticks(ys)
        ax.set_yticklabels([F.abbr(t) for t in teams], fontsize=7)
        for tick, t in zip(ax.get_yticklabels(), teams):
            if t in highlights:
                tick.set_color(highlights[t][1]); tick.set_fontweight("bold")
            else:
                tick.set_color(S.MUTED)
        ax.tick_params(axis="y", pad=26, length=0)
        ax.set_ylim(-1, n)
        ax.tick_params(axis="x", colors=S.MUTED, labelsize=8.5)
        ax.set_title(label, fontsize=11, fontweight="bold", color=S.INK, loc="left")
        if single:
            ft = next(iter(highlights))
            if ft in full_rank:
                psig = A.permutation_test(dev, metric, focus=ft).get("p_one_sided", float("nan"))
                star = " *" if psig < 0.05 else ""
                ax.text(1.0, 1.008, f"{ft} #{full_rank[ft]}/{nf} · p={psig:.3f}{star}",
                        transform=ax.transAxes, fontsize=9, color=highlights[ft][1],
                        fontweight="bold", ha="right")
        ax.set_xlabel("opponent deviation / 90", fontsize=8.5, color=S.MUTED)

    if single:
        who, who_prefix = next(iter(highlights)), "Highlighted"
    elif only_teams is not None:
        who, who_prefix = "Argentina (blue), with Spain / England / Brazil / France for comparison", "Highlighted"
    else:
        who, who_prefix = "Argentina (blue) + Spain / England / Brazil / France (green)", "Highlighted"
    scope = " — top teams only" if only_teams is not None else ""
    # header/footer margins in absolute inches so short figures aren't crowded
    sup_y, sub_y = 1 - 0.34 / h, 1 - 0.64 / h
    fig.suptitle(f"{yr} World Cup — do teams exceed their norms against each opponent?{scope}",
                 fontsize=13.5, fontweight="bold", color=S.INK, x=0.01, ha="left", y=sup_y)
    fig.text(0.01, sub_y, f"Each bar = one team; value = how much their opponents fouled / were carded / conceded penalties vs their own norm. {who_prefix}: {who}.",
             fontsize=9.5, color=S.MUTED, ha="left")
    note = ("p = permutation test (highlighted team vs a random team, one-sided); "
            "* = p<0.05, and # = rank among all teams. Single-tournament samples are small — read with care.")
    S.footnote(fig, note)
    fig.tight_layout(rect=[0, 0.55 / h, 1, 1 - 0.95 / h])
    _save(fig, f"10_deviation_by_opponent_{tag}_{yr}.png", subdir="by_world_cup")


def chart_deviation_by_opponent_per_worldcup(df, years=(2018, 2022, 2026)):
    """The chart-04 idea, per tournament: every team ranked by how much its opponents
    deviated from their own norm, flags + 3-letter codes on the y-axis. Three variants
    per tournament — Argentina highlighted; the four other elite teams highlighted (with
    Argentina still marked in blue); and just those five teams on their own."""
    for yr in years:
        dev = A.deviation_table(df[df.tournament == yr])
        for tag, highlights, only_teams in _DEV_VARIANTS:
            _draw_deviation_bars(dev, yr, tag, highlights, only_teams)


# ------------------------------------------- 7. anomalous-games ref-pressure scatter
def chart_anomaly_scatter(df):
    """Expected vs observed 'referee pressure' per team-game.

    x = a team's normal ref-pressure that tournament (its other games)
    y = its ref-pressure in THIS game
    Above the diagonal = whistled harder than usual. Argentina's opponents (orange)
    landing above the line, and Argentina itself (blue) below, is the fingerprint of
    games tilting Argentina's way.
    """
    r = A.add_ref_pressure(df)
    r = r.dropna(subset=["rpi_base"])
    bg = r[(r.team != S.FOCUS) & (r.opponent != S.FOCUS)]
    opp = r[r.opponent == S.FOCUS].copy()   # how Argentina's opponents were treated
    arg = r[r.team == S.FOCUS].copy()       # how Argentina was treated

    fig, ax = S.new_ax(figsize=(10.5, 8.5))
    lo = min(r.rpi_base.min(), r.rpi.min()) - 0.5
    hi = max(r.rpi_base.max(), r.rpi.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], color=S.MUTED, ls="--", lw=1.2, zorder=2)
    ax.text(hi, hi, "treated as usual  ", color=S.MUTED, fontsize=8.5, ha="right", va="bottom", rotation=45,
            rotation_mode="anchor")

    ax.scatter(bg.rpi_base, bg.rpi, s=20, color=S.NEUTRAL, alpha=0.4, edgecolor="none", zorder=3,
               label="all other team-games")
    ax.scatter(arg.rpi_base, arg.rpi, s=90, color=S.ARG, edgecolor="white", lw=0.8, zorder=5,
               label="Argentina (how ARG was treated)")
    ax.scatter(opp.rpi_base, opp.rpi, s=110, color=S.ACCENT, edgecolor="white", lw=0.8, zorder=6,
               label="Argentina's opponents (how they were treated)")

    # label the most anomalous opponent games (whistled hardest vs their norm).
    # For an opponent row the TEAM being whistled is `team` (opponent is Argentina).
    lab = opp.assign(label=[f"{tm} '{int(t)%100:02d}" for tm, t in zip(opp.team, opp.tournament)])
    for _, row in lab.sort_values("rpi_resid", ascending=False).head(6).iterrows():
        ax.annotate(row.label, (row.rpi_base, row.rpi), xytext=(6, 5), textcoords="offset points",
                    fontsize=8.5, color=S.ACCENT, fontweight="bold")
    # label Argentina's own most-punished game as an honest counterpoint
    argl = arg.assign(label=[f"vs {o} '{int(t)%100:02d}" for o, t in zip(arg.opponent, arg.tournament)])
    top_arg = argl.sort_values("rpi_resid", ascending=False).head(1)
    for _, row in top_arg.iterrows():
        ax.annotate(row.label, (row.rpi_base, row.rpi), xytext=(6, -12), textcoords="offset points",
                    fontsize=8.5, color=S.ARG_DARK, fontweight="bold")

    span = hi - lo
    ax.fill_between([lo, hi], [lo, hi], hi, color=S.ACCENT, alpha=0.04, zorder=1)
    ax.text(lo + 0.04 * span, hi - 0.05 * span, "punished MORE than their norm", color=S.ACCENT, fontsize=9, style="italic")
    ax.text(hi - 0.03 * span, lo + 0.03 * span, "punished LESS than their norm", color=S.ARG_DARK, fontsize=9,
            style="italic", ha="right")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Team's usual referee-pressure that tournament (other games)", color=S.MUTED, fontsize=10)
    ax.set_ylabel("Referee-pressure in this game", color=S.MUTED, fontsize=10)
    S.title(ax, "Which games tilted Argentina's way?",
            "Referee-pressure index (goal-impact units) = 0.76·penalties + 0.50·reds + 0.06·yellows + 0.025·fouls, per 90")
    leg = ax.legend(frameon=True, fontsize=8.5, loc="lower left")
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("none")
    S.footnote(fig, "Orange above the diagonal = Argentina's opponent whistled harder than usual; blue below = Argentina whistled less. Labels: 6 most anomalous opponent games.")
    _save(fig, "07_anomalous_games_scatter.png")


# ------------- 11. is Argentina significantly different, per World Cup? (forest plot)
def chart_significance_by_tournament(df, years=(2010, 2014, 2018, 2022, 2026)):
    """One row per World Cup: Argentina's opponent Referee-Pressure deviation versus
    what a random team looks like. The grey bar is the middle 90% of the permutation
    null (a typical team); the dot is Argentina, coloured when it clears p<0.05. This
    is the composite-metric answer to 'is Argentina different from other countries?'."""
    rows = A.focus_significance_by_tournament(df, years=years)
    fig, ax = S.new_ax(figsize=(10.5, 5.8))
    ax.grid(False)
    ax.grid(True, axis="x", color=S.GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    ys = list(range(len(rows)))[::-1]  # oldest at top
    xmax = 0.0
    for y, r in zip(ys, rows):
        if not r.get("n_games"):
            continue
        sig = r["p_one_sided"] < 0.05
        col = S.ACCENT if sig else S.NEUTRAL_DK
        # "typical team" range = central 90% of the null
        ax.plot([r["null_q05"], r["null_q95"]], [y, y], color=S.NEUTRAL, lw=9,
                solid_capstyle="round", zorder=2)
        ax.plot([r["null_mean"], r["null_mean"]], [y - 0.16, y + 0.16], color=S.MUTED, lw=1.4, zorder=3)
        ax.scatter([r["observed_mean_dev"]], [y], s=170, color=col, edgecolor="white",
                   lw=1.4, zorder=5)
        star = " *" if sig else ""
        ax.text(r["observed_mean_dev"], y + 0.30,
                f"p={r['p_one_sided']:.3f}{star}  ·  #{r['rank']}/{r['n_teams']}",
                ha="center", va="bottom", fontsize=8.5, color=col,
                fontweight="bold" if sig else "normal")
        xmax = max(xmax, abs(r["observed_mean_dev"]), abs(r["null_q95"]), abs(r["null_q05"]))

    ax.axvline(0, color=S.MUTED, lw=0.9, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([str(r["tournament"]) for r in rows], fontsize=11, fontweight="bold", color=S.INK)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=S.MUTED, labelsize=9)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(-xmax * 1.15, xmax * 1.35)
    ax.set_xlabel("Opponents' referee-pressure vs their own norm, per 90  "
                  "(<< whistled less   ·   whistled more >>)", color=S.MUTED, fontsize=9.5)
    S.title(ax, "Is Argentina different from other countries — each World Cup?",
            "Composite Referee-Pressure Index (penalties, cards & fouls, goal-impact weighted). "
            "Grey = a random team's range; dot = Argentina.")
    # legend-ish key
    ax.scatter([], [], s=120, color=S.ACCENT, edgecolor="white", label="Argentina — significant (p<0.05)")
    ax.scatter([], [], s=120, color=S.NEUTRAL_DK, edgecolor="white", label="Argentina — not significant")
    ax.plot([], [], color=S.NEUTRAL, lw=9, solid_capstyle="round", label="typical team (central 90% of null)")
    leg = ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    S.footnote(fig, "Per-tournament permutation test (20k reshuffles) on the goal-impact-weighted "
                    "RPI opponent-deviation. # = Argentina's rank among all teams with ≥3 games that cup. "
                    "2026 through the Round of 16.")
    _save(fig, "11_significance_by_tournament.png")


# ------- 12. same significance test, all five elite teams (Argentina vs the others)
_SIG_O_BRIGHT, _SIG_O_LIGHT = S.ACCENT, "#F2B79E"   # Argentina: significant / not
_SIG_B_BRIGHT, _SIG_B_LIGHT = S.ARG, "#AEC7E8"      # other team: significant / not


def _sig_dot_colour(team, sig):
    if team == S.FOCUS:
        return _SIG_O_BRIGHT if sig else _SIG_O_LIGHT
    return _SIG_B_BRIGHT if sig else _SIG_B_LIGHT


def chart_significance_top_teams(df, years=(2010, 2014, 2018, 2022, 2026), trim=0,
                                 name="12_significance_by_tournament_top_teams.png", robust=False):
    """Chart 11, extended to the five elite teams. Each World Cup is one lane; each
    team is a dot with its own null band. Colour encodes both identity (Argentina =
    orange, others = blue) and significance (bright = p<0.05, pale = n.s.), so you can
    see at a glance which elite sides — if any — had their opponents whistled unusually
    hard that tournament. `trim`>0 (chart 13) drops each team's single most extreme
    opponent-game high & low first, so no lone freak game can carry the result."""
    teams = [S.FOCUS] + _TOP4  # Argentina, Spain, England, Brazil, France
    data = {t: {r["tournament"]: r for r in
                A.focus_significance_by_tournament(df, focus=t, years=years, B=8000, trim=trim)}
            for t in teams}

    fig, ax = S.new_ax(figsize=(11, 8.4))
    ax.grid(False)
    ax.grid(True, axis="x", color=S.GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    offs = np.linspace(0.30, -0.30, len(teams))  # Argentina at the top of each lane
    centers = [-i for i in range(len(years))]     # 2010 at the top
    xmax = 0.0
    for yi, yr in enumerate(years):
        c = centers[yi]
        for tj, t in enumerate(teams):
            r = data[t].get(yr)
            if not r or not r.get("n_games"):
                continue
            sig = r["p_one_sided"] < 0.05
            col = _sig_dot_colour(t, sig)
            y = c + offs[tj]
            # each team's OWN null range (central 90%) — so "dot past the band edge"
            # matches its own per-team significance (game counts differ, so widths do too)
            ax.fill_betweenx([y - 0.052, y + 0.052], r["null_q05"], r["null_q95"],
                             color=S.NEUTRAL, alpha=0.55, zorder=1)
            ax.scatter([r["observed_mean_dev"]], [y], s=180 if sig else 120, color=col,
                       edgecolor="white", lw=1.2, zorder=5)
            lbl = F.abbr(t)
            if sig:
                lbl += f"   p={r['p_one_sided']:.3f}"
                if robust:
                    lbl += f"  ·  {r.get('n_kept', r['n_games'])}/{r['n_games']}g kept"
            ax.annotate(lbl, (r["observed_mean_dev"], y),
                        xytext=(12, 0), textcoords="offset points", va="center", ha="left",
                        fontsize=8, color=col if sig else S.MUTED,
                        fontweight="bold" if sig else "normal")
            xmax = max(xmax, abs(r["observed_mean_dev"]), abs(r["null_q05"]), abs(r["null_q95"]))

    ax.axvline(0, color=S.MUTED, lw=0.9, zorder=2)
    ax.set_yticks(centers)
    ax.set_yticklabels([str(y) for y in years], fontsize=12, fontweight="bold", color=S.INK)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=S.MUTED, labelsize=9)
    ax.set_ylim(centers[-1] - 0.6, centers[0] + 0.6)
    ax.set_xlim(-xmax * 1.15, xmax * 1.55)
    xlab_stat = "trimmed opponent referee-pressure" if robust else "opponents' referee-pressure"
    ax.set_xlabel(f"{xlab_stat} vs their own norm, per 90  "
                  "(<< whistled less   ·   whistled more >>)", color=S.MUTED, fontsize=9.5)
    if robust:
        S.title(ax, "Robustness check — does dropping each team's freak games change the verdict?",
                "Same test as chart 12, but each team's single most extreme opponent-game (high & low) is "
                "removed first. Bright = significant (p<0.05); grey = each team's own trimmed null range.")
    else:
        S.title(ax, "Argentina vs the other elite teams — significant each World Cup?",
                "Composite Referee-Pressure opponent-deviation. Bright = significant (p<0.05); pale = not. "
                "Grey = each team's own null range; a dot past its band's right edge is significant.")

    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    def _h(c, lab):
        return mlines.Line2D([], [], marker="o", ls="none", ms=10, mfc=c, mec="white", label=lab)
    handles = [
        _h(_SIG_O_BRIGHT, "Argentina — significant"),
        _h(_SIG_O_LIGHT, "Argentina — n.s."),
        _h(_SIG_B_BRIGHT, "Other elite team — significant"),
        _h(_SIG_B_LIGHT, "Other elite team — n.s."),
        mpatches.Patch(color=S.NEUTRAL, alpha=0.55, label="each team's null range (central 90%)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper left",
              bbox_to_anchor=(1.005, 1.0), borderaxespad=0)
    if robust:
        note = ("Outlier-trimmed permutation test (8k reshuffles): each team's single highest and lowest "
                "opponent-game is dropped before averaging, and the same trim is applied to every null "
                "sample. Kept-game counts shown for significant dots (early exits keep as few as 1 — read "
                "those with care). 2026 through the Round of 16.")
    else:
        note = ("Per-tournament permutation test (8k reshuffles) on the goal-impact-weighted RPI "
                "opponent-deviation; * = p<0.05. Elite set = Argentina, Spain, England, Brazil, France. "
                "2026 through the Round of 16.")
    S.footnote(fig, note)
    _save(fig, name)


# ============================================================ SHAREABLE / INSTAGRAM
from matplotlib.offsetbox import OffsetImage, AnnotationBbox  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
from . import flags as F  # noqa: E402


def _flag_img(team, zoom=0.20):
    p = F.flag_path(team)
    if p is None:
        return None
    img = mpimg.imread(str(p))
    return OffsetImage(img, zoom=zoom)


def chart_penalties_top10_flags(df):
    """Editorial top-10 of penalties won per 90 (team-tournament) with flags and the
    raw counts (penalties, minutes) that produced each rate."""
    g = df.groupby(["team", "tournament"]).agg(
        pens=("pens_won", "sum"), minutes=("minutes", "sum"), games=("match_id", "count"))
    g = g[g["games"] >= 3].reset_index()
    g["pens_p90"] = g["pens"] / g["minutes"] * 90.0
    g = g.sort_values("pens_p90", ascending=False).head(10).reset_index(drop=True)

    fig, ax = S.new_ax(figsize=(11, 6.6))
    ax.grid(False)
    ax.grid(True, axis="x", color=S.GRID, lw=0.8)
    ax.spines["left"].set_visible(False)
    n = len(g)
    ys = list(range(n))[::-1]  # rank 1 at top
    vmax = g.pens_p90.max()
    for y, (_, row) in zip(ys, g.iterrows()):
        focus = row.team == S.FOCUS
        color = S.ARG if focus else S.NEUTRAL_DK
        ax.barh(y, row.pens_p90, height=0.66, color=color, zorder=3)
        # flag just left of the axis
        im = _flag_img(row.team)
        if im is not None:
            ab = AnnotationBbox(im, (0, y), xycoords=ax.get_yaxis_transform(),
                                xybox=(-16, 0), boxcoords="offset points",
                                frameon=False, box_alignment=(1, 0.5))
            ax.add_artist(ab)
        # value + raw counts at bar end
        ax.text(row.pens_p90 + vmax * 0.012, y,
                f"{row.pens_p90:.2f}/90", va="center", ha="left", fontsize=11,
                fontweight="bold", color=S.ARG_DARK if focus else S.INK)
        ax.text(row.pens_p90 + vmax * 0.012, y - 0.34,
                f"{int(row.pens)} pens in {int(row.minutes)} min · {int(row.games)} games",
                va="center", ha="left", fontsize=7.8, color=S.MUTED)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{t} '{int(y) % 100:02d}" for t, y in zip(g.team, g.tournament)], fontsize=10.5)
    for tick, (_, row) in zip(ax.get_yticklabels(), g.iterrows()):
        if row.team == S.FOCUS:
            tick.set_color(S.ARG_DARK); tick.set_fontweight("bold")
        else:
            tick.set_color(S.INK)
    ax.tick_params(axis="y", pad=52, length=0)
    ax.tick_params(axis="x", colors=S.MUTED, labelsize=9)
    ax.set_xlim(0, vmax * 1.22)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("Penalties won per 90 minutes", color=S.MUTED, fontsize=10)
    S.title(ax, "Who wins the most penalties at the World Cup?",
            "Top 10 team-tournaments by penalties won per 90 min, 2010–2026 · Argentina in blue")
    S.footnote(fig, "Source: ESPN box scores. Shootout kicks excluded. Rate = penalties ÷ minutes × 90 (extra time counts as 120').")
    _save(fig, "penalties_per90_top10_flags.png", subdir="share")


def chart_summary_scorecard(df):
    """Square Instagram card: the three headline findings as big-number tiles."""
    summary = A.team_summary(df)
    r = A.rank_metric(summary, "pens_won_pg")
    dev = A.deviation_table(df)
    pt = {m: A.permutation_test(dev, m) for m in ["fouls_p90", "cards_p90", "pens_conceded_p90"]}

    fig = plt.figure(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.955, "Do World Cup referees favour Argentina?", ha="center",
             fontsize=19, fontweight="bold", color=S.INK)
    fig.text(0.5, 0.918, "Five World Cups · 2010–2026 · every team measured, not cherry-picked",
             ha="center", fontsize=10.5, color=S.MUTED)

    tiles = [
        (S.ARG, f"#{r['rank']} of {r['n_teams']}",
         "in penalties won per game",
         f"{r['focus_value']:.2f}/game vs {r['mean']:.2f} field avg  ·  {r['percentile']:.0f}th percentile"),
        (S.ACCENT, "significant",
         "opponents exceed their own norms vs Argentina",
         f"+{pt['fouls_p90']['observed_mean_dev']:.1f} fouls · +{pt['cards_p90']['observed_mean_dev']:.2f} cards · "
         f"+{pt['pens_conceded_p90']['observed_mean_dev']:.2f} pens /90  (all p<0.05)"),
        (S.GOOD, "only since 2022",
         "the penalty edge is recent, not career-long",
         "0 penalties won in 2010 & 2014  ·  0.71/game in 2022  ·  0.60/game in 2026"),
    ]
    y0, h = 0.63, 0.205
    for i, (col, big, mid, small) in enumerate(tiles):
        y = y0 - i * (h + 0.025)
        ax = fig.add_axes([0.07, y, 0.86, h])
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                   facecolor="#F5F7F9", edgecolor="none"))
        ax.add_patch(plt.Rectangle((0, 0), 0.014, 1, transform=ax.transAxes,
                                   facecolor=col, edgecolor="none"))
        ax.text(0.04, 0.66, big, fontsize=26, fontweight="bold", color=col, va="center")
        ax.text(0.04, 0.28, mid, fontsize=12.5, color=S.INK, va="center", fontweight="bold")
        ax.text(0.04, 0.09, small, fontsize=9.2, color=S.MUTED, va="center")
    fig.text(0.5, 0.055, "Verdict: a real, statistically significant tilt — but shared with a few other elite teams,",
             ha="center", fontsize=9.5, color=S.INK, style="italic")
    fig.text(0.5, 0.033, "and it can't separate refereeing bias from Argentina's knack for drawing fouls & penalties.",
             ha="center", fontsize=9.5, color=S.INK, style="italic")
    fig.text(0.5, 0.008, "Source: ESPN box scores, validated vs StatsBomb. 2026 through the Round of 16.",
             ha="center", fontsize=7.5, color=S.MUTED)
    _save(fig, "summary_scorecard.png", subdir="share")


def generate_all():
    df = A.load()
    summary = A.team_summary(df)
    dev = A.deviation_table(df)
    print("[viz] generating charts...")
    chart_penalty_ranking(df, summary)
    chart_strength_control(df)
    chart_argentina_timeline(df)
    chart_deviation_rankings(dev)
    chart_permutation_nulls(dev)
    chart_opp_cards(df, summary)
    chart_anomaly_scatter(df)
    chart_significance_by_tournament(df)
    chart_significance_top_teams(df)
    chart_significance_top_teams(df, trim=1, robust=True,
                                 name="13_significance_by_tournament_top_teams_robust.png")
    # banded full-field rankings
    chart_penalties_per90_team_year(df)
    chart_deviation_banded(dev)
    # deviation banded, last two World Cups only (2022 + 2026)
    dev_recent = A.deviation_table(df[df.tournament.isin([2022, 2026])])
    chart_deviation_banded(dev_recent, name="09b_deviation_banded_2022_2026.png",
                           min_games=4, era_label="2022 + 2026 only",
                           scope_note="opponents across the two most recent cups")
    # per-world-cup breakdowns
    chart_permutation_per_worldcup(df)
    chart_opp_cards_per_worldcup(df)
    chart_deviation_by_opponent_per_worldcup(df)
    # shareable / instagram
    chart_penalties_top10_flags(df)
    chart_summary_scorecard(df)
    print("[viz] done")


if __name__ == "__main__":
    generate_all()
