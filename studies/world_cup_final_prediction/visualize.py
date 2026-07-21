"""Charts for the World Cup final-prediction study.

Five figures:
  backtest_scorecard   - the proof: both models beat the base-rate baseline on the
                         untouched 2026 knockouts (four proper scoring rules).
  title_odds           - headline: each surviving team's chance of winning the cup.
  predicted_final      - the shareable hero card: the modal final + its result odds.
  model_comparison     - honesty: Elo vs Poisson title odds side by side.
  round_reach          - each team's odds of reaching the semis / final / lifting it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import config, style as S

GRAPH_DIR = config.GRAPH_DIR


def _save(fig, name):
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(GRAPH_DIR / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def knockout_scorecard(ko_scores: dict):
    """Out-of-sample metrics over all cups' knockouts (leave-one-cup-out CV)."""
    metrics = [("rps", "Ranked Probability Score", "lower is better"),
               ("log_loss", "Log loss", "lower is better"),
               ("brier", "Brier score", "lower is better"),
               ("accuracy", "Top-pick accuracy", "higher is better")]
    order = ["Elo", "Poisson", "BaseRate"]
    colors = {"Elo": S.ELO, "Poisson": S.POISSON, "BaseRate": S.NEUTRAL}
    n = ko_scores["Elo"]["n"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.patch.set_facecolor("white")
    for ax, (key, label, better) in zip(axes.flat, metrics):
        vals = [ko_scores[m][key] for m in order]
        bars = ax.bar(order, vals, color=[colors[m] for m in order], width=0.62, zorder=3)
        ax.set_facecolor("white")
        ax.grid(True, axis="y", color=S.GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=S.MUTED, labelsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold", color=S.INK, loc="left", pad=22)
        ax.text(0, 1.02, better, transform=ax.transAxes, fontsize=8, color=S.MUTED)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center",
                    va="bottom", fontsize=9, color=S.INK, fontweight="bold")
        ax.margins(y=0.18)
    fig.suptitle(f"Out-of-sample on every knockout game, 2010–2026 (n={n})",
                 fontsize=14, fontweight="bold", color=S.INK, x=0.02, ha="left")
    S.footnote(fig, "Leave-one-cup-out CV: each cup predicted with hyper-parameters tuned only on "
                    "the other cups. Both models beat the base-rate baseline; Elo generalises best.")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    _save(fig, "01_knockout_scorecard.png")


def _match_report(rows: list[dict], filename: str, title: str, sub: str, foot: str):
    """Shared renderer, built to be read in a couple of seconds (Instagram).

    One row per match, two bars (Elo, Poisson). Each bar = the probability the model
    gave the ACTUAL winner; the label is that winner as '84%  <flag>  GER'. Bars are
    GREEN if the model favoured the winner (>50%), RED if it favoured the loser, and
    FADED when the final went to penalties (a coin flip). The left margin shows the
    matchup, actual winner in bold.
    """
    import matplotlib.patches as mpatches
    GREEN, RED = "#2E9E5B", "#D14B3D"
    fig, ax = S.new_ax(figsize=(10.5, 1.5 * len(rows) + 2.0))
    ax.grid(True, axis="x", color=S.GRID, linewidth=0.8, zorder=0)
    for i, r in enumerate(rows):
        home, away, winner = r["home"], r["away"], r["winner"]
        pen = r.get("penalties", False)
        for m_key, off, mtag in (("elo", 0.24, "Elo"), ("poisson", -0.24, "Poisson")):
            m = r[m_key]
            if m is None or winner is None:
                continue
            p = m["p_winner"]                        # prob given the actual winner
            barcol = GREEN if p >= 0.5 else RED
            alpha = 0.45 if pen else 1.0             # penalties: light (coin flip)
            ax.barh(i + off, p, height=0.42, color=barcol, alpha=alpha, zorder=3)
            ax.text(0.015, i + off, mtag, va="center", ha="left", fontsize=8.5,
                    color=(S.INK if pen else "white"), fontweight="bold", zorder=5)
            # Prob given the winner, big, at the bar end: "84%    <flag>    GER".
            ax.text(p + 0.03, i + off, f"{p:.0%}", va="center", ha="left",
                    fontsize=13, color=S.INK, fontweight="bold")
            S.add_flag(ax, winner, p + 0.20, i + off, zoom=0.13)
            ax.text(p + 0.26, i + off, S.abbr(winner), va="center", ha="left",
                    fontsize=12, color=S.INK, fontweight="bold")

        # Left margin: year (big), then the matchup (winner's abbr bold).
        if r.get("label"):
            ax.text(-0.40, i, r["label"], va="center", ha="left", fontsize=14,
                    fontweight="bold", color=S.INK)
        for team, fx, tx in ((home, -0.262, -0.226), (away, -0.128, -0.092)):
            S.add_flag(ax, team, fx, i, zoom=0.12)
            won = (winner == team)
            ax.text(tx, i, S.abbr(team), va="center", ha="left", fontsize=9,
                    fontweight="bold" if won or not winner else "normal",
                    color=S.INK if won or not winner else S.NEUTRAL_DK)
        ax.text(-0.166, i, "v", va="center", ha="center", fontsize=8, color=S.MUTED)

    ax.axvline(0.5, color=S.MUTED, lw=1.0, ls=":", zorder=2)
    ax.text(0.5, len(rows) - 0.32, "coin-flip line", fontsize=7.5, color=S.MUTED,
            ha="center", va="bottom")
    ax.set_yticks([])
    ax.set_xlim(-0.44, 1.36)
    ax.set_xticks([0, 0.5, 1.0])
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}" if x >= 0 else "")
    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.legend(handles=[mpatches.Patch(color=GREEN, label="favoured the winner"),
                       mpatches.Patch(color=RED, label="favoured the loser"),
                       mpatches.Patch(color=GREEN, alpha=0.45, label="penalties (coin flip)")],
              frameon=False, fontsize=9, loc="lower right")
    S.title(ax, title, sub)
    S.footnote(fig, foot)
    _save(fig, filename)


def finals_report(finals: list[dict], best_model: str):
    """How each model would have called past World Cup finals, out of sample."""
    _match_report(
        list(reversed(finals)), "02_finals_report.png",
        "How much did the models back the eventual World Cup winner?",
        "Each bar = the probability a model gave the team that actually won — out-of-sample. "
        "Green = it favoured the winner; red = it favoured the loser; faded = decided on penalties.",
        "Leave-one-cup-out: each final predicted with hyper-parameters tuned only on the other cups. "
        "Left column shows the matchup (winner in bold). 2006 & 2022 were penalty shootouts.")


# --------------------------------------------------------------------------- #
def title_odds(team_table: list[dict], model_name: str):
    rows = [r for r in team_table if r["win_cup"] > 0.001][::-1]  # ascending for barh
    teams = [r["team"] for r in rows]
    vals = [r["win_cup"] for r in rows]
    fig, ax = S.new_ax(figsize=(9, 6))
    colors = [S.ELO_DARK if i == len(rows) - 1 else S.ELO for i in range(len(rows))]
    ax.barh(range(len(rows)), vals, color=colors, height=0.66, zorder=3)
    ax.set_yticks([])
    xmax = max(vals) * 1.18
    ax.set_xlim(-xmax * 0.24, xmax)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    for i, (t, v) in enumerate(zip(teams, vals)):
        ax.text(-xmax * 0.215, i, S.abbr(t), va="center", ha="left",
                fontsize=8.5, color=S.MUTED, fontweight="bold")
        S.add_flag(ax, t, -xmax * 0.10, i, zoom=0.16)
        ax.text(v + xmax * 0.012, i, f"{v:.0%}", va="center", ha="left",
                fontsize=10, color=S.INK, fontweight="bold")
    ax.set_xticks([t / 100 for t in range(0, int(xmax * 100) + 1, 10)])
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    S.title(ax, "Who wins the 2026 World Cup?",
            f"Probability of lifting the trophy — {model_name} model, from games through the Round of 16")
    S.footnote(fig, "Exact bracket propagation. Assumes the standard remaining draw (see config).")
    _save(fig, "02_title_odds.png")


# --------------------------------------------------------------------------- #
def predicted_final(a: str, b: str, md: dict, et_rate: float, scorelines: list[dict],
                    context: str, header: str = "PREDICTED FINAL"):
    """No-tie final. md = {'reg': (ph,pd,pa), 'win_a': .., 'win_b': ..}; a on the left."""
    ph, pd_, pa = md["reg"]
    win_a, win_b = md["win_a"], md["win_b"]
    fig = plt.figure(figsize=(9, 5.4))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.955, header, ha="center", fontsize=12, color=S.MUTED, fontweight="bold")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    S.add_flag(ax, a, 0.28, 0.78, zoom=0.6)
    S.add_flag(ax, b, 0.72, 0.78, zoom=0.6)
    ax.text(0.28, 0.63, a, ha="center", fontsize=14, fontweight="bold", color=S.INK)
    ax.text(0.72, 0.63, b, ha="center", fontsize=14, fontweight="bold", color=S.INK)
    ax.text(0.5, 0.75, "vs", ha="center", fontsize=12, color=S.MUTED)
    ax.text(0.5, 0.575, context, ha="center", fontsize=9.5, color=S.MUTED)

    # Hero bar: two-way "wins the World Cup" (no draw — a knockout has a winner).
    ax.text(0.5, 0.50, "Chance of winning the World Cup", ha="center", fontsize=10,
            color=S.INK, fontweight="bold")
    bx0, bx1, by, bh = 0.12, 0.88, 0.37, 0.085
    x = bx0
    for frac, col, lab in [(win_a, S.ELO, f"{a} {win_a:.0%}"), (win_b, S.POISSON, f"{b} {win_b:.0%}")]:
        w = (bx1 - bx0) * frac
        ax.add_patch(plt.Rectangle((x, by), w, bh, color=col, zorder=3))
        ax.text(x + w / 2, by + bh / 2, lab, ha="center", va="center",
                fontsize=10.5, color="white", fontweight="bold")
        x += w

    # How the tie is resolved (the added machinery), then scorelines.
    ax.text(0.5, by - 0.055,
            f"In 90'+ET: {a} {ph:.0%} · level {pd_:.0%} · {b} {pa:.0%}.  A level game is settled "
            f"~{et_rate:.0%} in extra time (edge to the stronger side), the rest on penalties (≈ coin flip).",
            ha="center", fontsize=8, color=S.MUTED)
    if scorelines:
        parts = [f"{s['home_goals']}–{s['away_goals']} ({s['prob']:.0%})" for s in scorelines[:5]]
        ax.text(0.5, 0.15, "Likeliest scorelines", ha="center", fontsize=9.5,
                color=S.INK, fontweight="bold")
        ax.text(0.5, 0.08, "    ".join(parts), ha="center", fontsize=9, color=S.MUTED)
    _save(fig, "03_predicted_final.png")


def final_model_comparison(a: str, b: str, per_model: dict):
    """Each model's no-tie winner split for the final.
    per_model: {name: {'win_a': .., 'win_b': .., 'reg': (ph,pd,pa)}}."""
    names = list(per_model)
    fig, ax = S.new_ax(figsize=(9, 3.4))
    ax.grid(False)
    for row, name in enumerate(names):
        wa, wb = per_model[name]["win_a"], per_model[name]["win_b"]
        x = 0.0
        for frac, col, lab in [(wa, S.ELO, f"{S.abbr(a)} {wa:.0%}"), (wb, S.POISSON, f"{S.abbr(b)} {wb:.0%}")]:
            ax.barh(row, frac, left=x, height=0.55, color=col, zorder=3)
            ax.text(x + frac / 2, row, lab, ha="center", va="center",
                    fontsize=9.5, color="white", fontweight="bold")
            x += frac
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10, color=S.INK, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_xticks([])
    ax.set_ylim(-0.6, len(names) - 0.4)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    favs = {n: (a if d["win_a"] > d["win_b"] else b) for n, d in per_model.items()}
    if len(set(favs.values())) == 1:
        sub = f"Chance of winning the World Cup (ties resolved). Both models favour {next(iter(favs.values()))}."
    else:
        leans = ", ".join(f"{n} leans {t}" for n, t in favs.items())
        sub = f"Chance of winning the World Cup (ties resolved). The models disagree — {leans}."
    S.title(ax, f"The final: {a} vs {b} — how the two models see it", sub)
    _save(fig, "04_final_model_comparison.png")


def final_both_models(a: str, b: str, per_model: dict, scorelines: list[dict],
                      context: str, header: str = "PREDICTED FINAL"):
    """The clean card of chart 03, but with a winner bar for BOTH models.

    per_model: {'Elo': {'win_a','win_b','reg'}, 'Poisson': {...}}; a on the left.
    """
    fig = plt.figure(figsize=(9, 6.0))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.96, header, ha="center", fontsize=12, color=S.MUTED, fontweight="bold")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Matchup header with flags (same look as chart 03).
    S.add_flag(ax, a, 0.28, 0.82, zoom=0.6)
    S.add_flag(ax, b, 0.72, 0.82, zoom=0.6)
    ax.text(0.28, 0.68, a, ha="center", fontsize=14, fontweight="bold", color=S.INK)
    ax.text(0.72, 0.68, b, ha="center", fontsize=14, fontweight="bold", color=S.INK)
    ax.text(0.5, 0.79, "vs", ha="center", fontsize=12, color=S.MUTED)
    ax.text(0.5, 0.625, context, ha="center", fontsize=9.5, color=S.MUTED)
    ax.text(0.5, 0.565, "Chance of winning the World Cup", ha="center", fontsize=10,
            color=S.INK, fontweight="bold")

    # One two-way winner bar per model, labelled on the left.
    bx0, bx1, bh = 0.24, 0.90, 0.085
    tops = {"Elo": 0.45, "Poisson": 0.31}
    for name, d in per_model.items():
        by = tops.get(name, 0.45)
        ax.text(0.205, by + bh / 2, name, ha="right", va="center", fontsize=10.5,
                fontweight="bold", color=S.INK)
        x = bx0
        for frac, col, lab in [(d["win_a"], S.ELO, f"{S.abbr(a)} {d['win_a']:.0%}"),
                               (d["win_b"], S.POISSON, f"{S.abbr(b)} {d['win_b']:.0%}")]:
            w = (bx1 - bx0) * frac
            ax.add_patch(plt.Rectangle((x, by), w, bh, color=col, zorder=3))
            if frac > 0.14:
                ax.text(x + w / 2, by + bh / 2, lab, ha="center", va="center",
                        fontsize=9.5, color="white", fontweight="bold")
            x += w

    favs = {n: (a if d["win_a"] > d["win_b"] else b) for n, d in per_model.items()}
    if len(set(favs.values())) == 1:
        note = f"Both models favour {next(iter(favs.values()))}."
    else:
        note = "The models disagree — " + ", ".join(f"{n} leans {t}" for n, t in favs.items()) + "."
    ax.text(0.5, 0.20, note, ha="center", fontsize=9.5, color=S.MUTED)
    if scorelines:
        parts = [f"{s['home_goals']}–{s['away_goals']} ({s['prob']:.0%})" for s in scorelines[:5]]
        ax.text(0.5, 0.11, "Likeliest scorelines:   " + "    ".join(parts),
                ha="center", fontsize=8.5, color=S.MUTED)
    _save(fig, "05_final_both_models.png")


# Horizontal placement of a flag within a wide segment. Spain's coat of arms sits
# toward the hoist (left), so left-align it; centred flags (Argentina's sun) stay put.
_FLAG_ANCHOR = {"Spain": "left"}


def _fit_fixed_height(img, seg_w_in: float, seg_h_in: float, anchor: str = "center"):
    """Frame a flag into a segment at FIXED full height (never trims top/bottom).

    Every flag then shows all its horizontal bands at the same vertical scale, so
    the two bars look consistent. A segment narrower than the flag trims the sides;
    a wider segment is filled by replicating the edge columns — this simply extends
    the flag's stripes outward, seamless for horizontally-banded flags (Argentina,
    Spain) and graceful otherwise. `anchor` (left/right/center) sets where the real
    flag sits so an off-centre emblem stays in frame.
    """
    h, w = img.shape[:2]
    target = seg_w_in / seg_h_in          # segment width : height
    native = w / h
    if target <= native:                  # narrower than flag -> trim sides
        new_w = max(1, round(h * target))
        x0 = 0 if anchor == "left" else (w - new_w if anchor == "right"
                                         else (w - new_w) // 2)
        return img[:, x0:x0 + new_w]
    total_w = max(w, round(h * target))   # wider -> pad sides, edge-clamped stripes
    pad = total_w - w
    left = 0 if anchor == "left" else (pad if anchor == "right" else pad // 2)
    return np.pad(img, ((0, 0), (left, pad - left), (0, 0)), mode="edge")


def final_flag_bars(a: str, b: str, per_model: dict, scorelines=None,
                    context: str = "", header: str = ""):
    """Two 'tug-of-war' bars (Elo, Poisson): each team's FLAG fills its win share.

    The split point IS the win probability. Flags are cropped (not stretched) to
    their share so proportions stay true, and the bars are tall enough that a
    narrow share (e.g. Argentina at 33%) trims the flag's sides rather than
    squashing it. per_model: {'Elo':{'win_a','win_b',...}, 'Poisson':{...}};
    team `a` is on the left. `scorelines`/`context`/`header` are accepted but
    unused — the chart is intentionally just the two bars.
    """
    import matplotlib.patches as mpatches
    import matplotlib.patheffects as pe

    FIG_W, FIG_H = 10.0, 5.6
    # Layered: soft drop shadow for depth, thin dark stroke for a crisp edge, then
    # the white fill on top — legible on the pale Argentina stripe and the Spain
    # yellow alike, without the flat "sticker outline" look.
    pct_fx = [pe.SimplePatchShadow(offset=(1.6, -1.6), shadow_rgbFace="#0A0F14",
                                   alpha=0.5, rho=0.5),
              pe.Stroke(linewidth=2.4, foreground="#141B21"),
              pe.Normal()]
    imgs = {t: (plt.imread(str(S.flag_path_hi(t))) if S.flag_path_hi(t) else None) for t in (a, b)}

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_autoscale_on(False)

    bx0, bx1 = 0.13, 0.97
    span = bx1 - bx0
    bh = 0.409                              # ~2.29in tall -> 33% share trims sideways
    tops = {"Elo": 0.53, "Poisson": 0.06}   # Elo on top
    for name, d in per_model.items():
        by = tops.get(name, 0.53)
        wa, wb = d["win_a"], d["win_b"]
        split = bx0 + span * wa

        # Rounded outline doubles as the clip path so the flags get rounded ends.
        bar = mpatches.FancyBboxPatch(
            (bx0, by), span, bh, boxstyle="round,pad=0,rounding_size=0.03",
            mutation_aspect=0.55, facecolor="none", edgecolor=S.INK,
            linewidth=1.4, zorder=6, transform=ax.transData)
        ax.add_patch(bar)
        for team, x0, x1 in ((a, bx0, split), (b, split, bx1)):
            img = imgs[team]
            if img is None or x1 - x0 <= 1e-4:
                continue
            crop = _fit_fixed_height(img, (x1 - x0) * FIG_W, bh * FIG_H,
                                     _FLAG_ANCHOR.get(team, "center"))
            im = ax.imshow(crop, extent=[x0, x1, by, by + bh], aspect="auto",
                           origin="upper", interpolation="lanczos", zorder=3)
            im.set_clip_path(bar)
        # Crisp divider at the probability boundary.
        ax.add_patch(mpatches.Rectangle((split - 0.004, by), 0.008, bh,
                                        color="white", zorder=5))

        disp = "ELO" if name.lower() == "elo" else name
        ax.text(0.105, by + bh / 2, disp, va="center", ha="right", fontsize=21,
                fontweight="bold", color=S.INK)
        ta = ax.text(bx0 + 0.025, by + bh / 2, f"{wa:.0%}", va="center", ha="left",
                     fontsize=30, color="white", fontweight="bold", zorder=7)
        tb = ax.text(bx1 - 0.025, by + bh / 2, f"{wb:.0%}", va="center", ha="right",
                     fontsize=30, color="white", fontweight="bold", zorder=7)
        ta.set_path_effects(pct_fx); tb.set_path_effects(pct_fx)

    _save(fig, "06_final_flag_bars.png")


# --------------------------------------------------------------------------- #
def model_comparison(elo_table, poisson_table):
    elo = {r["team"]: r["win_cup"] for r in elo_table}
    poi = {r["team"]: r["win_cup"] for r in poisson_table}
    teams = sorted(set(elo) | set(poi), key=lambda t: -(elo.get(t, 0) + poi.get(t, 0)))
    teams = teams[:9][::-1]
    y = np.arange(len(teams))
    fig, ax = S.new_ax(figsize=(9, 6))
    ax.barh(y + 0.2, [elo.get(t, 0) for t in teams], height=0.38, color=S.ELO,
            label="Elo (best model)", zorder=3)
    ax.barh(y - 0.2, [poi.get(t, 0) for t in teams], height=0.38, color=S.POISSON,
            label="Dixon–Coles Poisson", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([S.abbr(t) for t in teams], fontsize=9, color=S.INK)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    S.title(ax, "How much do the two models agree?",
            "Probability of winning the cup — the honest picture of model uncertainty")
    S.footnote(fig, "Elo concentrates on accumulated strength; the Poisson model shrinks harder and reads in-tournament goals.")
    _save(fig, "04_model_comparison.png")


# --------------------------------------------------------------------------- #
def round_reach(team_table: list[dict], model_name: str):
    rows = [r for r in team_table if (r["reach_sf"] > 0.001 or r["win_cup"] > 0.001)]
    rows = sorted(rows, key=lambda r: r["win_cup"])
    teams = [r["team"] for r in rows]
    cols = ["reach_sf", "reach_final", "win_cup"]
    col_labels = ["Reach semis", "Reach final", "Win cup"]
    M = np.array([[r[c] for c in cols] for r in rows])

    fig, ax = S.new_ax(figsize=(7.5, 6))
    ax.grid(False)
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(col_labels, fontsize=9.5, color=S.INK)
    ax.set_yticks(range(len(teams)))
    ax.set_yticklabels([S.abbr(t) for t in teams], fontsize=9, color=S.INK)
    ax.tick_params(length=0)
    for i in range(len(teams)):
        for j in range(len(cols)):
            v = M[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.5 else S.INK, fontweight="bold")
    S.title(ax, "Road to the final",
            f"Each team's odds of reaching each round — {model_name} model")
    S.footnote(fig, "Exact propagation through the remaining bracket.")
    _save(fig, "05_round_reach.png")
