"""Statistical analysis for the World Cup refereeing-bias study.

Three families of question, all computed for EVERY team so Argentina sits inside a
real distribution rather than being cherry-picked:

1. Absolute rates    -- penalties/cards/fouls per game and per-90, group vs knockout.
2. Strength control  -- do Argentina's penalties exceed what their dominance predicts?
3. Deviation-by-opponent (the centrepiece) -- for every team, how much do their
   OPPONENTS deviate from their own norms (more fouls/cards conceded, more penalties
   given away) specifically when facing that team. A permutation test asks whether
   the "facing Argentina" effect is a genuine outlier or just noise.

Everything is minute-adjusted: a stat in a 120-minute (extra-time) match is scaled
to a per-90 rate, so longer knockout games don't masquerade as high-event games.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import dataset_path

FOCUS = "Argentina"
RATE_METRICS = ["fouls", "cards", "yellows", "reds", "pens_won", "pens_conceded"]


# ----------------------------------------------------------------------------- load
def load() -> pd.DataFrame:
    df = pd.read_csv(dataset_path())
    # per-90 rates: scale by actual played regulation (90) vs extra-time (120).
    for m in RATE_METRICS:
        df[f"{m}_p90"] = df[m] / df["minutes"] * 90.0
    df["gd"] = df["gf"] - df["ga"]
    return df


# --------------------------------------------------------- referee pressure index
# "How much the whistle went against a team" in a game, in GOAL-IMPACT units.
# Each event is weighted by its approximate expected-goals swing against the offending
# team, so a penalty (~0.76 xG) counts ~30x a foul (~0.025 xG) rather than equally.
#   penalty conceded : 0.76  -- penalty conversion rate ~= opponent xG
#   red card         : 0.50  -- empirical net goal swing of going a man down
#   yellow card      : 0.06  -- minor (caution + second-yellow risk)
#   foul committed   : 0.025 -- an average free kick is low-xG; most fouls aren't dangerous
# All per-90 so extra-time games compare fairly. Weights are documented + adjustable.
RPI_WEIGHTS = {"pens_conceded_p90": 0.76, "reds_p90": 0.50, "yellows_p90": 0.06, "fouls_p90": 0.025}


def add_ref_pressure(df: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    """Add `rpi` (referee pressure index, goal-impact units) per team-game and
    `rpi_base` = the team's leave-one-out average RPI within the same tournament (its
    'normal' treatment), plus `rpi_resid` = this game minus that norm."""
    w = weights or RPI_WEIGHTS
    out = df.copy()
    z = np.zeros(len(out))
    for col, wt in w.items():
        z = z + wt * out[col].values.astype(float)
    out["rpi"] = z
    # leave-one-out tournament baseline per team
    base = np.full(len(out), np.nan)
    for _, idx in out.groupby(["team", "tournament"]).groups.items():
        idx = list(idx)
        vals = out.loc[idx, "rpi"].values.astype(float)
        n = len(vals)
        if n >= 2:
            loo = (vals.sum() - vals) / (n - 1)
            for j, i in enumerate(idx):
                base[out.index.get_loc(i)] = loo[j]
        else:
            for i in idx:
                base[out.index.get_loc(i)] = vals[0]
    out["rpi_base"] = base
    out["rpi_resid"] = out["rpi"] - out["rpi_base"]  # +ve = whistled harder than usual
    return out


def anomalous_games(df: pd.DataFrame, focus: str = FOCUS) -> pd.DataFrame:
    """Games involving `focus`, scored by net whistle swing in `focus`'s favour.

    For each focus match we take the OPPONENT's row (how the opponent was treated):
    a high opponent rpi_resid means the opponent was punished more than they usually
    are. We also attach the focus team's own rpi_resid (ideally negative)."""
    r = add_ref_pressure(df)
    focus_matches = r[r.team == focus][["match_id", "tournament", "stage", "stage_group", "opponent"]]
    rows = []
    for _, fm in focus_matches.iterrows():
        opp = r[(r.match_id == fm.match_id) & (r.team == fm.opponent)]
        foc = r[(r.match_id == fm.match_id) & (r.team == focus)]
        if not len(opp) or not len(foc):
            continue
        opp, foc = opp.iloc[0], foc.iloc[0]
        rows.append({
            "tournament": fm.tournament,
            "stage": fm.stage,
            "opponent": fm.opponent,
            "label": f"{fm.opponent} '{int(fm.tournament) % 100:02d}",
            "opp_rpi": opp.rpi, "opp_rpi_base": opp.rpi_base, "opp_rpi_resid": opp.rpi_resid,
            "focus_rpi": foc.rpi, "focus_rpi_base": foc.rpi_base, "focus_rpi_resid": foc.rpi_resid,
            "net_swing": opp.rpi_resid - foc.rpi_resid,  # opp punished more AND focus less
            "opp_fouls": opp.fouls, "opp_cards": opp.cards, "opp_pens_conceded": opp.pens_conceded,
            "focus_pens_won": foc.pens_won,
        })
    out = pd.DataFrame(rows).sort_values("net_swing", ascending=False).reset_index(drop=True)
    return out


# ------------------------------------------------------------------ team summaries
def team_summary(df: pd.DataFrame, min_games: int = 4) -> pd.DataFrame:
    """One row per team: per-game and per-90 averages across all its matches."""
    g = df.groupby("team")
    out = g.agg(
        games=("match_id", "count"),
        tournaments=("tournament", "nunique"),
        fouls_pg=("fouls", "mean"),
        cards_pg=("cards", "mean"),
        yellows_pg=("yellows", "mean"),
        pens_won_pg=("pens_won", "mean"),
        pens_conceded_pg=("pens_conceded", "mean"),
        fouls_p90=("fouls_p90", "mean"),
        cards_p90=("cards_p90", "mean"),
        pens_won_p90=("pens_won_p90", "mean"),
        pens_conceded_p90=("pens_conceded_p90", "mean"),
        opp_cards_pg=("cards", lambda s: np.nan),  # placeholder; filled below
        gd_pg=("gd", "mean"),
    )
    # cards drawn against the opposition: average of the OPPONENT's cards per game.
    opp_cards = df.groupby("team").apply(
        lambda x: _opp_mean(df, x, "cards"), include_groups=False
    )
    opp_fouls = df.groupby("team").apply(
        lambda x: _opp_mean(df, x, "fouls"), include_groups=False
    )
    out["opp_cards_pg"] = opp_cards
    out["opp_fouls_pg"] = opp_fouls
    out = out[out["games"] >= min_games].copy()
    return out.reset_index()


def _opp_mean(df: pd.DataFrame, team_rows: pd.DataFrame, metric: str) -> float:
    """Mean of the metric taken over the OPPONENTS a team faced (per game)."""
    # For each of the team's games, find the opponent's row value via match_id.
    vals = []
    for _, r in team_rows.iterrows():
        opp = df[(df.match_id == r.match_id) & (df.team == r.opponent)]
        if len(opp):
            vals.append(opp.iloc[0][metric])
    return float(np.mean(vals)) if vals else np.nan


def rank_metric(summary: pd.DataFrame, metric: str, focus: str = FOCUS) -> dict:
    """Percentile + z-score of `focus` for a metric across all qualifying teams."""
    s = summary.dropna(subset=[metric])
    vals = s[metric].values
    if focus not in set(s.team):
        return {}
    fv = float(s.loc[s.team == focus, metric].iloc[0])
    mu, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
    pct = float((vals < fv).mean() * 100)
    rank = int((vals > fv).sum()) + 1
    return {
        "metric": metric,
        "focus_value": fv,
        "mean": mu,
        "std": sd,
        "z": (fv - mu) / sd if sd else float("nan"),
        "percentile": pct,
        "rank": rank,
        "n_teams": len(vals),
    }


# ------------------------------------------------------- deviation-by-opponent core
def deviation_table(df: pd.DataFrame, metrics=("fouls_p90", "cards_p90", "pens_conceded_p90")) -> pd.DataFrame:
    """For every match-team row, deviation of `team`'s metric from its OWN norm.

    Baseline = leave-one-out mean within the SAME tournament (controls for era and
    roster), so we ask: relative to how this team usually plays THIS tournament, did
    it foul / get carded / concede penalties more in this specific match?
    """
    rows = df.copy()
    for m in metrics:
        base = np.full(len(rows), np.nan)
        # group by team+tournament, leave-one-out mean
        for (_, _), idx in rows.groupby(["team", "tournament"]).groups.items():
            idx = list(idx)
            vals = rows.loc[idx, m].values.astype(float)
            n = len(vals)
            if n >= 2:
                loo = (vals.sum() - vals) / (n - 1)
                for j, i in enumerate(idx):
                    base[rows.index.get_loc(i)] = loo[j]
        rows[f"base_{m}"] = base
        rows[f"dev_{m}"] = rows[m] - base
    return rows


def opponent_effects(dev: pd.DataFrame, metric: str, min_games: int = 6) -> pd.DataFrame:
    """Average deviation experienced by teams FACING each opponent.

    dev_col measures the ROW team's deviation; grouping by `opponent` tells us how
    much a team's opponents over/under-shoot their norm when they play that opponent.
    """
    col = f"dev_{metric}"
    d = dev.dropna(subset=[col])
    g = d.groupby("opponent")[col]
    out = pd.DataFrame({
        "opponent": g.mean().index,
        "mean_dev": g.mean().values,
        "n_games": g.count().values,
        "sem": g.std(ddof=1).values / np.sqrt(g.count().values),
    })
    out = out[out.n_games >= min_games].sort_values("mean_dev", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def _trimmed_mean(vals: np.ndarray, trim: int) -> float:
    """Mean after dropping the `trim` highest and `trim` lowest values. Falls back to
    the plain mean when there aren't enough values to trim both ends."""
    v = np.asarray(vals, dtype=float)
    if trim <= 0 or len(v) <= 2 * trim:
        return float(np.mean(v))
    s = np.sort(v)
    return float(np.mean(s[trim:len(s) - trim]))


def permutation_test(dev: pd.DataFrame, metric: str, focus: str = FOCUS, B: int = 20000,
                     seed: int = 12345, trim: int = 0) -> dict:
    """Is the (trimmed) mean deviation of teams facing `focus` an outlier vs facing a
    random team?

    Null: the deviations seen against `focus` are a random size-n sample of all
    deviations. One-sided p (upper tail): P(random sample stat >= observed). With
    `trim`>0 the statistic is a trimmed mean (drop the `trim` most extreme games each
    side), applied identically to the observed set and every null sample, so a single
    freak game (e.g. a 9-card final) can't drive the result.
    """
    col = f"dev_{metric}"
    d = dev.dropna(subset=[col])
    all_dev = d[col].values.astype(float)
    obs_vals = d.loc[d.opponent == focus, col].values.astype(float)
    n = len(obs_vals)
    if n == 0:
        return {}
    obs = _trimmed_mean(obs_vals, trim)
    rng = np.random.default_rng(seed)
    # sample n deviations at random (without replacement) B times; same trim applied
    null = np.array([_trimmed_mean(rng.choice(all_dev, size=n, replace=False), trim) for _ in range(B)])
    p_upper = (np.sum(null >= obs) + 1) / (B + 1)
    p_two = (np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean())) + 1) / (B + 1)
    return {
        "metric": metric,
        "focus": focus,
        "n_games": n,
        "n_kept": max(n - 2 * trim, n) if n <= 2 * trim else n - 2 * trim,
        "observed_mean_dev": obs,
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
        "null_q05": float(np.quantile(null, 0.05)),
        "null_q95": float(np.quantile(null, 0.95)),
        "z": (obs - null.mean()) / null.std(ddof=1) if null.std() else float("nan"),
        "p_one_sided": float(p_upper),
        "p_two_sided": float(p_two),
        "pooled_all_teams_mean_dev": float(all_dev.mean()),
    }


# --------------------------------------------- composite RPI deviation + significance
def deviation_table_rpi(df: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    """Deviation table using the composite Referee Pressure Index as the metric.

    Adds an `rpi` column (goal-impact-weighted sum of the per-90 event rates) and
    returns the standard deviation table with `dev_rpi` = each team's RPI in a match
    minus its own leave-one-out tournament norm. This is the single most appropriate
    summary metric for "how hard the whistle went": it folds penalties, cards and
    fouls into one number weighted by their goal impact (a penalty counts ~30x a foul),
    so one significance verdict replaces three correlated ones.
    """
    r = df.copy()
    z = np.zeros(len(r))
    for col, wt in (weights or RPI_WEIGHTS).items():
        z = z + wt * r[col].values.astype(float)
    r["rpi"] = z
    return deviation_table(r, metrics=("rpi",))


def focus_significance_by_tournament(df: pd.DataFrame, focus: str = FOCUS, years=None,
                                     B: int = 20000, min_games: int = 3, trim: int = 0) -> list[dict]:
    """Per World Cup: are `focus`'s opponents whistled above their own RPI norm more
    than a random team's opponents? One permutation test per tournament on the
    composite Referee Pressure Index, plus `focus`'s rank among all teams that cup.
    `trim`>0 drops the `trim` most extreme opponent-games each side (robustness)."""
    years = years or sorted(int(y) for y in df.tournament.unique())
    out = []
    for yr in years:
        devr = deviation_table_rpi(df[df.tournament == yr])
        pt = permutation_test(devr, "rpi", focus=focus, B=B, seed=1000 + int(yr), trim=trim)
        eff = opponent_effects(devr, "rpi", min_games=min_games)
        arg = eff[eff.opponent == focus]
        row = {"tournament": int(yr), "n_teams": int(len(eff)),
               "rank": int(arg["rank"].iloc[0]) if len(arg) else None}
        row.update(pt or {})
        out.append(row)
    return out


# ---------------------------------------------------------------- strength control
def strength_control(df: pd.DataFrame, focus: str = FOCUS, min_games: int = 4) -> dict:
    """Regress team penalties-won-per-game on average goal difference (a dominance
    proxy) and report where `focus` sits vs the prediction (residual)."""
    s = team_summary(df, min_games=min_games)
    x = s["gd_pg"].values.astype(float)
    y = s["pens_won_pg"].values.astype(float)
    b1, b0 = np.polyfit(x, y, 1)
    pred = b0 + b1 * x
    resid = y - pred
    s = s.assign(pred_pens=pred, resid_pens=resid)
    frow = s[s.team == focus]
    out = {
        "slope": float(b1),
        "intercept": float(b0),
        "table": s[["team", "gd_pg", "pens_won_pg", "pred_pens", "resid_pens"]],
    }
    if len(frow):
        out.update({
            "focus_gd_pg": float(frow.gd_pg.iloc[0]),
            "focus_pens_pg": float(frow.pens_won_pg.iloc[0]),
            "focus_pred": float(frow.pred_pens.iloc[0]),
            "focus_resid": float(frow.resid_pens.iloc[0]),
            "resid_percentile": float((resid < frow.resid_pens.iloc[0]).mean() * 100),
        })
    return out


if __name__ == "__main__":
    df = load()
    print("rows:", len(df), "| teams:", df.team.nunique(), "| tournaments:", sorted(df.tournament.unique()))
    summ = team_summary(df)
    for m in ["pens_won_pg", "pens_conceded_pg", "fouls_pg", "cards_pg", "opp_cards_pg"]:
        r = rank_metric(summ, m)
        if r:
            print(f"{m:18s} Arg={r['focus_value']:.3f}  mean={r['mean']:.3f}  z={r['z']:+.2f}  pct={r['percentile']:.0f}  rank={r['rank']}/{r['n_teams']}")
    print()
    dev = deviation_table(df)
    for m in ["fouls_p90", "cards_p90", "pens_conceded_p90"]:
        pt = permutation_test(dev, m)
        eff = opponent_effects(dev, m)
        arg = eff[eff.opponent == FOCUS]
        argrank = int(arg["rank"].iloc[0]) if len(arg) else -1
        print(f"[dev {m:18s}] Arg mean_dev={pt['observed_mean_dev']:+.3f} (n={pt['n_games']})  z={pt['z']:+.2f}  p1={pt['p_one_sided']:.3f}  rank {argrank}/{len(eff)}")
