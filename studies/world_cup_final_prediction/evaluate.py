"""Leakage-safe backtesting, proper scoring rules, and hyper-parameter search.

Every prediction for a knockout match is made from a model that has seen ONLY
strictly-earlier rounds of that same tournament (plus, optionally, past cups via
the carry-over prior). Both models are batched per (year, round) — one fit serves
every match in a round, since they share the same training window.

Scoring uses the Ranked Probability Score (the standard for ordered 1X2 outcomes),
with log-loss / Brier / accuracy alongside, always against a base-rate baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config, data
from .models import EloModel, PoissonModel, carryover_priors

OUTCOMES = {"H": 0, "D": 1, "A": 2}  # ordered: home win < draw < away win


# --------------------------------------------------------------------------- #
# Proper scoring rules (all lower = better, except accuracy)
# --------------------------------------------------------------------------- #
def _onehot(result: str) -> np.ndarray:
    v = np.zeros(3)
    v[OUTCOMES[result]] = 1.0
    return v


def rps(probs, result: str) -> float:
    """Ranked Probability Score for the ordered triple (H, D, A)."""
    p = np.asarray(probs)
    o = _onehot(result)
    cp, co = np.cumsum(p)[:-1], np.cumsum(o)[:-1]
    return float(np.mean((cp - co) ** 2))


def log_loss(probs, result: str) -> float:
    return float(-np.log(max(probs[OUTCOMES[result]], 1e-12)))


def brier(probs, result: str) -> float:
    return float(np.sum((np.asarray(probs) - _onehot(result)) ** 2))


def scores(preds: pd.DataFrame) -> dict:
    """Mean metrics over a table of predictions (cols pH,pD,pA,result)."""
    if preds.empty:
        return {"n": 0}
    prob = preds[["pH", "pD", "pA"]].to_numpy()
    hit = np.argmax(prob, axis=1) == preds.result.map(OUTCOMES).to_numpy()
    return {
        "n": len(preds),
        "rps": float(np.mean([rps(p, r) for p, r in zip(prob, preds.result)])),
        "log_loss": float(np.mean([log_loss(p, r) for p, r in zip(prob, preds.result)])),
        "brier": float(np.mean([brier(p, r) for p, r in zip(prob, preds.result)])),
        "accuracy": float(np.mean(hit)),
    }


# --------------------------------------------------------------------------- #
# Per-round batched predictors (leakage-safe)
# --------------------------------------------------------------------------- #
def _knockout_depths(df: pd.DataFrame, year: int) -> list[int]:
    ko = data.knockout_matches(df, year)
    return sorted(ko.depth.unique(), reverse=True)  # earliest KO round first


_POIS_CACHE: dict = {}


def poisson_predict_year(df: pd.DataFrame, year: int, kappa: float,
                         prior_weight: float) -> pd.DataFrame:
    key = (id(df), year, round(kappa, 6), round(prior_weight, 6))
    if key in _POIS_CACHE:
        return _POIS_CACHE[key]
    ko = data.knockout_matches(df, year)
    out = []
    for depth in _knockout_depths(df, year):
        train = data.tournament_games(df, year, before_depth=depth)
        if train.empty:
            continue
        teams = data.teams_in(train)
        pa, pd_ = carryover_priors(df, year, teams, prior_weight)
        model = PoissonModel(teams, kappa, pa, pd_).fit(train)
        for m in ko[ko.depth == depth].itertuples():
            if not model.can_predict(m.home, m.away):
                continue
            ph, pdw, paw = model.match_probs(m.home, m.away, m.host_side)
            out.append(_row(m, ph, pdw, paw))
    result = pd.DataFrame(out)
    _POIS_CACHE[key] = result
    return result


# Memoised: the same (year, k, mov, carryover) diffs are re-requested across many
# hyper-parameter combos and LOCO folds. Keyed on the df identity so a fresh dataset
# never hits stale entries.
_DIFFS_CACHE: dict = {}


def elo_diffs_year(df: pd.DataFrame, year: int, k: float, mov: float,
                   carryover: float) -> list[dict]:
    """Rating diffs for a cup's knockout matches (one replay, snapshot per round)."""
    key = (id(df), year, round(k, 6), round(mov, 6), round(carryover, 6))
    if key in _DIFFS_CACHE:
        return _DIFFS_CACHE[key]
    ko = data.knockout_matches(df, year)
    rows = []
    for depth in _knockout_depths(df, year):
        elo = EloModel(k, mov, carryover).fit(df, up_to_year=year, up_to_depth=depth)
        for m in ko[ko.depth == depth].itertuples():
            if not elo.can_predict(m.home, m.away):
                continue
            rows.append({"match_id": m.match_id, "home": m.home, "away": m.away,
                         "host_side": m.host_side, "result": m.result,
                         "diff": elo.diff(m.home, m.away, m.host_side)})
    _DIFFS_CACHE[key] = rows
    return rows


def _row(m, ph, pd_, pa) -> dict:
    return {"match_id": m.match_id, "stage": m.stage, "home": m.home, "away": m.away,
            "result": m.result, "pH": ph, "pD": pd_, "pA": pa}


# --------------------------------------------------------------------------- #
# Elo ordered-logit calibration (beta, tau) fit on the train cups
# --------------------------------------------------------------------------- #
def _ordered_probs(diff, beta, tau):
    z = beta * diff
    from math import exp
    sig = lambda t: 1 / (1 + exp(-t))
    pa = sig(-tau - z)
    ph = 1 - sig(tau - z)
    pdw = max(sig(tau - z) - sig(-tau - z), 1e-9)
    s = ph + pdw + pa
    return ph / s, pdw / s, pa / s


def fit_elo_calibration(diff_rows: list[dict]) -> tuple[float, float]:
    """Choose (beta, tau) minimising log-loss on train rating-diff/outcome pairs."""
    diffs = np.array([r["diff"] for r in diff_rows])
    outs = [r["result"] for r in diff_rows]

    def nll(p):
        beta, tau = abs(p[0]), abs(p[1])
        tot = 0.0
        for d, r in zip(diffs, outs):
            probs = _ordered_probs(d, beta, tau)
            tot += -np.log(max(probs[OUTCOMES[r]], 1e-12))
        return tot

    res = minimize(nll, [0.01, 0.4], method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 2000})
    return abs(res.x[0]), abs(res.x[1])


def elo_predict_year(df, year, k, mov, carryover, beta, tau) -> pd.DataFrame:
    out = []
    for r in elo_diffs_year(df, year, k, mov, carryover):
        ph, pdw, pa = _ordered_probs(r["diff"], beta, tau)
        out.append({"match_id": r["match_id"], "home": r["home"], "away": r["away"],
                    "result": r["result"], "pH": ph, "pD": pdw, "pA": pa})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Baseline: constant base-rate (H/D/A frequencies from the train cups)
# --------------------------------------------------------------------------- #
def base_rate_probs(df: pd.DataFrame, train_years: list[int]) -> tuple[float, float, float]:
    ko = df[df.tournament.isin(train_years) & df.is_knockout]
    n = len(ko)
    return (float((ko.result == "H").sum() / n),
            float((ko.result == "D").sum() / n),
            float((ko.result == "A").sum() / n))


def base_rate_predict_year(df, year, probs) -> pd.DataFrame:
    ko = data.knockout_matches(df, year)
    ph, pdw, pa = probs
    return pd.DataFrame([{"match_id": m.match_id, "home": m.home, "away": m.away,
                          "result": m.result, "pH": ph, "pD": pdw, "pA": pa}
                         for m in ko.itertuples()])


# --------------------------------------------------------------------------- #
# Hyper-parameter search on the validation cup
# --------------------------------------------------------------------------- #
def _pooled(df, years, predict_fn) -> pd.DataFrame:
    return pd.concat([predict_fn(y) for y in years], ignore_index=True)


def tune_poisson(df: pd.DataFrame, years=None) -> dict:
    """Choose (kappa, prior_weight) by pooled RPS over `years`' knockouts."""
    years = years or config.VALIDATION_YEARS
    best = None
    grid = []
    for kappa in config.POISSON_KAPPA_GRID:
        for pw in config.POISSON_PRIOR_WEIGHT_GRID:
            preds = _pooled(df, years, lambda y: poisson_predict_year(df, y, kappa, pw))
            s = scores(preds)
            grid.append({"kappa": kappa, "prior_weight": pw, **s})
            if best is None or s["rps"] < best["rps"]:
                best = {"kappa": kappa, "prior_weight": pw, **s}
    return {"best": best, "grid": grid}


def tune_elo(df: pd.DataFrame, years=None) -> dict:
    """Choose (k, mov, carryover) and calibrate (beta, tau) on `years`' knockouts."""
    years = years or config.VALIDATION_YEARS
    best = None
    grid = []
    for k in config.ELO_K_GRID:
        for mov in config.ELO_MOV_GRID:
            for co in config.ELO_CARRYOVER_GRID:
                calib_rows = []
                for y in years:
                    calib_rows += elo_diffs_year(df, y, k, mov, co)
                beta, tau = fit_elo_calibration(calib_rows)
                preds = _pooled(df, years,
                                lambda y: elo_predict_year(df, y, k, mov, co, beta, tau))
                s = scores(preds)
                combo = {"k": k, "mov": mov, "carryover": co, "beta": beta, "tau": tau, **s}
                grid.append(combo)
                if best is None or s["rps"] < best["rps"]:
                    best = combo
    return {"best": best, "grid": grid}


# --------------------------------------------------------------------------- #
# Leave-one-cup-out cross-validation
# --------------------------------------------------------------------------- #
def leave_one_cup_out(df: pd.DataFrame, cups: list[int]) -> dict:
    """For each cup, tune both models on the OTHER cups, then predict its knockouts.

    Every cup's knockout games (including its final) are thus predicted fully out of
    sample: the hyper-parameters never saw that cup. Team attack/defence are still
    re-fit within the cup, and priors stay strictly temporal (earlier cups only), so
    no future team data leaks in either.
    """
    out = {}
    for c in cups:
        others = [y for y in cups if y != c]
        pb = tune_poisson(df, others)["best"]
        eb = tune_elo(df, others)["best"]
        out[c] = {
            "pb": pb, "eb": eb,
            "poisson": poisson_predict_year(df, c, pb["kappa"], pb["prior_weight"]),
            "elo": elo_predict_year(df, c, eb["k"], eb["mov"], eb["carryover"],
                                    eb["beta"], eb["tau"]),
            "base": base_rate_predict_year(df, c, base_rate_probs(df, others)),
        }
    return out


def loco_scores(loco: dict) -> dict:
    """Pooled out-of-sample metrics over every cup's knockouts."""
    return {name: scores(pd.concat([loco[c][key] for c in loco], ignore_index=True))
            for name, key in (("Elo", "elo"), ("Poisson", "poisson"), ("BaseRate", "base"))}


def _match_row(loco: dict, cup: int, m, et_rate: float, label: str) -> dict:
    """One match's out-of-sample prediction (both models) vs its actual result."""
    from .simulate import resolve_draw
    actual = m.result  # H / A / D(=penalties)
    penalties = actual == "D"
    if actual == "H":
        winner = m.home
    elif actual == "A":
        winner = m.away
    else:  # penalties: winner not in the data — supplied from config if known
        winner = config.KNOWN_SHOOTOUT_WINNERS.get(int(cup))
    row = {"label": label, "cup": int(cup), "home": m.home, "away": m.away,
           "actual": actual, "winner": winner, "penalties": penalties,
           "went_to_et": bool(m.went_to_et)}
    for key in ("elo", "poisson"):
        pr = loco[cup][key]
        hit = pr[pr.match_id == m.match_id]
        if not len(hit):
            row[key] = None
            continue
        ph, pd_, pa = float(hit.pH.iloc[0]), float(hit.pD.iloc[0]), float(hit.pA.iloc[0])
        win_h, win_a = resolve_draw(ph, pd_, pa, et_rate)
        fav = m.home if win_h >= win_a else m.away
        p_winner = None if winner is None else (win_h if winner == m.home else win_a)
        row[key] = {
            "p_winner": p_winner,
            "reg": (ph, pd_, pa), "win_home": win_h, "win_away": win_a,
            "favorite": fav, "fav_prob": max(win_h, win_a),
            "rps": rps((ph, pd_, pa), actual),
            "called": (None if winner is None else (fav == winner)),
        }
    return row


def finals_table(df: pd.DataFrame, loco: dict, et_rate: float) -> list[dict]:
    """Per past-cup final: each model's out-of-sample prediction vs what happened."""
    rows = []
    for c in sorted(loco):
        fm = df[(df.tournament == c) & (df.stage_norm == "FINAL")]
        if len(fm):
            rows.append(_match_row(loco, c, fm.iloc[0], et_rate, label=str(int(c))))
    return rows


