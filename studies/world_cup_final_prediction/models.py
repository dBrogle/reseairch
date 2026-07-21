"""The two prediction models.

PoissonModel  - Dixon-Coles bivariate-Poisson goals model. Each team has an attack
                and a defence strength; expected goals in a match are your attack
                against their defence, so opponent strength and goal difference are
                built in. Fit by penalised MLE, the penalty shrinking the ~6 noisy
                games per team toward a prior (the field mean, or that team's past-
                World-Cup form). Outputs a full scoreline distribution.

EloModel      - A margin-of-victory Elo rating (beating a strong team, by a wide
                margin, moves you more) mapped to W/D/L via an ordered logit. The
                honest baseline the Poisson model has to beat.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config


# --------------------------------------------------------------------------- #
# Cross-tournament prior: a team's attack/defence seeded from PAST cups only.
# --------------------------------------------------------------------------- #
def carryover_priors(df: pd.DataFrame, year: int, teams: list[str],
                     weight: float) -> tuple[dict, dict]:
    """Per-team (attack, defence) priors from strictly-earlier tournaments.

    Leakage-free: only cups before `year` are used. `weight` in [0,1] scales how
    strongly past form is trusted (0 => everyone's prior is 0 => shrink to mean).
    Returned in the model's log-rate units, roughly zero-centred.
    """
    past = df[df.tournament < year]
    if weight == 0.0 or past.empty:
        return {t: 0.0 for t in teams}, {t: 0.0 for t in teams}

    # Global mean goals per team-per-match over the past cups = the "average team".
    mu0 = (past.hg.sum() + past.ag.sum()) / (2 * len(past))
    mu0 = max(mu0, 0.5)

    gf, ga, games = {}, {}, {}
    for _, m in past.iterrows():
        for team, scored, conceded in ((m.home, m.hg, m.ag), (m.away, m.ag, m.hg)):
            gf[team] = gf.get(team, 0) + scored
            ga[team] = ga.get(team, 0) + conceded
            games[team] = games.get(team, 0) + 1

    p_att, p_def = {}, {}
    for t in teams:
        if games.get(t, 0) == 0:
            p_att[t] = p_def[t] = 0.0
            continue
        gf_pg = max(gf[t] / games[t], 0.15)
        ga_pg = max(ga[t] / games[t], 0.15)
        p_att[t] = weight * math.log(gf_pg / mu0)   # scored more than avg => +attack
        p_def[t] = weight * math.log(mu0 / ga_pg)   # conceded less than avg => +defence
    return p_att, p_def


# --------------------------------------------------------------------------- #
# Dixon-Coles Poisson model
# --------------------------------------------------------------------------- #
class PoissonModel:
    def __init__(self, teams: list[str], kappa: float,
                 prior_att: dict | None = None, prior_def: dict | None = None):
        self.teams = list(teams)
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.T = len(self.teams)
        self.kappa = kappa
        self.prior_att = np.array([(prior_att or {}).get(t, 0.0) for t in self.teams])
        self.prior_def = np.array([(prior_def or {}).get(t, 0.0) for t in self.teams])
        self.att = np.zeros(self.T)
        self.dfn = np.zeros(self.T)
        self.mu = 0.0
        self.home_adv = config.HOST_HOME_ADV  # fixed (host only), not fitted
        self.rho = 0.0
        self._logfact = np.array([math.lgamma(k + 1) for k in range(config.MAX_GOALS + 1)])

    def _unpack(self, p):
        T = self.T
        return p[:T], p[T:2 * T], p[2 * T], p[2 * T + 1]

    def fit(self, matches: pd.DataFrame) -> "PoissonModel":
        i = matches.home.map(self.idx).to_numpy()
        j = matches.away.map(self.idx).to_numpy()
        x = matches.hg.to_numpy(float)
        y = matches.ag.to_numpy(float)
        host_h = (matches.host_side == "home").to_numpy(float)
        host_a = (matches.host_side == "away").to_numpy(float)

        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)

        hadv = config.HOST_HOME_ADV

        def nll(p):
            att, dfn, mu, rho = self._unpack(p)
            loglam_h = mu + att[i] - dfn[j] + hadv * host_h
            loglam_a = mu + att[j] - dfn[i] + hadv * host_a
            lam_h = np.exp(loglam_h)
            lam_a = np.exp(loglam_a)
            ll = x * loglam_h - lam_h + y * loglam_a - lam_a
            tau = np.ones_like(lam_h)
            tau[m00] = 1 - lam_h[m00] * lam_a[m00] * rho
            tau[m01] = 1 + lam_h[m01] * rho
            tau[m10] = 1 + lam_a[m10] * rho
            tau[m11] = 1 - rho
            ll = ll + np.log(np.clip(tau, 1e-10, None))
            pen = self.kappa * (np.sum((att - self.prior_att) ** 2)
                                + np.sum((dfn - self.prior_def) ** 2))
            return -np.sum(ll) + pen

        x0 = np.zeros(2 * self.T + 2)
        x0[2 * self.T] = math.log(max((x.mean() + y.mean()) / 2, 0.3))  # mu ~ log avg goals
        bounds = [(-3, 3)] * self.T + [(-3, 3)] * self.T + [(-3, 2), (-0.2, 0.2)]
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500})
        self.att, self.dfn, self.mu, self.rho = self._unpack(res.x)
        return self

    def rates(self, home: str, away: str, host_side: str | None) -> tuple[float, float]:
        i, j = self.idx[home], self.idx[away]
        lh = self.mu + self.att[i] - self.dfn[j] + (self.home_adv if host_side == "home" else 0.0)
        la = self.mu + self.att[j] - self.dfn[i] + (self.home_adv if host_side == "away" else 0.0)
        return math.exp(lh), math.exp(la)

    def score_matrix(self, home: str, away: str, host_side: str | None = None) -> np.ndarray:
        lam_h, lam_a = self.rates(home, away, host_side)
        k = np.arange(config.MAX_GOALS + 1)
        ph = np.exp(-lam_h + k * math.log(lam_h) - self._logfact)
        pa = np.exp(-lam_a + k * math.log(lam_a) - self._logfact)
        P = np.outer(ph, pa)
        # Dixon-Coles low-score dependence correction.
        r = self.rho
        P[0, 0] *= 1 - lam_h * lam_a * r
        P[0, 1] *= 1 + lam_h * r
        P[1, 0] *= 1 + lam_a * r
        P[1, 1] *= 1 - r
        P = np.clip(P, 0, None)
        return P / P.sum()

    def match_probs(self, home: str, away: str, host_side: str | None = None) -> tuple[float, float, float]:
        """(P home win, P draw, P away win) in regulation."""
        P = self.score_matrix(home, away, host_side)
        p_home = np.tril(P, -1).sum()   # rows(home) > cols(away)
        p_away = np.triu(P, 1).sum()
        p_draw = np.trace(P)
        return float(p_home), float(p_draw), float(p_away)

    def can_predict(self, home: str, away: str) -> bool:
        return home in self.idx and away in self.idx


# --------------------------------------------------------------------------- #
# Elo baseline
# --------------------------------------------------------------------------- #
ELO_HFA = 65.0  # host-nation home-field bonus, in Elo points (neutral otherwise)


class EloModel:
    def __init__(self, k: float, mov: float, carryover: float,
                 beta: float = 0.01, tau: float = 0.30):
        self.k = k
        self.mov = mov
        self.carryover = carryover
        self.beta = beta   # maps rating diff -> logit scale (fit on train cups)
        self.tau = tau     # draw half-width (fit on train cups)
        self.ratings: dict[str, float] = {}

    @staticmethod
    def _chrono(games: pd.DataFrame) -> pd.DataFrame:
        # Group stage first (depth 99) down to the final (depth 0).
        return games.sort_values(["depth", "match_id"], ascending=[False, True])

    def fit(self, df: pd.DataFrame, up_to_year: int, up_to_depth: int | None = None) -> "EloModel":
        """Run Elo through every cup up to (and partially into) `up_to_year`.

        For the target cup, only games strictly earlier than `up_to_depth` are
        consumed, so a prediction never sees its own or later rounds.
        """
        self.ratings = {}
        years = sorted(y for y in df.tournament.unique() if y <= up_to_year)
        for y in years:
            g = df[df.tournament == y]
            if y == up_to_year and up_to_depth is not None:
                g = g[g.depth > up_to_depth]
            self._regress_to_mean()
            for _, m in self._chrono(g).iterrows():
                self._update(m)
        return self

    def _regress_to_mean(self):
        for t in self.ratings:
            self.ratings[t] = 1500 + self.carryover * (self.ratings[t] - 1500)

    def _rating(self, t: str) -> float:
        return self.ratings.setdefault(t, 1500.0)

    def _update(self, m):
        rh, ra = self._rating(m.home), self._rating(m.away)
        hfa = ELO_HFA if m.host_side == "home" else (-ELO_HFA if m.host_side == "away" else 0.0)
        exp_h = 1 / (1 + 10 ** (-((rh + hfa) - ra) / 400))
        s_h = 1.0 if m.hg > m.ag else (0.0 if m.hg < m.ag else 0.5)
        mult = (1 + abs(m.hg - m.ag)) ** self.mov
        delta = self.k * mult * (s_h - exp_h)
        self.ratings[m.home] = rh + delta
        self.ratings[m.away] = ra - delta

    def diff(self, home: str, away: str, host_side: str | None) -> float:
        hfa = ELO_HFA if host_side == "home" else (-ELO_HFA if host_side == "away" else 0.0)
        return self._rating(home) + hfa - self._rating(away)

    def match_probs(self, home: str, away: str, host_side: str | None = None) -> tuple[float, float, float]:
        z = self.beta * self.diff(home, away, host_side)
        p_away = _sig(-self.tau - z)
        p_home = 1 - _sig(self.tau - z)
        p_draw = max(_sig(self.tau - z) - _sig(-self.tau - z), 1e-9)
        s = p_home + p_draw + p_away
        return p_home / s, p_draw / s, p_away / s

    def can_predict(self, home: str, away: str) -> bool:
        return home in self.ratings and away in self.ratings


def _sig(x: float) -> float:
    return 1 / (1 + math.exp(-x))
