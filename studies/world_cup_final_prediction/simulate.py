"""Forecast the rest of the 2026 bracket by exact probability propagation.

The remaining draw is eight quarter-finalists in a fixed binary tree, so instead
of Monte-Carlo we push team-reaches-this-node probabilities exactly up the tree.
A knockout tie is won in regulation/ET (match_probs) or, if drawn, on penalties
(a coin flip), so P(a advances) = P(a win) + P(draw) * SHOOTOUT_PROB.

A "model" here is anything exposing `.match_probs(home, away, host_side)` and
`.can_predict(a, b)` — both PoissonModel and EloModel qualify. All remaining games
are at neutral sites (no host is still alive), so host_side is always None.
"""

from __future__ import annotations

from collections import defaultdict

from . import config

# The fixed remaining tree. Slot names resolve to real teams via QUARTERFINALS,
# except UNRESOLVED_KEY, which is the winner of the undecided R16 penalty tie.
Dist = dict  # {team: probability}


def resolve_draw(ph: float, pd_: float, pa: float, et_rate: float) -> tuple[float, float]:
    """Split a regulation W/D/L into a no-tie (home_advances, away_advances).

    The draw mass `pd_` is settled: a fraction `et_rate` in extra time, where the
    stronger side (by its share of the non-draw probability) is favoured, and the
    rest on penalties (a coin flip). Returns probabilities that sum to 1.
    """
    denom = ph + pa
    norm_h = ph / denom if denom > 1e-12 else 0.5
    tie_to_h = et_rate * norm_h + (1 - et_rate) * config.SHOOTOUT_PROB
    return ph + pd_ * tie_to_h, pa + pd_ * (1 - tie_to_h)


def advance_prob(model, a: str, b: str, et_rate: float) -> float:
    """P(a beats b in a single knockout tie) — neutral site, draw resolved."""
    ph, pd_, pa = model.match_probs(a, b, None)
    adv_a, _ = resolve_draw(ph, pd_, pa, et_rate)
    return adv_a


def _resolve(model, dist_a: Dist, dist_b: Dist, et_rate: float) -> Dist:
    """Winner distribution of a tie whose two sides are themselves distributions."""
    out: Dist = defaultdict(float)
    for a, pa in dist_a.items():
        for b, pb in dist_b.items():
            joint = pa * pb
            adv = advance_prob(model, a, b, et_rate)
            out[a] += joint * adv
            out[b] += joint * (1 - adv)
    return dict(out)


def _merge(*dists: Dist) -> Dist:
    out: Dist = defaultdict(float)
    for d in dists:
        for t, p in d.items():
            out[t] += p
    return dict(out)


def forecast(model, et_rate: float) -> dict:
    """Propagate the bracket. Returns per-team round-reach probs + final matchups."""
    # Resolve the one undecided R16 tie into a distribution over its two teams.
    s, c = config.UNRESOLVED_R16
    sui_col: Dist = {s: advance_prob(model, s, c, et_rate),
                     c: advance_prob(model, c, s, et_rate)}

    def slot(name: str) -> Dist:
        return sui_col if name == config.UNRESOLVED_KEY else {name: 1.0}

    qfs = [(slot(h), slot(a)) for h, a in config.QUARTERFINALS]
    qf_winners = [_resolve(model, x, y, et_rate) for x, y in qfs]

    sf1 = _resolve(model, qf_winners[0], qf_winners[1], et_rate)
    sf2 = _resolve(model, qf_winners[2], qf_winners[3], et_rate)
    champion = _resolve(model, sf1, sf2, et_rate)

    reach_qf = _merge(*[_merge(x, y) for x, y in qfs])   # 1.0 for the 7 known, split for SUI/COL
    reach_sf = _merge(*qf_winners)
    reach_final = _merge(sf1, sf2)

    # Every possible final pairing and its probability (one side from each half).
    finals = {}
    for a, pa in sf1.items():
        for b, pb in sf2.items():
            finals[(a, b)] = pa * pb

    return {
        "reach_qf": reach_qf,
        "reach_sf": reach_sf,
        "reach_final": reach_final,
        "champion": champion,
        "finals": finals,
    }


def team_table(fc: dict) -> list[dict]:
    """Per-team probabilities of reaching each round, sorted by title odds."""
    teams = set(fc["reach_qf"]) | set(fc["reach_sf"]) | {t for m in fc["finals"] for t in m}
    rows = [{
        "team": t,
        "reach_sf": fc["reach_sf"].get(t, 0.0),
        "reach_final": fc["reach_final"].get(t, 0.0),
        "win_cup": fc["champion"].get(t, 0.0),
    } for t in teams]
    return sorted(rows, key=lambda r: -r["win_cup"])


def most_likely_final(fc: dict) -> tuple[str, str, float]:
    (a, b), p = max(fc["finals"].items(), key=lambda kv: kv[1])
    return a, b, p


def top_scorelines(model, a: str, b: str, n: int = 5) -> list[dict]:
    """Most probable exact scorelines for a matchup (Poisson only)."""
    if not hasattr(model, "score_matrix"):
        return []
    P = model.score_matrix(a, b, None)
    cells = [(int(x), int(y), float(P[x, y])) for x in range(P.shape[0]) for y in range(P.shape[1])]
    cells.sort(key=lambda c: -c[2])
    return [{"home_goals": x, "away_goals": y, "prob": p} for x, y, p in cells[:n]]
