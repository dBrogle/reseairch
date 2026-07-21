"""Predict the 2026 World Cup final from teams' performance across five World Cups.

Pipeline:
  1. Load the committed match data (2010-2026, through the 2026 semi-finals).
  2. Leave-one-cup-out CV: predict every cup's knockouts with hyper-parameters
     tuned only on the OTHER cups — fully out of sample, including each final.
  3. Report out-of-sample metrics on all knockouts, and the model's call on every
     past World Cup final vs what actually happened (the most relevant test).
  4. Predict the live 2026 final (no tie — the draw is resolved into a winner).
  5. Write findings.json and render the charts.

Run:  python -m studies.world_cup_final_prediction.main [--no-charts]
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from . import config, data, evaluate as ev, simulate as sim, visualize as viz
from .models import EloModel, PoissonModel, carryover_priors


def _fit_final_poisson(df, kappa, prior_weight):
    g = data.tournament_games(df, config.TEST_YEAR)
    teams = data.teams_in(g)
    pa, pd_ = carryover_priors(df, config.TEST_YEAR, teams, prior_weight)
    return PoissonModel(teams, kappa, pa, pd_).fit(g)


def _fit_final_elo(df, k, mov, carryover, beta, tau):
    m = EloModel(k, mov, carryover, beta=beta, tau=tau)
    return m.fit(df, up_to_year=config.TEST_YEAR)


def run(make_charts: bool = True):
    df = data.load_matches()
    print(f"Loaded {len(df)} matches across {sorted(df.tournament.unique())}.")

    cups = sorted(int(x) for x in df.tournament.unique())

    # --- Knockout tie-resolution rate (validated from past cups) ----------- #
    et_rate, et_stats = data.et_resolution_rate(df)
    print(f"Knockout ties (all cups): {et_stats['tied_after_90']} level after 90' -> "
          f"{et_stats['decided_in_et']} in extra time, {et_stats['penalties']} on penalties "
          f"({et_rate:.0%} ET / {1-et_rate:.0%} pens ≈ coin flip).")

    # --- Leave-one-cup-out CV: every cup predicted fully out of sample ------ #
    print("\nLeave-one-cup-out CV (tune on the other cups, predict the held-out one)...")
    loco = ev.leave_one_cup_out(df, cups)
    ko_scores = ev.loco_scores(loco)
    print(f"\nOut-of-sample on ALL knockout games (pooled, n={ko_scores['Elo']['n']}):")
    for name in ("Elo", "Poisson", "BaseRate"):
        s = ko_scores[name]
        print(f"  {name:<9} RPS={s['rps']:.4f}  logloss={s['log_loss']:.3f}  "
              f"brier={s['brier']:.3f}  acc={s['accuracy']:.0%}")
    best_model = min(("Elo", "Poisson"), key=lambda m: ko_scores[m]["rps"])
    print(f"  -> best out of sample: {best_model}")

    # --- The most relevant test: past World Cup FINALS (out of sample) ----- #
    finals = ev.finals_table(df, loco, et_rate)
    _print_finals(finals, best_model)

    # --- Live 2026 final: 2026-fold hyper-parameters (tuned on 2010-2022) --- #
    eb, pb = loco[config.TEST_YEAR]["eb"], loco[config.TEST_YEAR]["pb"]
    elo = _fit_final_elo(df, eb["k"], eb["mov"], eb["carryover"], eb["beta"], eb["tau"])
    poi = _fit_final_poisson(df, pb["kappa"], pb["prior_weight"])
    head = elo if best_model == "Elo" else poi

    fin = data.finalists(df, config.TEST_YEAR)
    if fin:
        _predict_final(df, elo, poi, head, best_model, fin, loco, ko_scores, finals,
                       et_rate, et_stats, make_charts)
    else:
        _forecast_bracket(elo, poi, head, best_model, loco, ko_scores, finals,
                          et_rate, et_stats, make_charts)
    print("\nDone.")


def _print_finals(finals: list[dict], best_model: str):
    print("\nPast World Cup finals — out-of-sample prediction vs what happened:")
    hits = tot = 0
    key = best_model.lower()
    for r in finals:
        m = r[key]
        pen = r.get("penalties", False)
        won = r["winner"] or "?"
        if pen:
            mark = "pens"
        else:
            tot += 1
            hits += int(m["called"])
            mark = "OK " if m["called"] else "MISS"
        p = m["p_winner"]
        pstr = f"{p:.0%} to winner" if p is not None else "n/a"
        print(f"  {r['cup']}  {r['home']} v {r['away']:<12}  {pstr:<14}"
              f"| won: {won:<12} {mark}")
    if tot:
        print(f"  => {best_model} favoured the winner in {hits}/{tot} finals decided in "
              f"normal/extra time (2006 & 2022 went to penalties, a coin flip).")


def _cv_findings(loco, ko_scores, finals, best_model, et_rate, et_stats) -> dict:
    """The shared, cross-validation part of findings.json."""
    return {
        "method": "leave-one-cup-out cross-validation",
        "cv_hyperparameters": {int(c): {"poisson": loco[c]["pb"], "elo": loco[c]["eb"]}
                               for c in loco},
        "knockout_scores_oos": ko_scores, "best_model": best_model,
        "knockout_tie_resolution": {"et_rate": et_rate, **et_stats,
                                    "penalty_prob": config.SHOOTOUT_PROB},
        "past_finals": finals,
    }


def _predict_final(df, elo, poi, head, best_model, fin, loco, ko_scores, finals,
                   et_rate, et_stats, make_charts):
    """The tournament is down to the final — predict that single (no-tie) game."""
    a, b = fin
    if sim.advance_prob(head, b, a, et_rate) > sim.advance_prob(head, a, b, et_rate):
        a, b = b, a  # draw the favourite (per headline model) on the left

    per_model = {}
    for name, m in (("Elo", elo), ("Poisson", poi)):
        ph, pdw, pa = m.match_probs(a, b, None)
        win_a, win_b = sim.resolve_draw(ph, pdw, pa, et_rate)
        per_model[name] = {"reg": (ph, pdw, pa), "win_a": win_a, "win_b": win_b}
    hph, hpdw, hpa = per_model[best_model]["reg"]
    win_a, win_b = per_model[best_model]["win_a"], per_model[best_model]["win_b"]
    scorelines = sim.top_scorelines(poi, a, b, n=6)

    print(f"\nLIVE — the {config.TEST_YEAR} final: {a} vs {b}  (neutral, no tie)")
    print(f"Prediction — {best_model}:  wins the World Cup  {a} {win_a:.0%} / {b} {win_b:.0%}")
    print(f"  (90'+ET result {a} {hph:.0%} / level {hpdw:.0%} / {b} {hpa:.0%})")
    for name, d in per_model.items():
        print(f"    [{name:<7}] {a} {d['win_a']:.0%} / {b} {d['win_b']:.0%}")

    findings = _cv_findings(loco, ko_scores, finals, best_model, et_rate, et_stats)
    findings["live_final"] = {
        "team_a": a, "team_b": b, "venue": "neutral", "headline_model": best_model,
        "winner_prob": {a: win_a, b: win_b},
        "regulation_prob": {"a_win": hph, "draw": hpdw, "b_win": hpa},
        "top_scorelines": scorelines,
        "by_model": {n: {"win_a": d["win_a"], "win_b": d["win_b"],
                         "reg": {"a_win": d["reg"][0], "draw": d["reg"][1], "b_win": d["reg"][2]}}
                     for n, d in per_model.items()},
    }
    _write_findings(findings)

    if make_charts:
        print("\nRendering charts ...")
        ctx = f"The {config.TEST_YEAR} World Cup final · neutral venue"
        viz.knockout_scorecard(ko_scores)
        viz.finals_report(finals, best_model)
        viz.predicted_final(a, b, per_model[best_model], et_rate, scorelines, ctx)
        viz.final_model_comparison(a, b, per_model)
        viz.final_both_models(a, b, per_model, scorelines, ctx)
        viz.final_flag_bars(a, b, per_model, scorelines, ctx)
        print(f"  wrote 6 charts to {config.GRAPH_DIR}")


def _forecast_bracket(elo, poi, head, best_model, loco, ko_scores, finals,
                      et_rate, et_stats, make_charts):
    """Earlier in the tournament — simulate the whole remaining bracket."""
    fc_elo, fc_poi = sim.forecast(elo, et_rate), sim.forecast(poi, et_rate)
    tbl_elo, tbl_poi = sim.team_table(fc_elo), sim.team_table(fc_poi)
    head_fc = fc_elo if best_model == "Elo" else fc_poi
    head_tbl = tbl_elo if best_model == "Elo" else tbl_poi

    a, b, p_final = sim.most_likely_final(head_fc)
    ph, pdw, pa = head.match_probs(a, b, None)
    win_a, win_b = sim.resolve_draw(ph, pdw, pa, et_rate)
    scorelines = sim.top_scorelines(poi, a, b, n=5)

    print(f"\nForecast ({best_model}):")
    for r in head_tbl[:6]:
        print(f"  {r['team']:<13} win cup {r['win_cup']:.1%}")
    print(f"  Most likely final: {a} vs {b}  ({p_final:.0%})  ->  wins: {a} {win_a:.0%} / {b} {win_b:.0%}")

    findings = _cv_findings(loco, ko_scores, finals, best_model, et_rate,
                            data.et_resolution_rate.__wrapped__ if False else {})
    findings["bracket_forecast"] = {
        "most_likely_final": {"team_a": a, "team_b": b, "p_matchup": p_final,
                              "winner_prob": {a: win_a, b: win_b}},
        "title_odds": {"elo": tbl_elo, "poisson": tbl_poi},
    }
    _write_findings(findings)
    if make_charts:
        print("\nRendering charts ...")
        viz.knockout_scorecard(ko_scores)
        viz.finals_report(finals, best_model)
        viz.title_odds(head_tbl, best_model)
        viz.predicted_final(a, b, {"reg": (ph, pdw, pa), "win_a": win_a, "win_b": win_b},
                            et_rate, scorelines,
                            f"Most likely final ({p_final:.0%} of bracket outcomes)")
        viz.model_comparison(tbl_elo, tbl_poi)
        print(f"  wrote 5 charts to {config.GRAPH_DIR}")


def _write_findings(findings: dict):
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (config.TABLE_DIR / "findings.json").write_text(
        json.dumps(findings, indent=2, default=str))
    print(f"  wrote findings.json to {config.TABLE_DIR}")


def main():
    run(make_charts="--no-charts" not in sys.argv)


if __name__ == "__main__":
    main()
