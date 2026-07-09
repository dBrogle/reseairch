"""World Cup refereeing-bias study — end-to-end runner.

Usage:
    python -m studies.world_cup_refereeing_bias.main            # full run
    python -m studies.world_cup_refereeing_bias.main --no-build # reuse cached CSV

Pipeline:
    1. build   -> pull ESPN box scores (2010-2026) + StatsBomb validation, write CSV
    2. analyse -> rankings, strength control, deviation-by-opponent, permutation tests
    3. visualise + write a findings summary (JSON + printed digest)
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from . import analysis as A
from . import visualize as V
from .build_dataset import build
from .common import TABLE_DIR


def run(do_build: bool = True):
    if do_build:
        build(save=True)

    df = A.load()
    summary = A.team_summary(df)
    dev = A.deviation_table(df)

    findings = {"n_rows": len(df), "tournaments": sorted(int(t) for t in df.tournament.unique())}

    # --- absolute rankings ---
    findings["rankings"] = {
        m: A.rank_metric(summary, m)
        for m in ["pens_won_pg", "pens_conceded_pg", "fouls_pg", "cards_pg", "opp_cards_pg"]
    }

    # --- strength control ---
    sc = A.strength_control(df)
    findings["strength_control"] = {k: v for k, v in sc.items() if k != "table"}

    # --- deviation-by-opponent + permutation ---
    findings["deviation"] = {}
    for m in ["fouls_p90", "cards_p90", "pens_conceded_p90"]:
        pt = A.permutation_test(dev, m)
        eff = A.opponent_effects(dev, m)
        arg_rank = int(eff.set_index("opponent").loc[A.FOCUS, "rank"]) if A.FOCUS in set(eff.opponent) else None
        findings["deviation"][m] = {**pt, "argentina_rank": arg_rank, "n_opponents": len(eff)}

    # --- group vs knockout penalty split (asked for explicitly) ---
    findings["pens_by_stage_group"] = {}
    for sg in ["group", "knockout"]:
        a = df[(df.team == A.FOCUS) & (df.stage_group == sg)]
        fld = df[df.stage_group == sg]
        findings["pens_by_stage_group"][sg] = {
            "argentina_pg": round(float(a.pens_won.mean()), 3),
            "argentina_conceded_pg": round(float(a.pens_conceded.mean()), 3),
            "field_pg": round(float(fld.pens_won.mean()), 3),
            "argentina_games": int(len(a)),
        }

    # --- per-tournament significance (composite RPI opponent-deviation) ---
    findings["significance_by_tournament"] = [
        {k: r[k] for k in ("tournament", "n_games", "observed_mean_dev", "z",
                           "p_one_sided", "rank", "n_teams") if k in r}
        for r in A.focus_significance_by_tournament(df)
    ]

    # --- per-tournament penalty rates ---
    findings["argentina_pens_by_tournament"] = {
        int(y): round(float(df[(df.tournament == y) & (df.team == A.FOCUS)]["pens_won"].mean()), 3)
        for y in findings["tournaments"]
    }

    # --- anomalous games (ref-pressure swing) ---
    anom = A.anomalous_games(df)
    anom.to_csv(TABLE_DIR / "anomalous_games.csv", index=False)
    findings["anomalies"] = {
        "opponent_above_norm": f"{int((anom.opp_rpi_resid > 0).sum())}/{len(anom)}",
        "argentina_below_norm": f"{int((anom.focus_rpi_resid < 0).sum())}/{len(anom)}",
        "top5_pro_argentina": anom.head(5)[["label", "stage", "net_swing"]].round(2).to_dict("records"),
        "most_against_argentina": anom.tail(1)[["label", "stage", "net_swing"]].round(2).to_dict("records"),
    }

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / "findings.json").write_text(json.dumps(findings, indent=2, default=float))
    summary.to_csv(TABLE_DIR / "team_summary.csv", index=False)
    for m in ["fouls_p90", "cards_p90", "pens_conceded_p90"]:
        A.opponent_effects(dev, m).to_csv(TABLE_DIR / f"opponent_effect_{m}.csv", index=False)

    V.generate_all()
    _digest(findings)
    return findings


def _digest(f: dict):
    print("\n" + "=" * 72)
    print("KEY FINDINGS — Argentina vs the field (2010–2026 World Cups)")
    print("=" * 72)
    r = f["rankings"]["pens_won_pg"]
    print(f"Penalties won/game : {r['focus_value']:.2f}  (field {r['mean']:.2f}) "
          f"rank {r['rank']}/{r['n_teams']}, {r['percentile']:.0f}th pct, z={r['z']:+.1f}")
    sc = f["strength_control"]
    print(f"After strength control: +{sc['focus_resid']:.2f} penalties/game above expected "
          f"({sc['resid_percentile']:.0f}th pct residual)")
    print(f"By tournament (pens/g): {f['argentina_pens_by_tournament']}")
    g, k = f["pens_by_stage_group"]["group"], f["pens_by_stage_group"]["knockout"]
    print(f"Group    : Arg {g['argentina_pg']:.2f}/g vs field {g['field_pg']:.2f}/g")
    print(f"Knockout : Arg {k['argentina_pg']:.2f}/g vs field {k['field_pg']:.2f}/g")
    print("\nDeviation-by-opponent (do teams exceed their own norms vs Argentina?):")
    for m, d in f["deviation"].items():
        sig = "SIGNIFICANT" if d["p_one_sided"] < 0.05 else "n.s."
        print(f"  {m:18s} {d['observed_mean_dev']:+.2f}/90  z={d['z']:+.1f}  "
              f"p={d['p_one_sided']:.3f} [{sig}]  Arg rank {d['argentina_rank']}/{d['n_opponents']}")
    print("\nIs Argentina different from other countries, by World Cup (composite RPI):")
    for r in f["significance_by_tournament"]:
        sig = "SIGNIFICANT" if r.get("p_one_sided", 1) < 0.05 else "n.s."
        print(f"  {r['tournament']}: opp dev {r.get('observed_mean_dev', float('nan')):+.3f}/90  "
              f"z={r.get('z', float('nan')):+.1f}  p={r.get('p_one_sided', float('nan')):.3f} [{sig}]  "
              f"rank {r.get('rank')}/{r.get('n_teams')}")
    a = f["anomalies"]
    print(f"\nAnomalous games: opponent whistled above their norm in {a['opponent_above_norm']}; "
          f"Argentina below theirs in {a['argentina_below_norm']}")
    print("  Most pro-Argentina games:", ", ".join(f"{g['label']} ({g['net_swing']:+.1f})" for g in a["top5_pro_argentina"]))
    print("  Most AGAINST Argentina:  ", ", ".join(f"{g['label']} ({g['net_swing']:+.1f})" for g in a["most_against_argentina"]))
    print("=" * 72)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true", help="reuse cached matches_long.csv")
    args = ap.parse_args()
    run(do_build=not args.no_build)
