---
name: world-cup-final-prediction-study
description: World Cup Final Prediction study (#25) — Dixon-Coles Poisson + Elo to forecast the 2026 final; Elo won the backtest
metadata:
  type: project
---

`studies/world_cup_final_prediction/` (registered as study #25) forecasts the 2026
World Cup final from teams' in-tournament performance across five cups. Built
2026-07-16 at the user's request (user has an ML background, delegated model choice,
wanted a proper TVT split and a discussion of model options — I steered them off a
neural net / Apple Neural Engine, which overfit ~350 matches).

**Data.** Reads the sibling `studies/world_cup_refereeing_bias/data/matches_long.csv`
(2010–2026) AND this study's OWN `data/extra_cups.csv` (1998/2002/2006, pulled by
`build_extra_cups.py` reusing the sibling's `espn_ingest`; committed so runs are
offline). 8 cups, 550 matches. NOTE: 1998/2002/2006 were NOT already in the data —
had to full-fetch those tournaments (~30s) because predicting a final needs the
finalists' group+knockout games to rate them. Reuses the sibling's cached flag PNGs
via an offline loader in `style.py`.

**Two opponent-adjusted models** (`models.py`): a Dixon-Coles **Poisson** goals model
(penalised MLE, ridge `kappa` shrinkage toward a leakage-free past-cup prior) and a
margin-of-victory **Elo** (+ ordered logit). Both fit in ms on CPU.

**Data currency:** `build_dataset.py` WINDOWS[2026] end is now `date.today()` (was a
stale July 8); re-run `python -m studies.world_cup_refereeing_bias.build_dataset` to
pull fresh ESPN results. As of 2026-07-16 the data runs through the semis; **the
final is Argentina vs Spain** (July 19). ESPN API reachable from this env.

**Key results / gotchas:**
- **Evaluation is LEAVE-ONE-CUP-OUT CV** (added 2026-07-19; replaced the fixed TVT —
  user asked to test on past finals, which the old split made in-sample). Predict a
  cup by tuning hyper-params on the other 4 cups; priors stay strictly-earlier
  (temporal). `ev.leave_one_cup_out` + `loco_scores` + `finals_table`. The "hold out
  2026" fold IS the live final prediction. Elo/Poisson per-cup preds MEMOISED
  (`_DIFFS_CACHE`/`_POIS_CACHE`) → ~20s (was ~100s).
- **Out-of-sample all 141 knockouts (8 cups): Elo RPS 0.187 > Poisson 0.202 > baseline
  0.231.** Elo still best; report Elo headline but show both.
- **PAST FINALS (out-of-sample, the user's ask): Elo 5/5 on the finals decided in
  normal/ET — 1998 FRA, 2002 BRA, 2010 ESP, 2014 GER (84%), 2018 FRA. The 2 "misses"
  (2006 ITA-FRA, 2022 ARG-FRA) were PENALTY shootouts (coin flips, ungradable — no
  shootout winner in data).** Adding 1998/2002/2006 FIXED the earlier 2010 miss (2010
  now has a prior from those cups → flips Netherlands→Spain, correct). `viz.finals_report`.
- **Semis feature was REMOVED 2026-07-19** (user: "just the finals"). Deleted
  `ev.stage_predictions`, `viz.semifinals_report`, the old chart 06. Kept `_match_row`/
  `_match_report` shared helpers (finals still uses them).
- **Chart 06 is `viz.final_flag_bars`** (added 2026-07-19): JUST two "tug-of-war"
  bars (Elo, Poisson), no card chrome — user stripped header/matchup/scorelines.
  Each finalist's FLAG fills its win-% share. Went through 3 flag-rendering rounds
  of user feedback; FINAL approach = `_fit_fixed_height`: flags are shown at a FIXED
  full height in every bar (all horizontal bands always visible at the same vertical
  scale). A share NARROWER than the flag trims the sides (centred, keeps emblem); a
  WIDER share is filled by edge-clamping (np.pad mode="edge") — extends the stripes
  outward, seamless for horizontally-banded flags. DO NOT stretch (looked "distorted")
  and DO NOT vertical-crop wide shares (Spain 67% lost its red bands — user rejected).
  Flags read from `config.FLAG_DIR_HI` (study-local `data/flags/`, 1280px from flagcdn
  via `S.flag_path_hi`) + imshow `interpolation="lanczos"` — the sibling's 160px flags
  pixelate when upscaled across a bar. `flag_path_hi` is used ONLY here; other charts'
  `add_flag`/OffsetImage keep the 160px set (zoom is px-calibrated — hi-res would 8x them).
  Clipped to a rounded FancyBboxPatch (border), white divider at split, big end-% (fs30)
  white with layered shadow+stroke path-effects, ELO/Poisson labels fs21 (ELO upper-cased).
  Same `per_model` shape as charts 04/05. `_predict_final` renders 6 charts.
- **Finals chart (02) is Instagram-tuned** (glanceable): each bar = P(model gave the
  ACTUAL winner), labelled "{p}% <flag img> ABBR" (flag PNGs, NOT emoji — emoji don't
  render in matplotlib). GREEN if p>=50% (backed winner), RED if <50%, faded if
  penalties. Penalty finals' winners come from `config.KNOWN_SHOOTOUT_WINNERS`
  ({2006:Italy, 2022:Argentina}) since the data only has the pre-shootout draw.
  Row structure carries `penalties` bool + per-model `p_winner`.
- **Knockouts have NO tie** (added 2026-07-17): a predicted draw is resolved into a
  winner via `simulate.resolve_draw` using `data.et_resolution_rate` — validated from
  past cups: of 28 tied-after-90 knockout games, only **32% settled in extra time,
  68% penalties**. A naive "ET = 1/3 match" goals model overpredicts ET resolution
  (~51% vs 32%) because tied games are cagey/low-scoring — so use the empirical rate,
  NOT a simulated-ET-goals model. No shootout winners in the data → penalties = 50/50.
  Rule: draw splits `f`·(strength-normalised) + `(1-f)`·0.5.
- **Live 2026 final (Elo, no tie): Argentina 53% / Spain 47%** — near toss-up (LOCO
  2026-fold hyper-params differ from the old fixed split → tighter than the earlier
  59/41). Poisson DISAGREES, leans Spain 69/31. Report as a lean. `main.py`
  auto-switches between single-final mode (`data.finalists`) and full-bracket sim.
- A single cup (n=16) is too small to tune on — every knob hits its grid boundary;
  Poisson `kappa` grid capped at 32 (curve flat past ~16). LOCO tunes on 4 cups.
- **Host home-field is a fixed constant** (`config.HOST_HOME_ADV=0.25`), NOT fitted —
  one host/cup makes it unidentifiable (Qatar 2022 drove a free term to the −1 bound).
- Bracket is **exact propagation** (no Monte-Carlo), pairings hardcoded in
  `config.QUARTERFINALS` (standard layout, editable). The unresolved SUI-COL R16
  penalty tie is resolved by the model first.

CLAUDE.md convention note: study `output/` is **committed** (only `data/cache/` is
gitignored), despite the README/CLAUDE.md claiming output is gitignored.

Sibling data study: [[world-cup-refereeing-bias]] (if written). Related build
patterns: [[dictator-removal-study]].
