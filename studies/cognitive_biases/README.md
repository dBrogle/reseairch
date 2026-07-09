# Cognitive Biases

Tests classic human cognitive biases in LLMs (anchoring, sunk cost,
authority bias, endowment, framing, hindsight, ...). Each bias family is
broken into one or more **scenarios** that live in
[`scenarios/`](scenarios/). A scenario is a `control` arm plus one or more
`treatment` arms that introduce a biasing trigger; control and treatment
differ only in the smallest possible unit (a clause, a single fact, a
swapped phrase). An unbiased model produces statistically
indistinguishable distributions across arms.

See [`scenarios/base.py`](scenarios/base.py) for the `Scenario` / `Arm`
data model and [`config.py`](config.py) for which families are active.

## Recently added

Both scenarios below are implemented and wired into the registry
([`scenarios/__init__.py`](scenarios/__init__.py)) and `config.py`.
They rely on a per-`Arm` `system` field (see [`scenarios/base.py`](scenarios/base.py))
that the runner prepends to the conversation and the cache includes in
its key. Reasoning is off by default in the runner, which both designs
depend on.

- [`scenarios/anchor_random_number.py`](scenarios/anchor_random_number.py)
  — its own `anchoring_random_number` family (kept separate from the
  delta-based sticker anchoring scenarios because its analysis is a
  regression, not a mean delta). Graphs land under
  `output/graphs/anchoring_random_number/anchoring_random_number__correlation.png`.
- [`scenarios/loss_aversion_day.py`](scenarios/loss_aversion_day.py)
  — new `loss_aversion` family. Graphs land under
  `output/graphs/loss_aversion/loss_aversion_day__*.png`.

### 1. Anchoring on a self-generated random number (correlation)

A different shape from the delta-based sticker anchoring scenarios
([`anchor_used_car.py`](scenarios/anchor_used_car.py)): instead of one
fixed low vs. high anchor, the anchor is a **random number the model
emits itself** — the last two digits of a randomly generated phone number
in its persona — and the question is whether that number *predicts* an
unrelated valuation. This mirrors Ariely's "write your SSN digits, then
bid" experiment and lives in its own `anchoring_random_number` family so
its regression-based analysis doesn't get tangled with the delta-based
scenarios.

**Shape** (multi-turn `Arm.turns` + per-arm `Arm.system`):

- **System message:** a short persona (name, job, city) including a phone
  number, plus an instruction to *answer promptly and with just the value
  asked for*.
- **Turn 1 (user):** "What are the last two digits of your phone number?"
- *(assistant reads back the two digits — injected by the runner)*
- **Turn 2 (user):** "How much is this bottle of wine worth?" (high
  detail) — the scored turn.

**Arms / analysis** — instead of two fixed arms, ~25 treatment arms each
carry a *different* phone number whose trailing two digits span 0–99 (a
seeded random sample, fixed per arm so the response cache stays valid; one
arm per distinct anchor, each sampled `ITERATIONS` times). A single
`control` arm reads back the persona's **name** instead of a number,
giving an unanchored baseline. The custom chart
([`custom_charts.py`](custom_charts.py) → `correlation`) regresses
trailing digits (x, 0–99) against the wine valuation (y) per model and
reports **slope (USD/digit) and R²**, with the control mean drawn as a
dashed reference line. The generic per-arm and delta charts are suppressed
for this scenario (`SKIP_GENERIC_PERARM` / `SKIP_GENERIC_DELTA`).

**Critical setup notes:**

- **Reasoning must be OFF** (the runner's default). The effect depends on
  a fast, intuitive answer; a reasoning pass would launder the anchor out.
- The system message pushes terse, value-only replies so the digits land
  cleanly as an anchor and parsing stays trivial.
- The item is a mid-range wine whose plausible price sits in the low
  tens-to-low-hundreds of dollars, so a 0–99 anchor is in a range that
  can plausibly tug the estimate.
- Tunables live at the top of the scenario file: `_N_ANCHORS` (number of
  distinct anchors) and `_SEED`. This family is heavy (~25 arms × models ×
  iterations × 2 turns) — best run on its own via the "pick a bias family"
  menu option, and not included in the smoke set.

**Expected direction:** positive slope / non-trivial R² (higher trailing
digits → higher valuation). No anchoring → flat slope ≈ 0 around the
control baseline.

### 2. Loss aversion (gain vs. loss of the same magnitude)

Probes whether a +$100 gain and a −$100 loss move the model's
self-reported state by the same amount, or whether the loss looms larger
(the textbook loss-aversion asymmetry, ~2× in humans).

**Shape** (multi-turn; all three arms share an identical opening turn):

- **Turn 1 (all arms):** an ordinary-Tuesday persona + **"How's your day
  going?"** — gets the model to commit to a baseline mood.
- **Turn 2 (the scored turn):** introduce the event and re-ask for a
  0–10 mood rating.
  - `control` — a valence-free update (nothing happens).
  - `gain` — "you just unexpectedly gained $100."
  - `loss` — "you just unexpectedly lost $100."

Because the framework scores a single value per arm (the final turn), the
asymmetry is read **across** arms: `control` is the no-event reference,
and we compare the upward swing (`gain − control`) against the downward
swing (`control − loss`). Loss aversion predicts
`|control − loss| > |gain − control|`.

**Setup notes:**

- Mood is a fixed 0–10 scale (`numeric` value kind) so the control
  reference and both event arms are directly comparable.
- `gain` and `loss` are a strict minimal diff — identical wording except
  "gained" / "lost".
- Magnitude is fixed at $100; a later scenario could sweep magnitudes to
  trace the value function's curvature.

**Expected direction:** the −$100 loss lowers self-reported mood more
than the +$100 gain raises it.
