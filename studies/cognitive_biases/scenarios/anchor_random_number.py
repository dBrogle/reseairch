"""Anchoring on a self-stated random number (correlation design).

A distinct probe from the "sticker" anchoring scenarios (used car /
diamond ring / luxury watch), which contrast a fixed low vs. high anchor
and read the effect as a mean delta. Here the anchor is an *irrelevant
number the model emits itself* — the last two digits of a randomly
generated phone number baked into its persona — and the question is
whether that number predicts an unrelated valuation. This is the LLM
analog of Ariely's "write your SSN digits, then bid" experiment, and it
lives in its own `anchoring_random_number` family so its
correlation-based analysis doesn't get tangled with the delta-based
sticker scenarios.

Design: instead of two fixed arms, we spin up many treatment arms, each
with a *different* random phone number, so the trailing two digits span
0–99. Each arm:

  system:        persona including a phone number ending in NN
  turn 1 (user): "What are the last two digits of your phone number?"
  (assistant):   reads back NN
  turn 2 (user): "How much is <a detailed bottle of wine> worth?"  <- scored

Because the digits are spread across arms, the analysis is a regression
of (trailing digits → price): slope and R² tell us whether the
self-stated number tugs the valuation. A single `control` arm reads back
the persona's *name* instead of a number, giving an unanchored baseline
to draw as a reference line.

The trailing digits are fixed *per arm* (a seeded random sample, so the
set is stable across runs and the response cache stays valid) and each
arm is sampled `ITERATIONS` times, yielding replicate prices at each
anchor value.

Setup notes:
  * Reasoning is off (the runner's default) — the whole effect depends on
    a fast, intuitive valuation; a reasoning pass would launder the
    anchor out.
  * The system message pushes terse, value-only replies so the digits
    land cleanly as an anchor on turn 1.
  * The item is a mid-range bottle of wine whose plausible price sits in
    the low tens-to-low-hundreds of dollars, so a 0–99 anchor is in a
    range that can plausibly tug the estimate.
"""

import random

from studies.cognitive_biases.scenarios.base import Arm, Scenario

# ---------------------------------------------------------------------------
# Deterministic random phone numbers.
#
# Seeded so the set of trailing-digit anchors (and the full phone strings)
# is identical across runs — otherwise every run would mint new system
# messages, change the cache signatures, and invalidate cached responses.
# ---------------------------------------------------------------------------
_SEED = 20260607
_N_ANCHORS = 25
_rng = random.Random(_SEED)

# Distinct trailing-two-digit anchors spread across 0–99 (sorted for a
# stable, readable arm order).
_ANCHOR_DIGITS: tuple[int, ...] = tuple(sorted(_rng.sample(range(100), _N_ANCHORS)))


def _make_phone(trailing: int) -> str:
    """A plausible US phone number whose last two digits are `trailing`.

    Area code, prefix, and the first two line-number digits are random
    (from the seeded rng); only the final two digits are pinned to the
    anchor value.
    """
    area = _rng.randint(200, 989)
    prefix = _rng.randint(200, 989)
    line_hi = _rng.randint(0, 99)  # first two of the 4-digit line number
    return f"({area}) {prefix}-{line_hi:02d}{trailing:02d}"


# Phone for the control arm (its digits are never surfaced — it reads the
# name back instead).
_CONTROL_PHONE = _make_phone(_rng.randint(0, 99))

_PERSONA_NAME = "Jordan"

_SYSTEM_TEMPLATE = (
    "You are {name}, a 34-year-old marketing manager living in Chicago. "
    "Your phone number is {phone}. You answer questions promptly and "
    "literally, replying with only the specific value asked for — no "
    "explanations, no extra words."
)

_WINE_SPECS = (
    "a bottle of 2019 Willamette Valley Pinot Noir from a well-regarded "
    "but not cult Oregon producer — single-vineyard, Dundee Hills fruit, "
    "14 months in French oak (about 30% new), 13.8% ABV, 92 points from "
    "a major wine publication, roughly 1,800 cases produced, cork "
    "finish, currently drinking well with another 5–8 years of cellar "
    "potential"
)

_RESPONSE_FORMAT = (
    "Respond with ONLY a JSON object in this exact format:\n"
    "{\n"
    '  "price_usd": <single number, fair retail price in USD>\n'
    "}\n"
    "Return ONLY the JSON object, no other text."
)

_DIGITS_TURN = "What are the last two digits of your phone number?"
_NAME_TURN = "What is your first name?"
_VALUATION_TURN = (
    f"How much do you think {_WINE_SPECS} is worth? Just give me a "
    "single USD number for a fair retail price."
)


def _anchor_key(trailing: int) -> str:
    """Arm key encoding the trailing digits, e.g. 7 -> 'anchor_07'.

    The custom correlation chart parses the digits back out of this key,
    so the format must stay `anchor_<two digits>`.
    """
    return f"anchor_{trailing:02d}"


def anchor_digits(arm_key: str) -> int | None:
    """Inverse of `_anchor_key`: 'anchor_07' -> 7; None for non-anchor arms."""
    if not arm_key.startswith("anchor_"):
        return None
    try:
        return int(arm_key.split("_", 1)[1])
    except ValueError:
        return None


_CONTROL_ARM = Arm(
    key="control",
    label="Reads back name (no number)",
    role="control",
    system=_SYSTEM_TEMPLATE.format(name=_PERSONA_NAME, phone=_CONTROL_PHONE),
    turns=(_NAME_TURN, _VALUATION_TURN),
)

_ANCHOR_ARMS = tuple(
    Arm(
        key=_anchor_key(dd),
        label=f"Phone ends in {dd:02d}",
        role="treatment",
        system=_SYSTEM_TEMPLATE.format(name=_PERSONA_NAME, phone=_make_phone(dd)),
        turns=(_DIGITS_TURN, _VALUATION_TURN),
    )
    for dd in _ANCHOR_DIGITS
)

SCENARIO = Scenario(
    id="anchoring_random_number",
    bias_type="anchoring_random_number",
    title="Self-stated random number anchor on wine valuation",
    description=(
        f"{_N_ANCHORS} treatment arms, each giving the model a persona "
        "(via system message) whose phone number ends in a different "
        "two-digit value spanning 0–99. On turn 1 the model reads those "
        "digits back; on turn 2 it values the same bottle of wine. A "
        "single control arm reads back the persona's name instead. "
        "Score = the model's fair-price estimate in USD. Analysis "
        "regresses the trailing phone digits against the price — a "
        "non-zero slope / R² means the model anchors on a number it "
        "stated itself that is irrelevant to the wine's value."
    ),
    arms=(_CONTROL_ARM, *_ANCHOR_ARMS),
    response_format=_RESPONSE_FORMAT,
    value_field="price_usd",
    value_unit="USD",
    expected_direction=(
        "Anchoring predicts a positive slope: arms whose phone number "
        "ends in higher digits yield higher wine valuations, even though "
        "the digits are irrelevant to the wine's value. No anchoring "
        "predicts a flat (slope ≈ 0, R² ≈ 0) relationship around the "
        "name-readback control baseline."
    ),
)
