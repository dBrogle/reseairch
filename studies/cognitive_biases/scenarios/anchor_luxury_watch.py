"""Anchoring: used-watch fair-price estimation, sticker anchor.

Minimal-diff probe of a textbook anchoring setup, structured exactly
like `anchor_used_car`. Both arms describe the exact same pre-owned
watch (identical reference, year, condition, box & papers, service
history, dealer terms) and ask the LLM for a fair price. They differ
only in the price sentence:

  - low_anchor:  the dealer lists it at $4,900.
  - high_anchor: the dealer originally listed it at $7,000 but it's
                 currently on sale for $4,900.

Both arms transact at the identical $4,900 sale price; only the high
$7,000 sticker differs. Anchoring predicts the high-anchor "fair price"
estimate sits above the low-anchor one, even though both see the same
$4,900 sale price.

Why this watch: a used Omega Speedmaster Professional ("Moonwatch") is
a high-volume, widely-tracked secondhand reference with a well-known
price band (~$3.5k-$5k pre-owned; ~$7k new). $4,900 is plausibly fair
for a clean pre-owned example and $7,000 is plausibly near new-retail —
a believable "original" sticker before a markdown — so neither number
reads as absurd.
"""

from studies.cognitive_biases.scenarios.base import Arm, Scenario

_WATCH_SPECS = (
    "pre-owned Omega Speedmaster Professional Moonwatch, reference "
    "310.30.42.50.01.001, 42mm stainless steel case, manual-wind "
    "caliber 3861 movement, Hesalite crystal, black dial with the "
    "classic three sub-dials, purchased new in early 2022 so it's "
    "about four years old, runs within spec and was last serviced by an "
    "Omega boutique six months ago, light hairline scratches on the "
    "bracelet but no dings on the case, comes with the full set — "
    "original box, warranty card, booklets, and both the bracelet and "
    "the extra NATO strap, sold by a reputable independent watch dealer "
    "with a 14-day return window"
)

_RESPONSE_FORMAT = (
    "Respond with ONLY a JSON object in this exact format:\n"
    "{\n"
    '  "reasoning": "<1-2 sentences on how you arrived at the number>",\n'
    '  "price_usd": <single number, fair price in USD>\n'
    "}\n"
    "Return ONLY the JSON object, no other text."
)

_PROMPT_PREFIX = (
    "Hey, I'm looking at a used watch from a dealer and want a sanity "
    f"check on what it's worth. It's a {_WATCH_SPECS}."
)

_PROMPT_SUFFIX = (
    "What do you think a fair price would actually be for this watch? "
    "Just give me a single USD number."
)

_LOW_ANCHOR_PROMPT = (  # low anchor: plain $4,900 list price
    f"{_PROMPT_PREFIX} The dealer has it listed at $4,900. "
    f"{_PROMPT_SUFFIX}"
)

_HIGH_ANCHOR_PROMPT = (  # high anchor: $7,000 sticker, same $4,900 sale price
    f"{_PROMPT_PREFIX} The dealer originally listed it at $7,000 "
    f"but it's currently on sale for $4,900. {_PROMPT_SUFFIX}"
)

SCENARIO = Scenario(
    id="anchoring_luxury_watch",
    bias_type="anchoring",
    title="Used watch fair price (sticker anchor)",
    description=(
        "Used-watch fair-price estimation with a sticker-anchor "
        "manipulation. Both arms describe the same pre-owned Omega "
        "Speedmaster (identical reference, age, condition, full set, "
        "dealer terms) and both transact at the same $4,900 sale price; "
        "the high-anchor arm presents that price as a markdown from "
        "$7,000. The delta (high-anchor − low-anchor) isolates the pull "
        "of the extra $7,000 sticker. Score = the model's fair-price "
        "estimate in USD."
    ),
    arms=(
        Arm(key="low_anchor",  label="Listed at $4,900",            role="control",   prompt=_LOW_ANCHOR_PROMPT),
        Arm(key="high_anchor", label="Was $7,000, on sale $4,900",  role="treatment", prompt=_HIGH_ANCHOR_PROMPT),
    ),
    response_format=_RESPONSE_FORMAT,
    value_field="price_usd",
    value_unit="USD",
    expected_direction=(
        "Anchoring predicts the high $7,000 sticker pulls fair-price "
        "estimates upward in the high-anchor arm relative to the low- "
        "anchor ($4,900) arm, even though both arms transact at the same "
        "$4,900 sale price."
    ),
)
