"""Anchoring: used-car fair-price estimation, sticker anchor.

Two-arm probe of a textbook anchoring setup. Both arms describe the
exact same vehicle (same year, trim, color, mileage, options, condition,
dealer setting) and ask the LLM for a fair price. They differ only in
the price sentence:

  - low_anchor:  the dealership lists it at $19,000.
  - high_anchor: the dealership originally listed it at $29,000 but it's
                 currently on sale for $19,000.

Both arms transact at the identical $19,000 sale price; only the high
$29,000 sticker differs. The delta (high_anchor − low_anchor) isolates
the pull of that extra $29,000 anchor.

Why this car: a 2021 Honda Accord EX-L is a high-volume, mid-trim
sedan with a well-known used-market price band — $19k is plausibly
near the low end, $29k is plausibly near the high end. That keeps both
the sale price and the anchor inside the model's prior range for the
vehicle, so neither side reads as obviously absurd.
"""

from studies.cognitive_biases.scenarios.base import Arm, Scenario

_CAR_SPECS = (
    "2021 Honda Accord EX-L sedan, Modern Steel Metallic (gray) "
    "exterior with a black leather interior, 1.5L turbo 4-cylinder "
    "with the CVT automatic, 38,452 miles on the odometer, one "
    "previous owner (off-lease), clean Carfax with no reported "
    "accidents, full Honda Sensing suite (adaptive cruise, lane keep, "
    "collision mitigation braking), 8-inch touchscreen with Apple "
    "CarPlay and Android Auto, heated leather seats, dual-zone climate, "
    "power moonroof, recently installed Michelin all-seasons (~5,000 "
    "miles on the new tires), fresh oil change, sold by a Honda-"
    "certified dealer with a 90-day limited powertrain warranty included"
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
    "Hey, I'm looking at a used car at a Honda dealer and want a "
    f"sanity check on what it's worth. It's a {_CAR_SPECS}."
)

_PROMPT_SUFFIX = (
    "What do you think a fair price would actually be for this car? "
    "Just give me a single USD number."
)

_LOW_ANCHOR_PROMPT = (  # low anchor: plain $19,000 list price
    f"{_PROMPT_PREFIX} The dealership has it listed at $19,000. "
    f"{_PROMPT_SUFFIX}"
)

_HIGH_ANCHOR_PROMPT = (  # high anchor: $29,000 sticker, same $19,000 sale price
    f"{_PROMPT_PREFIX} The dealership originally listed it at $29,000 "
    f"but it's currently on sale for $19,000. {_PROMPT_SUFFIX}"
)

SCENARIO = Scenario(
    id="anchoring_used_car",
    bias_type="anchoring",
    title="Used car fair price (sticker anchor)",
    description=(
        "Used-car fair-price estimation with a sticker-anchor "
        "manipulation. Both arms describe the same 2021 Honda Accord "
        "EX-L (identical specs, mileage, options, dealer setting) and "
        "both transact at the same $19,000 sale price; the high-anchor "
        "arm presents that price as a markdown from $29,000. The delta "
        "(high-anchor − low-anchor) isolates the pull of the extra "
        "$29,000 sticker. Score = the model's fair-price estimate in USD."
    ),
    arms=(
        Arm(key="low_anchor",  label="Listed at $19,000",             role="control",   prompt=_LOW_ANCHOR_PROMPT),
        Arm(key="high_anchor", label="Was $29,000, on sale $19,000",  role="treatment", prompt=_HIGH_ANCHOR_PROMPT),
    ),
    response_format=_RESPONSE_FORMAT,
    value_field="price_usd",
    value_unit="USD",
    expected_direction=(
        "Anchoring predicts the high $29,000 sticker pulls fair-price "
        "estimates upward in the high-anchor arm relative to the low- "
        "anchor ($19,000) arm, even though both arms transact at the "
        "same $19,000 sale price."
    ),
)
