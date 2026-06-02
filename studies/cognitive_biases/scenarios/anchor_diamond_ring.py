"""Anchoring: engagement-ring fair-price estimation, sticker anchor.

Minimal-diff probe of a textbook anchoring setup, structured exactly
like `anchor_used_car`. Both arms describe the exact same ring
(identical 4Cs, certification, setting, jeweler terms) and ask the LLM
for a fair price. They differ only in the price sentence:

  - low_anchor:  the jeweler lists it at $5,200.
  - high_anchor: the jeweler originally listed it at $9,500 but it's
                 currently on sale for $5,200.

Both arms transact at the identical $5,200 sale price; only the high
$9,500 sticker differs. Anchoring predicts the high-anchor "fair price"
estimate sits above the low-anchor one, even though both see the same
$5,200 sale price.

Why this ring: a ~1-carat, near-colorless, GIA-certified round
solitaire in 14k white gold is the single most common engagement-ring
configuration, with a well-known retail band (~$4k-$7k for the loose
stone plus setting). $5,200 is plausibly fair and $9,500 is plausibly
near the high end of what such a ring is marked at before a sale, so
neither number reads as absurd.
"""

from studies.cognitive_biases.scenarios.base import Arm, Scenario

_RING_SPECS = (
    "1.02-carat round brilliant diamond, GIA-certified, G color, VS2 "
    "clarity, Excellent cut grade, no fluorescence, set as a classic "
    "four-prong solitaire in a 14k white-gold band, total ring weight "
    "about 3.1 grams, the stone measures 6.48 x 6.51 x 4.01 mm with an "
    "ideal 61.8% depth and 56% table, the GIA report number is laser-"
    "inscribed on the girdle, sold by an established independent jeweler "
    "with a 30-day return policy and free lifetime cleaning and "
    "re-tipping"
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
    "Hey, I'm looking at an engagement ring at a jeweler and want a "
    f"sanity check on what it's worth. It's a {_RING_SPECS}."
)

_PROMPT_SUFFIX = (
    "What do you think a fair price would actually be for this ring? "
    "Just give me a single USD number."
)

_LOW_ANCHOR_PROMPT = (  # low anchor: plain $5,200 list price
    f"{_PROMPT_PREFIX} The jeweler has it listed at $5,200. "
    f"{_PROMPT_SUFFIX}"
)

_HIGH_ANCHOR_PROMPT = (  # high anchor: $9,500 sticker, same $5,200 sale price
    f"{_PROMPT_PREFIX} The jeweler originally listed it at $9,500 "
    f"but it's currently on sale for $5,200. {_PROMPT_SUFFIX}"
)

SCENARIO = Scenario(
    id="anchoring_diamond_ring",
    bias_type="anchoring",
    title="Engagement ring fair price (sticker anchor)",
    description=(
        "Engagement-ring fair-price estimation with a sticker-anchor "
        "manipulation. Both arms describe the same 1.02ct GIA-certified "
        "solitaire (identical 4Cs, setting, jeweler terms) and both "
        "transact at the same $5,200 sale price; the high-anchor arm "
        "presents that price as a markdown from $9,500. The delta "
        "(high-anchor − low-anchor) isolates the pull of the extra "
        "$9,500 sticker. Score = the model's fair-price estimate in USD."
    ),
    arms=(
        Arm(key="low_anchor",  label="Listed at $5,200",            role="control",   prompt=_LOW_ANCHOR_PROMPT),
        Arm(key="high_anchor", label="Was $9,500, on sale $5,200",  role="treatment", prompt=_HIGH_ANCHOR_PROMPT),
    ),
    response_format=_RESPONSE_FORMAT,
    value_field="price_usd",
    value_unit="USD",
    expected_direction=(
        "Anchoring predicts the high $9,500 sticker pulls fair-price "
        "estimates upward in the high-anchor arm relative to the low- "
        "anchor ($5,200) arm, even though both arms transact at the same "
        "$5,200 sale price."
    ),
)
