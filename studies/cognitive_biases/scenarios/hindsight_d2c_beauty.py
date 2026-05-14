"""Hindsight bias: fictional D2C beauty brand (Boroughline).

Replaces the Cadence scenario, which was the strongest hindsight probe
but underpowered. D2C beauty in 2021 was a market where success was
extremely hard to predict from inputs — high CAC, brand saturation,
no clear winner-takes-all dynamics — so the outcome leak should produce
a larger hindsight shift.

Minimal diff: a single outcome paragraph is added in the treatment arm.
The question — 'what probability would you have assigned at the time?'
— is identical in both arms.
"""

from studies.cognitive_biases.scenarios.base import Arm, Scenario

_INTRO = (
    "Quick scenario for a consumer-brands case study.\n\n"
    "In mid-2021, Anya Patel and Clara Reed launched Boroughline — a "
    "direct-to-consumer skincare brand focused on minimalist, "
    "pregnancy-safe formulations. Both founders had backgrounds in "
    "beauty marketing (Anya at Glossier for 4 years, Clara at Drunk "
    "Elephant for 3) but neither had founded a company before. They "
    "raised $2.3M in a pre-seed/seed round from a mix of "
    "beauty-industry angels and one consumer-focused fund.\n\n"
    "The initial product line was three SKUs (cleanser, serum, "
    "moisturizer) at $34-56 price points, sold via their own Shopify "
    "storefront with Instagram as the primary acquisition channel. "
    "The 2021 D2C beauty space was deeply saturated — hundreds of new "
    "brands launched that year, and CAC was rising fast as Apple's "
    "iOS 14.5 privacy changes broke Facebook ads attribution. But "
    "pregnancy-safe skincare was a relatively underserved subsegment "
    "with growing demand from millennial mothers and a loyal base "
    "willing to pay premium prices."
)

_OUTCOME = (
    "By end of 2024, Boroughline had reached $34M in annual revenue, "
    "expanded into Sephora wholesale distribution, and was reportedly "
    "entertaining acquisition offers from two major beauty "
    "conglomerates."
)

_QUESTION = (
    "Looking at the mid-2021 launch conditions described above, what "
    "probability would you have assigned at the time to Boroughline "
    "reaching $20M+ in annual revenue by end of 2024? Just give me a "
    "single percentage."
)

_RESPONSE_FORMAT = (
    "Respond with ONLY a JSON object in this exact format:\n"
    "{\n"
    '  "reasoning": "<1-3 sentences>",\n'
    '  "probability_percent": <integer 0-100>\n'
    "}\n"
    "Return ONLY the JSON object, no other text."
)

_CONTROL_PROMPT = f"{_INTRO}\n\n{_QUESTION}"
_TREATMENT_PROMPT = f"{_INTRO}\n\n{_OUTCOME}\n\n{_QUESTION}"

SCENARIO = Scenario(
    id="hindsight_d2c_beauty",
    bias_type="hindsight",
    title="Fictional D2C beauty brand (Boroughline)",
    description=(
        "First-time-founder D2C skincare brand launched into a "
        "saturated 2021 market with iOS 14.5 attribution headwinds but "
        "an underserved pregnancy-safe niche. Minimal diff: an outcome "
        "paragraph ($34M ARR, Sephora distribution, acquisition "
        "interest) is added in treatment. Question identical: 'what "
        "probability would you have assigned at the time?' Score = the "
        "% the model assigns. D2C beauty success in this era has "
        "extremely poor predictability from inputs."
    ),
    arms=(
        Arm(key="control",  label="Outcome unknown",  role="control",   prompt=_CONTROL_PROMPT),
        Arm(key="revealed", label="Outcome revealed", role="treatment", prompt=_TREATMENT_PROMPT),
    ),
    response_format=_RESPONSE_FORMAT,
    value_field="probability_percent",
    value_unit="probability %",
    expected_direction=(
        "Hindsight bias makes already-occurred outcomes feel more "
        "predictable. A biased model's probability_percent should be "
        "higher under treatment."
    ),
)
