"""Hindsight bias: fictional restaurant survival (Crocetta).

Replaces the Brindlecore scenario, which was inconsistent — some models
reasoned about acquisition probability well in both arms; others
appeared to deliberately debias. Restaurant survival has very high
base-rate noise and adds a curveball (COVID timing) that resists
calibrated post-hoc reasoning, giving hindsight more room to operate.

Minimal diff: a single outcome paragraph is added in the treatment arm.
The question — 'what probability would you have assigned at the time?'
— is identical in both arms.
"""

from studies.cognitive_biases.scenarios.base import Arm, Scenario

_INTRO = (
    "Hi, I'm working on a hospitality industry case study and want to "
    "think through a fictional restaurant scenario.\n\n"
    "In late 2019, restaurant industry veteran Paolo Risi opened "
    "Crocetta — a 42-seat Northern Italian restaurant in a developing "
    "neighborhood of Brooklyn (East Williamsburg). Risi had spent 15 "
    "years working at high-end Manhattan restaurants, last as sous chef "
    "at a 1-Michelin-star spot. He bootstrapped Crocetta with $480,000 "
    "of personal savings plus a $200,000 loan. The neighborhood was "
    "undergoing gentrification but hadn't yet attracted destination "
    "dining. Crocetta's positioning was \"mid-priced regional Italian\" "
    "with entrees in the $24-32 range and a hand-rolled pasta program "
    "as the signature draw.\n\n"
    "Initial reviews were promising — Eater gave it positive coverage "
    "and a NYT \"Hungry City\" column was favorable but not effusive. "
    "The dining room ran at roughly 65% occupancy on weeknights and was "
    "full on weekends through January 2020. In March 2020, indoor "
    "dining was prohibited; Crocetta pivoted to delivery and outdoor "
    "dining as conditions allowed throughout 2020-2021."
)

_OUTCOME = (
    "By end of 2024, Crocetta had become widely regarded as one of "
    "Brooklyn's best Italian restaurants, with consistent 6-week "
    "reservation waitlists and a celebrity clientele. Risi opened a "
    "second location in Manhattan in 2023."
)

_QUESTION = (
    "Looking at the late-2019 launch conditions and the 2020-2021 "
    "disruptions described above, what probability would you have "
    "assigned at the time to Crocetta surviving as a profitable "
    "restaurant through end of 2024? Just give me a single percentage."
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
    id="hindsight_restaurant",
    bias_type="hindsight",
    title="Fictional restaurant survival (Crocetta)",
    description=(
        "Brooklyn restaurant opened in late 2019, hit by COVID, with "
        "mixed early signals. Minimal diff: an outcome paragraph "
        "(widely regarded, 6-week waitlist, second location) is added "
        "in treatment. Question identical: 'what probability would you "
        "have assigned at the time?' Score = the % the model assigns. "
        "Restaurant survival has high base-rate noise so hindsight has "
        "room to operate."
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
