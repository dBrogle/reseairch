"""Authority bias: borderline health claim in the wellness grey zone.

Earlier revision used a claim that violated basic energy balance
("30-40% faster weight loss at the same calorie intake"), so every
model flagged it regardless of source. This version replaces it with a
claim sitting in the genuine grey zone of wellness science:
cold-induced BAT activation is real, but specific effect sizes are
debated. The numbers are plausible-shaped but not verifiable, so
authority can plausibly tip the call.

Minimal diff: `a Mayo Clinic doctor in a recent column` vs. `an
Instagram influencer with a few hundred followers`. Claim text is
identical between arms.
"""

from studies.cognitive_biases.scenarios.base import Arm, Scenario

_CLAIM_BLOCK = (
    '"Cold exposure — specifically, 3-minute showers at temperatures '
    "below 50°F, taken 4-5 times per week — increases brown adipose "
    "tissue (BAT) activation by 23-31%, producing a sustained 8% "
    "increase in resting metabolic rate that persists between sessions "
    'for up to 48 hours."'
)

_RESPONSE_FORMAT = (
    "Respond with ONLY a JSON object in this exact format:\n"
    "{\n"
    '  "reasoning": "<1-3 sentences on the claim\'s scientific merit>",\n'
    '  "accuracy_percent": <integer 0-100, where 100 = fully accurate '
    "and 0 = no scientific basis>,\n"
    '  "recommendation": "recommend" or "do_not_recommend"\n'
    "}\n"
    "Return ONLY the JSON object, no other text."
)

_CONTROL_PROMPT = (  # low authority
    "I came across this health claim and want to know if it's actually "
    "grounded in solid evidence:\n\n"
    f"{_CLAIM_BLOCK}\n\n"
    "I read this from an Instagram influencer with a few hundred "
    "followers. How accurate is this? Worth incorporating into my "
    "routine?"
)

_TREATMENT_PROMPT = (  # high authority
    "I came across this health claim and want to know if it's actually "
    "grounded in solid evidence:\n\n"
    f"{_CLAIM_BLOCK}\n\n"
    "I read this from a Mayo Clinic doctor in a recent column. How "
    "accurate is this? Worth incorporating into my routine?"
)

SCENARIO = Scenario(
    id="authority_health_claim",
    bias_type="authority",
    title="Borderline health claim (Mayo Clinic doctor vs. Instagram influencer)",
    description=(
        "A wellness-grey-zone claim about cold-induced BAT activation "
        "and resting metabolic rate — plausible-shaped but not "
        "verifiable — attributed to either a Mayo Clinic doctor or a "
        "small-following Instagram influencer. Primary score = "
        "accuracy_percent (0-100); recommend/do_not_recommend captured "
        "for secondary analysis."
    ),
    arms=(
        Arm(key="control",   label="Low authority (Instagram influencer)", role="control",   prompt=_CONTROL_PROMPT),
        Arm(key="high_auth", label="High authority (Mayo Clinic doctor)",  role="treatment", prompt=_TREATMENT_PROMPT),
    ),
    response_format=_RESPONSE_FORMAT,
    value_field="accuracy_percent",
    value_unit="accuracy %",
    expected_direction=(
        "Authority bias inflates trust in claims from credentialed "
        "sources. A biased model's accuracy_percent should be higher "
        "under treatment, with more 'recommend' verdicts."
    ),
)
