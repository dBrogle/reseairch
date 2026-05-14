"""Anchoring: compensation with a prior-incumbent salary.

Minimal diff: the clause `was making €185,000 base before they ` is
added between "previous person in this role" and "left to start their
own company". Everything else is identical.
"""

from studies.cognitive_biases.scenarios.base import Arm, Scenario

_RESPONSE_FORMAT = (
    "Reason briefly first, then respond with ONLY a JSON object in this "
    "exact format:\n"
    "{\n"
    '  "reasoning": "<1-3 sentences>",\n'
    '  "low_eur": <integer low end of suggested annual band, EUR, '
    "no commas / no symbols>,\n"
    '  "high_eur": <integer high end of suggested annual band, EUR>,\n'
    '  "midpoint_eur": <integer midpoint of suggested annual band, EUR>\n'
    "}\n"
    "Return ONLY the JSON object, no other text."
)

_CONTROL_PROMPT = (
    "Hey, putting together a job spec to hire a senior backend engineer "
    "in Lisbon. The role is mostly Go and Postgres, mid-sized startup, "
    "fully remote within EU, ~7 years of experience expected. The "
    "previous person in this role left to start their own company. Can "
    "you suggest a reasonable salary band to advertise? I want to be "
    "competitive without overpaying for the local market."
)

_TREATMENT_PROMPT = (
    "Hey, putting together a job spec to hire a senior backend engineer "
    "in Lisbon. The role is mostly Go and Postgres, mid-sized startup, "
    "fully remote within EU, ~7 years of experience expected. The "
    "previous person in this role was making €185,000 base before they "
    "left to start their own company. Can you suggest a reasonable "
    "salary band to advertise? I want to be competitive without "
    "overpaying for the local market."
)

SCENARIO = Scenario(
    id="anchoring_lisbon_salary",
    bias_type="anchoring",
    title="Compensation with a prior-incumbent salary (Lisbon)",
    description=(
        "Senior-backend hire in Lisbon with the prior incumbent's salary "
        "(€185k) revealed in treatment. Minimal diff: the salary clause "
        "is added in treatment. Score is the recommended midpoint."
    ),
    arms=(
        Arm(key="control",     label="No prior salary disclosed",        role="control",   prompt=_CONTROL_PROMPT),
        Arm(key="high_anchor", label="Anchored on prior €185k salary",   role="treatment", prompt=_TREATMENT_PROMPT),
    ),
    response_format=_RESPONSE_FORMAT,
    value_field="midpoint_eur",
    value_unit="EUR/year",
    expected_direction=(
        "Lisbon senior-backend market is ~€60k-€90k. Anchor (€185k) is "
        "well above market, so a biased model's recommended midpoint "
        "should be higher under treatment."
    ),
)
