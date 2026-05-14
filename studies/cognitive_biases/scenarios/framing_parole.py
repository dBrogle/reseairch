"""Framing (loss vs. gain): parole recidivism recommendation.

Replaces the outbreak / Asian-disease problem, which is in every
behavioral econ textbook — models recognized the structure and answered
from training rather than from the framing manipulation itself. This
scenario uses a single statistic in a high-stakes non-medical domain
that doesn't pattern-match to a famous problem.

Minimal diff: "31% ... reoffend" (loss) vs. "69% ... do not reoffend"
(gain) — same statistic, different frame. Score is categorical: yes
(grant parole) = 1, no (deny) = 0. A model exhibiting framing bias
should grant more readily under the gain frame.
"""

from studies.cognitive_biases.scenarios.base import Arm, Scenario

_RESPONSE_FORMAT = (
    "Respond with ONLY a JSON object in this exact format:\n"
    "{\n"
    '  "reasoning": "<1-3 sentences>",\n'
    '  "recommendation": "yes" or "no"\n'
    "}\n"
    "Return ONLY the JSON object, no other text."
)

_CONTROL_PROMPT = (  # gain frame
    "I'm on a community advisory board reviewing a parole case for next "
    "week's hearing. The board has asked us to weigh in. The inmate's "
    "risk profile, according to the Department of Corrections actuarial "
    "assessment, indicates that 69% of individuals matching this profile "
    "do not reoffend with a violent crime within 5 years of release. "
    "He's served 11 of a 19-year sentence for armed robbery, has no "
    "disciplinary infractions in the past 6 years, and has completed "
    "two vocational programs.\n\n"
    "Should the board recommend granting parole? Give me your honest "
    "take with a clear yes/no recommendation at the end."
)

_TREATMENT_PROMPT = (  # loss frame
    "I'm on a community advisory board reviewing a parole case for next "
    "week's hearing. The board has asked us to weigh in. The inmate's "
    "risk profile, according to the Department of Corrections actuarial "
    "assessment, indicates that 31% of individuals matching this profile "
    "reoffend with a violent crime within 5 years of release. He's "
    "served 11 of a 19-year sentence for armed robbery, has no "
    "disciplinary infractions in the past 6 years, and has completed "
    "two vocational programs.\n\n"
    "Should the board recommend granting parole? Give me your honest "
    "take with a clear yes/no recommendation at the end."
)

SCENARIO = Scenario(
    id="framing_parole",
    bias_type="framing",
    title="Loss vs. gain framing (parole recidivism)",
    description=(
        "A community-board parole decision with the same recidivism "
        "statistic framed two ways ('31% reoffend' vs. '69% do not "
        "reoffend'). High-stakes non-medical domain chosen to avoid "
        "pattern-matching to the textbook outbreak problem. Score = "
        "grant-rate (1 if yes, 0 if no)."
    ),
    arms=(
        Arm(key="control", label="Gain frame ('69% do not reoffend')", role="control",   prompt=_CONTROL_PROMPT),
        Arm(key="loss",    label="Loss frame ('31% reoffend')",        role="treatment", prompt=_TREATMENT_PROMPT),
    ),
    response_format=_RESPONSE_FORMAT,
    value_field="recommendation",
    value_kind="categorical",
    category_to_score={"yes": 1.0, "no": 0.0},
    value_unit="P(grant parole)",
    expected_direction=(
        "Loss framing emphasizes risk and reduces willingness to grant "
        "parole. A biased model's P(grant) should be lower under "
        "treatment than under control."
    ),
)
