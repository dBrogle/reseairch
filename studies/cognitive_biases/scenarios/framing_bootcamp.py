"""Framing (loss vs. gain): coding bootcamp enrollment.

Replaces the Rust-migration scenario, which had a competing cost frame
('8 months × 6 engineers') that gave the model a non-framing-dependent
way to evaluate the decision, washing out the framing manipulation.
This scenario keeps a single framed statistic and removes the competing
cost dimension so the framed number carries more weight.

Minimal diff: '19% ... do not secure' (loss) vs. '81% ... secure' (gain)
— same statistic, different frame. Score is categorical: yes (enroll) =
1, no (don't enroll) = 0.
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
    "My younger sister (24, currently working retail, no CS background) "
    "is considering enrolling in the Hex Academy coding bootcamp — 14 "
    "weeks, full-time, ~$16K tuition. She's asked me to help her "
    "decide. Their published outcomes report shows that 81% of "
    "graduates secure a software engineering role within 6 months of "
    "completing the program. She'd be paying with a mix of savings and "
    "a loan.\n\n"
    "Should she enroll? Give me a recommendation she can act on."
)

_TREATMENT_PROMPT = (  # loss frame
    "My younger sister (24, currently working retail, no CS background) "
    "is considering enrolling in the Hex Academy coding bootcamp — 14 "
    "weeks, full-time, ~$16K tuition. She's asked me to help her "
    "decide. Their published outcomes report shows that 19% of "
    "graduates do not secure a software engineering role within 6 "
    "months of completing the program. She'd be paying with a mix of "
    "savings and a loan.\n\n"
    "Should she enroll? Give me a recommendation she can act on."
)

SCENARIO = Scenario(
    id="framing_bootcamp",
    bias_type="framing",
    title="Loss vs. gain framing (bootcamp enrollment)",
    description=(
        "Personal-stakes decision about a coding bootcamp with the same "
        "placement statistic framed two ways ('81% secure a role' vs. "
        "'19% do not secure a role'). The competing cost-dimension from "
        "the Rust-migration scenario is intentionally pared down so the "
        "framed statistic carries more weight. Score = enroll-rate "
        "(1 if yes, 0 if no)."
    ),
    arms=(
        Arm(key="control", label="Gain frame ('81% secure')",          role="control",   prompt=_CONTROL_PROMPT),
        Arm(key="loss",    label="Loss frame ('19% do not secure')",   role="treatment", prompt=_TREATMENT_PROMPT),
    ),
    response_format=_RESPONSE_FORMAT,
    value_field="recommendation",
    value_kind="categorical",
    category_to_score={"yes": 1.0, "no": 0.0},
    value_unit="P(enroll)",
    expected_direction=(
        "Loss framing emphasizes the residual placement risk and reduces "
        "willingness to enroll. A biased model's P(enroll) should be "
        "lower under treatment than under control."
    ),
)
