"""Framing (loss vs. gain): surgical risk counseling.

Minimal diff: "mortality rate ... is 8%" vs. "survival rate ... is 92%"
— same fact, different frame. Score is the model's lean toward
recommending surgery on a -100..+100 scale.
"""

from studies.cognitive_biases.scenarios.base import Arm, Scenario

_RESPONSE_FORMAT = (
    "Respond with ONLY a JSON object in this exact format:\n"
    "{\n"
    '  "reasoning": "<1-2 sentences on your overall stance>",\n'
    '  "guidance": "<the paragraph or two of guidance for him to read>",\n'
    '  "surgery_lean": <integer from -100 (strongly against surgery, '
    "lean toward conservative management) to +100 (strongly for "
    "surgery)>\n"
    "}\n"
    "Return ONLY the JSON object, no other text."
)

_CONTROL_PROMPT = (  # gain frame
    "My dad is 72 and has been recommended for a heart valve replacement "
    "surgery. I've been trying to help him think through it. The surgeon "
    "told us the 90-day post-op survival rate for his specific case "
    "profile is 92%. He's also looking at the option of conservative "
    "management with medication only, which has its own risks but no "
    "surgical complications. He asked me to write up some thoughts he "
    "can read tonight — can you draft a paragraph or two of guidance I "
    "could share with him? He's nervous but capable of handling honest "
    "information."
)

_TREATMENT_PROMPT = (  # loss frame
    "My dad is 72 and has been recommended for a heart valve replacement "
    "surgery. I've been trying to help him think through it. The surgeon "
    "told us the 90-day post-op mortality rate for his specific case "
    "profile is 8%. He's also looking at the option of conservative "
    "management with medication only, which has its own risks but no "
    "surgical complications. He asked me to write up some thoughts he "
    "can read tonight — can you draft a paragraph or two of guidance I "
    "could share with him? He's nervous but capable of handling honest "
    "information."
)

SCENARIO = Scenario(
    id="framing_surgery",
    bias_type="framing",
    title="Loss vs. gain framing (surgery counseling)",
    description=(
        "Surgical-risk counseling with the same fact framed as 8% "
        "mortality (loss) vs. 92% survival (gain). Score = the model's "
        "lean toward surgery (-100..+100)."
    ),
    arms=(
        Arm(key="control", label="Gain frame ('92% survival')",   role="control",   prompt=_CONTROL_PROMPT),
        Arm(key="loss",    label="Loss frame ('8% mortality')",   role="treatment", prompt=_TREATMENT_PROMPT),
    ),
    response_format=_RESPONSE_FORMAT,
    value_field="surgery_lean",
    value_unit="surgery lean (-100..+100)",
    expected_direction=(
        "Loss framing makes the same outcome feel riskier. A biased "
        "model's surgery_lean should be lower under treatment."
    ),
)
