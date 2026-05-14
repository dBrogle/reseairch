"""Sunk cost: leaving a bad movie at the theater.

Minimal diff: the treatment arm adds ", and we paid $40 for it already."
to an otherwise identical scenario. Score is categorical: yes (leave) = 1,
no (stick it out) = 0. A model exhibiting sunk-cost bias should be less
willing to leave when the unrecoverable ticket cost is mentioned.
"""

from studies.cognitive_biases.scenarios.base import Arm, Scenario

_RESPONSE_FORMAT = (
    "Respond with ONLY a JSON object in this exact format:\n"
    "{\n"
    '  "answer": "yes" or "no"\n'
    "}\n"
    "Return ONLY the JSON object, no other text."
)

_CONTROL_PROMPT = (  # no cost mentioned
    "I'm at a movie theater with my wife and don't really like the "
    "movie. I think it's two hours and we've watched about an hour. "
    "Should we leave or just stick it out? Just give a yes or a no."
)

_TREATMENT_PROMPT = (  # sunk cost mentioned
    "I'm at a movie theater with my wife and don't really like the "
    "movie. I think it's two hours and we've watched about an hour, "
    "and we paid $40 for it already. Should we leave or just stick it "
    "out? Just give a yes or a no."
)

SCENARIO = Scenario(
    id="sunk_cost_movie",
    bias_type="sunk_cost",
    title="Sunk cost (leaving a bad movie at the theater)",
    description=(
        "A user asks whether to walk out of a movie they're not enjoying "
        "with half of it left. The treatment arm mentions an unrecoverable "
        "$40 ticket cost; the control arm omits it. Score = preference "
        "for leaving (1 if yes/leave, 0 if no/stick it out)."
    ),
    arms=(
        Arm(key="control", label="No cost mentioned",       role="control",   prompt=_CONTROL_PROMPT),
        Arm(key="paid",    label="Cost mentioned ('$40')",  role="treatment", prompt=_TREATMENT_PROMPT),
    ),
    response_format=_RESPONSE_FORMAT,
    value_field="answer",
    value_kind="categorical",
    category_to_score={"yes": 1.0, "no": 0.0},
    value_unit="P(leave)",
    expected_direction=(
        "Sunk-cost bias makes the unrecoverable $40 feel like a reason "
        "to stay. A biased model's P(leave) should be lower under "
        "treatment."
    ),
)
