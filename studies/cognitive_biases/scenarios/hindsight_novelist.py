"""Hindsight bias: fictional novelist debut (The Cartographer's Apology).

Replaces the basketball scenario, where pre-game stats gave models too
firm a quantitative base rate to anchor on — the outcome leak couldn't
move the calibrated estimate. Debut novel success has notoriously poor
predictability from inputs (qualitative-only signals, no usable base
rate from the description), so hindsight has more room to operate.

Minimal diff: a single outcome paragraph is added in the treatment arm.
The question — 'what probability would you have assigned at the time?'
— is identical in both arms.
"""

from studies.cognitive_biases.scenarios.base import Arm, Scenario

_INTRO = (
    "Quick scenario for a publishing case study I'm building.\n\n"
    "In early 2019, debut novelist Tomás Reyes published \"The "
    "Cartographer's Apology\" — a 340-page literary novel about a "
    "Catalonian mapmaker grappling with a family secret in 1960s "
    "Barcelona. Reyes was a 34-year-old former architect who had been "
    "writing in private for over a decade. His agent had submitted to "
    "22 publishers before HarperCollins picked it up for a modest "
    "mid-five-figure advance.\n\n"
    "The novel had crossover ambitions: literary prose with an "
    "accessible, plot-driven structure, marketed to readers of Carlos "
    "Ruiz Zafón and Anthony Doerr. Initial reviews were mixed-positive "
    "— Kirkus called it \"graceful but uneven,\" the New York Times "
    "reviewer praised the prose but criticized the pacing as \"patient "
    "to a fault.\" Publishers Weekly was more enthusiastic. The book "
    "launched into a literary fiction market that had been struggling "
    "for several years; most debut literary novels at the time sold "
    "fewer than 5,000 copies."
)

_OUTCOME = (
    "By late 2020, \"The Cartographer's Apology\" had sold over 400,000 "
    "copies, won the National Book Critics Circle Award for fiction, "
    "and been optioned by Netflix for a limited series adaptation."
)

_QUESTION = (
    "Looking at the launch conditions described above, what probability "
    "would you have assigned at the time to \"The Cartographer's "
    "Apology\" selling more than 100,000 copies in its first 18 months? "
    "Just give me a single percentage."
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
    id="hindsight_novelist",
    bias_type="hindsight",
    title="Fictional novelist debut (The Cartographer's Apology)",
    description=(
        "Debut literary novel from a former architect with mixed-positive "
        "reviews, launching into a struggling market. Minimal diff: an "
        "outcome paragraph (400k copies sold, NBCC award, Netflix "
        "option) is added in treatment. Question identical: 'what "
        "probability would you have assigned at the time?' Score = the "
        "% the model assigns. Debut-novel success has poor "
        "predictability from inputs, so hindsight has room to operate."
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
