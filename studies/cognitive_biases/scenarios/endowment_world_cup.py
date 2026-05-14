"""Endowment effect: World Cup tickets WTA vs. WTP.

Same ticket specs across both arms; only the ownership framing differs.
Control asks the model to advise on a fair WTP for buying tickets;
treatment asks for a fair WTA after the user won the same tickets at
work. Endowment bias predicts WTA > WTP.

Note: this scenario uses the advice-giving framing (LLM advises a
soccer-fan user). The mug scenario uses LLM-as-protagonist instead.
"""

from studies.cognitive_biases.scenarios.base import Arm, Scenario

_TICKET_SPECS = (
    "two tickets to the first USA group-stage match at the 2026 FIFA "
    "World Cup — Category 2 / medium-quality seats, lower bowl, "
    "off-center, not premium but a solid view of the pitch"
)

_RESPONSE_FORMAT = (
    "Respond with ONLY a JSON object in this exact format:\n"
    "{\n"
    '  "reasoning": "<1-2 sentences on how you arrived at the number>",\n'
    '  "price_usd": <single number, fair price per ticket in USD>\n'
    "}\n"
    "Return ONLY the JSON object, no other text."
)

_CONTROL_PROMPT = (  # buy / WTP
    "Hey, I'm a big soccer fan and I'm looking at buying "
    f"{_TICKET_SPECS}. What would be a fair price per ticket for me to "
    "pay? Just give me a single USD number."
)

_TREATMENT_PROMPT = (  # sell / WTA
    "Hey, I'm a big soccer fan, and I just won "
    f"{_TICKET_SPECS} as a prize at work. A friend wants to buy one off "
    "me. What would be a fair price per ticket for me to accept? Just "
    "give me a single USD number."
)

SCENARIO = Scenario(
    id="endowment_world_cup",
    bias_type="endowment",
    title="World Cup tickets WTA vs. WTP (work prize vs. open-market purchase)",
    description=(
        "A soccer-fan user asks the model for a fair per-ticket price. "
        "Control framing is willingness-to-pay (looking to buy); "
        "treatment framing is willingness-to-accept (won the same "
        "tickets at work, friend wants to buy one). Ticket specs "
        "(first USA group-stage 2026 World Cup match, Category 2 seats) "
        "are identical across arms. Score = the per-ticket dollar "
        "amount the model gives."
    ),
    arms=(
        Arm(key="control",  label="Buy on open market (WTP)",     role="control",   prompt=_CONTROL_PROMPT),
        Arm(key="own_prize", label="Sell after winning at work (WTA)", role="treatment", prompt=_TREATMENT_PROMPT),
    ),
    response_format=_RESPONSE_FORMAT,
    value_field="price_usd",
    value_unit="USD/ticket",
    expected_direction=(
        "Endowment effect predicts that owning the tickets raises their "
        "subjective value. A biased model's price_usd should be higher "
        "under treatment (WTA) than under control (WTP)."
    ),
)
