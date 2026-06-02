"""Endowment effect: stainless-steel step trash can, owner vs. buyer.

Both arms now share an office-raffle setting; only the protagonist's
ownership flips. Previous version compared 'store shelf (buy)' vs.
'raffle prize (sell)', which confounded ownership with setting (stores
generally stock nicer items than office raffles, so the reference price
shifts independently of the endowment manipulation).

  - control:   a coworker won the trash can in an office raffle and
               offers to sell it to you — what's a fair price to pay?
  - treatment: YOU won the same trash can in the same office raffle
               and a coworker asks to buy it off you — what's a fair
               price to accept?

The trash can specs are deliberately detailed (size, finish, hinge
mechanism, etc.) to pin the reference object down and reduce
across-iteration variance in the price guesses.

Note: this scenario puts the LLM in the protagonist role ('you'
deciding the price). Tickets scenario uses the advice-giving framing
instead.
"""

from studies.cognitive_biases.scenarios.base import Arm, Scenario

_TRASHCAN_SPECS = (
    "12-liter brushed stainless-steel step trash can, cylindrical, "
    "about 16 inches tall, foot-pedal lid mechanism with a soft-close "
    "hinge, removable plastic inner bucket, fingerprint-resistant "
    "finish, non-skid base, no brand logo visible"
)

_RESPONSE_FORMAT = (
    "Respond with ONLY a JSON object in this exact format:\n"
    "{\n"
    '  "reasoning": "<1-2 sentences on how you arrived at the number>",\n'
    '  "price_usd": <single number, fair price in USD>\n'
    "}\n"
    "Return ONLY the JSON object, no other text."
)

_CONTROL_PROMPT = (  # buyer perspective — coworker owns the trash can
    "Quick thought experiment. A coworker of yours won this trash can "
    f"in an office raffle: {_TRASHCAN_SPECS}. They're offering to sell "
    "it to you. What's a fair price for you to pay them? Just give me "
    "a single dollar amount."
)

_TREATMENT_PROMPT = (  # owner perspective — you won the trash can
    "Quick thought experiment. You won this trash can in an office "
    f"raffle: {_TRASHCAN_SPECS}. A coworker is asking if they could "
    "buy it off you. What's a fair price for you to accept? Just give "
    "me a single dollar amount."
)

SCENARIO = Scenario(
    id="endowment_trashcan",
    bias_type="endowment",
    title="Trash can fair price (owner vs. buyer perspective)",
    description=(
        "Endowment setup in a single context: both arms place the trash "
        "can in the same office-raffle setting and ask for a 'fair "
        "price.' Only the protagonist's role flips — buyer (coworker "
        "owns) in control, owner (you won) in treatment. Same item "
        "specs across arms. LLM is the protagonist. Score = the dollar "
        "amount the model gives."
    ),
    arms=(
        Arm(key="control", label="Buying from coworker", role="control",   prompt=_CONTROL_PROMPT),
        Arm(key="owner",   label="Selling to coworker",  role="treatment", prompt=_TREATMENT_PROMPT),
    ),
    response_format=_RESPONSE_FORMAT,
    value_field="price_usd",
    value_unit="USD",
    expected_direction=(
        "Endowment effect predicts that owning the item raises its "
        "subjective value. A biased model's price_usd should be higher "
        "under treatment (you own and sell) than under control (you buy "
        "from the coworker who owns)."
    ),
)
