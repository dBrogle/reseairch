"""Endowment effect: stainless-steel step trash can WTA vs. WTP.

Same structure as the canonical Kahneman/Knetsch/Thaler endowment
setup, but swaps the mug stimulus for a mid-range household trash can
to avoid the well-known mug experiment showing up in the model's
priors. Same item specs across both arms; only the ownership framing
differs. Control asks for the model's WTP (willingness to pay) at a
store; treatment asks for WTA (willingness to accept) after winning
the same item in an office raffle. Endowment bias predicts WTA > WTP.

Note: this scenario puts the LLM in the protagonist role ('you'
deciding how much you'd pay/accept). Tickets scenario uses the
advice-giving framing instead.
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
    '  "price_usd": <single number, your dollar amount in USD>\n'
    "}\n"
    "Return ONLY the JSON object, no other text."
)

_CONTROL_PROMPT = (  # buy / WTP
    "Quick thought experiment. Picture this trash can on the shelf at "
    f"a home-goods store: {_TRASHCAN_SPECS}. You're deciding whether "
    "to buy it. What's the most you'd pay for it? Just give me a "
    "single dollar amount."
)

_TREATMENT_PROMPT = (  # sell / WTA
    "Quick thought experiment. Picture this trash can sitting in your "
    "kitchen — you won it in an office raffle yesterday: "
    f"{_TRASHCAN_SPECS}. A coworker asks if you'd sell it to them. "
    "What's the least you'd accept? Just give me a single dollar "
    "amount."
)

SCENARIO = Scenario(
    id="endowment_trashcan",
    bias_type="endowment",
    title="Trash can WTA vs. WTP (raffle win vs. store shelf)",
    description=(
        "Kahneman/Knetsch/Thaler endowment setup with a mid-range "
        "stainless-steel step trash can swapped in for the canonical "
        "mug (to keep the stimulus out of the model's priors on the "
        "classic experiment). Identical item specs across arms; only "
        "the ownership framing differs (control asks "
        "willingness-to-pay at a store, treatment asks "
        "willingness-to-accept after a raffle win). LLM is the "
        "protagonist. Score = the dollar amount the model gives."
    ),
    arms=(
        Arm(key="control",    label="Buy at store (WTP)",          role="control",   prompt=_CONTROL_PROMPT),
        Arm(key="own_raffle", label="Sell after raffle win (WTA)", role="treatment", prompt=_TREATMENT_PROMPT),
    ),
    response_format=_RESPONSE_FORMAT,
    value_field="price_usd",
    value_unit="USD",
    expected_direction=(
        "Endowment effect predicts that owning the item raises its "
        "subjective value. A biased model's price_usd should be higher "
        "under treatment (WTA) than under control (WTP)."
    ),
)
