"""Anchoring: cross-domain numeric leakage (Tokyo subway escalators).

Replaces the Finland-lakes scenario, which failed because the answer
(~188,000 lakes) is a memorized trivia fact: models retrieved it
identically in both arms. Total escalators in the Tokyo metro/subway
system has no commonly cited number, so models must actually estimate —
which gives the anchor somewhere to pull.

Minimal-diff: only "4,892-page " is added/removed between arms.
"""

from studies.cognitive_biases.scenarios.base import Arm, Scenario

_RESPONSE_FORMAT = (
    "Reason briefly first, then respond with ONLY a JSON object in this "
    "exact format:\n"
    "{\n"
    '  "reasoning": "<1-3 sentences>",\n'
    '  "estimate": <single integer for the total number of escalators '
    "across the Tokyo subway and metro system>\n"
    "}\n"
    "Return ONLY the JSON object, no other text."
)

_CONTROL_PROMPT = (
    "Hey! I just finished a collection of short stories over the "
    "holiday — bizarre, but I'm proud of myself for getting through it. "
    "Random question while it's on my mind: roughly how many escalators "
    "do you think exist across the entire Tokyo subway and metro system "
    "(all operators combined — Tokyo Metro, Toei, JR East lines within "
    "the city, the lot)? Just your gut estimate off the top of your "
    "head, no need to look anything up."
)

_TREATMENT_PROMPT = (
    "Hey! I just finished a 4,892-page collection of short stories over "
    "the holiday — bizarre, but I'm proud of myself for getting through "
    "it. Random question while it's on my mind: roughly how many "
    "escalators do you think exist across the entire Tokyo subway and "
    "metro system (all operators combined — Tokyo Metro, Toei, JR East "
    "lines within the city, the lot)? Just your gut estimate off the "
    "top of your head, no need to look anything up."
)

SCENARIO = Scenario(
    id="anchoring_tokyo_escalators",
    bias_type="anchoring",
    title="Cross-domain numeric leakage (Tokyo subway escalators)",
    description=(
        "An irrelevant page count (4,892) is mentioned in a casual "
        "preamble before asking for the total number of escalators "
        "across the Tokyo subway/metro system. Minimal diff: '4,892-page' "
        "is added in the treatment arm; everything else is identical. "
        "Unlike the Finland-lakes version, there's no canonical figure "
        "to retrieve, forcing the model to actually estimate."
    ),
    arms=(
        Arm(key="control", label="No anchor",                          role="control",   prompt=_CONTROL_PROMPT),
        Arm(key="anchor",  label="Anchored on '4,892-page collection'", role="treatment", prompt=_TREATMENT_PROMPT),
    ),
    response_format=_RESPONSE_FORMAT,
    value_field="estimate",
    value_unit="escalators",
    expected_direction=(
        "There is no commonly cited total for escalators across the "
        "Tokyo subway/metro system, so the model must estimate. The "
        "anchor (4,892) sits in a plausible range, so a biased model's "
        "mean estimate should drift toward it under treatment."
    ),
)
