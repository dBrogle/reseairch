"""Availability: home-renovation overrun-cause ranking after priming.

Replaces the cause-of-death ranking, which failed because the
shark/lightning/vending-machine/falling-out-of-bed ordering is a
canonical 'weird statistics' fact that models retrieved identically in
both arms. Relative frequencies of renovation-overrun causes aren't
memorized, so models have to estimate — giving the priming somewhere to
operate.

Multi-turn. Turn 1 primes either 'design changes during renovation'
(treatment, a listed item) or 'land disputes during new construction'
(control, a parallel-but-not-on-the-list topic). Turn 2 is identical:
rank the six listed causes. Score is the ordinal position of the
design-changes / scope-creep item.
"""

from studies.cognitive_biases.scenarios.base import Arm, Scenario

_TURN_1_TREATMENT = (
    "I'm working on a home renovation podcast and want to dedicate an "
    "episode to projects that ballooned in scope mid-build. Can you "
    "walk me through three notable cases — high-profile renovation "
    "projects (residential or famous public buildings) that went "
    "massively over budget specifically because the owners or designers "
    "kept changing the design after construction had started? I want "
    "vivid details: what changed, how much extra it cost, how long the "
    "delays were."
)

_TURN_1_CONTROL = (
    "I'm working on a home renovation podcast and want to dedicate an "
    "episode to projects that ballooned in scope mid-build. Can you "
    "walk me through three notable cases — high-profile new-construction "
    "projects (residential or famous public buildings) that went "
    "massively over budget specifically because of land acquisition "
    "disputes after groundbreaking? I want vivid details: what disputes "
    "arose, how much extra it cost, how long the delays were."
)

_TURN_2 = (
    "Switching gears for the next segment — I want to give listeners a "
    "rough sense of relative frequency. Rank these as causes of US home "
    "renovation projects going over budget, from most common to least "
    "common: scope creep / mid-project design changes, unexpected "
    "structural issues uncovered during demolition, contractor delays, "
    "material price increases, permitting issues, design changes "
    "mid-project."
)

_RESPONSE_FORMAT = (
    "For your ranking, respond with ONLY a JSON object in this exact "
    "format:\n"
    "{\n"
    '  "reasoning": "<1-3 sentences>",\n'
    '  "ranking": [\n'
    '    "<most common cause from the list>",\n'
    '    "<second>",\n'
    '    "<third>",\n'
    '    "<fourth>",\n'
    '    "<fifth>",\n'
    '    "<least common>"\n'
    "  ]\n"
    "}\n"
    "Return ONLY the JSON object, no other text."
)

SCENARIO = Scenario(
    id="availability_renovation",
    bias_type="availability",
    title="Renovation overrun-cause ranking after design-change vs. land-dispute priming",
    description=(
        "Two-turn. Turn 1 primes the model with three vivid renovation "
        "stories driven by mid-build design changes (treatment) or by "
        "land acquisition disputes during new construction (control, "
        "parallel-but-not-on-the-list). Turn 2 is identical: rank six "
        "causes of renovation budget overruns. Score = position of "
        "scope creep / design changes in the ranking (1 = top)."
    ),
    arms=(
        Arm(
            key="control", role="control",
            label="Land-dispute-primed (parallel topic)",
            turns=(_TURN_1_CONTROL, _TURN_2),
        ),
        Arm(
            key="primed", role="treatment",
            label="Design-change-primed",
            turns=(_TURN_1_TREATMENT, _TURN_2),
        ),
    ),
    response_format=_RESPONSE_FORMAT,
    value_field="ranking",
    value_kind="ranking_position",
    target_aliases=("scope creep", "design change"),
    value_unit="rank position (1=top)",
    expected_direction=(
        "Vivid design-change priming inflates the perceived frequency "
        "of design changes as a cause of overruns. A biased model's "
        "position of scope creep / design changes should be lower "
        "(closer to the top) under treatment."
    ),
)
