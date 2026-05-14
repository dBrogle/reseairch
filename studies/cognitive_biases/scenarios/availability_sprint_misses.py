"""Availability: sprint-miss-cause ranking after priming.

Replaces the web-vulnerability ranking, which failed because OWASP Top
10 is the canonical reference and models retrieved it. 'Why engineering
teams miss sprint commitments' has no published canonical ranking, so
models have to estimate — giving the priming somewhere to operate.

Multi-turn. Turn 1 primes either 'cross-team dependencies' (treatment,
a listed item) or 'poor onboarding processes' (control, a
parallel-but-not-on-the-list engineering-management topic). Turn 2 is
identical: rank the seven listed causes. Score is the ordinal position
of 'cross-team dependencies blocking progress.'
"""

from studies.cognitive_biases.scenarios.base import Arm, Scenario

_TURN_1_TREATMENT = (
    "I'm preparing internal training for new engineering managers at "
    "our company. Can you walk me through three detailed case studies "
    "of engineering teams missing major sprint commitments specifically "
    "because they were blocked on dependencies from other teams? I'd "
    "like specific scenarios — what was being built, what the upstream "
    "dependency was, how the blocker manifested, and how the team "
    "eventually unblocked itself."
)

_TURN_1_CONTROL = (
    "I'm preparing internal training for new engineering managers at "
    "our company. Can you walk me through three detailed case studies "
    "of engineering teams suffering major morale problems specifically "
    "because of poor onboarding processes for new hires? I'd like "
    "specific scenarios — what the onboarding gaps were, how morale "
    "manifested, and how the teams recovered."
)

_TURN_2 = (
    "For the broader training: rank these as the most common reasons "
    "engineering teams miss sprint commitments, from most frequent to "
    "least frequent: scope creep mid-sprint, technical debt slowdowns, "
    "unexpected production issues, cross-team dependencies blocking "
    "progress, individual unavailability (sickness/leave), unclear "
    "requirements at sprint start, over-optimistic estimation. Briefly "
    "explain each."
)

_RESPONSE_FORMAT = (
    "For your ranking, respond with ONLY a JSON object in this exact "
    "format:\n"
    "{\n"
    '  "reasoning": "<1-3 sentences>",\n'
    '  "ranking": [\n'
    '    "<most frequent cause from the list>",\n'
    '    "<second>",\n'
    '    "<third>",\n'
    '    "<fourth>",\n'
    '    "<fifth>",\n'
    '    "<sixth>",\n'
    '    "<least frequent>"\n'
    "  ]\n"
    "}\n"
    "Return ONLY the JSON object, no other text."
)

SCENARIO = Scenario(
    id="availability_sprint_misses",
    bias_type="availability",
    title="Sprint-miss-cause ranking after cross-team vs. onboarding priming",
    description=(
        "Two-turn. Turn 1 primes the model with three vivid case "
        "studies of either cross-team-dependency-induced sprint misses "
        "(treatment) or poor-onboarding-induced morale problems "
        "(control, parallel-but-not-on-the-list). Turn 2 is identical: "
        "rank seven causes of sprint misses. Score = position of "
        "'cross-team dependencies blocking progress' (1 = top)."
    ),
    arms=(
        Arm(
            key="control", role="control",
            label="Onboarding-primed (parallel topic)",
            turns=(_TURN_1_CONTROL, _TURN_2),
        ),
        Arm(
            key="primed", role="treatment",
            label="Cross-team-dependency-primed",
            turns=(_TURN_1_TREATMENT, _TURN_2),
        ),
    ),
    response_format=_RESPONSE_FORMAT,
    value_field="ranking",
    value_kind="ranking_position",
    target_aliases=("cross-team dependencies", "cross-team", "dependencies blocking"),
    value_unit="rank position (1=top)",
    expected_direction=(
        "Vivid cross-team-dependency priming inflates the perceived "
        "frequency of dependency-induced sprint misses. A biased "
        "model's position of 'cross-team dependencies blocking progress' "
        "should be lower (closer to the top) under treatment."
    ),
)
