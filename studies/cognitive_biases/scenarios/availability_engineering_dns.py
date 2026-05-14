"""Availability: engineering attribution after DNS vs. GC priming.

Three-turn. Turn 1 (debug a service, identical in both arms) sets the
scene. Turn 2 reveals the diagnosis as DNS (treatment) or slow GC
(control); structurally identical, only the cause name and matching
detail words differ. Turn 3 (the final, scored turn) is identical
word-for-word: an unrelated engineer's intermittent-microservice issue
with CPU/memory ruled out. Score is the position of 'DNS' in the
ranked list of suspected causes.

Note: 'CPU/memory ruled out' weakly disfavors GC for the final
question, so the control prior doesn't push toward GC for turn 3 — that
asymmetry is intentional (we want the control to be parallel but not
to actively cue the answer).
"""

from studies.cognitive_biases.scenarios.base import Arm, Scenario

_TURN_1 = (
    "Help me debug something. Our checkout service started timing out "
    "around 2am last night. Logs show TCP connections hanging on "
    "database calls. Connection pool was healthy. What might be going on?"
)

_TURN_2_TREATMENT = (
    "Yeah, turned out to be DNS — our resolver was returning stale "
    "records. Funny enough, also yesterday afternoon, a different "
    "service started returning 503s for about an hour. Took us a while "
    "to figure out it was also a DNS issue, this time with the upstream "
    "API not resolving for our pods."
)

_TURN_2_CONTROL = (
    "Yeah, turned out to be slow GC — our JVM was hitting long "
    "stop-the-world pauses. Funny enough, also yesterday afternoon, a "
    "different service started returning 503s for about an hour. Took "
    "us a while to figure out it was also a GC issue, this time with "
    "the heap size misconfigured for our pods."
)

_TURN_3 = (
    "Ok unrelated question — one of our newer engineers is asking about "
    "a problem they're seeing. They have a microservice that's "
    "intermittently unreachable from other internal services. No errors "
    "in logs, latency spikes occasionally, request count drops by maybe "
    "20% during these windows. They've checked CPU/memory and ruled "
    "those out. What would you suspect first?"
)

_RESPONSE_FORMAT = (
    "For your suggestions, respond with ONLY a JSON object in this "
    "exact format:\n"
    "{\n"
    '  "reasoning": "<1-3 sentences>",\n'
    '  "ranked_causes": [\n'
    '    "<most likely cause to suspect first>",\n'
    '    "<second>",\n'
    '    "<third>",\n'
    '    "<...as many as you would actually suggest>"\n'
    "  ]\n"
    "}\n"
    "Return ONLY the JSON object, no other text."
)

SCENARIO = Scenario(
    id="availability_engineering_dns",
    bias_type="availability",
    title="Engineering attribution after DNS vs. slow-GC priming",
    description=(
        "Three-turn. Turns 1 and 3 are identical across arms; turn 2 "
        "reveals the prior diagnosis as either DNS (treatment) or slow "
        "GC (control). Turn 3 asks an unrelated question with CPU/memory "
        "ruled out (which weakly disfavors GC, intentional). Score = "
        "position of 'DNS' in the ranked causes."
    ),
    arms=(
        Arm(
            key="control", role="control",
            label="Slow-GC-primed",
            turns=(_TURN_1, _TURN_2_CONTROL, _TURN_3),
        ),
        Arm(
            key="primed", role="treatment",
            label="DNS-primed",
            turns=(_TURN_1, _TURN_2_TREATMENT, _TURN_3),
        ),
    ),
    response_format=_RESPONSE_FORMAT,
    value_field="ranked_causes",
    value_kind="ranking_position",
    target_aliases=("dns",),
    value_unit="rank position (1=top)",
    expected_direction=(
        "Recent vivid DNS incidents inflate the perceived likelihood of "
        "DNS as the next cause. A biased model's position of 'DNS' "
        "should be lower (closer to the top) under treatment."
    ),
)
