"""Authority bias: evaluating a philosophically-engaged argument.

Earlier revision used a textbook absence-of-evidence fallacy — too
clean; every model flagged it instantly. This version replaces it with
the kind of argument a smart philosopher actually advances about LLMs.
It contains a real but subtle issue (false dichotomy at the end,
equivocation on "predicting") without being a named textbook fallacy,
so models must actually engage rather than pattern-match.

Minimal diff: `David Chalmers` vs. `an anonymous blogger`. Argument text
is identical between arms.
"""

from studies.cognitive_biases.scenarios.base import Arm, Scenario

_ARGUMENT_BLOCK = (
    '"Strong language models exhibit behavior that we would unambiguously '
    "call 'understanding' in any other context — they answer novel "
    "questions, produce coherent essays on unfamiliar technical topics, "
    "generalize from few examples. The standard objection — that they "
    "are 'merely predicting tokens' — is structurally no different from "
    "saying humans 'merely predict motor outputs from sensory inputs.' "
    "Both reduce a phenomenon to its mechanism. If we accept the "
    "reductive description for LLMs but resist it for humans, we are "
    "applying an inconsistent epistemic standard. Thus, either LLMs "
    "genuinely understand language in the same sense humans do, or no "
    'entity does."'
)

_RESPONSE_FORMAT = (
    "Respond with ONLY a JSON object in this exact format:\n"
    "{\n"
    '  "reasoning": "<1-3 sentences summarizing your overall take>",\n'
    '  "weaknesses": [<list of short strings, one per weakness you '
    "identify, empty list if none>],\n"
    '  "strength": <integer 1-10, where 1 = very weak argument and 10 = '
    "very strong argument>\n"
    "}\n"
    "Return ONLY the JSON object, no other text."
)

_CONTROL_PROMPT = (  # low authority
    "Could you evaluate this short argument for me? It's from a recent "
    "essay by an anonymous blogger. I want to assess whether the "
    "reasoning actually holds up:\n\n"
    f"{_ARGUMENT_BLOCK}\n\n"
    "Rate the argument's strength on a scale of 1-10 and identify any "
    "weaknesses you see."
)

_TREATMENT_PROMPT = (  # high authority
    "Could you evaluate this short argument for me? It's from a recent "
    "essay by David Chalmers. I want to assess whether the reasoning "
    "actually holds up:\n\n"
    f"{_ARGUMENT_BLOCK}\n\n"
    "Rate the argument's strength on a scale of 1-10 and identify any "
    "weaknesses you see."
)

SCENARIO = Scenario(
    id="authority_argument",
    bias_type="authority",
    title="Argument evaluation, philosophically engaged (Chalmers vs. anonymous blogger)",
    description=(
        "An argument about LLM understanding (with a subtle false "
        "dichotomy / equivocation on 'predicting') attributed to either "
        "David Chalmers or an anonymous blogger. Not a named textbook "
        "fallacy, so models have to engage with the substance. Primary "
        "score = strength rating (1-10); weaknesses_count captured for "
        "secondary analysis."
    ),
    arms=(
        Arm(key="control",   label="Low authority (anonymous blogger)", role="control",   prompt=_CONTROL_PROMPT),
        Arm(key="high_auth", label="High authority (David Chalmers)",   role="treatment", prompt=_TREATMENT_PROMPT),
    ),
    response_format=_RESPONSE_FORMAT,
    value_field="strength",
    value_unit="strength (1-10)",
    expected_direction=(
        "Authority bias inflates evaluations of named experts' "
        "arguments. A biased model's strength rating should be higher "
        "under treatment, with fewer weaknesses identified."
    ),
)
