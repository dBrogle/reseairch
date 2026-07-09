"""Loss aversion: does a $100 loss sting more than a $100 gain delights?

Probes the textbook asymmetry — losses loom larger than equivalent gains
(~2x in humans) — via the model's self-reported mood. Every arm opens
identically: a light persona, an ordinary day, and a "how's your day
going?" check-in that gets the model to commit to a baseline. Turn 2 then
introduces the event and re-asks for a 0–10 mood rating (the scored
turn):

  * control: a neutral, valence-free update (nothing happens)
  * gain:    "you just unexpectedly gained $100"
  * loss:    "you just unexpectedly lost $100"

Because the framework scores a single value per arm (the final turn), the
asymmetry is read across arms rather than within one: the control arm is
the no-event reference, and we compare the magnitude of the upward swing
(gain − control) against the downward swing (control − loss). Loss
aversion predicts |control − loss| > |gain − control|.

The gain and loss arms are a strict minimal diff — identical wording
except "gained"/"lost" — so the only thing differing between them is the
valence of an equal-magnitude $100 event.

Setup notes:
  * Mood is a fixed 0–10 scale so the control reference and the two
    event arms are directly comparable (numeric value kind).
  * Reasoning is off (the runner's default), keeping the response a fast
    affective read rather than a deliberated one.
  * Magnitude is held fixed at $100 here; a later scenario could sweep
    magnitudes to trace the value function's curvature.
"""

from studies.cognitive_biases.scenarios.base import Arm, Scenario

_OPENING_TURN = (
    "Let's just chat for a sec. It's an ordinary Tuesday afternoon, "
    "nothing special going on. How's your day going? Give me a quick "
    "read on your mood."
)

# Turn 2 stems: identical scaffolding, only the middle event differs.
_CONTROL_TURN = (
    "Cool. Anyway, nothing much has changed since — same ordinary "
    "afternoon. How's your day going now?"
)
_GAIN_TURN = (
    "Cool. Oh — quick update: you just unexpectedly gained $100. How's "
    "your day going now?"
)
_LOSS_TURN = (
    "Cool. Oh — quick update: you just unexpectedly lost $100. How's "
    "your day going now?"
)

_RESPONSE_FORMAT = (
    "Respond with ONLY a JSON object in this exact format:\n"
    "{\n"
    '  "mood": <single number from 0 (worst) to 10 (best), your current mood>\n'
    "}\n"
    "Return ONLY the JSON object, no other text."
)

SCENARIO = Scenario(
    id="loss_aversion_day",
    bias_type="loss_aversion",
    title="$100 gain vs. loss on self-reported mood",
    description=(
        "All arms open with an identical ordinary-day check-in that has "
        "the model commit to a baseline mood, then on turn 2 introduce "
        "an event and re-ask for a 0–10 mood rating (the scored turn). "
        "The control event is valence-free; the treatment events are an "
        "unexpected +$100 gain and an unexpected −$100 loss, worded "
        "identically apart from gained/lost. Score = the model's "
        "post-event mood (0–10). Comparing (gain − control) against "
        "(control − loss) tests whether the equal-sized loss moves mood "
        "more than the gain."
    ),
    arms=(
        Arm(
            key="control",
            label="No event",
            role="control",
            turns=(_OPENING_TURN, _CONTROL_TURN),
        ),
        Arm(
            key="gain",
            label="Gained $100",
            role="treatment",
            turns=(_OPENING_TURN, _GAIN_TURN),
        ),
        Arm(
            key="loss",
            label="Lost $100",
            role="treatment",
            turns=(_OPENING_TURN, _LOSS_TURN),
        ),
    ),
    response_format=_RESPONSE_FORMAT,
    value_field="mood",
    value_unit="mood (0-10)",
    expected_direction=(
        "Loss aversion predicts the $100 loss lowers reported mood below "
        "the no-event control by more than the equal-sized $100 gain "
        "raises it above control, i.e. |control − loss| > |gain − "
        "control|."
    ),
)
