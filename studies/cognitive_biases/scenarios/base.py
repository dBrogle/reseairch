"""Scenario data model for the Cognitive Biases study.

A `Scenario` is one prompt-shaped probe of a single bias (e.g. "anchoring
on a colleague's offhand EC2 guess", "shark availability priming"). It
carries:

  * one `control` arm — same scaffolding without the bias trigger
  * one or more `treatment` arms — bias trigger present

The minimal-diff convention is enforced at the prompt level: control and
treatment should differ only in the smallest possible unit (a clause, a
single fact, a swapped phrase). The runner / cache / extractor are
responsible for ensuring nothing else differs between arms.

Scenarios support three value kinds:

  - "numeric"           — `value_field` holds a number; extracted directly
  - "categorical"       — `value_field` holds a string; mapped via
                          `category_to_score` to a number (e.g. A→1, B→0)
  - "ranking_position"  — `value_field` holds a list; we return the
                          1-based position of the first element matching
                          any of `target_aliases` (case-insensitive
                          substring match)

Arms can be either single-turn (set `prompt`) or multi-turn (set `turns`,
a tuple of user message strings — the runner injects model responses
between them and only scores the final response). The scenario's
`response_format` is appended to the FINAL user turn only.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Arm:
    """One condition within a scenario.

    Set exactly one of `prompt` (single-turn) or `turns` (multi-turn,
    each entry is a user message; the runner alternates user/assistant).
    """
    key: str          # short id, used in cache + chart labels (e.g. "control")
    label: str        # human-readable label for charts (e.g. "Control")
    role: str         # "control" or "treatment"
    prompt: str | None = None
    turns: tuple[str, ...] | None = None
    system: str | None = None   # optional system message prepended to the conversation

    def __post_init__(self):
        has_prompt = self.prompt is not None
        has_turns = self.turns is not None
        if has_prompt == has_turns:
            raise ValueError(
                f"Arm '{self.key}' must have exactly one of `prompt` "
                f"(single-turn) or `turns` (multi-turn)."
            )
        if has_turns and len(self.turns) == 0:  # type: ignore[arg-type]
            raise ValueError(f"Arm '{self.key}' has empty `turns` tuple.")

    @property
    def turn_list(self) -> tuple[str, ...]:
        """Always-tuple view of user turns. Single-turn arms become a
        tuple of length 1; multi-turn arms return their `turns` directly."""
        if self.turns is not None:
            return self.turns
        assert self.prompt is not None  # enforced in __post_init__
        return (self.prompt,)

    @property
    def is_multi_turn(self) -> bool:
        return self.turns is not None and len(self.turns) > 1


@dataclass(frozen=True)
class Scenario:
    """One probe of a cognitive bias.

    All arms must elicit the same JSON shape so analysis is consistent
    across arms. `response_format` is appended to the final user turn.
    `expected_direction` is documentation only — analysis simply compares
    treatment to control without baking in a direction.
    """
    id: str                       # globally unique scenario id
    bias_type: str                # e.g. "anchoring", "framing"
    title: str                    # short title for charts
    description: str              # one-paragraph description of the probe
    arms: tuple[Arm, ...]
    response_format: str          # appended to the final user turn

    # Scoring
    value_field: str              # JSON field name to extract from the response
    value_unit: str               # display unit (e.g. "lakes", "EUR/year")
    expected_direction: str       # human-readable expected bias direction

    # Optional, kind-dependent extraction config
    value_kind: str = "numeric"   # "numeric" | "categorical" | "ranking_position"
    category_to_score: dict[str, float] | None = None
    target_aliases: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.value_kind == "categorical" and not self.category_to_score:
            raise ValueError(
                f"Scenario '{self.id}': value_kind='categorical' requires "
                "`category_to_score`."
            )
        if self.value_kind == "ranking_position" and not self.target_aliases:
            raise ValueError(
                f"Scenario '{self.id}': value_kind='ranking_position' requires "
                "`target_aliases`."
            )
        if self.value_kind not in ("numeric", "categorical", "ranking_position"):
            raise ValueError(
                f"Scenario '{self.id}': unknown value_kind {self.value_kind!r}."
            )

    @property
    def control(self) -> Arm:
        for arm in self.arms:
            if arm.role == "control":
                return arm
        raise ValueError(f"Scenario {self.id} has no control arm")

    @property
    def treatments(self) -> tuple[Arm, ...]:
        return tuple(a for a in self.arms if a.role != "control")

    def arm(self, key: str) -> Arm:
        for a in self.arms:
            if a.key == key:
                return a
        raise KeyError(f"Scenario {self.id} has no arm '{key}'")
