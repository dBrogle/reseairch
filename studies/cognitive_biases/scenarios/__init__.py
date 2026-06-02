"""Scenario registry for the Cognitive Biases study.

Each cognitive bias gets one or more scenario files in this folder. To
add a new bias or scenario:

  1. Create `scenarios/<scenario_id>.py` exposing a top-level `SCENARIO`
     of type `studies.cognitive_biases.scenarios.base.Scenario`.
  2. Import it here and append to `ALL_SCENARIOS`.

Scenarios are grouped by bias family in the order shown in CLI menus
and graphs.
"""

from studies.cognitive_biases.scenarios import (
    anchor_diamond_ring,
    anchor_luxury_watch,
    anchor_used_car,
    authority_argument,
    authority_health_claim,
    endowment_trashcan,
    endowment_world_cup,
    framing_bootcamp,
    framing_parole,
    framing_surgery,
    hindsight_d2c_beauty,
    hindsight_novelist,
    hindsight_restaurant,
    sunk_cost_movie,
)
from studies.cognitive_biases.scenarios.base import Arm, Scenario

ALL_SCENARIOS: tuple[Scenario, ...] = (
    # ---- Anchoring ----
    anchor_used_car.SCENARIO,
    anchor_diamond_ring.SCENARIO,
    anchor_luxury_watch.SCENARIO,
    # ---- Framing (loss vs. gain) ----
    framing_parole.SCENARIO,
    framing_bootcamp.SCENARIO,
    framing_surgery.SCENARIO,
    # ---- Authority ----
    authority_argument.SCENARIO,
    authority_health_claim.SCENARIO,
    # ---- Hindsight ----
    hindsight_restaurant.SCENARIO,
    hindsight_novelist.SCENARIO,
    hindsight_d2c_beauty.SCENARIO,
    # ---- Sunk cost ----
    sunk_cost_movie.SCENARIO,
    # ---- Endowment ----
    endowment_trashcan.SCENARIO,
    endowment_world_cup.SCENARIO,
)

SCENARIOS_BY_ID: dict[str, Scenario] = {s.id: s for s in ALL_SCENARIOS}

BIAS_TYPES: tuple[str, ...] = tuple(
    dict.fromkeys(s.bias_type for s in ALL_SCENARIOS).keys()
)


def scenarios_by_bias(bias_type: str) -> tuple[Scenario, ...]:
    return tuple(s for s in ALL_SCENARIOS if s.bias_type == bias_type)


__all__ = [
    "ALL_SCENARIOS",
    "SCENARIOS_BY_ID",
    "BIAS_TYPES",
    "Arm",
    "Scenario",
    "scenarios_by_bias",
]
