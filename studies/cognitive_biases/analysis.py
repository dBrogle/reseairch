"""Statistics for the Cognitive Biases study.

For each (scenario, model, arm), compute:
  - n / mean / 95% t-CI of the per-iteration values

For each (scenario, model, treatment_arm) compute the bias delta:
  - mean_diff = treatment_mean - control_mean
  - 95% CI on the difference using Welch's two-sample t formula

CIs are computed with hard-coded t-table critical values so we don't need
scipy.
"""

import math

from studies.cognitive_biases.cache import (
    get_extraction,
    get_responses,
    load_extraction_cache,
    load_response_cache,
    request_signature,
)
from studies.cognitive_biases.config import EXTRACTOR_MODEL
from studies.cognitive_biases.scenarios.base import Scenario

# t-distribution critical values for 95% two-sided CI, df = n-1.
_T_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    40: 2.021, 50: 2.009, 60: 2.000, 80: 1.990, 100: 1.984,
    200: 1.972,
}


def _t_critical_95(df: float) -> float:
    if df <= 0 or math.isnan(df):
        return float("nan")
    if df >= 200:
        return 1.960
    keys = sorted(_T_95.keys())
    df_int = int(round(df))
    if df_int in _T_95:
        return _T_95[df_int]
    for k in keys:
        if k > df_int:
            return _T_95[k]
    return 1.960


def t_ci_95(values: list[float]) -> tuple[float, float, float]:
    """Mean and t-distribution 95% CI. Returns (mean, lo, hi).

    For n == 0 returns NaNs; for n == 1 returns (mean, mean, mean).
    """
    n = len(values)
    if n == 0:
        nan = float("nan")
        return (nan, nan, nan)
    mean = sum(values) / n
    if n == 1:
        return (mean, mean, mean)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(var)
    sem = sd / math.sqrt(n)
    margin = _t_critical_95(n - 1) * sem
    return (mean, mean - margin, mean + margin)


def _welch_diff_ci(
    treat: list[float], control: list[float]
) -> tuple[float, float, float] | None:
    """Welch two-sample 95% CI on (mean_treat - mean_control)."""
    n1, n2 = len(treat), len(control)
    if n1 < 2 or n2 < 2:
        return None
    m1 = sum(treat) / n1
    m2 = sum(control) / n2
    v1 = sum((v - m1) ** 2 for v in treat) / (n1 - 1)
    v2 = sum((v - m2) ** 2 for v in control) / (n2 - 1)
    diff = m1 - m2
    se2 = v1 / n1 + v2 / n2
    if se2 <= 0:
        return (diff, diff, diff)
    se = math.sqrt(se2)
    # Welch–Satterthwaite degrees of freedom
    num = (v1 / n1 + v2 / n2) ** 2
    denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df = num / denom if denom > 0 else min(n1, n2) - 1
    margin = _t_critical_95(df) * se
    return (diff, diff - margin, diff + margin)


# ---------------------------------------------------------------------------
# Score collection
# ---------------------------------------------------------------------------

def collect_values(
    model: str,
    scenario: Scenario,
    temperature: float,
    iterations: int,
) -> dict[str, list[float]]:
    """Return {arm_key: [values]} for one (model, scenario)."""
    response_cache = load_response_cache(model)
    extraction_cache = load_extraction_cache(model)

    out: dict[str, list[float]] = {arm.key: [] for arm in scenario.arms}
    for arm in scenario.arms:
        sig = request_signature(scenario, arm)
        responses = get_responses(response_cache, sig, temperature)[:iterations]
        for r in responses:
            if r.get("error") is not None or r.get("response") is None:
                continue
            ext = get_extraction(
                extraction_cache,
                scenario.id, arm.key, r["response"],
                scenario.value_field, EXTRACTOR_MODEL,
            )
            if ext is None:
                continue
            v = ext.get("value")
            if v is None:
                continue
            try:
                out[arm.key].append(float(v))
            except (TypeError, ValueError):
                continue
    return out


def compute_scenario_stats(
    model: str,
    scenario: Scenario,
    temperature: float,
    iterations: int,
) -> dict:
    """Per-arm summary stats + per-treatment delta vs control.

    Shape:
    {
      "model":     str,
      "scenario":  Scenario,
      "per_arm": {
          arm_key: {"mean", "ci_low", "ci_high", "n", "values"}
      },
      "deltas": {
          treatment_arm_key: {"diff", "ci_low", "ci_high"} | None
      },
    }
    """
    values = collect_values(model, scenario, temperature, iterations)

    per_arm: dict[str, dict] = {}
    for arm in scenario.arms:
        vals = values[arm.key]
        if not vals:
            per_arm[arm.key] = {
                "mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": 0, "values": [],
            }
            continue
        mean, lo, hi = t_ci_95(vals)
        per_arm[arm.key] = {
            "mean": mean, "ci_low": lo, "ci_high": hi,
            "n": len(vals), "values": list(vals),
        }

    control_vals = values[scenario.control.key]
    deltas: dict[str, dict | None] = {}
    for arm in scenario.treatments:
        treat_vals = values[arm.key]
        if not treat_vals or not control_vals:
            deltas[arm.key] = None
            continue
        ci = _welch_diff_ci(treat_vals, control_vals)
        if ci is None:
            deltas[arm.key] = None
        else:
            diff, lo, hi = ci
            deltas[arm.key] = {"diff": diff, "ci_low": lo, "ci_high": hi}

    return {
        "model":    model,
        "scenario": scenario,
        "per_arm":  per_arm,
        "deltas":   deltas,
    }
