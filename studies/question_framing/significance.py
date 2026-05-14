"""Statistical significance testing for the Question Framing study.

Each HLE question is asked under every framing, so the design is paired:

  - Cochran's Q  (omnibus): "Do any framings differ?" — chi-square distributed
                  with k-1 df, generalises McNemar to >2 conditions.
  - McNemar      (post-hoc): "Does this framing differ from control?" —
                  exact two-sided binomial on the b/c discordant pairs.
  - Holm         (multiplicity): step-down adjustment for the k-1
                  framing-vs-control comparisons.

All stats kernels are implemented from scratch to avoid a scipy dependency.
"""

import math

from studies.question_framing.config import (
    FRAMING_KEYS,
    TEMPERATURE,
    ITERATIONS,
    GRADER_MODEL,
)
from studies.question_framing.runner import build_messages
from studies.question_framing.cache import (
    load_response_cache,
    get_responses,
    load_grading_cache,
    get_grading,
)


# ---------------------------------------------------------------------------
# Per-(subject, framing) binary outcome matrices
# ---------------------------------------------------------------------------

def _verdict_for(response_cache, grading_cache, framing_key, question) -> int | None:
    """Return 1 (CORRECT), 0 (INCORRECT), or None (no graded response)."""
    messages = build_messages(framing_key, question)
    responses = get_responses(response_cache, messages, TEMPERATURE)[:ITERATIONS]
    for result in responses:
        if result.get("error") is not None or result.get("response") is None:
            continue
        verdict = get_grading(
            grading_cache, question["id"], result["response"], GRADER_MODEL,
        )
        if verdict == "CORRECT":
            return 1
        if verdict == "INCORRECT":
            return 0
    return None


def build_per_model_matrix(
    model: str,
    questions: list[dict],
    framings: list[str],
) -> tuple[list[list[int]], list[dict]]:
    """N×K binary matrix for one model. Only questions with a graded
    response under EVERY framing are kept (Cochran's Q requires complete cases)."""
    response_cache = load_response_cache(model)
    grading_cache  = load_grading_cache(model)
    matrix: list[list[int]] = []
    kept:   list[dict] = []
    for q in questions:
        row = [_verdict_for(response_cache, grading_cache, f, q) for f in framings]
        if any(v is None for v in row):
            continue
        matrix.append(row)
        kept.append(q)
    return matrix, kept


def build_pooled_matrix(
    models: list[str],
    questions: list[dict],
    framings: list[str],
) -> tuple[list[list[int]], list[tuple[str, str]]]:
    """Pooled (model, question) × framing matrix. Each row is a (model, question)
    pair with complete coverage across all framings."""
    matrix: list[list[int]] = []
    keys:   list[tuple[str, str]] = []
    for model in models:
        sub, kept = build_per_model_matrix(model, questions, framings)
        for row, q in zip(sub, kept):
            matrix.append(row)
            keys.append((model, q["id"]))
    return matrix, keys


# ---------------------------------------------------------------------------
# Stats kernels
# ---------------------------------------------------------------------------

def _regularized_gamma_p(s: float, x: float) -> float:
    """Regularized lower incomplete gamma P(s, x). Series + continued fraction."""
    if x <= 0:
        return 0.0
    log_prefix = -x + s * math.log(x) - math.lgamma(s)
    if x < s + 1:
        term = 1.0 / s
        total = term
        for n in range(1, 2000):
            term *= x / (s + n)
            total += term
            if abs(term) < 1e-15 * abs(total):
                break
        return total * math.exp(log_prefix)
    # Lentz's continued fraction for Q(s, x), then return 1 - Q
    b = x + 1.0 - s
    c = 1e300
    d = 1.0 / b
    h = d
    for n in range(1, 2000):
        an = -n * (n - s)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return 1.0 - h * math.exp(log_prefix)


def chi2_sf(x: float, df: int) -> float:
    """Survival function of chi-square: P(X > x) where X ~ chi2(df)."""
    if df <= 0:
        return float("nan")
    if x <= 0:
        return 1.0
    return 1.0 - _regularized_gamma_p(df / 2.0, x / 2.0)


def cochrans_q(matrix: list[list[int]]) -> tuple[float, int, float]:
    """Cochran's Q test. matrix is N×K binary, rows=subjects, cols=treatments.
    Returns (Q, df, p)."""
    n = len(matrix)
    if n == 0:
        return float("nan"), 0, float("nan")
    k = len(matrix[0])
    if k < 2:
        return float("nan"), max(0, k - 1), float("nan")
    col_sums = [sum(matrix[i][j] for i in range(n)) for j in range(k)]
    row_sums = [sum(matrix[i]) for i in range(n)]
    T = sum(col_sums)
    sum_col_sq = sum(s * s for s in col_sums)
    sum_row_sq = sum(s * s for s in row_sums)
    denom = k * T - sum_row_sq
    if denom == 0:
        return 0.0, k - 1, 1.0
    Q = (k - 1) * (k * sum_col_sq - T * T) / denom
    return Q, k - 1, chi2_sf(Q, k - 1)


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. Under H0 the b+c discordant pairs
    are 50/50, so min(b,c) ~ Binomial(b+c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cum = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * cum)


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment. Returns adjusted p-values
    in the original input order."""
    n = len(pvalues)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [0.0] * n
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, pvalues[idx] * (n - rank))
        running = max(running, adj)
        adjusted[idx] = running
    return adjusted


# ---------------------------------------------------------------------------
# Significance computation
# ---------------------------------------------------------------------------

def compute_significance(
    matrix: list[list[int]],
    framings: list[str],
    control_key: str = "control",
) -> dict:
    """Cochran's Q over all framings + pairwise McNemar vs control with Holm.

    Returns:
        {
          "n_subjects": int,
          "framings":   list[str],
          "control_key": str,
          "cochran":    {"Q", "df", "p"} | None,
          "pairwise":   {framing_key: {"delta", "b", "c", "p_raw", "p_holm"}},
        }
    """
    out = {
        "n_subjects":  len(matrix),
        "framings":    list(framings),
        "control_key": control_key,
        "cochran":     None,
        "pairwise":    {},
    }
    if not matrix:
        return out

    Q, df, p_q = cochrans_q(matrix)
    out["cochran"] = {"Q": Q, "df": df, "p": p_q}

    if control_key not in framings:
        return out
    ctrl_idx = framings.index(control_key)
    n = len(matrix)
    ctrl_acc = sum(row[ctrl_idx] for row in matrix) / n

    p_raws: list[float] = []
    keys:   list[str]   = []
    for j, fk in enumerate(framings):
        if j == ctrl_idx:
            continue
        b = sum(1 for row in matrix if row[j] == 1 and row[ctrl_idx] == 0)
        c = sum(1 for row in matrix if row[j] == 0 and row[ctrl_idx] == 1)
        framing_acc = sum(row[j] for row in matrix) / n
        p = mcnemar_exact_p(b, c)
        out["pairwise"][fk] = {
            "delta":  framing_acc - ctrl_acc,
            "b":      b,
            "c":      c,
            "p_raw":  p,
            "p_holm": p,  # overwritten below
        }
        p_raws.append(p)
        keys.append(fk)

    for fk, padj in zip(keys, holm_adjust(p_raws)):
        out["pairwise"][fk]["p_holm"] = padj

    return out
