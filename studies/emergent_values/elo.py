"""Bradley-Terry model fitting for the Emergent Values study.

Given pairwise win probabilities from LLM comparisons, fits a Bradley-Terry
model to estimate utility scores (Elo-like ratings) for each option.

The Bradley-Terry model assumes P(A beats B) = sigma(mu_A - mu_B) where
sigma is the logistic function. We optimize mu values via gradient descent
to minimize binary cross-entropy loss.
"""

import math
from collections import defaultdict


def _sigmoid(x: float) -> float:
    """Logistic sigmoid, clamped to avoid overflow."""
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def compute_win_probabilities(
    cache: dict,
    options: list[tuple[str, str]],
    pairs: list[tuple[int, int]],
    bidirectional: bool,
) -> list[tuple[int, int, float, int]]:
    """Compute empirical P(A > B) for each pair from cached results.

    Returns [(idx_a, idx_b, p_a_wins, n_total), ...] for pairs with data.
    """
    edges = []
    for idx_a, idx_b in pairs:
        label_a, text_a = options[idx_a]
        label_b, text_b = options[idx_b]

        # Forward direction: option A presented first
        forward_results = []
        for key, entry in cache.items():
            if entry.get("option_a") == text_a and entry.get("option_b") == text_b:
                forward_results = entry.get("results", [])
                break

        # Reverse direction: option B presented first
        reverse_results = []
        if bidirectional:
            for key, entry in cache.items():
                if entry.get("option_a") == text_b and entry.get("option_b") == text_a:
                    reverse_results = entry.get("results", [])
                    break

        # Accumulate P(idx_a wins) using soft p_a when available, binary otherwise
        p_a_sum = 0.0
        n_total = 0

        for r in forward_results:
            if r.get("error") is not None:
                continue
            if "p_a" in r and r["p_a"] is not None:
                # Logprobs: p_a = P(chose A) = P(idx_a wins)
                p_a_sum += r["p_a"]
                n_total += 1
            elif r.get("choice") == "A":
                p_a_sum += 1.0
                n_total += 1
            elif r.get("choice") == "B":
                p_a_sum += 0.0
                n_total += 1

        for r in reverse_results:
            if r.get("error") is not None:
                continue
            if "p_a" in r and r["p_a"] is not None:
                # Reverse: p_a = P(chose A) = P(idx_b wins), so P(idx_a wins) = 1 - p_a
                p_a_sum += (1.0 - r["p_a"])
                n_total += 1
            elif r.get("choice") == "B":
                # In reverse, choosing B = choosing text_a = idx_a wins
                p_a_sum += 1.0
                n_total += 1
            elif r.get("choice") == "A":
                # In reverse, choosing A = choosing text_b = idx_b wins
                p_a_sum += 0.0
                n_total += 1

        if n_total > 0:
            p_a = p_a_sum / n_total
            edges.append((idx_a, idx_b, p_a, n_total))

    return edges


def fit_bradley_terry(
    n_options: int,
    edges: list[tuple[int, int, float, int]],
    num_epochs: int = 1000,
    learning_rate: float = 0.05,
) -> tuple[list[float], float, float]:
    """Fit Bradley-Terry model via gradient descent.

    Args:
        n_options: Total number of options.
        edges: [(idx_a, idx_b, p_a_wins, n_observations), ...]
        num_epochs: Gradient descent iterations.
        learning_rate: Step size.

    Returns:
        (utilities, train_loss, train_accuracy)
        where utilities[i] is the Elo-like score for option i.
    """
    if not edges:
        return [0.0] * n_options, 0.0, 0.0

    # Initialize utilities to zero
    mu = [0.0] * n_options

    for epoch in range(num_epochs):
        # Normalize to mean=0
        mean_mu = sum(mu) / n_options
        mu = [m - mean_mu for m in mu]

        # Compute gradients
        grad = [0.0] * n_options
        for idx_a, idx_b, p_a, n_obs in edges:
            pred = _sigmoid(mu[idx_a] - mu[idx_b])
            error = (p_a - pred) * n_obs
            grad[idx_a] += error
            grad[idx_b] -= error

        # Update
        for i in range(n_options):
            mu[i] += learning_rate * grad[i]

    # Normalize final: mean=0, std proportional to Elo scale (400)
    mean_mu = sum(mu) / n_options
    mu = [m - mean_mu for m in mu]
    if n_options > 1:
        var = sum(m * m for m in mu) / n_options
        std = math.sqrt(var) if var > 0 else 1.0
        mu = [m / std * 400.0 for m in mu]

    # Compute training metrics
    total_loss = 0.0
    correct = 0
    total = 0
    for idx_a, idx_b, p_a, n_obs in edges:
        pred = _sigmoid((mu[idx_a] - mu[idx_b]) / 400.0)
        # Binary cross-entropy
        pred_clamped = max(1e-7, min(1 - 1e-7, pred))
        loss = -(p_a * math.log(pred_clamped) + (1 - p_a) * math.log(1 - pred_clamped))
        total_loss += loss
        if (pred >= 0.5 and p_a >= 0.5) or (pred < 0.5 and p_a < 0.5):
            correct += 1
        total += 1

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return mu, avg_loss, accuracy


def aggregate_entity_scores(
    options: list[tuple[str, str]],
    utilities: list[float],
) -> dict[str, float]:
    """Aggregate per-option utilities into per-entity scores.

    Options are labeled like 'India_10', 'India_50', etc.
    Returns {entity: mean_utility} averaged across all measure values.
    """
    entity_scores = defaultdict(list)
    for (label, _text), utility in zip(options, utilities):
        # Label format: "Entity_N" - split on last underscore
        parts = label.rsplit("_", 1)
        entity = parts[0]
        entity_scores[entity].append(utility)

    return {entity: sum(scores) / len(scores) for entity, scores in entity_scores.items()}
