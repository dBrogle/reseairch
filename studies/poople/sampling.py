"""Deterministic, prefix-stable sampling of test words by difficulty.

For each distance bucket we sort that bucket's words (reproducible base order),
shuffle once with a fixed seed, and take the first N. Because we always take a
*prefix* of the same shuffled order, raising SAMPLE_PER_BUCKET keeps every word
(and its cached LLM results) that a smaller N already produced, and only appends
new words — so growing the test set never wastes prior work.
"""

import random

from studies.poople.config import (
    SAMPLE_BUCKETS,
    SAMPLE_PER_BUCKET,
    SAMPLE_SEED,
)


def _shuffled_bucket(words_at_dist: list[str], distance: int) -> list[str]:
    """All words at `distance`, in a fixed seed-derived shuffled order.

    The seed is combined with the distance so different buckets get independent
    orders while staying reproducible run-to-run.
    """
    ordered = sorted(words_at_dist)
    rng = random.Random(f"{SAMPLE_SEED}:{distance}")
    rng.shuffle(ordered)
    return ordered


def sample_test_words(
    dist: dict[str, int],
    buckets: tuple[int, ...] = SAMPLE_BUCKETS,
    per_bucket: int = SAMPLE_PER_BUCKET,
) -> dict[int, list[str]]:
    """Return {distance: [sampled words]} for each requested bucket.

    `dist` is the solver's word->distance oracle. If a bucket has fewer than
    `per_bucket` words, all of them are returned.
    """
    by_dist: dict[int, list[str]] = {d: [] for d in buckets}
    for word, d in dist.items():
        if d in by_dist:
            by_dist[d].append(word)

    return {
        d: _shuffled_bucket(by_dist[d], d)[:per_bucket]
        for d in buckets
    }


def flat_test_words(
    dist: dict[str, int],
    buckets: tuple[int, ...] = SAMPLE_BUCKETS,
    per_bucket: int = SAMPLE_PER_BUCKET,
) -> list[tuple[str, int]]:
    """Flatten the sample into an ordered list of (word, par) pairs."""
    sample = sample_test_words(dist, buckets, per_bucket)
    out: list[tuple[str, int]] = []
    for d in buckets:
        for word in sample[d]:
            out.append((word, d))
    return out
