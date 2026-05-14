"""
HLE (Humanity's Last Exam) dataset loading and stratified sampling.

Downloads cais/hle from HuggingFace, filters to text-only questions, and
samples them proportionally across categories. Results are cached to disk
so subsequent runs do not re-download.
"""

import hashlib
import random
from collections import defaultdict
from pathlib import Path
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def _stable_priority(seed: int, category: str, qid: str) -> str:
    """Per-item hash priority. Independent of bucket size, so adding or
    removing items from upstream HLE never reorders the surviving items."""
    return hashlib.sha256(f"{seed}:{category}:{qid}".encode()).hexdigest()


_SLIM_KEYS = ("id", "question", "answer", "answer_type", "raw_subject", "category")


def _slim(q: dict) -> dict:
    return {k: q[k] for k in _SLIM_KEYS}


def load_hle_stratified_sample(
    per_category: int = 5,
    seed: int = 42,
    base_questions: list[dict] | None = None,
) -> list[dict]:
    """Download HLE and sample exactly `per_category` questions from each category.

    Prefix-consistent across both per_category and upstream dataset changes:
    each item's rank within its category is `sha256(seed:cat:id)`, which depends
    only on the item itself. Adding or removing items elsewhere in the bucket
    never reorders the survivors.

    If `base_questions` is given, those are kept as a per-category prefix and
    the function only samples (per_category - existing_in_category) more from
    the current dataset, excluding any IDs already in the base. This lets a
    smaller cached sample anchor a larger one so existing response caches
    remain valid.
    """
    print("Loading HLE dataset from Hugging Face (cais/hle)...")
    dataset = load_dataset("cais/hle", split="test")

    # Filter to text-only questions — image field is "" when no image is needed
    text_only = [item for item in dataset if not item["image"]]
    print(f"Text-only questions: {len(text_only)} of {len(dataset)} total")

    # Group by category
    category_buckets = defaultdict(list)
    for item in text_only:
        category_buckets[item["category"]].append(item)

    base_by_cat: dict[str, list[dict]] = defaultdict(list)
    base_ids: set[str] = set()
    if base_questions:
        for q in base_questions:
            base_by_cat[q["category"]].append(_slim(q))
            base_ids.add(q["id"])

    categories = sorted(category_buckets.keys())
    print(f"Across {len(categories)} categories:")
    for cat in categories:
        count = len(category_buckets[cat])
        taking = min(per_category, count)
        anchored = len(base_by_cat.get(cat, []))
        new = max(0, taking - anchored)
        print(f"  {cat}: {count} questions "
              f"(keeping {anchored} cached, sampling {new} new)")

    sampled: list[dict] = []
    for cat in categories:
        existing = base_by_cat.get(cat, [])
        sampled.extend(existing)
        need = per_category - len(existing)
        if need <= 0:
            continue
        candidates = sorted(
            (q for q in category_buckets[cat] if q["id"] not in base_ids),
            key=lambda q: _stable_priority(seed, cat, q["id"]),
        )
        for q in candidates[:need]:
            sampled.append(_slim(q))

    # Final shuffle only affects iteration order — the response cache is keyed
    # by message-hash, not position, so this does not break cache reuse.
    final_rng = random.Random(f"{seed}:final")
    final_rng.shuffle(sampled)

    return sampled


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    questions = load_hle_stratified_sample(per_category=5, seed=42)

    print(f"\n--- Sampled {len(questions)} questions ---\n")
    for i, q in enumerate(questions, 1):
        print(f"[{i:02d}] [{q['category']}] {q['raw_subject']}")
        print(f"      {q['question'][:120]}{'...' if len(q['question']) > 120 else ''}")
        print()
