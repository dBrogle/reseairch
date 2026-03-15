"""
HLE (Humanity's Last Exam) dataset loading and stratified sampling.

Downloads cais/hle from HuggingFace, filters to text-only questions, and
samples them proportionally across categories. Results are cached to disk
so subsequent runs do not re-download.
"""

import random
from collections import defaultdict
from pathlib import Path
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def load_hle_stratified_sample(per_category: int = 5, seed: int = 42) -> list[dict]:
    """Download HLE and sample exactly `per_category` questions from each category."""
    random.seed(seed)

    print("Loading HLE dataset from Hugging Face (cais/hle)...")
    dataset = load_dataset("cais/hle", split="test")

    # Filter to text-only questions — image field is "" when no image is needed
    text_only = [item for item in dataset if not item["image"]]
    print(f"Text-only questions: {len(text_only)} of {len(dataset)} total")

    # Group by category
    category_buckets = defaultdict(list)
    for item in text_only:
        category_buckets[item["category"]].append(item)

    categories = sorted(category_buckets.keys())
    print(f"Across {len(categories)} categories:")
    for cat in categories:
        count = len(category_buckets[cat])
        taking = min(per_category, count)
        print(f"  {cat}: {count} questions (sampling {taking})")

    # Sample exactly per_category from each category
    sampled = []
    for cat in categories:
        bucket = category_buckets[cat]
        take = min(per_category, len(bucket))
        sampled.extend(random.sample(bucket, take))
    random.shuffle(sampled)

    return [
        {
            "id":          q["id"],
            "question":    q["question"],
            "answer":      q["answer"],
            "answer_type": q["answer_type"],
            "raw_subject": q["raw_subject"],
            "category":    q["category"],
        }
        for q in sampled
    ]


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
