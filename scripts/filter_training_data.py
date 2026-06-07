"""
Filter and clean the raw synthetic training pairs.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


# Quality Filters
def is_too_short(query: str, min_length: int = 15) -> bool:
    """
    Reject questions shorter than min_length characters.
    """
    return len(query.strip()) < min_length


def is_too_generic(query: str) -> bool:
    """
    Reject questions that are too vague to be useful training examples.
    If the question doesn't help the model learn to distinguish between passages, it's noise.
    """
    generic_patterns = [
        r"^what is this",
        r"^what does this",
        r"^tell me about",
        r"^can you explain",
        r"^what are the main points",
        r"^summarize",
        r"^what information",
        r"^describe the",
        r"^what does the passage",
        r"^according to the passage",
        r"^based on the text",
        r"^what is mentioned",
        r"^what are some",
    ]
    query_lower = query.lower().strip()
    return any(re.match(pattern, query_lower) for pattern in generic_patterns)


def reference_passage(query: str) -> bool:
    """
    Reject questions that explicitly reference the source passages.
    Real users never mention it.
    These leak the training setup into the data and confused the model.
    """
    passage_refs = [
        "passage",
        "text",
        "article",
        "paragraph",
        "excerpt",
        "above",
        "mentioned",
        "stated",
        "described here",
    ]
    query_lower = query.lower().strip()
    return any(ref in query_lower for ref in passage_refs)


def is_duplicate(query: str, seen: set[str]) -> bool:
    """
    Reject exact and near-duplicate questions.
    """
    # Remove punctuation, collapse whitespace
    normalized = re.sub(r"[^\w\s]", "", query.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized in seen:
        return True
    seen.add(normalized)
    return False


# Main Filter Pipeline
def filter_training_data(
    input_path: str = "data/training/raw_pairs.jsonl",
    output_path: str = "data/training/filtered_pairs.jsonl",
    min_length: int = 15,
    sample_size: int = 0,
) -> dict:
    """
    Apply all quality filters to the raw training pairs.
    Parameters
    ----------
    input_path : str
        Path to the raw JSONL from the generation step.
    output_path : str
        Where to save the filtered pairs.
    min_length : int
        Minimum question length in characters.
    sample_size : int
        If > 0, print this many random filtered pairs for manual review.

    Returns
    -------
    dict
        Statistics about what was filtered and why.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw pairs file not found at {input_path}. " f"Run generate_training_data.py first."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Track filiter stats
    stats = Counter()
    seen_queries: set[str] = set()
    kept_pairs: list[str] = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1

            pair = json.loads(line)
            query = pair.get("query", "")

            # Apply filters in order (cheapest first)
            if is_too_short(query, min_length):
                stats["removed_too_short"] += 1
                continue
            if is_too_generic(query):
                stats["removed_too_generic"] += 1
                continue
            if reference_passage(query):
                stats["removed_passage_ref"] += 1
                continue
            if is_duplicate(query, seen_queries):
                stats["removed_duplicate"] += 1
                continue

            kept_pairs.append(pair)
            stats["kept"] += 1

    # Write filtered pairs
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in kept_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # Summary
    print("=" * 60)
    print("FILTERING RESULTS")
    print("=" * 60)
    print(f"  Total raw pairs:        {stats['total']}")
    print(
        f"  Kept:                   {stats['kept']} ({stats['kept']/max(stats['total'],1)*100:.1f}%)"
    )
    print(f"  Removed (too short):    {stats['removed_too_short']}")
    print(f"  Removed (too generic):  {stats['removed_too_generic']}")
    print(f"  Removed (passage ref):  {stats['removed_passage_ref']}")
    print(f"  Removed (duplicate):    {stats['removed_duplicate']}")
    print(f"\n  Output: {output_path}")
    print("=" * 60)

    # Show random sample for manual review
    if sample_size > 0 and kept_pairs:
        sample = random.sample(kept_pairs, min(sample_size, len(kept_pairs)))
        print(f"\n{'=' * 60}")
        print(f"RANDOM SAMPLE ({len(sample)} pairs for manual review)")
        print(f"{'=' * 60}")
        for i, pair in enumerate(sample, 1):
            print(f"\n--- Pair {i} ---")
            print(f"  Q: {pair['query']}")
            print(f"  Source: {pair.get('source', 'N/A')}")
            print(f"  Passage: {pair['positive'][:150]}...")

    return dict(stats)


def main():
    parser = argparse.ArgumentParser(description="Filter raw training pairs for quality")
    parser.add_argument(
        "--min-length",
        type=int,
        default=15,
        help="Minimum question length in characters (default: 15)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=20,
        help="Number of random pairs to show for manual review (default: 20)",
    )
    parser.add_argument(
        "--input", type=str, default="data/training/raw_pairs.jsonl", help="Path to raw pairs JSONL"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training/filtered_pairs.jsonl",
        help="Path to save filtered pairs JSONL",
    )
    args = parser.parse_args()

    filter_training_data(
        input_path=args.input,
        output_path=args.output,
        min_length=args.min_length,
        sample_size=args.sample,
    )


if __name__ == "__main__":
    main()
