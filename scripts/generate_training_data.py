"""
Run synthetic question generation for embedding fine-tunning.
"""

import argparse
import sys
import time
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from felinet.data.loader import load_corpus
from felinet.embeddings.chunker import chunk_corpus
from felinet.embeddings.training_data import TrainingDataConfig, run_generation
from felinet.schemas import ChunkingConfig


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training pairs for embedding fine-tuning"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only the first N chunks (for testing)"
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=3,
        help="Number of questions to generate per chunk (default: 3)",
    )
    args = parser.parse_args()
    print("=" * 60)
    print("FeliNet - Synthetic Training Data Generation")
    print("=" * 60)

    # Step 1
    print("\n[1/3] Loading corpus...")
    documents = load_corpus()
    print(f"{len(documents)} documents loaded")

    print("\n[2/3] Chunking...")
    config = ChunkingConfig()
    chunks = chunk_corpus(documents, config)
    print(f"{len(chunks)} chunks created")

    # Conver DocumentChunk objects to dicts for generator
    chunk_dicts = []
    for c in chunks:
        chunk_dicts.append(
            {
                "id": c.id,
                "content": c.content,
                "source": c.source.value,
                "title": c.metadata.get("title", ""),
            }
        )

    # Apply limit if specified
    if args.limit:
        chunk_dicts = chunk_dicts[: args.limit]
        print(f"Limited to first {args.limit} chunks")

    # Step 2: Generate training pairs
    print(f"\n[3/3] Generating {args.questions} questions per chunk...")
    print(f"  Estimated API calls: {len(chunk_dicts)}")
    print(f"  Estimated cost: ~${len(chunk_dicts) * 0.001:.2f}")
    print(f"  Estimated time: ~{len(chunk_dicts) * 0.5 / 60:.0f} minutes")
    print()

    gen_config = TrainingDataConfig()

    start = time.time()
    output_path = run_generation(chunk_dicts, gen_config)
    elapsed = time.time() - start

    # Summary
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
        print(f"\nTotal training pairs in {output_path}: {total_lines}")
    print(f"Total time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
