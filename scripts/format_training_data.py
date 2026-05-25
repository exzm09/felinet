"""
Format filtered training pairs for sentence-transformers fine-tuning.
Take the filtered JSON and convert it into the format that sentence-transformers expects for training with MultipleNegativesRankingLoss.
Output is a Hugging Face Dataset saved to disk.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

def format_for_training(
    input_path: str = "data/training/filtered_pairs.jsonl",
    output_dir: str = "data/training/dataset",
    val_split: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Convert filtered JSONL pairs into train/val datasets for sentence-transformers.
    Parameters
    ----------
    input_path : str
        Path to filtered JSONL (output of filter_training_data.py).
    output_dir : str
        Directory to save the formatted datasets.
    val_split : float
        Fraction of data to hold out for validation (default: 10%).
    seed : int
        Random seed for reproducible splits. Same seed = same split every time.

    Returns
    -------
    dict
        Counts of train/val examples.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Filtered pairs not found at {input_path}. "
        )

    # Load all filtered pairs
    pairs = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                pairs.append(
                    {
                        "anchor": data["query"],
                        "positive": data["positive"]
                    }
                )
    print(f"Loaded {len(pairs)} filtered pairs")

    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(pairs)

    # Split into train and validation
    val_size = int(len(pairs) * val_split)
    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]
    print(f"Train: {len(train_pairs)} pairs ({(1 - val_split) * 100:.0f}%)")
    print(f"Val: {len(val_pairs)} pairs ({val_split * 100:.0f}%)")

    # Save as JSONL files
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    for path, data in [(train_path, train_pairs), (val_path, val_pairs)]:
        with open(path, "w", encoding="utf-8") as f:
            for pair in data:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # Have a metadata file recording 
    meta = {
        "source": str(input_path),
        "total_pairs": len(pairs),
        "train_size": len(train_pairs),
        "val_size": len(val_pairs),
        "val_split": val_split,
        "seed": seed,
        "format": "jsonl with 'anchor' and 'positive' columns",
        "intended_loss": "MultipleNegativesRankingLoss",
        "intended_model": "all-MiniLM-L6-v2",
    }
    meta_path = output_dir / "dataset_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"DATASET READY FOR FINE-TUNING")
    print(f"{'=' * 60}")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"  Meta:  {meta_path}")
    print(f"{'=' * 60}")

    return {
        "train_size": len(train_pairs),
        "val_size": len(val_pairs),
        "total": len(pairs),
    }

def main():
    parser = argparse.ArgumentParser(
        description="Format training pairs for sentence-transformers"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Validation split fraction (default: 0.1)"
    )
    parser.add_argument(
        "--input", type=str, default="data/training/filtered_pairs.jsonl",
    )
    parser.add_argument(
        "--output", type=str, default="data/training/dataset",
    )
    args = parser.parse_args()

    format_for_training(
        input_path=args.input,
        output_dir=args.output,
        val_split=args.val_split,
    )


if __name__ == "__main__":
    main()