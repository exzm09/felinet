"""
Log the Week 4 baseline experiment to MLflow.

Record chunking config, embedding model, corpus stats, and retrieval quality observations so we have a reference point for every future improvement.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mlflow

from felinet.data.loader import load_corpus
from felinet.embeddings.chunker import chunk_corpus, count_tokens
from felinet.schemas import ChunkingConfig

EXPERIMENT_NAME = "felinet-rag-pipeline"

def main():
    # Set up experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load and chunk (reuse existing pipeline)
    documents = load_corpus()
    config = ChunkingConfig()
    chunks = chunk_corpus(documents, config)
    token_counts = [c.token_count for c in chunks]

    with mlflow.start_run(run_name="week4-baseline-dense-retrieval"):
        # Chunking parameters
        mlflow.log_params({
            "chunk_size_tokens": config.chunk_size,
            "chunk_overlap_tokens": config.chunk_overlap,
            "splitter": "RecursiveCharacterTextSplitter",
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dim": 384,
            "distance_metric": "cosine",
            "vector_store": "qdrant",
            "retrieval_method": "dense_only",
            "top_k": 5
        })

        # Corpus & chunk stats
        mlflow.log_metrics({
            "num_documents": len(documents),
            "num_chunks": len(chunks),
            "avg_chunks_per_doc": len(chunks) / len(documents),
            "token_count_min": min(token_counts),
            "token_count_max": max(token_counts),
            "token_count_mean": sum(token_counts) / len(token_counts)
        })

        # Retrieval quality observations (from test)
        mlflow.log_metrics({
            "manual_kidney_top1_score": 0.7281,
            "manual_leukemia_top1_score": 0.5805,
            "manual_feeding_top1_score": 0.7126,
            "manual_lilies_top1_score": 0.7471,
            "manual_siamese_top1_score": 0.6115,
            "manual_avg_top1_score": (0.7281 + 0.5805 + 0.7126 + 0.7471 + 0.6115) / 5,
        })

        # Tag for easy filtering later
        mlflow.set_tags({
            "week": "4",
            "phase": "baseline",
            "retrieval_type": "dense_only",
            "notes": "First baseline. Bibliography chunks pollute results. "
                     "Leukemia and Siamese queries underperform."
        })

    print("Experiment logged! Run `mlflow ui` to view.")

if __name__ == "__main__":
    main()