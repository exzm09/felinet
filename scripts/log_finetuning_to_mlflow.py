"""
Log-fine-tuning experiment to MLflow.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def log_finetuning_experiment(
    results_path: str = "models/felinet-embedding-v1/training_results.json",
    experiment_name: str = "felinet-embeddings",
) -> None:
    """
    Log the fine-tuning experiment results to MLflow.

    Parameters
    ----------
    results_path : str
        Path to the training_results.json saved from Colab.
    experiment_name : str
        MLflow experiment name.
    """
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found at {results_path}. ")

    with open(results_path) as f:
        results = json.load(f)

    # Set or create the experiment
    mlflow.set_experiment(experiment_name)

    # Log baseline run
    with mlflow.start_run(run_name="baseline-all-MiniLM-L6-v2"):
        mlflow.log_param("model", "all-MiniLM-L6-v2")
        mlflow.log_param("type", "baseline")
        mlflow.log_param("embedding_dim", 384)

        for key, value in results.get("baseline_metrics", {}).items():
            # Clean up the metric name for MLflow
            clean_key = key.replace("felinet-val_", "").replace("@", "_at_")
            mlflow.log_metric(clean_key, value)

        logger.info("Logged baseline run to MLflow")

    # Log fine-tuned run
    with mlflow.start_run(run_name="finetuned-felinet-v1"):
        # Training parameters
        mlflow.log_param("model", "felinet-embedding-v1")
        mlflow.log_param("base_model", results.get("base_model", "all-MiniLM-L6-v2"))
        mlflow.log_param("type", "fine-tuned")
        mlflow.log_param("training_pairs", results.get("training_pairs", 0))
        mlflow.log_param("validation_pairs", results.get("validation_pairs", 0))
        mlflow.log_param("epochs", results.get("epochs", 4))
        mlflow.log_param("batch_size", results.get("batch_size", 32))
        mlflow.log_param("learning_rate", results.get("learning_rate", 2e-5))
        mlflow.log_param("embedding_dim", 384)

        for key, value in results.get("finetuned_metrics", {}).items():
            clean_key = key.replace("felinet-val_", "").replace("@", "_at_")
            mlflow.log_metric(clean_key, value)

        # Also log the improvement deltas
        baseline = results.get("baseline_metrics", {})
        finetuned = results.get("finetuned_metrics", {})
        for key in baseline:
            if key in finetuned:
                delta = finetuned[key] - baseline[key]
                clean_key = key.replace("felinet-val_", "").replace("@", "_at_")
                mlflow.log_metric(f"improvement_{clean_key}", delta)

        # Log the results file as an artifact
        mlflow.log_artifact(str(results_path))

        logger.info("Logged fine-tuned run to MLflow")

    # Print summary
    print("\n" + "=" * 60)
    print("MLFLOW LOGGING COMPLETE")
    print("=" * 60)
    print(f"  Experiment: {experiment_name}")
    print("  Runs logged: 2 (baseline + fine-tuned)")
    print("    Then open http://localhost:5000 in your browser")
    print("=" * 60)


if __name__ == "__main__":
    log_finetuning_experiment()
