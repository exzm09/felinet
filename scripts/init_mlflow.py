"""
Verify MLflow setup by logging a dummy experiment.
"""

import mlflow

def main():
    experiment_name = "felinet-rag-pipeline"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="week1-scaffold-verification"):
        mlflow.log_params({
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "retrieval_top_k": 30,
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "llm_model": "llama-3.3-70b-versatile"
        })

        mlflow.log_metrics(
            {
                "faithfullness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "latency_p50_ms": 0.0
            }
        )

        mlflow.set_tags(
            {
                "phase": "week1_scaffold",
                "pipeline_version": "0.1.0",
                "status": "verification-only"
            }
        )
    
    print(f"MLflow experiment '{experiment_name}' created successfully.")
    print(f"Run 'mlflow ui --port 5000' and open http://localhost:5000")


if __name__ == "__main__":
    main()