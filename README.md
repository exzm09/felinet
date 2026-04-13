# FeliNet 🐱

A feline health and breed knowledge assistant powered by a deep RAG pipeline with full MLOps infrastructure.

## What is this?

FeliNet is an open-source RAG system built over a curated corpus of veterinary and breed information. It combines:

- **Hybrid search** (BM25 + dense retrieval with reciprocal rank fusion)
- **Cross-encoder reranking** for precision
- **Fine-tuned domain embeddings** on feline veterinary text
- **Guardrailed LLM generation** with source citations
- **Full MLOps stack**: experiment tracking, data versioning, CI/CD quality gates, observability, and drift detection

## Project structure

```
felinet/
├── src/felinet/          # Source code
│   ├── data/             # Ingestion, scraping, ETL
│   ├── rag/              # Retrieval, reranking, generation
│   ├── embeddings/       # Fine-tuning, evaluation
│   ├── api/              # FastAPI endpoints
│   ├── evaluation/       # DeepEval / RAGAS test suites
│   ├── mlops/            # Drift detection, monitoring
│   └── schemas.py        # Pydantic data models
├── tests/                # Unit and integration tests
├── configs/              # YAML configuration files
├── data/                 # DVC-tracked data (not in git)
├── notebooks/            # Exploration and analysis
├── scripts/              # One-off setup scripts
└── docs/                 # Architecture docs and ADRs
```

## Quick start

```bash
# Clone and set up
git clone https://github.com/exzm09/felinet.git
cd felinet
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Copy and fill in environment variables
cp .env.example .env

# Initialize DVC
dvc init

# Verify MLflow
python scripts/init_mlflow.py
mlflow ui --port 5000

# Run tests
pytest tests/
```

## Current status

- [x] **Week 1**: Project scaffold, schemas, DVC, MLflow
- [x] Week 2: Data ingestion pipeline
- [ ] Week 3: ETL with Prefect + data quality
- [ ] Week 4: Chunking + vector store
- [ ] Week 5: End-to-end naive RAG
- [ ] Week 6: Evaluation framework
- [ ] Week 7: Hybrid search
- [ ] Week 8: Reranking + context engineering
- [ ] Week 9: Synthetic training data
- [ ] Week 10: Embedding fine-tuning
- [ ] Week 11: Guardrails
- [ ] Week 12: CI/CD quality gates
- [ ] Week 13: Drift detection + monitoring
- [ ] Week 14: A/B testing + feedback loops
- [ ] Week 15: Frontend + deployment
- [ ] Week 16: Documentation + polish

## Tech stack

| Layer | Tool |
|---|---|
| Embeddings | sentence-transformers (fine-tuned) |
| Vector store | Qdrant |
| Hybrid search | rank_bm25 + Qdrant dense + RRF |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Groq (Llama 3.3 70B) |
| API | FastAPI |
| Frontend | Gradio |
| Experiment tracking | MLflow |
| Data versioning | DVC |
| Observability | Langfuse |
| CI/CD | GitHub Actions + DeepEval |
| Orchestration | Prefect |

## License

MIT
