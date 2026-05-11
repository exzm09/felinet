"""
FeliNet FastAPI application.
Serves RAG pipeline as an HTTP API with:
- POST /query (ask a feline health question, get a cited answer)
- GET  /health (check if the server and its dependencies are running)
"""
from __future__ import annotations
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from felinet.embeddings.embedder import load_embedding_model
from felinet.rag.pipeline import query_rag
from felinet.schemas import RAGConfig, RAGResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App setup

app = FastAPI(
    title="FeliNet API",
    description="Feline health and breed knowledge assiant powered by RAG",
    version="0.5.0"
)

# Startup: pre-load heavy resources once, reuse for every request

_embedding_model: SentenceTransformer | None = None
_rag_config: RAGConfig | None = None

@app.on_event("startup")
async def startup_load_models():
    """
    Load embedding model and config once when the server starts.
    """
    global _embedding_model, _rag_config
    logger.info("Loading RAG config...")
    _rag_config = RAGConfig()   # uses defaults; could load from YAML later

    logger.info("Loading embedding model...")
    _embedding_model = load_embedding_model(_rag_config.embedding_model)

    logger.info(f"FeliNet API ready!")


# Request / Response models for API

class QueryRequest(BaseModel):
    """
    What the user sends to POST /query.
    """
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="A question about cat health, breeds, or behavior",
        json_schema_extra={"examples": ["Why is my cat sneezing?"]}
    )

class SourceInfo(BaseModel):
    """
    One cited source in the response.
    """
    source: str
    title: str | None = None
    url: str | None = None
    relevance_score: float

class QueryResponse(BaseModel):
    """
    What the API sends back - a simplified view of RAGResponse.
    """
    answer: str
    source: list[SourceInfo]
    model_used: str
    latency_ms: float
    trace_id: str | None = None

# Endpoints

@app.post("/query", response_model = QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Ask FeliNet a question about cats.
    Pipeline:
    1. Embeds question into a vector.
    2. Searches the knowledge base for relevant passages
    3. Sends the passages + questiono to Llama 3.3 70B
    4. Returns a cited answer
    """
    try:
        rag_response: RAGResponse = query_rag(
            query=request.question,
            config=_rag_config,
            embedding_model=_embedding_model
        )

        # Convert internal RAGResponse -> API QueryResponse
        sources = [
            SourceInfo(
                source=chunk.source.value,
                title=chunk.document_title,
                url=chunk.url,
                relevance_score=round(chunk.score, 4)
            )
            for chunk in rag_response.retrieved_chunks
        ]

        return QueryResponse(
            answer=rag_response.answer,
            source=sources,
            model_used=rag_response.model_used,
            latency_ms=round(rag_response.latency_ms, 1),
            trace_id=rag_response.trace_id
        )
    
    except Exception as e:
        logger.error(f"RAG pipeline error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
        )
    
@app.get("/health")
async def health_check():
    """
    Quick check that the server is running and models are loaded.
    Returns the embedding model name and config status.
    """
    return {
        "status": "healthy",
        "embedding_model_loaded": _embedding_model is not None,
        "config_loaded": _rag_config is not None,
        "collection_name": _rag_config.collection_name if _rag_config else None
    }