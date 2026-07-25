"""
FeliNet FastAPI application.
Serves RAG pipeline as an HTTP API with:
- POST /query (ask a feline health question, get a cited answer)
- GET  /health (check if the server and its dependencies are running)
"""

from __future__ import annotations

import logging
import os

import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from felinet.embeddings.embedder import load_embedding_model
from felinet.experiments.feedback import record_feedback
from felinet.rag.pipeline import query_rag
from felinet.rag.retriever import BM25Index
from felinet.schemas import RAGConfig, RAGResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App setup

app = FastAPI(
    title="FeliNet API",
    description="Feline health and breed knowledge assiant powered by RAG",
    version="0.5.0",
)

# Startup: pre-load heavy resources once, reuse for every request

_embedding_model: SentenceTransformer | None = None
_rag_config: RAGConfig | None = None
_bm25_index: BM25Index | None = None


@app.on_event("startup")
async def startup_load_models():
    """
    Load embedding model and config once when the server starts.
    """
    global _embedding_model, _rag_config
    logger.info("Loading RAG config...")
    _rag_config = RAGConfig()  # uses defaults; could load from YAML later

    logger.info("Loading embedding model...")
    _embedding_model = load_embedding_model(_rag_config.embedding_model)

    # Warm the vector store: builds the in-memory Qdrant
    from felinet.embeddings.vector_store import get_client

    get_client()
    parquet_path = os.getenv("QDRANT_EXPORT_PATH", "data/qdrant_export.parquet")
    _bm25_index = BM25Index.from_parquet(parquet_path)

    logger.info("FeliNet API ready.")


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
        json_schema_extra={"examples": ["Why is my cat sneezing?"]},
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


@app.post("/query", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Ask FeliNet a question about cats.
    Pipeline:
    1. Embeds question into a vector.
    2. Searches the knowledge base for relevant passages
    3. Sends the passages + question
    4. Returns a cited answer
    """
    try:
        rag_response: RAGResponse = query_rag(
            query=request.question,
            config=_rag_config,
            embedding_model=_embedding_model,
            retrieval_mode="hybrid",
            bm25_index=_bm25_index,
        )

        # Convert internal RAGResponse -> API QueryResponse
        sources = [
            SourceInfo(
                source=chunk.source.value,
                title=chunk.document_title,
                url=chunk.url,
                relevance_score=round(chunk.score, 4),
            )
            for chunk in rag_response.retrieved_chunks
        ]

        return QueryResponse(
            answer=rag_response.answer,
            source=sources,
            model_used=rag_response.model_used,
            latency_ms=round(rag_response.latency_ms, 1),
            trace_id=rag_response.trace_id,
        )

    except Exception as e:
        logger.error(f"RAG pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


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
        "bm25_index_loaded": _bm25_index is not None,
        "collection_name": _rag_config.collection_name if _rag_config else None,
    }


# Gradio chat UI

EXAMPLE_QUESTIONS = [
    "What are the early signs of kidney disease in cats?",
    "Is it safe for my cat to be around lilies?",
    "What is the temperament of a Maine Coon?",
    "How often should I feed a senior cat?",
]


def _format_sources(chunks, max_show: int) -> str:
    if not chunks:
        return "_No sources retrieved._"
    lines = []
    for i, c in enumerate(chunks[:max_show], start=1):
        title = c.document_title or c.source.value
        snippet = c.content.strip().replace("\n", " ")
        snippet = snippet[:240] + "…" if len(snippet) > 240 else snippet
        header = f"**{i}. {title}**  ·  _{c.source.value}_"
        if c.url:
            header += f"  ·  [link]({c.url})"
        lines.append(f"{header}\n\n> {snippet}")
    return "\n\n---\n\n".join(lines)


def chat_respond(message, history, max_sources, trace_meta):
    if not message or not message.strip():
        return history, "_Ask a question to see sources._", trace_meta, ""
    response = query_rag(
        query=message,
        config=_rag_config,
        embedding_model=_embedding_model,
        retrieval_mode="hybrid",
        bm25_index=_bm25_index,
    )
    history = (history or []) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response.answer},
    ]
    trace_meta = (trace_meta or []) + [
        {"trace_id": response.trace_id, "query": message, "answer": response.answer}
    ]
    return history, _format_sources(response.retrieved_chunks, int(max_sources)), trace_meta, ""


def on_like(evt: gr.LikeData, trace_meta):
    turn = (evt.index - 1) // 2 if isinstance(evt.index, int) else 0
    if 0 <= turn < len(trace_meta):
        m = trace_meta[turn]
        record_feedback(
            trace_id=m["trace_id"], query=m["query"], answer=m["answer"], liked=evt.liked
        )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="FeliNet", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# FeliNet\nAsk about cat health, breeds, nutrition, and behavior. "
            "Answers are grounded in cited veterinary sources."
        )
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(type="messages", height=460, show_copy_button=True)
                box = gr.Textbox(placeholder="Ask about cats…", show_label=False, autofocus=True)
                gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=box, label="Try one")
                with gr.Accordion("Sources for the latest answer", open=False):
                    sources = gr.Markdown("_Ask a question to see sources._")
            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                max_sources = gr.Slider(1, 5, value=5, step=1, label="Sources to display")
                gr.Markdown(
                    "_FeliNet answers only from retrieved context and says "
                    "“I don't know” when the sources don't cover your question._"
                )
        trace_meta = gr.State([])
        box.submit(
            chat_respond,
            inputs=[box, chatbot, max_sources, trace_meta],
            outputs=[chatbot, sources, trace_meta, box],
        )
        chatbot.like(on_like, [trace_meta], None)
    return demo


app = gr.mount_gradio_app(app, build_demo(), path="/")
