"""
FeliNet RAG pipeline.

Connects retrieval (Qdrant) -> context formatting -> generation (Groq) with with optional Langfuse observability on every step.

Single-stage retrieval + direct generation.
"""
from __future__ import annotations
import logging
import os
import time

import requests as http_requests

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from felinet.embeddings.vector_store import get_client, search
from felinet.schemas import (
    RAGConfig,
    RAGResponse,
    RetrievedChunk,
    DataSource
)

# Load .env so API keys are available as environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Optional Langfuse integration
# If Langfuse is disables or can not connect, use a no-op decorator so the pipeline runs identically just without tracing.
_LANGFUSE_AVAILABLE = False
if os.getenv("LANGFUSE_ENABLED", "true").lower() != "false":
    try:
        from langfuse.decorators import langfuse_context, observe
        _LANGFUSE_AVAILABLE = True
        logger.info("Langfuse observability enabled")
    except Exception as e:
        logger.warning(f"Langfuse not available: {e}")
if not _LANGFUSE_AVAILABLE:
    # No-op decorator: @observe() does nothing
    def observe(*args, **kwargs):
        def decorator(fn):
            return fn
        # Handle both @observe and @observe()
        if args and callable(args[0]):
            return args[0]
        return decorator
    
    class _DummyContext:
        def get_current_trace_id(self):
            return None
        def update_current_observation(self, **kwargs):
            pass
    langfuse_context = _DummyContext()
    logger.info("Langfuse disabled - running without observability")

# Component 1: Embed the user's query
@observe()  # Langfuse traces automatically
def embed_query(
    query: str,
    model: SentenceTransformer
) -> list[float]:
    """
    Turn user's question into a vector (same space as stored chunks).
    Must be the SAME model that embedded the corpus
    """
    vector = model.encode(query, normalize_embeddungs=True)
    return vector.tolist()

# Component 2: Retrieve relevant chunks from Qdrant

@observe()
def retrieve_chunks(
    query_vector: list[float],
    config: RAGConfig,
    qdrant_url: str = "http://localhost:6333"
) -> list[RetrievedChunk]:
    """
    Search Qdrant for chunks that are the most similar to the query vector.

    Returns RetrievedChunk objects so the rest of the pipeline works with typed, validated data.
    Parameters
    ----------
    query_vector : list[float]
        The embedded query.
    config : RAGConfig
        Pipeline config (tcollection name, top_k, etc.).
    qdrant_url : str
        Where Qdrant is running.
    """
    client = get_client(url=qdrant_url)
    # For now use dense-only search
    # For naice RAG (no reranker), use top_k_reranked(5) instead of 30
    top_k = config.retrieval.top_k_initial if not config.retrieval.use_reranker else config.retrieval.top_k_initial
    # Override to 5 for now
    top_k = config.retrieval.top_k_reranked
    raw_results = search(
        client=client,
        query_vector=query_vector,
        collection_name=config.collection_name,
        top_k=top_k
    )

    retrieved = []
    for hit in raw_results:
        chunk = RetrievedChunk(
            chunk_id=hit.get("chunk_id", str(hit["id"])),
            content=hit["content"],
            source=DataSource(hit["source"]),
            score=hit["score"],
            document_title=hit.get("title"),
            url=hit.get("url")
        )
        retrieved.append(chunk)

    logger.info(f"Retrieved {len(retrieved)} chunks (top score: {retrieved[0].score:.3f})" if retrieved else "No chunks retrieved")
    return retrieved

# Component 3: Format the context for the LLM
@observe()
def format_context(chunks: list[RetrievedChunk]) -> str:
    """
    Turn retrieved chunks into a numbered, readable block of text that the LLM will use.
    Each chunk is labeled with its source and title so the LLM can cite them in its answer.
    Example output:
    [1] Source: cornell_feline_health | Title: Feline Asthma
        Feline asthma is a condition in which the airways ...

    [2] Source: wikipedia_cat_breeds | Title: Persian Cat
        The Persian cat is a long-haired breed ...
    """
    if not chunks:
        return "No relevant context was found for this query."
    
    sections = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[{i}] Source: {chunk.source.value}"
        if chunk.document_title:
            header += f" | Title: {chunk.document_title}"
        sections.append(f"{header}\n{chunk.content}")

    return "\n\n".join(sections)

# Component 4: Generate an answer via Groq (Llama 3.3 70B)

@observe(as_type="generation")  # tells Langfuse this is an LLM call
def generate_answer(
    query: str,
    context: str,
    config: RAGConfig
) -> str:
    """
    Send the user's question + retrieved context to the LLM and get an answer back.
    Uses requests library to call Groq's OpenAI-compatible API directly,
    because the groq Python client has connectivity issues on some setups.

     Parameters
    ----------
    query : str
        The user's original question.
    context : str
        Formatted context string from format_context().
    config : RAGConfig
        Pipeline config (model name, temperature, etc.).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env file")

    # Build the message list.
    messages = [
        {
            "role": "system",
            "content": config.generation.system_prompt
        },
        {
            "role": "user",
            "content": (
                f"Context: \n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer the question using ONLY the context above. "
                "Cite sources by thier number (e.g., [1], [2])."
            )
        }
    ]
    # Update Langfuse with I/O for this generation
    langfuse_context.update_current_observation(
        input=messages,
        model=config.generation.model_name
    )

    # Call Groq API directly via requests
    response = http_requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        json={
            "model": str(config.generation.model_name),
            "messages": messages,
            "temperature": float(config.generation.temperature),
            "max_tokens": int(config.generation.max_tokens),
        },
        timeout=60,  # 60 second timeout for long answers
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Groq API error {response.status_code}: {response.text}"
        )

    data = response.json()
    answer = data["choices"][0]["message"]["content"]

    # Log token usage to Langfuse
    usage = data.get("usage", {})
    langfuse_context.update_current_observation(
        output=answer,
        usage={
            "input": usage.get("prompt_tokens", 0),
            "output": usage.get("completion_tokens", 0),
        },
    )

    return answer

# Component 5: The full pipeline

@observe()
def query_rag(
    query: str,
    config: RAGConfig | None = None,
    embedding_model: SentenceTransformer | None = None,
    qdrant_url: str = "http://localhost:6333"
) -> RAGResponse:
    """
    End-to-end RAG: question in -> cited answer out.
    The function FastAPI will call.
    Langfuse traces the entire flow as a single trace with child spans for each step (embed -> retrieve -> format -> generate).
    Parameters
    ----------
    query : str
        The user's question about cats.
    config : RAGConfig, optional
        Pipeline configuration. Uses defaults if not provided.
    embedding_model : SentenceTransformer, optional
        Pre-loaded model. If None, loads it fresh (slower on first call).
    qdrant_url : str
        Qdrant server address.

    Returns
    -------
    RAGResponse
        Structured response with answer, citations, latency, and trace ID.    
    """

    start_time = time.time()
    if config is None:
        config = RAGConfig()

    # Step 1: Load embedding model if not provided
    if embedding_model is None:
        from felinet.embeddings.embedder import load_embedding_model
        embedding_model = load_embedding_model(config.embedding_model)
    
    # Step 2: Embed the query
    query_vector = embed_query(query, embedding_model)

    # Step 3: Retrieve relevant chunks
    retrieved = retrieve_chunks(query_vector, config, qdrant_url)

    # Step 4: Format context
    context = format_context(retrieved)

    # Step 5: Generate answer
    answer = generate_answer(query, context, config)

    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000

    # Get Langfuse trace ID
    trace_id = langfuse_context.get_current_trace_id()

    # Build structured response
    response = RAGResponse(
        answer=answer,
        retrieved_chunks=retrieved,
        query=query,
        model_used=config.generation.model_name,
        latency_ms=latency_ms,
        config_snapshot=config,
        trace_id=trace_id
    )

    logger.info(f"RAG query complete | latency={latency_ms:.0f} ms | "
                f"chunks={len(retrieved)} | trace={trace_id}")
    
    return response