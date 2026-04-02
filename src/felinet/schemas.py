"""
FeliNet core data schemas.

Pydantic models define the data contracts used throughout the pipeline:
- DocumentChunk: represents a chunk of text stored in the vecoter database
- RAGConfig: configuration for the RAG pipeline (chunking, retrieval, generation)
- RAGResponse: structured response form the RAG pipeline with citations
"""
from datetime import datetime, timezone
# A fixed list of allowed options catching mistakes before running code
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# Custom data types that restrict values to a fixed set.
class DataSource(str, Enum):
    """Known data sources in the FeliNet corpus."""
    CORNELL = "cornel_feline_health"
    WIKIPEDIA = "wikipedia_cat_breeds"
    CFA = "cfa_breed_profiles"
    ASPCA = "aspca_toxicology"
    MERCK = "merck_vet_manual"

class ContentType(str, Enum):
    """Types of content in the corpus"""
    ARTICLE = "article"
    BREED_PROFILE = "breed_profile"
    TOXICOLOGY = "toxicology"
    NUTRITION = "nutrition"
    DISEASE = "disease"
    BEHAVIOR = "behavior"

# Document & Chunk Models
class SourceDocument(BaseModel):
    """A raw document before chunking"""
    # Required (...)
    id: str = Field(..., description="Unique document identifier")
    source: DataSource
    url: str = Field(..., description="Source URL for provenance")
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=50, description="Full text content")
    content_type: ContentType
    # Auto-computed - factory function creates a fresh default
    scraped_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = Field(default_factory=dict, description="Source-specific metadata")

class DocumentChunk(BaseModel):
    """A chunk of text ready for embedding and storage in the vector DB"""
    id: str = Field(..., description="Unique chunk identifier (doc_id + chunk_index)")
    document_id: str = Field(..., description="Parent document ID")
    source: DataSource
    content: str = Field(..., min_length=10, description="Chunk text content")
    content_type: ContentType
    chunk_index: int = Field(..., ge=0, description="Position within the parent document")
    token_count: int = Field(..., gt=0, description="Number of tokens in the chunk")
    # Hold as None when embedding not avaliable
    embedding: Optional[list[float]] = Field(None, description="Vector embedding")
    # Avoiding embedding drift
    embedding_model: Optional[str] = Field(None, description="Model used to generate embedding")
    pipeline_version: str = Field(..., description="Pipeline version that produced this chunk")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = Field(default_factory=dict)

    # Customized error message; last line of defense catching anything that slipped through
    @field_validator("token_count")
    @classmethod
    def token_count_within_bounds(cls, v: int) -> int:
        if v > 2048:
            raise ValueError(f"Chunk has {v} tokens - exceeds the 2048 max. Re-chunk this document.")
        return v
    
# RAG Pipeline Configuration

class ChunkingConfig(BaseModel):
    """Configuration for the text chunking step"""
    chunk_size: int = Field(512, ge=64, le=2048, description="Target chunk size in tokens")
    chunk_overlap: int = Field(50, ge=0, description="Overlap between consecutive chunks in tokens")
    separators: list[str] = Field(
        default=["\n\n", "\n", ". ", " ", ""],
        description="Separators for RecursiveCharacterTextSplitter"
    )

class RetrievalConfig(BaseModel):
    """Configuration for the retrieval step"""
    top_k_initial: int = Field(30, ge=5, le=100, description="Candidates from hybrid search")
    top_k_reranked: int = Field(5, ge=1, le=20, description="Final chunk after reranking")
    bm25_weight: float = Field(0.5, ge=0.0, le=1.0, description="Weight for BM25 in hybrid fusion")
    dense_weight: float = Field(0.5, ge=0.0, le=1.0, description="Weight for dense retrieval")
    use_reranker: bool = Field(True, description="Whether to apply cross-encoder reranking")
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class GenerationConfig(BaseModel):
    """Configuration for the LLM generation step"""
    model_name: str = Field("llama-3.3-70b-versatile", description="Groq model identifier")
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=64, le=4096)
    system_prompt: str = Field(
        default=(
            "You are FeliNet, a feline health and breed knowledge assistant. "
            "Answer questions about cats using ONLY the provided context. "
            "Cite your sources. If the context does not contain enough information, "
            "say so clearly - do not guess or hallucinate."
        )
    )

class RAGConfig(BaseModel):
    """Compile RAG pipeline configuration - one object controls the whole pipeline"""
    # Type + Default
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    embedding_model: str = Field(
        "all-MiniLM-L6-v2",
        description="Sentence-transformer model for embeddings"
    )
    collection_name: str = Field("felinet_chunks", description="Qdrant collection name")

# RAG Response Models
class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval, with its relevance score"""
    chunk_id: str
    content: str
    source: DataSource
    score: float = Field(..., description="Relevance score")
    document_title: Optional[str] = None
    url: Optional[str] = None

class RAGResponse(BaseModel):
    """Structured response from the RAG pipeline"""
    answer: str = Field(..., description="Generated answer text")
    retrieved_chunks: list[RetrievedChunk] = Field(
        ..., description="Chunks used to generate the answer"
    )
    query: str = Field(..., description="Original user query")
    model_used: str = Field(..., description="LLM model that generated the answer")
    latency_ms: float = Field(..., ge=0, description="End-to-end latency in milliseconds")
    config_snapshot: RAGConfig = Field(..., description="Pipeline config used for the query")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: Optional[str] = Field(None, description="Langfuse trace ID for observability")