"""Tests for FeliNet core schemas"""

import pytest
from pydantic import ValidationError

from felinet.schemas import(
    ChunkingConfig,
    ContentType,
    DataSource,
    DocumentChunk,
    RAGConfig,
    RAGResponse,
    RetrievedChunk,
    SourceDocument
)

class TestSourceDocument:
    def test_valid_document(self):
        doc = SourceDocument(
            id="cornell_001",
            source=DataSource.CORNELL,
            url="https://www.vet.cornell.edu/example",
            title="Feline Lower Urinary Tract Disease",
            content="Feline lower urinary tract disease (FLUTD) describes a variety of conditions..." * 5,
            content_type=ContentType.DISEASE
        )
        # Test that Pydantic STORED it correctly
        assert doc.source == DataSource.CORNELL
        assert doc.scraped_at is not None

    def test_content_too_short_rejected(self):
        with pytest.raises(ValidationError):
            SourceDocument(
                id="bad",
                source=DataSource.CORNELL,
                url="https://example.com",
                title="Short",
                content="Too short", # Below min_length=50
                content_type=ContentType.ARTICLE
            )

    def test_empty_title_rejected(self):
        with pytest.raises(ValidationError):
            SourceDocument(
                id="bad",
                source=DataSource.CORNELL,
                url="https://example.com",
                title="",  # min_length=1
                content="x" * 100,
                content_type=ContentType.ARTICLE
            )
    

class TestDocumentChunk:
    def test_valid_chunk(self):
        chunk = DocumentChunk(
            id="cornell_001_chunk_0",
            document_id="cornell_001",
            source=DataSource.CORNELL,
            content="FLUTD affects the bladder and urethra of cats...",
            content_type=ContentType.DISEASE,
            chunk_index=0,
            token_count=128,
            pipeline_version="0.1.0"
        )
        assert chunk.embedding is None # embedding is optional at creation
        assert chunk.chunk_index == 0

    def test_oversized_chunk_rejected(self):
        with pytest.raises(ValidationError, match="exceeds the 2048 max"):
            DocumentChunk(
                id="bad",
                document_id="doc",
                source=DataSource.CORNELL,
                content="Some text",
                content_type=ContentType.ARTICLE,
                chunk_index=0,
                token_count=3000,  # above 2048 limit
                pipeline_version="0.1.0"
            )
    
    def test_negative_chunk_size_index_rejected(self):
        with pytest.raises(ValidationError):
            DocumentChunk(
                id="bad",
                document_id="doc",
                source=DataSource.CORNELL,
                content="Some text here",
                content_type=ContentType.ARTICLE,
                chunk_index=-1,  # ge=0
                token_count=50,
                pipeline_version="0.1.0"
            )

class TestRAGConfig:
    def test_defaults_are_sane(self):
        config = RAGConfig()
        assert config.chunking.chunk_size == 512
        assert config.retrieval.top_k_initial == 30
        assert config.retrieval.top_k_reranked == 5
        assert config.generation.temperature == 0.1
        assert config.embedding_model == "all-MiniLM-L6-v2"

    def test_override_nested_confid(self):
        config = RAGConfig(
            chunking=ChunkingConfig(chunk_size=256, chunk_overlap=25),
            embedding_model="BAAI/bge-base-en-v1.5"
        )
        assert config.chunking.chunk_size == 256
        assert config.embedding_model == "BAAI/bge-base-en-v1.5"

    def test_invalid_chunk_size_rejected(Self):
        with pytest.raises(ValidationError):
            ChunkingConfig(chunk_size=10) # below ge=64

class TestRAGResponses:
    def test_valid_response(self):
        response = RAGResponse(
            answer="FLUTD is a group of conditions affecting the cat's bladder...",
            retrieved_chunks=[
                RetrievedChunk(
                    chunk_id="cornell_001_chunk_0",
                    content="FLUTD affects the bladder...",
                    source=DataSource.CORNELL,
                    score=0.92,
                    document_title="Feline Lower Urinary Tract Disease",
                    url="https://www.vet.cornell.edu/example",
                )
            ],
            query="What is FLUTD in cats?",
            model_used="llama-3.3-70b-versatile",
            latency_ms=1250.5,
            config_snapshot=RAGConfig()
        )
        assert len(response.retrieved_chunks) == 1
        assert response.latency_ms > 0