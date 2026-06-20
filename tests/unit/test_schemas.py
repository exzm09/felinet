"""Tests for FeliNet core schemas"""

import pytest
from pydantic import ValidationError

from felinet.schemas import (
    ChunkingConfig,
    ContentType,
    DataSource,
    DocumentChunk,
    GenerationConfig,
    RAGConfig,
    RAGResponse,
    RetrievalConfig,
    RetrievedChunk,
    SourceDocument,
)


class TestSourceDocument:
    """
    Tests for the SourceDocument model (raw docs before chunking)
    """

    def test_valid_document(self, sample_source_document):
        """
        A well-formed doc should be accepted without errors
        """
        doc = sample_source_document
        # Test that Pydantic STORED it correctly
        assert doc.id == "test_doc_001"
        assert doc.source == DataSource.CORNELL
        assert doc.scraped_at is not None
        assert doc.content_type == ContentType.DISEASE
        assert len(doc.content) > 50

    def test_empty_title_rejected(self):
        """
        Title must have at least 1 character
        """
        with pytest.raises(ValidationError) as exc_info:
            SourceDocument(
                id="bad_doc",
                source=DataSource.CORNELL,
                url="https://example.com",
                title="",  # Empty - should fail
                content="A" * 60,
                content_type=ContentType.ARTICLE,
            )
        # Check that the error is about title length
        assert "title" in str(exc_info.value).lower()

    def test_content_too_short_rejected(self):
        """
        Content must be at least 50 characters (catches empty/broken scrapes)
        """
        with pytest.raises(ValidationError):
            SourceDocument(
                id="bad_doc",
                source=DataSource.CORNELL,
                url="https://example.com",
                title="Short",
                content="Too short",  # Below min_length=50
                content_type=ContentType.ARTICLE,
            )

    def test_invalid_source_rejected(self):
        """
        Source must be one of the defined DataSource values.
        """
        with pytest.raises(ValidationError):
            SourceDocument(
                id="bad_doc",
                source="random_blog",  # Not a valid DataSource
                url="https://example.com",
                title="Good Title",
                content="A" * 60,
                content_type=ContentType.ARTICLE,
            )

    def test_metadata_defaults_to_empty_dict(self):
        """
        Metadata should default to an empty dict if not provided.
        """
        doc = SourceDocument(
            id="doc",
            source=DataSource.CORNELL,
            url="https://example.com",
            title="Test",
            content="A" * 60,
            content_type=ContentType.ARTICLE,
        )
        assert doc.metadata == {}


class TestDocumentChunk:
    """
    Tests for the DocumentChunk model (chunks stored in vector DB).
    """

    def test_valid_chunk(self, sample_chunk):
        """
        A well-formed chunk should be accepted.
        """
        chunk = sample_chunk
        assert chunk.chunk_index == 0
        assert chunk.token_count == 30
        assert chunk.embedding is None  # Embedding not set yet

    def test_negative_chunk_index_rejected(self):
        """
        chunk_index must be >= 0.
        """
        with pytest.raises(ValidationError):
            DocumentChunk(
                id="chunk",
                document_id="doc",
                source=DataSource.CORNELL,
                content="A" * 20,
                content_type=ContentType.ARTICLE,
                chunk_index=-1,  # Negative -> should fail
                token_count=10,
                pipeline_version="0.1.0",
            )

    def test_zero_token_count_rejected(self):
        """
        token_count must be > 0 (a chunk with 0 tokens makes no sense).
        """
        with pytest.raises(ValidationError):
            DocumentChunk(
                id="chunk",
                document_id="doc",
                source=DataSource.CORNELL,
                content="A" * 20,
                content_type=ContentType.ARTICLE,
                chunk_index=0,
                token_count=0,  # Zero -> should fail
                pipeline_version="0.1.0",
            )

    def test_huge_token_count_rejected(self):
        """
        Chunks over 2048 tokens should be rejected (custom validator).
        """
        with pytest.raises(ValidationError, match="exceeds the 2048 max"):
            DocumentChunk(
                id="chunk",
                document_id="doc",
                source=DataSource.CORNELL,
                content="A" * 20,
                content_type=ContentType.ARTICLE,
                chunk_index=0,
                token_count=5000,  # Way too big
                pipeline_version="0.1.0",
            )

    def test_short_content_rejected(self):
        """
        Chunk content must be at least 10 characters.
        """
        with pytest.raises(ValidationError):
            DocumentChunk(
                id="chunk",
                document_id="doc",
                source=DataSource.CORNELL,
                content="Hi",  # Too short
                content_type=ContentType.ARTICLE,
                chunk_index=0,
                token_count=1,
                pipeline_version="0.1.0",
            )


class TestRAGConfig:
    """
    Tests for the pipeline configuration models.
    """

    def test_default_config(self, sample_rag_config):
        """
        Default config should have sensible values.
        """
        config = sample_rag_config
        assert config.chunking.chunk_size == 512
        assert config.chunking.chunk_overlap == 50
        assert config.retrieval.top_k_initial == 30
        assert config.retrieval.top_k_reranked == 5
        assert config.retrieval.use_reranker is True
        assert config.embedding_model == "models/felinet-embedding-v1"
        assert config.collection_name == "felinet_chunks"

    def test_chunking_config_bounds(self):
        """
        chunk_size must be between 64 and 2048.
        """
        with pytest.raises(ValidationError):
            ChunkingConfig(chunk_size=10)  # Below 64

        with pytest.raises(ValidationError):
            ChunkingConfig(chunk_size=5000)  # Above 2048

    def test_retrieval_config_bounds(self):
        """
        top_k values must be within defined bounds.
        """
        with pytest.raises(ValidationError):
            RetrievalConfig(top_k_initial=2)  # Below 5

        with pytest.raises(ValidationError):
            RetrievalConfig(top_k_reranked=0)  # Below 1

    def test_temperature_bounds(self):
        """
        Temperature must be between 0.0 and 2.0.
        """
        with pytest.raises(ValidationError):
            GenerationConfig(temperature=-0.5)

        with pytest.raises(ValidationError):
            GenerationConfig(temperature=3.0)

    def test_bm25_weight_bounds(self):
        """
        BM25 weight must be between 0.0 and 1.0.
        """
        with pytest.raises(ValidationError):
            RetrievalConfig(bm25_weight=1.5)

    def test_custom_config(self):
        """
        Config can be customized without errors.
        """
        config = RAGConfig(
            embedding_model="custom-model",
            collection_name="test_collection",
        )
        config.chunking.chunk_size = 256
        assert config.embedding_model == "custom-model"


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
            config_snapshot=RAGConfig(),
        )
        assert len(response.retrieved_chunks) == 1
        assert response.latency_ms > 0


class TestRetrievedChunk:
    """
    Tests for the retrieval response models.
    """

    def test_valid_retrieved_chunk(self):
        """
        A well-formed RetrievedChunk should be accepted.
        """
        chunk = RetrievedChunk(
            chunk_id="test_001",
            content="Cats need taurine for heart health.",
            source=DataSource.CORNELL,
            score=0.85,
            document_title="Feline Nutrition",
            url="https://example.com",
        )
        assert chunk.score == 0.85
        assert chunk.document_title == "Feline Nutrition"

    def test_optional_fields_can_be_none(self):
        """
        document_title and url are optional.
        """
        chunk = RetrievedChunk(
            chunk_id="test_001",
            content="Some content here.",
            source=DataSource.CORNELL,
            score=0.5,
        )
        assert chunk.document_title is None
        assert chunk.url is None
