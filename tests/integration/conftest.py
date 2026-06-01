"""
FeliNet test fixtures - shared helpers used across all test files.
"""
import pytest
from felinet.schemas import (
    SourceDocument,
    DocumentChunk,
    DataSource,
    ContentType,
    RAGConfig,
    RetrievedChunk,
)

@pytest.fixture
def sample_source_document():
    """
    A valid SourceDocument for testing.
    """
    return SourceDocument(
        id="test_doc_001",
        source=DataSource.CORNELL,
        url="https://example.com/cat-health",
        title="Feline Kidney Disease",
        content=(
            "Chronic kidney disease (CKD) is one of the most common conditions affecting older cats. It occurs when the kidneys gradually lose their ability to filter waste products from the blood. Early signs include increased thirst and urination, weight loss, and decreased appetite. Diagnosis involves blood tests measuring BUN and creatinine levels."
        ),
        content_type=ContentType.DISEASE,
        metadata={"category": "nephrology"},
    )

@pytest.fixture
def sample_chunk():
    """A valid DocumentChunk for testing."""
    return DocumentChunk(
        id="test_doc_001_chunk_0",
        document_id="test_doc_001",
        source=DataSource.CORNELL,
        content=(
            "Chronic kidney disease (CKD) is one of the most common conditions affecting older cats. Early signs include increased thirst."
        ),
        content_type=ContentType.DISEASE,
        chunk_index=0,
        token_count=30,
        pipeline_version="0.1.0",
        metadata={"title": "Feline Kidney Disease"},
    )


@pytest.fixture
def sample_rag_config():
    """A default RAGConfig for testing."""
    return RAGConfig()


@pytest.fixture
def sample_retrieved_chunks():
    """A list of RetrievedChunk objects simulating retrieval results."""
    return [
        RetrievedChunk(
            chunk_id="chunk_001",
            content=(
                "Cats require taurine, an amino acid found in meat. Taurine deficiency can cause dilated cardiomyopathy and blindness."
            ),
            source=DataSource.CORNELL,
            score=0.85,
            document_title="Feline Nutrition",
            url="https://example.com/nutrition",
        ),
        RetrievedChunk(
            chunk_id="chunk_002",
            content=(
                "Regular veterinary checkups are important for early detection of common feline diseases like CKD, diabetes, and hyperthyroidism."
            ),
            source=DataSource.WIKIPEDIA,
            score=0.72,
            document_title="Cat Health Overview",
            url="https://example.com/health",
        ),
        RetrievedChunk(
            chunk_id="chunk_003",
            content=(
                "The Persian cat is a long-haired breed known for its flat face and calm temperament. They require daily grooming."
            ),
            source=DataSource.CFA,
            score=0.45,
            document_title="Persian Cat Profile",
            url="https://example.com/persian",
        ),
    ]
