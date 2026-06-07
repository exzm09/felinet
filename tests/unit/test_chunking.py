"""
Unit tests for FeliNet chunking logic.

Tests that the chunker:
    - Produces valid DocumentChunk objects
    - Respects token count limits
    - Generates deterministic chunk IDs
    - Handles edge cases (very short docs, very long docs)

"""

import pytest

from felinet.embeddings.chunker import chunk_corpus as chunk_document
from felinet.schemas import (
    ChunkingConfig,
    ContentType,
    DataSource,
    DocumentChunk,
    SourceDocument,
)

# Fixtures specific to chunking tests


@pytest.fixture
def long_document():
    """
    A document long enough to produce multiple chunks.
    """
    # Create content that's clearly longer than 512 tokens (~4 chars per token on average, so 3000 chars ≈ 750 tokens)
    paragraph = (
        "Chronic kidney disease (CKD) is one of the most common conditions "
        "affecting older cats, with studies suggesting that up to 30-40% of "
        "cats over the age of 10 may be affected. The kidneys play a crucial "
        "role in filtering waste products from the blood, concentrating urine, "
        "and maintaining fluid and electrolyte balance. Early signs of CKD in "
        "cats can be subtle and may include increased thirst, increased urination, "
        "weight loss, decreased appetite, and occasional vomiting. As the disease "
        "progresses, cats may develop mouth ulcers, bad breath, and lethargy."
    )
    content = "\n\n".join([paragraph] * 8)

    return SourceDocument(
        id="ckd_article",
        source=DataSource.CORNELL,
        url="https://example.com/ckd",
        title="Chronic Kidney Disease in Cats",
        content=content,
        content_type=ContentType.DISEASE,
    )


@pytest.fixture
def short_document():
    """
    A document short enough to fit in a single chunk.
    """
    return SourceDocument(
        id="short_article",
        source=DataSource.WIKIPEDIA,
        url="https://example.com/tabby",
        title="Tabby Cat Coat Pattern",
        content=(
            "A tabby is any domestic cat with a distinctive 'M' shaped marking "
            "on its forehead. Tabbies are not a breed but a coat pattern found "
            "across many cat breeds and mixed breed cats."
        ),
        content_type=ContentType.BREED_PROFILE,
    )


@pytest.fixture
def default_chunking_config():
    return ChunkingConfig()


# Tests


class TestChunkDocuments:
    def test_produces_valid_document_chunks(self, long_document, default_chunking_config):
        """
        chunk_document should return a list of DocumentChunk objects.
        """
        chunks = chunk_document([long_document], default_chunking_config)
        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_long_doc_produces_multiple_chunks(self, long_document, default_chunking_config):
        """
        A long document should be split into multiple chunks.
        """
        chunks = chunk_document([long_document], default_chunking_config)
        assert len(chunks) >= 2, f"Expected ≥2 chunks from long doc, got {len(chunks)}"

    def test_short_doc_produces_one_chunk(self, short_document, default_chunking_config):
        """
        A short document should produce exactly one chunk.
        """
        chunks = chunk_document([short_document], default_chunking_config)
        assert len(chunks) == 1

    def test_chunk_token_count_within_bounds(self, long_document, default_chunking_config):
        """
        Every chunk's token_count should be > 0 and ≤ 2048.
        """
        chunks = chunk_document([long_document], default_chunking_config)
        for chunk in chunks:
            assert chunk.token_count > 0
            assert chunk.token_count <= 2048

    def test_chunk_ids_are_unique(self, long_document, default_chunking_config):
        """
        Every chunk should have a unique ID.
        """
        chunks = chunk_document([long_document], default_chunking_config)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs are not unique"

    def test_chunk_ids_are_deterministic(self, long_document, default_chunking_config):
        """
        Same input should produce same chunk IDs (SHA-256 based).
        """
        chunks_a = chunk_document([long_document], default_chunking_config)
        chunks_b = chunk_document([long_document], default_chunking_config)
        ids_a = [c.id for c in chunks_a]
        ids_b = [c.id for c in chunks_b]
        assert ids_a == ids_b, "Chunk IDs should be deterministic"

    def test_chunk_preserves_source(self, long_document, default_chunking_config):
        """
        Chunks should inherit the source from their parent document.
        """
        chunks = chunk_document([long_document], default_chunking_config)
        for chunk in chunks:
            assert chunk.source == DataSource.CORNELL
            assert chunk.document_id == "ckd_article"

    def test_chunk_indices_are_sequential(self, long_document, default_chunking_config):
        """
        chunk_index should start at 0 and increment for each doc.
        """
        chunks = chunk_document([long_document], default_chunking_config)
        indices = [c.chunk_index for c in chunks]
        expected = list(range(len(chunks)))
        assert indices == expected

    def test_chunk_content_is_nonempty(self, long_document, default_chunking_config):
        """
        Every chunk should have non-empty content.
        """
        chunks = chunk_document([long_document], default_chunking_config)
        for chunk in chunks:
            assert len(chunk.content.strip()) >= 10

    def test_multiple_docs_all_chunked(
        self, long_document, short_document, default_chunking_config
    ):
        """
        Chunking multiple documents should produce chunks for all of them.
        """
        chunks = chunk_document([long_document, short_document], default_chunking_config)
        doc_ids = {c.document_id for c in chunks}
        assert "ckd_article" in doc_ids
        assert "short_article" in doc_ids

    def test_empty_list_returns_empty(self, default_chunking_config):
        """
        Chunking an empty list should return an empty list.
        """
        chunks = chunk_document([], default_chunking_config)
        assert chunks == []

    def test_custom_chunk_size(self, long_document):
        """
        Smaller chunk_size should produce more chunks.
        """
        small_config = ChunkingConfig(chunk_size=128, chunk_overlap=10)
        large_config = ChunkingConfig(chunk_size=512, chunk_overlap=50)
        small_chunks = chunk_document([long_document], small_config)
        large_chunks = chunk_document([long_document], large_config)
        assert len(small_chunks) >= len(
            large_chunks
        ), "Smaller chunk_size should produce at least as many chunks"
