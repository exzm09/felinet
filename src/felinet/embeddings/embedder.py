"""
Embedding module for FeliNet.

Wraps sentence-transformers to convert DocumentChunks into vectors.

1. The model tokenized text (splits into WordPiece tokens)
2. Feeds tokens through a small transformer (6 layers, ~23M params)
3. Mean-pools the last hidden states into a single 384-dim vector
4. Vector captures the "meaning" off the text in a space where cosine similarity is semantic similarity
"""

from __future__ import annotations
import logging

from sentence_transformers import SentenceTransformer
from felinet.schemas import DocumentChunk

logger = logging.getLogger(__name__)

def load_embedding_model(
        model_name: str = "all-MiniLM-L6-v2"
) -> SentenceTransformer:
    """
    Load a sentence-transformer model.
    First call downloads the model weights to a local cache. Subsequent calls load from cache instantly.

    Parameters
    ----------
    model_name : str
    """
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info(f"Model loaded - embedding dimensions: {model.get_sentence_embedding_dimension()}")
    return model

def embed_chunks(
        chunks: list[DocumentChunk],
        model: SentenceTransformer,
        batch_size: int = 64
) -> list[DocumentChunk]:
    """
    Embed a list of DocumentChunks in-place and return them.

    Parameters
    ----------
    chunks : list[DocumentChunk]
        Chunks with content but no embedding yet.
    model : SentenceTransformer
        Loaded embedding model.
    batch_size : int
        How many chunks to embed at once. 64 is a good default for CPU.
        GPU can handle 256+.

    Returns
    -------
    list[DocumentChunk]
        Same chunks, now with embedding and embedding_model filled in.
    """
    texts = [chunk.content for chunk in chunks]
    model_name = model[0].auto_model.config.name_or_path

    logger.info(f"Embedding {len(texts)} chunks with {model_name} (batch size = {batch_size})...")

    # .encode() return a numpy array of shape (n_chunks, embedding_dim)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embedding=True    # L2-normalize so cosine similiarity = dot product
    )
    for chunk, vector in zip(chunks, embeddings):
        chunk.embedding = vector.tolist()
        chunk.embedding_model = model_name

    logger.info("Embedding complete")
    return chunks