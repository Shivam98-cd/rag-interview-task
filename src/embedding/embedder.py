"""
src/embedding/embedder.py

Generates dense embedding using the open-source sentence-transformers library.

Model: sentence-transformers/all-MiniLM-L6-v2
  - Completely free, Apache 2.0 license
  - 384-dimensional output
  - Fast on CPU (~14k sentences/sec)
  - Excellent semantic similarity performance

The model is downloaded automatically on first use (~22MB).
Subsequent runs use the cached version.
"""

from __future__ import annotations
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config


# ── singleton model loader ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load (or return cached) SentenceTransformer model."""
    print(f"[Embedder] Loading model '{config.EMBEDDING_MODEL}' ...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    print(f"[Embedder] Model loaded. Embedding dim = {config.EMBEDDING_DIMENSION}")
    return model


# ── public API ────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Embed a list of strings.

    Args:
        texts:      List of text strings.
        batch_size: How many to embed at once (tune for memory).

    Returns:
        List of float vectors, one per input text.
    """
    model = _get_model()
    all_embeddings: list[list[float]] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i : i + batch_size]
        vectors = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.extend(vectors.tolist())

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.
    Convenience wrapper around embed_texts.
    """
    return embed_texts([query])[0]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add an 'embedding' field to each chunk dict in-place.

    Input:  [{"chunk_id": str, "text": str, ...}, ...]
    Output: same list with 'embedding' key added
    """
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    for chunk, vector in zip(chunks, embeddings):
        chunk["embedding"] = vector

    print(f"[Embedder] Embedded {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    print("Testing embedder...")
    test_texts = [
        "What is business communication?",
        "The 7 Cs of effective communication are clarity and conciseness.",
    ]
    vectors = embed_texts(test_texts)
    print(f"Embedded {len(vectors)} texts")
    print(f"Vector dimension: {len(vectors[0])}")
    print(f"First 5 values of vector 1: {vectors[0][:5]}")
    print("✅ Embedder working correctly!")