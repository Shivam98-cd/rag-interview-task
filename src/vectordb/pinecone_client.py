"""
src/vectordb/pinecone_client.py

Handles all Pinecone operations:
  - Index creation (serverless, free tier)
  - Upsert vectors in batches
  - Query for top-k similar vectors
"""

from __future__ import annotations
import time

from pinecone import Pinecone, ServerlessSpec

import config


# ── index management ──────────────────────────────────────────────────────────

def get_or_create_index() -> object:
    """
    Connect to Pinecone and return the index.
    Creates the index if it doesn't exist yet.
    """
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]

    if config.PINECONE_INDEX_NAME not in existing:
        print(f"[Pinecone] Creating index '{config.PINECONE_INDEX_NAME}' ...")
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=config.PINECONE_CLOUD,
                region=config.PINECONE_REGION,
            ),
        )
        # Wait until the index is ready
        while not pc.describe_index(config.PINECONE_INDEX_NAME).status["ready"]:
            print("[Pinecone] Waiting for index to become ready ...")
            time.sleep(2)
        print("[Pinecone] Index created and ready.")
    else:
        print(f"[Pinecone] Using existing index '{config.PINECONE_INDEX_NAME}'.")

    return pc.Index(config.PINECONE_INDEX_NAME)


# ── upsert ────────────────────────────────────────────────────────────────────

def upsert_chunks(chunks: list[dict], batch_size: int = 100) -> None:
    """
    Upsert embedded chunks into Pinecone.

    Each chunk must have:
        chunk_id  (str)  — unique vector ID
        embedding (list) — float vector
        text      (str)  — stored as metadata
        page      (int)  — stored as metadata
        source    (str)  — stored as metadata
    """
    index = get_or_create_index()

    vectors = [
        {
            "id": chunk["chunk_id"],
            "values": chunk["embedding"],
            "metadata": {
                "text": chunk["text"],
                "page": chunk["page"],
                "source": chunk["source"],
            },
        }
        for chunk in chunks
    ]

    # Upsert in batches (Pinecone recommends ≤100 per request)
    total = 0
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        total += len(batch)
        print(f"[Pinecone] Upserted {total}/{len(vectors)} vectors ...")

    print(f"[Pinecone] All {len(vectors)} vectors upserted successfully.")


# ── query ─────────────────────────────────────────────────────────────────────

def query_index(
    query_vector: list[float],
    top_k: int = None,
) -> list[dict]:
    """
    Query Pinecone for the most similar vectors.

    Returns:
        List of dicts: [{"text": str, "page": int, "score": float}, ...]
        Sorted by descending similarity score.
    """
    top_k = top_k or config.RETRIEVAL_TOP_K
    index = get_or_create_index()

    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    results = [
        {
            "chunk_id": match["id"],
            "text": match["metadata"]["text"],
            "page": match["metadata"].get("page"),
            "source": match["metadata"].get("source"),
            "score": round(match["score"], 4),
        }
        for match in response["matches"]
    ]

    return results


if __name__ == "__main__":
    import sys

    sys.path.insert(0, r"D:\rag-interview-task")

    import config

    config.validate()

    print("Testing Pinecone connection...")

    # Test 1 — create/connect to index
    index = get_or_create_index()
    print("✅ Connected to Pinecone index!")

    # Test 2 — upsert a dummy vector
    dummy_chunks = [
        {
            "chunk_id": "test_chunk_0001",
            "embedding": [0.1] * 384,
            "text": "This is a test chunk for Business Communication.",
            "page": 1,
            "source": "test.pdf",
        }
    ]
    upsert_chunks(dummy_chunks)
    print("✅ Upsert working!")

    # Test 3 — query it back
    results = query_index([0.1] * 384, top_k=1)
    print(f"✅ Query working! Got {len(results)} result(s)")
    for r in results:
        print(f"   score={r['score']} | {r['text'][:60]}...")

    print("\n✅ Pinecone fully working!")