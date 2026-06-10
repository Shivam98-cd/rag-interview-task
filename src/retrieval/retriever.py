"""
src/retrieval/retriever.py

Orchestrates semantic retrieval:
  1. Embed the user's question
  2. Query Pinecone for top-k relevant chunks
  3. Return ranked context chunks
"""

from __future__ import annotations

from src.embedding.embedder import embed_query
from src.vectordb.pinecone_client import query_index
import config


def retrieve(question: str, top_k: int = None) -> list[dict]:
    """
    Retrieve the most relevant text chunks for a given question.

    Args:
        question: The user's natural language question.
        top_k:    Number of chunks to return (defaults to config.RETRIEVAL_TOP_K).

    Returns:
        List of chunk dicts sorted by relevance score (highest first):
        [{"chunk_id", "text", "page", "source", "score"}, ...]
    """
    top_k = top_k or config.RETRIEVAL_TOP_K

    # Step 1: Embed the query
    query_vector = embed_query(question)

    # Step 2: Search Pinecone
    results = query_index(query_vector, top_k=top_k)

    print(
        f"[Retriever] Retrieved {len(results)} chunks for query: "
        f"'{question[:60]}...'"
    )
    for r in results:
        print(f"  → page={r['page']} score={r['score']} | {r['text'][:80]}...")

    return results


def build_context(chunks: list[dict]) -> str:
    """
    Concatenate retrieved chunks into a single context string for the LLM.

    Adds page references so the LLM can cite sources if needed.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[Chunk {i} | Page {chunk['page']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)

if __name__ == "__main__":
    import sys
    sys.path.insert(0, r"D:\rag-interview-task")

    import config
    config.validate()

    test_question = "What is business communication?"

    print(f"Testing retriever with question: '{test_question}'\n")

    # Retrieve relevant chunks
    chunks = retrieve(test_question)

    print(f"\nTop {len(chunks)} retrieved chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n[{i}] score={chunk['score']} | page={chunk['page']}")
        print(f"    {chunk['text'][:150]}...")

    # Build context
    context = build_context(chunks)
    print(f"\n{'='*50}")
    print("Built context string (first 300 chars):")
    print(context[:300])
    print("\n✅ Retriever working!")