"""
src/ingestion/chunker.py

Splits page-level text into overlapping chunks suitable for embedding.

Strategy: Recursive character splitting
  1. Try to split on double-newlines (paragraph boundary)
  2. Fall back to single newline
  3. Fall back to sentence boundary (". ")
  4. Hard-split by character count

Preserves metadata (page number, source) in every chunk.
"""

from __future__ import annotations
import config


# ── helpers ───────────────────────────────────────────────────────────────────

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Recursively split `text` into chunks of at most `chunk_size` characters
    with `chunk_overlap` character overlap between consecutive chunks.
    """
    chunks: list[str] = []

    def _recurse(text: str, separators: list[str]) -> None:
        if len(text) <= chunk_size:
            if text.strip():
                chunks.append(text.strip())
            return

        sep = separators[0] if separators else ""
        parts = text.split(sep) if sep else list(text)

        current = ""
        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                # If a single part is too large, recurse with next separator
                if len(part) > chunk_size and len(separators) > 1:
                    _recurse(part, separators[1:])
                    current = ""
                else:
                    current = part.strip()

        if current.strip():
            chunks.append(current.strip())

    _recurse(text, SEPARATORS)

    # Apply overlap: merge small consecutive chunks
    return _apply_overlap(chunks, chunk_size, chunk_overlap)


def _apply_overlap(chunks: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Re-assemble chunks with overlap so each chunk shares `chunk_overlap`
    characters with its predecessor.
    """
    if not chunks or chunk_overlap == 0:
        return chunks

    result: list[str] = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            result.append(chunk)
        else:
            # Prepend tail of previous chunk as context
            prev_tail = result[-1][-chunk_overlap:]
            merged = (prev_tail + " " + chunk).strip()
            # If merged still fits within chunk_size, great; otherwise just use chunk
            if len(merged) <= chunk_size:
                result.append(merged)
            else:
                result.append(chunk)
    return result


# ── public API ────────────────────────────────────────────────────────────────

def chunk_pages(
    pages: list[dict],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> list[dict]:
    """
    Split a list of page dicts into chunk dicts.

    Input:  [{"page": int, "text": str, "source": str}, ...]
    Output: [{"chunk_id": str, "text": str, "page": int, "source": str}, ...]
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    all_chunks: list[dict] = []
    chunk_idx = 0

    for page_dict in pages:
        text_chunks = _split_text(page_dict["text"], chunk_size, chunk_overlap)
        for chunk_text in text_chunks:
            all_chunks.append(
                {
                    "chunk_id": f"chunk_{chunk_idx:04d}",
                    "text": chunk_text,
                    "page": page_dict["page"],
                    "source": page_dict["source"],
                }
            )
            chunk_idx += 1

    print(
        f"[Chunker] Created {len(all_chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return all_chunks

if __name__ == "__main__":
    sample_pages = [
        {
            "page": 1,
            "text": "Business communication is the process of sharing information. " * 20,
            "source": "test.pdf"
        }
    ]
    chunks = chunk_pages(sample_pages, chunk_size=200, chunk_overlap=50)
    for c in chunks:
        print(f"[{c['chunk_id']}] page={c['page']} | {c['text'][:80]}...")
    print(f"\nTotal chunks: {len(chunks)}")