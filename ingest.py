"""
ingest.py — One-time pipeline to ingest the PDF into Pinecone.

Usage:
    python ingest.py                          # uses data/business_communication.pdf
    python ingest.py --pdf /path/to/file.pdf  # use any PDF
    python ingest.py --pdf /path/to/file.pdf --no-ocr  # skip OCR
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
config.validate()

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_pages
from src.embedding.embedder import embed_chunks
from src.vectordb.pinecone_client import upsert_chunks


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest a PDF into the RAG pipeline.")
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to the PDF file. Defaults to data/business_communication.pdf (or PDF_PATH in .env).",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR fallback for image-heavy pages.",
    )
    return parser.parse_args()


def run_ingestion(pdf_path: str = None, ocr: bool = True) -> None:
    # Resolve path: CLI arg → env var → default
    pdf_path = pdf_path or config.PDF_PATH

    if not os.path.exists(pdf_path):
        print(
            f"\n[Ingest] ERROR: PDF not found at '{pdf_path}'\n\n"
            "  Fix options:\n"
            "  1. Copy your PDF to:  rag_app/data/business_communication.pdf\n"
            "  2. Set in .env:       PDF_PATH=/full/path/to/your.pdf\n"
            "  3. Pass as argument:  python ingest.py --pdf /full/path/to/your.pdf\n"
        )
        sys.exit(1)

    print("=" * 60)
    print("RAG Ingestion Pipeline")
    print(f"PDF     : {pdf_path}")
    print(f"OCR     : {'enabled' if ocr else 'disabled'}")
    print("=" * 60)

    print("\n[Step 1/4] Loading PDF ...")
    pages = load_pdf(pdf_path, ocr_fallback=ocr)

    print("\n[Step 2/4] Chunking pages ...")
    chunks = chunk_pages(pages)

    print("\n[Step 3/4] Embedding chunks ...")
    chunks = embed_chunks(chunks)

    print("\n[Step 4/4] Upserting to Pinecone ...")
    upsert_chunks(chunks)

    print("\n" + "=" * 60)
    print(f"✅ Ingestion complete! {len(chunks)} chunks stored in Pinecone.")
    print("   Start the API: python src/api/app.py")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    run_ingestion(pdf_path=args.pdf, ocr=not args.no_ocr)