"""
src/ingestion/pdf_loader.py

Extracts ALL content from every PDF page using a two-pass strategy:

Pass 1 — pdfplumber  : extracts native text layer + tables
Pass 2 — OCR fallback: for pages where images/charts dominate and
                        pdfplumber yields little or no text, the page
                        is rasterized and run through pytesseract OCR
                        so that text inside flowcharts, bubble diagrams,
                        and infographics is captured too.

Requires system packages:
    apt install tesseract-ocr poppler-utils   (Linux)
    brew install tesseract poppler            (macOS)

Returns:
    [{"page": int, "text": str, "source": str, "extraction": str}, ...]
    where "extraction" is "native", "ocr", or "native+ocr"
"""

from __future__ import annotations
import re
import io

import pdfplumber

# OCR dependencies — imported lazily so the loader still works if they're
# missing (native-text PDFs don't need OCR at all).
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


# ── tunables ──────────────────────────────────────────────────────────────────

# If pdfplumber yields fewer than this many characters, we consider the page
# image-heavy and trigger OCR on it.
OCR_TRIGGER_THRESHOLD = 100   # characters


# ── helpers ───────────────────────────────────────────────────────────────────

def _table_to_text(table: list[list]) -> str:
    """Convert a pdfplumber table (list-of-lists) to readable text."""
    rows = []
    for row in table:
        cells = [str(cell).strip() if cell is not None else "" for cell in row]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def _clean(text: str) -> str:
    """Collapse whitespace and remove common PDF artefacts."""
    text = re.sub(r"[ \t]+", " ", text)        # horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)      # excessive blank lines
    text = re.sub(r"[^\x20-\x7E\n]", " ", text) # non-ASCII junk → space
    return text.strip()


def _ocr_page(pdf_path: str, page_number: int, dpi: int = 200) -> str:
    """
    Rasterize a single PDF page and run Tesseract OCR on it.

    Args:
        pdf_path:    Path to the PDF file.
        page_number: 1-indexed page number.
        dpi:         Render resolution. 200 is a good balance of speed/accuracy.

    Returns:
        OCR text string (may be empty if the image has no readable text).
    """
    if not OCR_AVAILABLE:
        return ""

    try:
        # convert_from_path uses poppler under the hood
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_number,
            last_page=page_number,
        )
        if not images:
            return ""

        ocr_text = pytesseract.image_to_string(images[0], lang="eng")
        return _clean(ocr_text)

    except Exception as exc:
        print(f"[Loader] OCR failed on page {page_number}: {exc}")
        return ""


# ── public API ────────────────────────────────────────────────────────────────

def load_pdf(pdf_path: str, ocr_fallback: bool = True) -> list[dict]:
    """
    Load a PDF and return a list of page dicts with all extracted content.

    Strategy per page:
      1. pdfplumber  → native text + tables
      2. If text < OCR_TRIGGER_THRESHOLD chars AND ocr_fallback=True
         → rasterize & OCR the page
      3. If both yield text, concatenate (native text first, OCR supplement)

    Args:
        pdf_path:     Absolute or relative path to the PDF.
        ocr_fallback: Enable OCR for image-heavy pages (default True).
                      Set False to skip OCR even if available.

    Returns:
        List of dicts:
        {
            "page":       int,   # 1-indexed page number
            "text":       str,   # all extracted text for this page
            "source":     str,   # filename
            "extraction": str,   # "native" | "ocr" | "native+ocr"
        }
    """
    if ocr_fallback and not OCR_AVAILABLE:
        print(
            "[Loader] WARNING: OCR requested but dependencies missing.\n"
            "         Install with: pip install pdf2image pytesseract\n"
            "         And system:   apt install tesseract-ocr poppler-utils\n"
            "         Falling back to native-text extraction only."
        )
        ocr_fallback = False

    pages: list[dict] = []
    source_name = pdf_path.split("/")[-1]

    native_count = ocr_count = combined_count = 0

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"[Loader] Processing {total_pages} pages from '{source_name}' ...")

        for i, page in enumerate(pdf.pages, start=1):

            # ── Pass 1: native text layer ──────────────────────────────────
            raw_text = page.extract_text() or ""

            # ── Pass 1b: tables ────────────────────────────────────────────
            tables = page.extract_tables() or []
            table_parts = [_table_to_text(t) for t in tables]
            table_text = ""
            if table_parts:
                table_text = "\n\n[TABLE]\n" + "\n\n[TABLE]\n".join(table_parts)

            native_combined = _clean(raw_text + table_text)

            # ── Pass 2: OCR fallback for image-heavy pages ─────────────────
            ocr_text = ""
            extraction_method = "native"

            if ocr_fallback and len(native_combined) < OCR_TRIGGER_THRESHOLD:
                print(
                    f"[Loader] Page {i}: only {len(native_combined)} chars from "
                    f"native extraction — running OCR ..."
                )
                ocr_text = _ocr_page(pdf_path, page_number=i)

                if ocr_text and native_combined:
                    extraction_method = "native+ocr"
                    combined_count += 1
                elif ocr_text:
                    extraction_method = "ocr"
                    ocr_count += 1
                else:
                    extraction_method = "native"
                    native_count += 1
            else:
                native_count += 1

            # ── Combine and store ──────────────────────────────────────────
            final_text = native_combined
            if ocr_text:
                # Append OCR text as supplement, tagged so we can trace it
                final_text = (
                    native_combined + "\n\n[OCR-EXTRACTED]\n" + ocr_text
                ).strip()

            if final_text:
                pages.append(
                    {
                        "page": i,
                        "text": final_text,
                        "source": source_name,
                        "extraction": extraction_method,
                    }
                )

    # ── Summary ────────────────────────────────────────────────────────────
    print(
        f"[Loader] Done. {len(pages)}/{total_pages} pages extracted.\n"
        f"         native={native_count}  ocr-only={ocr_count}  "
        f"native+ocr={combined_count}"
    )
    if not OCR_AVAILABLE and ocr_fallback is False:
        print(
            "[Loader] NOTE: Charts/diagrams that are pure images were NOT extracted.\n"
            "         Install pdf2image + pytesseract to enable OCR."
        )

    return pages

if __name__ == "__main__":
    import sys

    # Use provided path or default
    path = sys.argv[1] if len(sys.argv) > 1 else r"D:\rag-interview-task\data\business_communication.pdf"

    pages = load_pdf(path, ocr_fallback=False)  # set True if tesseract is installed

    print(f"\nTotal pages extracted: {len(pages)}")
    for p in pages:
        print(f"\n{'='*50}")
        print(f"Page {p['page']} | extraction={p['extraction']}")
        print(p['text'][:300])
        print("...")