"""
pdf_ingest.py — PDF ingestion for AcumenAI.

Extracts text from PDF files (WSJ articles, books, docs, etc.)
and feeds them into the evolutionary brain as training data.

Usage (inside AcumenAI):
    /pdf C:\\Users\\You\\Downloads\\wsj_article.pdf
    /pdf-dir C:\\Users\\You\\Downloads\\wsj_pdfs

Or in Python:
    from pdf_ingest import ingest_pdf_to_brain, ingest_pdf_dir_to_brain
    ingest_pdf_to_brain(brain, "article.pdf")
"""

from __future__ import annotations

import re
from pathlib import Path


def _extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Try pypdf first (lightweight)
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        if pages:
            return "\n\n".join(pages)
    except ImportError:
        pass
    except Exception:
        pass

    # Try pdfplumber (better layout handling)
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text.strip())
        if pages:
            return "\n\n".join(pages)
    except ImportError:
        pass
    except Exception:
        pass

    # Try pdfminer as last resort
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        text = pdfminer_extract(str(path))
        if text and text.strip():
            return text.strip()
    except ImportError:
        pass
    except Exception:
        pass

    raise RuntimeError(
        "Could not read PDF. Install a PDF library:\n"
        "  pip install pypdf\n"
        "  pip install pdfplumber\n"
        "  pip install pdfminer.six"
    )


def _clean_pdf_text(text: str) -> str:
    """Clean up extracted PDF text."""
    # Remove multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove page numbers (common in WSJ: "Page 1 of 4" etc.)
    text = re.sub(r"Page \d+ of \d+", "", text, flags=re.IGNORECASE)
    # Remove very short lines (headers/footers)
    lines = text.split("\n")
    lines = [l for l in lines if len(l.strip()) > 15 or l.strip() == ""]
    return "\n".join(lines).strip()


def ingest_pdf_to_brain(brain, pdf_path: str) -> str:
    """
    Extract text from a PDF and add it to the brain's training corpus.

    Returns a status message.
    """
    try:
        raw_text = _extract_text_from_pdf(pdf_path)
    except FileNotFoundError as e:
        return f"Error: {e}"
    except RuntimeError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Unexpected error reading PDF: {e}"

    if not raw_text or len(raw_text.strip()) < 50:
        return f"PDF appears to be empty or unreadable: {pdf_path}"

    clean = _clean_pdf_text(raw_text)
    char_count = len(clean)
    word_count = len(clean.split())

    # Feed into brain
    result = brain.add_text(clean, max_chars=500_000)

    filename = Path(pdf_path).name
    return (
        f"Ingested PDF: {filename}\n"
        f"  {word_count:,} words / {char_count:,} chars extracted\n"
        f"  Brain says: {result}"
    )


def ingest_pdf_dir_to_brain(brain, dir_path: str) -> str:
    """
    Ingest all PDF files in a directory into the brain.
    """
    folder = Path(dir_path)
    if not folder.exists():
        return f"Directory not found: {dir_path}"

    pdfs = list(folder.glob("*.pdf")) + list(folder.glob("*.PDF"))
    if not pdfs:
        return f"No PDF files found in: {dir_path}"

    results = []
    for pdf in pdfs:
        msg = ingest_pdf_to_brain(brain, str(pdf))
        results.append(msg)

    return f"Processed {len(pdfs)} PDFs:\n\n" + "\n\n".join(results)
