"""
PDF ingestion and cleaning stage of the RAG pipeline.

Reads every PDF in PDF_FOLDER, extracts text per page, and removes
structural extraction noise (repeated headers/footers, standalone page
numbers, excess blank lines) before the text is handed off to the
chunking stage. Noise is detected structurally (by repetition/pattern),
not by filtering on word content, so this works across any PDF.
"""

import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from config import PDF_FOLDER

# a repeated line must show up on at least this fraction of pages
# to be treated as a header/footer
HEADER_FOOTER_THRESHOLD = 0.6


def extract_pages(pdf_path: Path) -> list[str]:
    """Read a PDF and return a list of raw text, one entry per page."""
    reader = PdfReader(pdf_path)
    return [page.extract_text() or "" for page in reader.pages]


def find_repeated_lines(pages_lines: list[list[str]], threshold: float = HEADER_FOOTER_THRESHOLD) -> set[str]:
    """
    Find lines that repeat across most pages of the SAME document.
    These are treated as headers/footers, regardless of what they say.

    Args:
        pages_lines: one list of text lines per page (same document).
        threshold: minimum fraction of pages a line must appear on
            (e.g. 0.6 = 60%+ of pages) to be treated as a header/footer.

    Returns:
        The set of distinct lines classified as repeated header/footer noise.
    """
    line_counts = Counter()

    for lines in pages_lines:
        # set() so a line repeated twice on one page only counts once
        unique_lines_on_page = set(line.strip() for line in lines if line.strip())
        for line in unique_lines_on_page:
            line_counts[line] += 1

    total_pages = len(pages_lines)
    if total_pages <= 1:
        return set()

    return {
        line for line, count in line_counts.items()
        if count / total_pages >= threshold
    }


def clean_pages(pages_text: list[str]) -> tuple[list[str], set[str]]:
    """
    Remove structural PDF noise (repeated headers/footers, standalone
    page numbers, excess blank lines) from a list of per-page text.

    Args:
        pages_text: raw extracted text, one entry per page, in page order.

    Returns:
        A tuple of (cleaned_pages, repeated_lines):
        - cleaned_pages: same length/order as pages_text, with noise removed.
        - repeated_lines: the header/footer lines that were stripped out,
          returned for visibility/debugging.
    """
    pages_lines = [page.split("\n") for page in pages_text]
    repeated_lines = find_repeated_lines(pages_lines)

    cleaned_pages = []
    for lines in pages_lines:
        kept_lines = []
        for line in lines:
            stripped = line.strip()

            if stripped in repeated_lines:
                continue  # header/footer

            if re.fullmatch(r"(page\s+)?\d+(\s+of\s+\d+)?", stripped, flags=re.IGNORECASE):
                continue  # standalone page number

            kept_lines.append(line)

        page_clean = "\n".join(kept_lines)
        page_clean = re.sub(r"\n{3,}", "\n\n", page_clean)  # collapse blank-line runs
        cleaned_pages.append(page_clean)

    return cleaned_pages, repeated_lines


if __name__ == "__main__":
    for pdf_file in PDF_FOLDER.glob("*.pdf"):
        pages_text = extract_pages(pdf_file)
        cleaned_pages, repeated_lines = clean_pages(pages_text)

        original_chars = sum(len(p) for p in pages_text)
        cleaned_chars = sum(len(p) for p in cleaned_pages)

        print(f"File: {pdf_file.name}")
        print(f"Pages: {len(pages_text)}")
        print(f"Characters before cleaning: {original_chars}")
        print(f"Characters after cleaning:  {cleaned_chars}")
        print(f"Removed as header/footer noise ({len(repeated_lines)} distinct lines):")
        for line in repeated_lines:
            print(f"  - {line!r}")
        print("-" * 50)
