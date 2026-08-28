"""
PDF ingestion and cleaning stage of the RAG pipeline (LangChain branch).

Loads every PDF in PDF_FOLDER as LangChain Documents (one per page) via
langchain_service, then removes structural extraction noise -- repeated
headers/footers, standalone page numbers, excess blank lines -- before the
text is handed off to the chunking stage.

Division of labour worth noting: LangChain's PyPDFLoader replaces the raw
pypdf page-reading loop, but LangChain ships no equivalent for the
noise-cleaning below. Detecting a running header by how often a line repeats
across a document is bespoke logic, so it stays hand-written here and simply
operates on Document.page_content instead of plain strings.
"""

import re
from collections import Counter

from langchain_core.documents import Document

from config import PDF_FOLDER
from langchain_service import load_pdf_pages

# a repeated line must show up on at least this fraction of pages
# to be treated as a header/footer
HEADER_FOOTER_THRESHOLD = 0.6


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


def clean_documents(pages: list[Document]) -> tuple[list[Document], set[str]]:
    """
    Remove structural PDF noise from a document's page-level Documents.

    Args:
        pages: LangChain Documents, one per page, in page order, all from
            the same PDF.

    Returns:
        A tuple of (cleaned_pages, repeated_lines):
        - cleaned_pages: new Documents with noise stripped from page_content,
          metadata carried through unchanged.
        - repeated_lines: the header/footer lines that were stripped out,
          returned for visibility/debugging.
    """
    pages_lines = [page.page_content.split("\n") for page in pages]
    repeated_lines = find_repeated_lines(pages_lines)

    cleaned_pages = []
    for page, lines in zip(pages, pages_lines):
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

        cleaned_pages.append(Document(page_content=page_clean, metadata=dict(page.metadata)))

    return cleaned_pages, repeated_lines


def load_and_clean(pdf_path) -> tuple[list[Document], set[str]]:
    """Load one PDF and clean it in a single call -- what chunking.py uses."""
    return clean_documents(load_pdf_pages(pdf_path))


if __name__ == "__main__":
    for pdf_file in sorted(PDF_FOLDER.glob("*.pdf")):
        pages = load_pdf_pages(pdf_file)
        cleaned_pages, repeated_lines = clean_documents(pages)

        original_chars = sum(len(p.page_content) for p in pages)
        cleaned_chars = sum(len(p.page_content) for p in cleaned_pages)

        print(f"File: {pdf_file.name}")
        print(f"Pages: {len(pages)}")
        print(f"Characters before cleaning: {original_chars}")
        print(f"Characters after cleaning:  {cleaned_chars}")
        print(f"Removed as header/footer noise ({len(repeated_lines)} distinct lines):")
        for line in repeated_lines:
            print(f"  - {line!r}")
        print("-" * 50)
