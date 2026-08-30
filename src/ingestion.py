"""
PDF ingestion and cleaning stage of the RAG pipeline.

Reads every PDF in PDF_FOLDER, extracts text per page, and removes
structural extraction noise (repeated headers/footers, standalone page
numbers, excess blank lines) before the text is handed off to the
chunking stage. Noise is detected structurally (by repetition/pattern),
not by filtering on word content, so this works across any PDF.

WHY CLEANING IS NEEDED AT ALL

A PDF is a page-layout format, not a text format. When you pull text out of
one you also get whatever repeats on every page: the paper's running title,
page numbers, footers. If that noise stayed in, it would end up inside the
chunks, get embedded as if it were content, and pollute search results --
every chunk of a paper would look slightly "about" that paper's title.

The trick used here is to spot noise by BEHAVIOUR rather than by wording: a
line appearing on 60%+ of a document's pages is almost certainly a header,
whatever it happens to say. That means no hand-written list of titles to
maintain, and it works the same on any PDF you throw at it.
"""

import re                          # regular expressions, like System.Text.RegularExpressions
from collections import Counter    # a dict that counts things for you
from pathlib import Path           # object-oriented file paths

from pypdf import PdfReader        # the PDF text-extraction library

from config import PDF_FOLDER

# a repeated line must show up on at least this fraction of pages
# to be treated as a header/footer
HEADER_FOOTER_THRESHOLD = 0.6


def extract_pages(pdf_path: Path) -> list[str]:
    """Read a PDF and return a list of raw text, one entry per page."""
    reader = PdfReader(pdf_path)

    # A list comprehension over the PDF's pages. `or ""` substitutes an empty
    # string when extract_text() returns None -- which happens on pages that
    # are pure images with no text layer. Without it, later code would crash
    # trying to call string methods on None.
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
    # Counter is a dict that starts every key at 0 automatically, so you can
    # do counts[x] += 1 without checking whether x exists first. In C# you
    # would need: if (!d.ContainsKey(k)) d[k] = 0; d[k]++;
    line_counts = Counter()

    for lines in pages_lines:
        # set() so a line repeated twice on one page only counts once.
        # A set holds no duplicates -- C#'s HashSet<T>.
        # The trailing "if line.strip()" filters out blank lines.
        unique_lines_on_page = set(line.strip() for line in lines if line.strip())
        for line in unique_lines_on_page:
            line_counts[line] += 1

    total_pages = len(pages_lines)
    if total_pages <= 1:
        return set()        # a single page proves nothing about repetition

    # A set comprehension: keep only the lines appearing on enough pages.
    # .items() iterates key/value pairs, like a C# KeyValuePair loop.
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
    # Split every page into its individual lines first, so we can look at
    # them one at a time.
    pages_lines = [page.split("\n") for page in pages_text]

    # Work out the noise ONCE for the whole document, before cleaning any
    # page -- you cannot tell a header from body text by looking at one page.
    repeated_lines = find_repeated_lines(pages_lines)

    cleaned_pages = []
    for lines in pages_lines:
        kept_lines = []
        for line in lines:
            stripped = line.strip()

            if stripped in repeated_lines:
                continue  # header/footer -- `continue` skips to the next line

            # A regex matching a line that is ONLY a page number: "12",
            # "Page 12", "12 of 20". fullmatch requires the WHOLE line to
            # match, so a sentence merely containing a number is left alone.
            #   \s+ = one or more spaces, \d+ = one or more digits,
            #   (...)? = optional group, r"..." = raw string so backslashes
            #   are literal (C#'s @"...")
            if re.fullmatch(r"(page\s+)?\d+(\s+of\s+\d+)?", stripped, flags=re.IGNORECASE):
                continue  # standalone page number

            kept_lines.append(line)

        page_clean = "\n".join(kept_lines)      # put the surviving lines back
        # Removing lines leaves gaps behind. Collapse any run of 3+ newlines
        # down to 2, so paragraph breaks stay meaningful for the chunker.
        page_clean = re.sub(r"\n{3,}", "\n\n", page_clean)  # collapse blank-line runs
        cleaned_pages.append(page_clean)

    # Return the cleaned text AND what was removed, so the __main__ block
    # below can show exactly what got stripped from each PDF.
    return cleaned_pages, repeated_lines


# Runs only when executed directly: python src/ingestion.py
# This is a debugging view, not a pipeline step -- chunking.py imports the
# functions above rather than running this. Use it to eyeball what the
# cleaner is actually removing from each PDF.
if __name__ == "__main__":
    # .glob("*.pdf") finds matching files in that folder, like
    # Directory.GetFiles(path, "*.pdf").
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
