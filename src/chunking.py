"""
Chunking stage of the RAG pipeline (LangChain branch).

Takes the cleaned, page-level Documents produced by ingestion.py and splits
them into ~TARGET_WORDS chunks with OVERLAP_WORDS of overlap, using
LangChain's RecursiveCharacterTextSplitter (configured in langchain_service
to count words rather than characters, so it stays comparable to the
hand-written implementation on `main`).

What LangChain replaced: the ~80 lines of paragraph accumulation, sentence
fallback splitting, and manual overlap-tail carrying are now one configured
splitter object.

What stayed hand-written: section detection. Tagging a chunk with the paper
section it came from ("3.3.1 Task #1: Masked LM") is bespoke metadata this
project wants for citations, and LangChain has no equivalent -- so it runs
as a post-processing pass over the split Documents' metadata.

chunks.json is still written out, but now purely as a human-readable
artifact for inspection and for the report. The FAISS store built in
vectorstore.py carries its own copy of the text and metadata, so nothing
downstream depends on this file's row order.
"""

import json
import re

from langchain_core.documents import Document

from config import CHUNKS_FILE, PDF_FOLDER, PROCESSED_DIR
from ingestion import load_and_clean
from langchain_service import split_documents

SECTION_PATTERN = re.compile(
    r"^(abstract|introduction|related work|background|method|model architecture|"
    r"experiments|results|discussion|conclusion|references|\d+(\.\d+)*\s+[A-Z])",
    re.IGNORECASE,
)


def detect_section(text: str, current_section: str | None) -> str | None:
    """
    Best-effort section heading detection. If the chunk's first line looks
    like a heading, that becomes the current section; otherwise the previous
    value carries forward.
    """
    lines = text.strip().splitlines()
    if not lines:
        return current_section
    first_line = lines[0]
    if len(first_line) < 60 and SECTION_PATTERN.match(first_line):
        return first_line
    return current_section


def tag_chunk_metadata(chunks: list[Document], doc_id: str) -> list[Document]:
    """
    Add this project's own metadata to each split Document: a stable chunk_id,
    the detected section, and a word count. Runs in document order, since
    section detection carries forward from one chunk to the next.
    """
    current_section = None
    for index, chunk in enumerate(chunks):
        current_section = detect_section(chunk.page_content, current_section)
        chunk.metadata["chunk_id"] = f"{doc_id}-{index:04d}"
        chunk.metadata["section"] = current_section
        chunk.metadata["word_count"] = len(chunk.page_content.split())
    return chunks


def build_chunks_for_pdf(pdf_path) -> list[Document]:
    """Load, clean, split, and tag one PDF -- returns chunk-level Documents."""
    cleaned_pages, _ = load_and_clean(pdf_path)
    chunks = split_documents(cleaned_pages)
    return tag_chunk_metadata(chunks, pdf_path.stem)


def build_all_chunks() -> list[Document]:
    """Chunk every PDF in PDF_FOLDER -- what vectorstore.py calls."""
    all_chunks = []
    for pdf_file in sorted(PDF_FOLDER.glob("*.pdf")):
        all_chunks.extend(build_chunks_for_pdf(pdf_file))
    return all_chunks


def chunks_to_records(chunks: list[Document]) -> list[dict]:
    """Flatten Documents into plain dicts for the human-readable chunks.json."""
    return [
        {
            "chunk_id": c.metadata.get("chunk_id"),
            "doc_id": c.metadata.get("doc_id"),
            "source_file": c.metadata.get("source_file"),
            "page": c.metadata.get("page"),
            "section": c.metadata.get("section"),
            "text": c.page_content,
            "word_count": c.metadata.get("word_count"),
        }
        for c in chunks
    ]


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    for pdf_file in sorted(PDF_FOLDER.glob("*.pdf")):
        chunks = build_chunks_for_pdf(pdf_file)
        all_chunks.extend(chunks)

        word_counts = [c.metadata["word_count"] for c in chunks]
        no_section = sum(1 for c in chunks if c.metadata["section"] is None)
        print(f"{pdf_file.name}: {len(chunks)} chunks, "
              f"words min/avg/max = {min(word_counts)}/{sum(word_counts)//len(word_counts)}/{max(word_counts)}, "
              f"{no_section} with no detected section")

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks_to_records(all_chunks), f, indent=2)

    print(f"\nWrote {len(all_chunks)} total chunks to {CHUNKS_FILE}")
