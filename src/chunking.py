"""
Chunking stage of the RAG pipeline.

Takes the cleaned, per-page text produced by ingestion.py and combines
paragraphs (falling back to sentences when a block of text is too big --
some PDFs don't reliably preserve blank lines between paragraphs) into
semantically coherent chunks, each tagged with metadata (document, page
range, section, chunk id) needed later for embeddings and citations.
Chunk row order in the output file is the canonical order that
embeddings.npy and the FAISS index will rely on later.
"""

import json
import re

from config import CHUNKS_FILE, PDF_FOLDER, PROCESSED_DIR
from ingestion import clean_pages, extract_pages

TARGET_WORDS = 250
MAX_WORDS = 400
OVERLAP_WORDS = 45

SECTION_PATTERN = re.compile(
    r"^(abstract|introduction|related work|background|method|model architecture|"
    r"experiments|results|discussion|conclusion|references|\d+(\.\d+)*\s+[A-Z])",
    re.IGNORECASE,
)


SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def split_into_sentences(text):
    """Split text into sentences, breaking after . ! or ? followed by a capital letter."""
    sentences = SENTENCE_PATTERN.split(text)
    return [s.strip() for s in sentences if s.strip()]


def split_into_segments(pages_text):
    """
    Turn a document's cleaned per-page text into one ordered list of
    (page_number, segment_text) tuples across the whole document.
    Page numbers start at 1.

    Splits on blank lines first (real paragraph breaks, when present).
    Any resulting block still bigger than MAX_WORDS gets broken down
    further by sentence -- some PDFs don't reliably preserve blank
    lines between paragraphs, so this guarantees no unit handed to
    build_chunks() is ever oversized, regardless of PDF formatting.
    """
    segments = []
    for page_number, page_text in enumerate(pages_text, start=1):
        for block in page_text.split("\n\n"):
            block = block.strip()
            if not block:
                continue

            if len(block.split()) <= MAX_WORDS:
                segments.append((page_number, block))
                continue

            for sentence in split_into_sentences(block):
                if len(sentence.split()) <= MAX_WORDS:
                    segments.append((page_number, sentence))
                else:
                    words = sentence.split()
                    for i in range(0, len(words), MAX_WORDS):
                        piece = " ".join(words[i:i + MAX_WORDS])
                        segments.append((page_number, piece))

    return segments


def detect_section(paragraph_text, current_section):
    """
    Best-effort section heading detection. If the paragraph's first line
    looks like a heading, update the current section; otherwise keep
    whatever section was last detected.
    """
    first_line = paragraph_text.strip().splitlines()[0]
    if len(first_line) < 60 and SECTION_PATTERN.match(first_line):
        return first_line
    return current_section


def make_chunk(segments_in_chunk, doc_id, source_file, section, overlap_text, chunk_index):
    """Build one chunk dict from a list of (page_number, text) segments."""
    text = overlap_text + " ".join(s for _, s in segments_in_chunk)
    page_numbers = [pg for pg, _ in segments_in_chunk]
    return {
        "chunk_id": f"{doc_id}-{chunk_index:04d}",
        "doc_id": doc_id,
        "source_file": source_file,
        "page_start": min(page_numbers),
        "page_end": max(page_numbers),
        "section": section,
        "text": text,
        "word_count": len(text.split()),
    }


def build_chunks(segments, doc_id, source_file):
    """
    Greedily combine consecutive segments into chunks of roughly
    TARGET_WORDS, carrying a tail of OVERLAP_WORDS into the next chunk.
    A segment longer than MAX_WORDS on its own becomes its own chunk
    (guaranteed not to happen in practice, since split_into_segments()
    already caps every segment at MAX_WORDS -- kept as a safety net).
    """
    chunks = []
    current_segments = []
    current_words = 0
    current_section = None
    overlap_text = ""

    for page_number, segment_text in segments:
        current_section = detect_section(segment_text, current_section)
        segment_words = len(segment_text.split())

        if segment_words > MAX_WORDS:
            if current_segments:
                chunks.append(make_chunk(current_segments, doc_id, source_file, current_section, overlap_text, len(chunks)))
                overlap_text = ""
                current_segments = []
                current_words = 0
            chunks.append(make_chunk([(page_number, segment_text)], doc_id, source_file, current_section, "", len(chunks)))
            continue

        if current_segments and current_words + segment_words > TARGET_WORDS:
            chunk = make_chunk(current_segments, doc_id, source_file, current_section, overlap_text, len(chunks))
            chunks.append(chunk)
            words = chunk["text"].split()
            overlap_text = " ".join(words[-OVERLAP_WORDS:]) + " "
            current_segments = []
            current_words = 0

        current_segments.append((page_number, segment_text))
        current_words += segment_words

    if current_segments:
        chunks.append(make_chunk(current_segments, doc_id, source_file, current_section, overlap_text, len(chunks)))

    return chunks


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    for pdf_file in PDF_FOLDER.glob("*.pdf"):
        doc_id = pdf_file.stem
        pages_text = extract_pages(pdf_file)
        cleaned_pages, _ = clean_pages(pages_text)
        segments = split_into_segments(cleaned_pages)
        chunks = build_chunks(segments, doc_id, pdf_file.name)
        all_chunks.extend(chunks)

        word_counts = [c["word_count"] for c in chunks]
        no_section = sum(1 for c in chunks if c["section"] is None)
        print(f"{pdf_file.name}: {len(chunks)} chunks, "
              f"words min/avg/max = {min(word_counts)}/{sum(word_counts)//len(word_counts)}/{max(word_counts)}, "
              f"{no_section} with no detected section")

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nWrote {len(all_chunks)} total chunks to {CHUNKS_FILE}")
