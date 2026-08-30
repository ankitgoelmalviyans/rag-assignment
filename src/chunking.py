"""
Chunking stage of the RAG pipeline.

Takes the cleaned, per-page text produced by ingestion.py and combines
paragraphs (falling back to sentences when a block of text is too big --
some PDFs don't reliably preserve blank lines between paragraphs) into
semantically coherent chunks, each tagged with metadata (document, page
range, section, chunk id) needed later for embeddings and citations.
Chunk row order in the output file is the canonical order that
embeddings.npy and the FAISS index will rely on later.

WHY CHUNK AT ALL?

You cannot embed a whole 15-page paper as one vector -- a single list of 768
numbers cannot represent that much distinct meaning, and even if it could,
retrieval would only ever return "the whole paper" instead of the specific
paragraph that answers the question. So the text is cut into pieces small
enough to represent one idea, and big enough to be useful on their own.

Chunk size is a genuine trade-off:
  too small -> the chunk lacks the surrounding context needed to make sense
  too large -> its vector becomes an average of several ideas and matches
               nothing precisely
~250 words sits in the middle for this kind of writing.
"""

import json
import re

from config import CHUNKS_FILE, PDF_FOLDER, PROCESSED_DIR
from ingestion import clean_pages, extract_pages   # reuse the cleaning stage

# Soft goal: stop adding to a chunk once it would exceed this.
TARGET_WORDS = 250

# Hard cap: no single piece of text handed to the chunk builder may be
# longer than this.
MAX_WORDS = 400

# How many words from the end of one chunk get repeated at the start of the
# next. See build_chunks() for why this matters.
OVERLAP_WORDS = 45

# Matches a line that looks like a section heading: either a known section
# name, or a numbered heading such as "3.2 Model Architecture".
#   ^      = must match at the START of the text
#   a|b|c  = alternatives ("or")
#   \d+    = one or more digits, (\.\d+)* = repeated ".2" parts
# re.compile() parses the pattern once and reuses it, instead of re-parsing
# on every call -- like caching a Regex object in C# rather than newing one
# up each time.
SECTION_PATTERN = re.compile(
    r"^(abstract|introduction|related work|background|method|model architecture|"
    r"experiments|results|discussion|conclusion|references|\d+(\.\d+)*\s+[A-Z])",
    re.IGNORECASE,
)


# Finds sentence boundaries: whitespace that comes right AFTER . ! or ? and
# right BEFORE a capital letter.
#   (?<=...) "lookbehind" -- must be preceded by this, but don't consume it
#   (?=...)  "lookahead"  -- must be followed by this, but don't consume it
# Using both means the split happens at the space only, keeping the full
# stop attached to the sentence it ends.
SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def split_into_sentences(text):
    """Split text into sentences, breaking after . ! or ? followed by a capital letter.

    A rough heuristic, not a proper sentence tokeniser -- "Fig. 3" would fool
    it. That is acceptable because this is only a fallback for oversized
    blocks, not the main splitting strategy.
    """
    sentences = SENTENCE_PATTERN.split(text)
    # Trim each piece and drop any that are empty after trimming.
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

    # enumerate() gives you the index alongside the item; start=1 makes page
    # numbers human-friendly (1, 2, 3...) instead of starting at 0.
    for page_number, page_text in enumerate(pages_text, start=1):
        # "\n\n" (a blank line) usually marks a paragraph break.
        for block in page_text.split("\n\n"):
            block = block.strip()
            if not block:          # empty string -> falsy -> skip it
                continue

            # .split() with no argument splits on any whitespace, so
            # len(text.split()) is a quick word count.
            if len(block.split()) <= MAX_WORDS:
                # Store a tuple (page_number, text). A tuple is a fixed
                # grouping of values -- C#'s ValueTuple. Keeping the page
                # number attached is what lets chunks cite a page later.
                segments.append((page_number, block))
                continue

            # FALLBACK 1: this block is too big, which means the PDF didn't
            # give us usable paragraph breaks. Split it into sentences.
            for sentence in split_into_sentences(block):
                if len(sentence.split()) <= MAX_WORDS:
                    segments.append((page_number, sentence))
                else:
                    # FALLBACK 2: even one "sentence" is over the cap (badly
                    # extracted text, tables, formulas). Chop it by word
                    # count. Crude, but guarantees nothing oversized escapes.
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
    # splitlines()[0] takes the first line. The length check filters out
    # ordinary sentences that happen to begin with a section-like word --
    # a real heading is short.
    first_line = paragraph_text.strip().splitlines()[0]
    if len(first_line) < 60 and SECTION_PATTERN.match(first_line):
        return first_line
    # Not a heading: carry the previous section forward, so every chunk
    # inherits whichever section it sits under.
    return current_section


def make_chunk(segments_in_chunk, doc_id, source_file, section, overlap_text, chunk_index):
    """Build one chunk dict from a list of (page_number, text) segments."""
    # "s for _, s in ..." unpacks each (page_number, text) tuple and keeps
    # only the text. The underscore is the conventional name for "a value I
    # have to unpack but don't need" -- C#'s discard, _.
    text = overlap_text + " ".join(s for _, s in segments_in_chunk)

    # Same idea, keeping only the page numbers this time.
    page_numbers = [pg for pg, _ in segments_in_chunk]

    return {
        # {chunk_index:04d} zero-pads to 4 digits: 7 -> "0007". Keeps ids
        # sorting correctly as text. C#: chunkIndex.ToString("D4")
        "chunk_id": f"{doc_id}-{chunk_index:04d}",
        "doc_id": doc_id,
        "source_file": source_file,
        # A chunk can span a page break, so record both ends of the range.
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

    WHY THE OVERLAP MATTERS. Imagine a key sentence landing exactly on a
    boundary: "The base model uses 8 attention heads, | each with dimension
    64." Without overlap, neither chunk contains the complete fact, and
    whichever one is retrieved gives a partial answer. Repeating the last 45
    words at the start of the next chunk means the full sentence survives
    intact in at least one of them.
    """
    chunks = []
    current_segments = []      # segments accumulating into the current chunk
    current_words = 0          # running word count for the current chunk
    current_section = None     # last heading seen, carried forward
    overlap_text = ""          # tail of the previous chunk, prepended to this

    for page_number, segment_text in segments:
        current_section = detect_section(segment_text, current_section)
        segment_words = len(segment_text.split())

        # Safety net: a single oversized segment becomes a chunk by itself.
        # split_into_segments() should already prevent this.
        if segment_words > MAX_WORDS:
            if current_segments:          # flush whatever is pending first
                chunks.append(make_chunk(current_segments, doc_id, source_file, current_section, overlap_text, len(chunks)))
                overlap_text = ""
                current_segments = []
                current_words = 0
            chunks.append(make_chunk([(page_number, segment_text)], doc_id, source_file, current_section, "", len(chunks)))
            continue

        # THE MAIN RULE: if adding this segment would push us past the target,
        # close the current chunk off first.
        if current_segments and current_words + segment_words > TARGET_WORDS:
            chunk = make_chunk(current_segments, doc_id, source_file, current_section, overlap_text, len(chunks))
            chunks.append(chunk)

            # Save the last OVERLAP_WORDS words to prepend to the next chunk.
            # words[-45:] is negative slicing: "the last 45 items".
            words = chunk["text"].split()
            overlap_text = " ".join(words[-OVERLAP_WORDS:]) + " "

            current_segments = []         # start a fresh chunk
            current_words = 0

        # Add the segment to whichever chunk is currently open.
        current_segments.append((page_number, segment_text))
        current_words += segment_words

    # The loop ends with a partly-filled chunk still pending -- don't lose it.
    if current_segments:
        chunks.append(make_chunk(current_segments, doc_id, source_file, current_section, overlap_text, len(chunks)))

    return chunks


# Runs only when executed directly: python src/chunking.py
# This IS a pipeline step -- it writes chunks.json, which vectorstore.py
# then reads.
if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    for pdf_file in PDF_FOLDER.glob("*.pdf"):
        # .stem is the filename without its extension: "1706.03762.pdf"
        # becomes "1706.03762", which is used as the document id.
        doc_id = pdf_file.stem

        # The full per-document pipeline, one stage per line:
        pages_text = extract_pages(pdf_file)            # PDF  -> raw text
        cleaned_pages, _ = clean_pages(pages_text)      # strip header noise
                                                        # (_ discards the
                                                        # removed-lines set)
        segments = split_into_segments(cleaned_pages)   # -> paragraph pieces
        chunks = build_chunks(segments, doc_id, pdf_file.name)   # -> chunks

        all_chunks.extend(chunks)     # extend adds each item; append would
                                       # have nested the whole list

        # Print size statistics per document. Worth doing: a wildly large max
        # would mean the splitting logic failed on this PDF's formatting, and
        # that is far easier to spot here than to debug later via bad answers.
        word_counts = [c["word_count"] for c in chunks]
        # sum(1 for ... if ...) counts matching items -- LINQ's .Count(x => ...)
        no_section = sum(1 for c in chunks if c["section"] is None)
        print(f"{pdf_file.name}: {len(chunks)} chunks, "
              # // is integer division (discards the remainder)
              f"words min/avg/max = {min(word_counts)}/{sum(word_counts)//len(word_counts)}/{max(word_counts)}, "
              f"{no_section} with no detected section")

    # Write every document's chunks to ONE file, in the order they were
    # generated. That order is load-bearing: vectorstore.py embeds this list
    # top to bottom, and FAISS row i then corresponds to chunk i. Reordering
    # this file without rebuilding the index would silently break every
    # citation, so always run chunking.py and vectorstore.py together.
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        # json.dump writes to a file (json.dumps returns a string instead).
        # indent=2 pretty-prints it so a human can read the output.
        json.dump(all_chunks, f, indent=2)

    print(f"\nWrote {len(all_chunks)} total chunks to {CHUNKS_FILE}")
