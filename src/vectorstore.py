"""
Embedding + FAISS vector store stage of the RAG pipeline.

Turns every chunk in chunks.json into a vector via Ollama's nomic-embed-text
model, and builds a FAISS index over those vectors for fast similarity
search. embed_texts() is also reused later by bot.py to embed a user's
question the same way, before searching this index.

FAISS row i always corresponds to chunks[i] in chunks.json -- that
positional link is how a search result (a row number) gets mapped back
to the actual chunk text/metadata. There is no separate metadata file;
chunks.json IS the metadata sidecar.

WHAT IS AN EMBEDDING?

A model reads a piece of text and outputs a fixed-length list of numbers --
768 of them here -- that represents its MEANING. Text about similar topics
produces similar numbers. That is the whole trick: once text is numbers, you
can measure how close two pieces of text are with arithmetic instead of
keyword matching. "How is BERT trained?" and "BERT's pretraining objective"
share almost no words but land close together.

The embedding model is small (137M parameters) and only does this one job.
It never writes sentences -- that is llama3.2's job, over in bot.py.
"""

import json
import time

import faiss           # the vector index
import numpy as np     # array maths library; think strongly-typed double[,]
                       # with built-in vector operations
import requests

from config import (
    CHUNKS_FILE,
    EMBED_MODEL,
    EMBEDDINGS_FILE,
    INDEX_FILE,
    OLLAMA_BASE_URL,
    PROCESSED_DIR,
)

# How many chunks to send to Ollama per HTTP request. 398 chunks in batches
# of 32 means ~13 requests instead of 398 -- far less network overhead.
BATCH_SIZE = 32

# How many times to retry a failed batch before giving up entirely.
MAX_RETRIES = 3


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each vector to unit length, so FAISS inner product == cosine similarity.

    "Unit length" means the arrow has length exactly 1, so only its DIRECTION
    carries meaning. That matters because we want to compare what text is
    about, not how long it happens to be.

    Once every vector has length 1, the cheap "inner product" calculation
    FAISS uses gives exactly the same answer as the more expensive standard
    similarity measure (cosine similarity).
    """
    # axis=1 means "compute one length per row" (per chunk), not one number
    # for the entire array. keepdims=True keeps the result shaped (398, 1) so
    # it can be divided straight back into the (398, 768) array below.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    # Dividing a whole array by another array divides element-by-element, no
    # loop needed. This is NumPy's big convenience over plain Python lists.
    return vectors / norms


def embed_texts(texts: list[str], model: str = EMBED_MODEL, batch_size: int = BATCH_SIZE) -> np.ndarray:
    """
    Turn a list of strings into a 2D array of normalized embedding vectors,
    one row per input text, in the same order. Calls Ollama's /api/embed in
    batches to keep the number of HTTP requests small, retrying each batch
    a few times before giving up.

    Used in TWO places, which is deliberate:
      * here, to embed all 398 chunks once when building the index
      * in bot.py, to embed the user's live question at query time
    Both must use the same model and the same normalisation, or the numbers
    would not be comparable -- like measuring one in miles and one in km.
    """
    all_vectors = []

    # range(start, stop, step) counts 0, 32, 64, ... up to len(texts).
    for start in range(0, len(texts), batch_size):
        # texts[start:start + batch_size] is "slicing": take that portion of
        # the list. Roughly LINQ's .Skip(start).Take(batch_size).
        batch = texts[start:start + batch_size]
        batch_vectors = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Same idea as calling the chat model, different endpoint:
                # /api/embed returns numbers instead of text.
                response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/embed",
                    json={"model": model, "input": batch},   # a whole batch
                                                              # in one request
                    timeout=120,
                )
                response.raise_for_status()

                # Reply shape: {"embeddings": [[0.1, ...], [0.4, ...], ...]}
                # -- one inner list of 768 numbers per input text.
                batch_vectors = response.json()["embeddings"]
                break        # success, stop retrying this batch

            except requests.exceptions.RequestException as error:
                if attempt == MAX_RETRIES:
                    raise    # out of attempts -- fail loudly
                print(f"  embed batch failed (attempt {attempt}/{MAX_RETRIES}): {error} -- retrying")
                # 2 ** attempt is "2 to the power of attempt" -> waits 2s,
                # then 4s. Backing off gives a busy server time to recover.
                time.sleep(2 ** attempt)

        # .extend() adds every item of a list; .append() would have added the
        # list itself as a single nested element.
        all_vectors.extend(batch_vectors)

    # Convert the plain Python lists into one NumPy array. float32 (rather
    # than Python's default float64) is what FAISS expects.
    vectors = np.array(all_vectors, dtype="float32")
    return normalize(vectors)


# Runs only when executed directly: python src/vectorstore.py
if __name__ == "__main__":
    # parents=True creates missing parent folders; exist_ok=True means "don't
    # error if it's already there". Equivalent to Directory.CreateDirectory().
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    # A "list comprehension": build a new list by pulling one field out of
    # every chunk. Same as LINQ's chunks.Select(c => c["text"]).ToList().
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embed_texts(texts)     # the slow part: real model calls
    # .shape is (rows, columns) -> (398, 768): 398 chunks, 768 numbers each.
    print(f"Computed embeddings: shape = {embeddings.shape}")

    # Save the raw numbers. Kept for inspection/debugging only -- nothing
    # downstream reads this file; the FAISS index below is what gets used.
    np.save(EMBEDDINGS_FILE, embeddings)
    print(f"Saved embeddings to {EMBEDDINGS_FILE}")

    # --- build the searchable index ---
    dimension = embeddings.shape[1]      # 768
    # IndexFlatIP: "Flat" = compare against every vector (exact, no shortcuts),
    # "IP" = inner product. Exact search is fine at ~400 vectors; the clever
    # approximate index types only pay off at millions.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)                # row order here == chunks.json order
    faiss.write_index(index, str(INDEX_FILE))
    print(f"Saved FAISS index ({index.ntotal} vectors, dimension {dimension}) to {INDEX_FILE}")

    # --- sanity checks ---
    # Prove the saved index can be reloaded and still lines up with chunks.json.
    reloaded_index = faiss.read_index(str(INDEX_FILE))
    # assert crashes immediately with this message if the condition is false.
    # A blunt internal check, not user-facing validation.
    assert reloaded_index.ntotal == len(chunks), "index size doesn't match chunk count!"
    print(f"Reload check passed: {reloaded_index.ntotal} vectors match {len(chunks)} chunks")

    # Run one real query end-to-end, so a broken index is obvious right away
    # rather than surfacing later inside the chatbot.
    sample_query = "self-attention mechanism formula"
    query_vector = embed_texts([sample_query])
    scores, row_indices = reloaded_index.search(query_vector, k=3)

    print(f"\nSample query: {sample_query!r}")
    for score, row in zip(scores[0], row_indices[0]):
        chunk = chunks[row]              # row number -> real chunk
        preview = chunk["text"][:100].replace("\n", " ")   # first 100 chars
        print(f"  score={score:.3f} | {chunk['doc_id']} p.{chunk['page_start']}-{chunk['page_end']} | {preview}...")
