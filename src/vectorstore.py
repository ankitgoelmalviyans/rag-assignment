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
"""

import json
import time

import faiss
import numpy as np
import requests

from config import (
    CHUNKS_FILE,
    EMBED_MODEL,
    EMBEDDINGS_FILE,
    INDEX_FILE,
    OLLAMA_BASE_URL,
    PROCESSED_DIR,
)

BATCH_SIZE = 32
MAX_RETRIES = 3


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each vector to unit length, so FAISS inner product == cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def embed_texts(texts: list[str], model: str = EMBED_MODEL, batch_size: int = BATCH_SIZE) -> np.ndarray:
    """
    Turn a list of strings into a 2D array of normalized embedding vectors,
    one row per input text, in the same order. Calls Ollama's /api/embed in
    batches to keep the number of HTTP requests small, retrying each batch
    a few times before giving up.
    """
    all_vectors = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        batch_vectors = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/embed",
                    json={"model": model, "input": batch},
                    timeout=120,
                )
                response.raise_for_status()
                batch_vectors = response.json()["embeddings"]
                break
            except requests.exceptions.RequestException as error:
                if attempt == MAX_RETRIES:
                    raise
                print(f"  embed batch failed (attempt {attempt}/{MAX_RETRIES}): {error} -- retrying")
                time.sleep(2 ** attempt)

        all_vectors.extend(batch_vectors)

    vectors = np.array(all_vectors, dtype="float32")
    return normalize(vectors)


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)
    print(f"Computed embeddings: shape = {embeddings.shape}")

    np.save(EMBEDDINGS_FILE, embeddings)
    print(f"Saved embeddings to {EMBEDDINGS_FILE}")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_FILE))
    print(f"Saved FAISS index ({index.ntotal} vectors, dimension {dimension}) to {INDEX_FILE}")

    # --- sanity checks ---
    reloaded_index = faiss.read_index(str(INDEX_FILE))
    assert reloaded_index.ntotal == len(chunks), "index size doesn't match chunk count!"
    print(f"Reload check passed: {reloaded_index.ntotal} vectors match {len(chunks)} chunks")

    sample_query = "self-attention mechanism formula"
    query_vector = embed_texts([sample_query])
    scores, row_indices = reloaded_index.search(query_vector, k=3)

    print(f"\nSample query: {sample_query!r}")
    for score, row in zip(scores[0], row_indices[0]):
        chunk = chunks[row]
        preview = chunk["text"][:100].replace("\n", " ")
        print(f"  score={score:.3f} | {chunk['doc_id']} p.{chunk['page_start']}-{chunk['page_end']} | {preview}...")
