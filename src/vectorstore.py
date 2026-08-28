"""
Embedding + FAISS vector store stage of the RAG pipeline (LangChain branch).

Builds a LangChain FAISS store over the chunk Documents produced by
chunking.py and persists it to VECTORSTORE_DIR.

The biggest structural change from the hand-written implementation on `main`:
that version maintained TWO artifacts -- faiss.index (vectors) and
chunks.json (text/metadata) -- linked only by matching row position, an
invariant the code had to protect by hand. LangChain's FAISS object owns the
index and the docstore together, so a search returns Documents carrying their
own text and metadata. There is no positional link left to break.

Embedding batching and retry, previously hand-written here, are handled
inside OllamaEmbeddings.
"""

from config import VECTORSTORE_DIR
from chunking import build_all_chunks
from langchain_service import (
    build_vectorstore,
    load_vectorstore,
    save_vectorstore,
    search_with_scores,
)

if __name__ == "__main__":
    chunks = build_all_chunks()
    print(f"Built {len(chunks)} chunks from the source PDFs")

    print("Embedding chunks via Ollama (nomic-embed-text)...")
    store = build_vectorstore(chunks)

    save_vectorstore(store, VECTORSTORE_DIR)
    print(f"Saved FAISS store ({store.index.ntotal} vectors, "
          f"dimension {store.index.d}) to {VECTORSTORE_DIR}")

    # --- sanity checks ---
    reloaded = load_vectorstore(VECTORSTORE_DIR)
    assert reloaded.index.ntotal == len(chunks), "reloaded store size doesn't match chunk count!"
    print(f"Reload check passed: {reloaded.index.ntotal} vectors match {len(chunks)} chunks")

    sample_query = "self-attention mechanism formula"
    results = search_with_scores(reloaded, sample_query, k=3)

    print(f"\nSample query: {sample_query!r}")
    for document, score in results:
        preview = document.page_content[:100].replace("\n", " ")
        print(f"  score={score:.3f} | {document.metadata.get('doc_id')} "
              f"p.{document.metadata.get('page')} | {preview}...")
