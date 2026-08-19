"""
Shared constants for the RAG pipeline: file paths and Ollama settings.

Kept in one place so every stage (ingestion, chunking, embeddings,
vector store, bot, evaluation) references the same values instead of
each duplicating its own copy.
"""

from pathlib import Path

# --- Data locations ---
PDF_FOLDER = Path("data/pdfs")
PROCESSED_DIR = Path("data/processed")
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"
EMBEDDINGS_FILE = PROCESSED_DIR / "embeddings.npy"
INDEX_FILE = PROCESSED_DIR / "faiss.index"

EVAL_DIR = Path("data/eval")
QUESTIONS_FILE = EVAL_DIR / "questions.json"
RESULTS_FILE = EVAL_DIR / "results.json"

# --- Ollama settings ---
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"
