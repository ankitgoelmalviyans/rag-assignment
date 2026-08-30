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

# Context window, in tokens, requested on every chat call.
#
# Ollama's own default is 2048. Measured against this corpus, a prompt carries
# 5 retrieved chunks (~207 words each, so ~1,400 tokens) plus the system
# prompt, the question, and up to 4 turns of conversation memory -- around
# 2,100 tokens once the memory window fills. That is over the default, and
# Ollama truncates silently rather than raising an error, so leaving this
# unset would quietly drop the oldest part of the prompt on later questions.
NUM_CTX = 8192

# How long Ollama keeps the model loaded between calls. An evaluation run
# makes roughly 20 calls; without this, the model can be unloaded and reloaded
# from disk repeatedly.
KEEP_ALIVE = "30m"
