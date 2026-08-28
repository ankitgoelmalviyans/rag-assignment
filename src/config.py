"""
Shared constants for the RAG pipeline: file paths, Ollama settings, and the
tuning knobs every stage reads.

Kept in one place so every stage (ingestion, chunking, embeddings, vector
store, bot, evaluation) references the same values instead of each
duplicating its own copy.
"""

from pathlib import Path

# --- Data locations ---
PDF_FOLDER = Path("data/pdfs")
PROCESSED_DIR = Path("data/processed")
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"

# LangChain's FAISS wrapper persists a *folder* (index.faiss + index.pkl),
# not a single file -- it stores the vectors and the document/metadata
# store together, so there is no hand-maintained positional link between
# a row number and its chunk text.
VECTORSTORE_DIR = PROCESSED_DIR / "faiss_store"

EVAL_DIR = Path("data/eval")
QUESTIONS_FILE = EVAL_DIR / "questions.json"
RESULTS_FILE = EVAL_DIR / "results.json"

# --- Ollama settings ---
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"

# How long Ollama keeps the model resident in memory between calls. Loading
# llama3.2 off disk takes seconds; a long evaluation run makes dozens of
# calls, so holding the model between them is the single biggest local
# speed win available here.
OLLAMA_KEEP_ALIVE = "30m"

# Ollama's own default context window is 2048 tokens, which is smaller than
# a prompt carrying 5 retrieved chunks plus 4 turns of conversation memory --
# leaving it at the default silently truncates the oldest part of the prompt.
OLLAMA_NUM_CTX = 8192

# Generous per-call ceiling: local CPU inference on longer prompts has been
# measured in minutes, not seconds.
OLLAMA_TIMEOUT = 300

# --- Chunking ---
TARGET_WORDS = 250
OVERLAP_WORDS = 45

# --- Embedding ---
# OllamaEmbeddings.embed_documents() sends every text it is given in a single
# HTTP request. Handing it all ~500 chunks at once exceeds OLLAMA_TIMEOUT on
# CPU-only inference, so the store is built in batches of this size instead --
# the same approach the hand-written implementation used, and it gives
# progress output during what is otherwise a silent multi-minute wait.
EMBED_BATCH_SIZE = 32

# --- Retrieval / memory ---
TOP_K = 5
MEMORY_TURNS = 4
