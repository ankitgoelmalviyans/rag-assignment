"""
Shared constants for the RAG pipeline: file paths and Ollama settings.

Kept in one place so every stage (ingestion, chunking, embeddings,
vector store, bot, evaluation) references the same values instead of
each duplicating its own copy. Change a path or a model name here and
every file picks it up -- the classic bug this avoids is updating a value
in one file and forgetting the other four.

Roughly the role of appsettings.json in a .NET project, except it is real
Python code, so other files just import these names directly.
"""

from pathlib import Path      # Path objects handle / vs \ across OSes

# --- Data locations ---
#
# IMPORTANT: these are RELATIVE paths. They are resolved against whatever
# folder you are standing in when you run a script -- NOT against where this
# file lives. That is why every command must be run from the project root:
#
#     python src/bot.py          works    ("data/pdfs" exists from here)
#     cd src && python bot.py    breaks   (no "data/pdfs" inside src/)
PDF_FOLDER = Path("data/pdfs")
PROCESSED_DIR = Path("data/processed")

# Path supports / as a join operator, so this reads naturally and stays
# correct on any OS. Equivalent to Path.Combine() in .NET.
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"        # chunk text + metadata
EMBEDDINGS_FILE = PROCESSED_DIR / "embeddings.npy" # raw vectors (debug only)
INDEX_FILE = PROCESSED_DIR / "faiss.index"         # the searchable index

EVAL_DIR = Path("data/eval")
QUESTIONS_FILE = EVAL_DIR / "questions.json"       # the 10 test questions
RESULTS_FILE = EVAL_DIR / "results.json"           # written by evaluate.py

# --- Ollama settings ---
#
# Ollama runs a small web server on your machine that hosts the models. It is
# not a cloud service and needs no API key -- "calling the LLM" throughout
# this project just means an HTTP request to this address.
OLLAMA_BASE_URL = "http://localhost:11434"

# TWO models, doing two completely different jobs:
EMBED_MODEL = "nomic-embed-text"   # text -> 768 numbers, so text can be FOUND
CHAT_MODEL = "llama3.2"            # reads text and WRITES the answer

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
