# RAG Assignment — Conversational PDF Q&A Bot

A Retrieval-Augmented Generation (RAG) application that answers questions about 5 NLP research
papers (Transformer, BERT, RoBERTa, T5, GPT-3), remembers the last 4 turns of conversation, and
runs entirely locally using [Ollama](https://ollama.ai) + [FAISS](https://github.com/facebookresearch/faiss) — no cloud API, no API key.

Built for the "Generative AI Fundamentals" assignment. Every stage — PDF extraction,
chunking, embeddings, vector search, and conversational memory — is implemented from scratch in
plain Python (no LangChain/LlamaIndex), so the mechanics of RAG are fully visible rather than
hidden behind a framework.

---

## Table of contents

- [How it works, in one picture](#how-it-works-in-one-picture)
- [Terminology](#terminology)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Running the pipeline](#running-the-pipeline)
- [Stage-by-stage details](#stage-by-stage-details)
- [Design decisions](#design-decisions)
- [Current status](#current-status)
- [Known limitations](#known-limitations)

---

## How it works, in one picture

RAG has two phases that run at completely different times: **indexing** (once, offline) and
**query** (every time the user asks something).

```
INDEXING (run once, offline — src/chunking.py then src/vectorstore.py)
────────────────────────────────────────────────────────────────────────
  5 PDFs
    │  ingestion.py: pypdf extracts raw text, strips headers/footers/page numbers
    ▼
  cleaned per-page text
    │  chunking.py: splits into ~250-word chunks with 45-word overlap
    ▼
  chunks.json   (398 chunks: {chunk_id, doc_id, page_start, section, text, ...})
    │  vectorstore.py: embed_texts() calls Ollama's nomic-embed-text model
    ▼
  faiss.index   (one 768-dim vector per chunk, same row order as chunks.json)


QUERY (every time the user asks a question — src/bot.py)
────────────────────────────────────────────────────────────────────────
  "How does BERT's masked language modeling work?"
    │  embed_texts([question])  — same embedding model as above
    ▼
  query vector (768 numbers)
    │  faiss index.search(query_vector, k=5)  — pure vector math, NO model call
    ▼
  top-5 matching rows + similarity scores  →  looked up in chunks.json
    │  format_context() — stitches the 5 chunks into one labeled text block
    ▼
  "[Source: BERT, page 4]\n...\n\n[Source: RoBERTa, page 3]\n...\n\nQuestion: ..."
    │  + last 4 turns of conversation history prepended
    ▼
  ONE call to llama3.2 via Ollama's /api/chat
    ▼
  final answer (single string) — also appended to memory for the next turn
```

The key thing this diagram is meant to make obvious: **the embedding model and the LLM are two
different models used for two different jobs**, and they never talk to each other directly.
The embedding model turns text into vectors so similar text can be *found*. The LLM reads
whatever text was found and *reasons* over it to produce an answer. FAISS itself runs no model at
all — it's pure numeric similarity search.

---

## Terminology

| Term | Meaning in this project |
|---|---|
| **Chunk** | A ~250-word slice of a paper's text, small enough to embed meaningfully and retrieve precisely. See `chunks.json`. |
| **Embedding** | A list of numbers (a vector, 768 of them here) that represents a piece of text's *meaning* in a way that can be mathematically compared. Produced by the `nomic-embed-text` model. |
| **Vector store / vector database** | A structure optimized for finding the vectors most similar to a query vector. Here: FAISS's `IndexFlatIP`. |
| **FAISS** | Facebook AI Similarity Search — the library used as the vector store. `IndexFlatIP` does exact (brute-force) inner-product search, which is fine at this corpus size (~400 vectors). |
| **Cosine similarity / inner product** | A measure of how "close" two vectors are in direction. Vectors here are L2-normalized (`vectorstore.py: normalize()`) so FAISS's inner product search is equivalent to cosine similarity. |
| **Top-k retrieval** | Asking the vector store for the `k` closest chunks to a query (here `k=5`, `bot.py: TOP_K`). It's a setting you choose, unrelated to the model itself. |
| **LLM (Large Language Model)** | The model that reads retrieved text and generates a natural-language answer. Here: `llama3.2`, run locally via Ollama. |
| **Retrieval** | The search step (embed question → FAISS search) — finds *candidate* relevant text. Does not itself answer the question. |
| **Generation** | The LLM step — reads the retrieved context and actually produces the answer, including deciding what's relevant, ignoring noise, and phrasing a direct response. |
| **Context window / prompt context** | The block of retrieved chunk text + conversation history that gets sent to the LLM alongside the question. |
| **Conversational memory** | The last `MEMORY_TURNS=4` (question, answer) pairs, held in a `collections.deque(maxlen=4)` in `ChatSession`, replayed as prior turns in each new prompt. |
| **Hallucination** | When an LLM states something not actually supported by the given context — either invented, or recalled from its own pretraining instead of the retrieved text. A real risk this pipeline's system prompt tries to guard against. |
| **System prompt** | The fixed instruction given to the LLM on every call, telling it how to behave (answer only from context, cite sources, admit when it doesn't know). See `bot.py: SYSTEM_PROMPT`. |
| **Positional join** | This project's specific design detail: FAISS row `i` always corresponds to `chunks[i]` in `chunks.json` — there's no separate ID-based metadata lookup, just matching array position. |

---

## Project structure

```
rag-assignment/
├── README.md                    # this file
├── TECHNICAL_DETAILS.md         # full function-by-function deep dive + design reasoning
├── PYTHON_KNOWLEDGEBASE.md      # Python syntax reference (for readers coming from C#/.NET)
├── data/
│   ├── pdfs/                    # the 5 source papers (Transformer, BERT, RoBERTa, T5, GPT-3)
│   ├── processed/                # generated at runtime — chunks.json, embeddings.npy, faiss.index
│   └── eval/
│       ├── questions.json        # the 10 test questions
│       └── results.json           # generated by evaluate.py — answers + LLM-as-judge scores
└── src/
    ├── config.py                 # shared paths + Ollama model names, used by every stage
    ├── ingestion.py               # PDF → raw text → cleaned text (strip headers/footers/page numbers)
    ├── chunking.py                # cleaned text → chunks.json (paragraph/sentence-aware, with overlap)
    ├── vectorstore.py             # chunks.json → embeddings → faiss.index
    ├── bot.py                     # the conversational bot: retrieval + prompt building + LLM call + memory
    └── evaluate.py                # runs the 10 questions + LLM-as-judge scoring → results.json
```

`data/processed/` is regenerated by running the pipeline — it's not committed to git.

---

## Setup

**1. Install and start Ollama**, then pull the two models this project uses:
```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```
Ollama must be running and reachable at `http://localhost:11434` (the default) — see
`src/config.py: OLLAMA_BASE_URL` if yours differs.

**2. Create a virtual environment and install Python dependencies:**
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install pypdf faiss-cpu numpy requests
```

**3. Confirm the 5 source PDFs are in `data/pdfs/`** (already included in this repo).

---

## Running the pipeline

Run every command from the **project root** (not from inside `src/`) — the paths in
`config.py` (`data/pdfs`, `data/processed`, ...) are relative to the root, and running a
script as `python src/whatever.py` (rather than `cd`-ing into `src/` first) is what lets
Python resolve both the sibling imports (e.g. `bot.py` importing `config`) and those data
paths correctly at the same time. Each script also has useful debug output when run directly.

```bash
# 1. Build the chunk file (also runs ingestion internally)
python src/chunking.py
#    → writes data/processed/chunks.json

# 2. Build embeddings + FAISS index
python src/vectorstore.py
#    → writes data/processed/embeddings.npy and data/processed/faiss.index
#    → also runs a sample query at the end as a sanity check

# 3. Chat with the bot
python src/bot.py
#    → interactive REPL: type a question, type "quit" to exit

# 4. Run the 10-question evaluation (see Stage 5 below)
python src/evaluate.py
#    → writes data/eval/results.json and prints a scoring summary
```

---

## Stage-by-stage details

### 1. PDF ingestion & cleaning — `src/ingestion.py`
- `extract_pages()`: uses `pypdf.PdfReader` to pull raw text out of each page.
- `find_repeated_lines()` / `clean_pages()`: PDFs extract with recurring noise — running headers,
  footers, standalone page numbers. This is detected **structurally**, not by keyword: any line
  that repeats on ≥60% of a document's pages is treated as noise and stripped, regardless of what
  it actually says. Standalone page numbers are caught with a regex. Runs of 3+ blank lines are
  collapsed.

### 2. Chunking — `src/chunking.py`
- `split_into_segments()`: splits each page on blank lines (paragraph breaks). Since not every
  PDF preserves those reliably, any block still over `MAX_WORDS=400` gets broken down further by
  sentence (regex-based sentence splitting), and as a last resort by a hard word-count slice —
  so nothing oversized ever reaches the next step, regardless of source formatting.
- `build_chunks()`: greedily accumulates segments up to `TARGET_WORDS=250` per chunk. When a
  chunk closes, its last `OVERLAP_WORDS=45` words are carried into the *start* of the next chunk
  — this overlap means a sentence that would otherwise get cut in half across a chunk boundary
  still has full context in at least one chunk.
- `detect_section()`: best-effort tagging of which paper section (Introduction, Method, etc.) a
  chunk falls in, used for citation/debugging, not for retrieval logic.
- Output: `chunks.json` — the chunk order here is load-bearing (see "positional join" below).

### 3. Embeddings + vector store — `src/vectorstore.py`
- `embed_texts()`: batches text (32 at a time) into calls to Ollama's `/api/embed` with
  `nomic-embed-text`, retrying failed batches with exponential backoff. This exact function is
  reused unchanged by `bot.py` to embed the user's live question — same model, same code path,
  so the query vector lands in the same vector space as the indexed chunks.
- `normalize()`: L2-normalizes every vector so that FAISS's inner-product search behaves like
  cosine similarity.
- `faiss.IndexFlatIP`: builds an exact (brute-force) similarity index — chosen deliberately over
  approximate indexes (IVF, HNSW, PQ) since ~400 vectors is small enough that exact search is
  fast and there's no need for the extra complexity/accuracy trade-off those bring.
- **No separate metadata database.** `chunks.json[row]` *is* the metadata for FAISS row `row` —
  a hand-maintained positional link rather than an ID-based join.

### 4. Conversational bot with memory — `src/bot.py`
- `retrieve()`: embeds the incoming question, searches FAISS for the top `TOP_K=5` chunks, and
  maps each result row back to its full chunk dict + similarity score. **All 5 results are
  returned unconditionally — there is no relevance-score filtering.**
- `format_context()`: concatenates the 5 chunks into one labeled text block
  (`[Source: doc_id, page X]` + text), which becomes part of a single prompt.
- `ChatSession`: owns a `deque(maxlen=4)` of past `(question, answer)` pairs. Each `ask()` call
  retrieves fresh context for the *current* question, replays the last 4 turns as prior
  user/assistant messages, appends the new context + question, sends **one** combined prompt to
  `llama3.2` via `call_llama()`, and stores the new turn — automatically evicting the oldest turn
  once more than 4 have accumulated.
- The LLM only ever makes **one call per question**. It receives the 5 chunks already merged into
  a single block of text — it has no awareness that "5 separate documents were retrieved"; it
  just reads one prompt and reasons over everything in it, the same way a person would read a
  page with several pasted excerpts before answering a question about them.

### 5. Evaluation — `src/evaluate.py`
- `run_interactions()`: asks all 10 questions from `data/eval/questions.json` through **one**
  continuous `ChatSession`, in order — so the 4-turn memory window fills and evicts exactly as it
  would in real use. It snapshots the memory available *before* each question is asked (needed by
  the judge afterward) and records the retrieved chunks alongside the answer.
- **LLM-as-judge scoring**: `judge_answer()` sends each `(question, retrieved context, memory
  snapshot, answer)` to `llama3.2` a second time, with a rubric (`JUDGE_SYSTEM_PROMPT`) asking it
  to score 1–5 on the assignment's four named criteria — Relevance, Accuracy, Contextual
  Awareness, Response Quality — each with a one-sentence justification. This is the project's
  self-devised metric: the rubric is fully defined in `JUDGE_SYSTEM_PROMPT`, not delegated to
  RAGAS/Trulens. Parsing includes a fallback that repairs a truncated JSON response (a real
  failure mode observed with this local model) before giving up on that one record.
- The 10 questions (`data/eval/questions.json`) cover all 5 papers, and several are deliberately
  written as pronoun-based follow-ups ("it"/"that" referring to a prior question's topic) so
  answering them correctly requires the bot to actually resolve the reference from
  `ChatSession.history` — not just from the retrieved text.
- Output: `data/eval/results.json` (every question, answer, retrieved chunk, and judge score) plus
  a printed summary of average scores per criterion.
- **A real RAGAS-based evaluation was also attempted**, in response to a specific instructor
  requirement naming exact RAGAS metrics (Faithfulness, Answer Correctness, Context Recall,
  Context Precision) with minimum thresholds. It was ultimately removed from the codebase after
  confirming it wasn't practical on this local CPU + `llama3.2` setup — see
  `TECHNICAL_DETAILS.md`'s "RAGAS investigation" section for the full account (version conflicts,
  concurrency overload, per-call latency, and unreliable structured output from a small local
  model). The self-devised metric above is this project's actual, submitted evaluation
  methodology.

---

## Design decisions

- **No LangChain / LlamaIndex anywhere in the project.** Every RAG primitive (PDF loader, text
  splitter, embeddings wrapper, vector store wrapper, prompt template, memory) is hand-written.
  This is deliberate: the assignment doesn't require a framework, and writing each piece by hand
  is what makes the mechanics of RAG (chunk boundaries, the positional metadata link, prompt
  construction, memory eviction) visible instead of hidden behind abstractions. A real LangChain +
  `ragas`-based evaluation attempt was made at one point (in response to an instructor requirement
  naming exact RAGAS metrics), but it was removed after confirming it wasn't practical on this
  hardware — see `TECHNICAL_DETAILS.md`'s "RAGAS investigation" section for the full account.
- **Self-devised evaluation metric, not RAGAS/Trulens.** Both are valid per the assignment, but
  RAGAS in particular pulls in a heavy dependency chain (including LangChain) and its default
  metrics assume an OpenAI model unless manually rewired — friction this project avoids by
  documenting its own evaluation methodology instead (see `TECHNICAL_DETAILS.md`).
- **Exact FAISS search (`IndexFlatIP`) over approximate indexes** — at ~400 vectors, brute-force
  search is already fast; approximate methods would trade accuracy for speed the project doesn't
  need.
- **Retrieval always returns exactly `k` chunks, with no score threshold.** This is a known,
  intentional simplification — see [Known limitations](#known-limitations).
- **LLM-as-judge for evaluation, not manual scoring.** `evaluate.py` reuses `llama3.2` (the same
  model that generates answers) as an automated judge against a documented rubric — the same
  underlying technique RAGAS uses internally, but implemented directly rather than pulled in as a
  dependency. See [Known limitations](#known-limitations) for the self-bias caveat this implies.

---

## Current status

| Stage | Status |
|---|---|
| PDF extraction + noise cleaning | ✅ Done |
| Chunking | ✅ Done (398 chunks) |
| Embeddings | ✅ Done |
| FAISS vector store | ✅ Done |
| Retrieval + LLM integration | ✅ Done |
| Conversational memory (4 turns) | ✅ Done |
| Test harness (10 questions) | ✅ Done (`data/eval/questions.json`) |
| Evaluation scoring (self-devised) | ✅ Done — LLM-as-judge, avg scores: Relevance 4.89, Accuracy 5.00, Contextual Awareness 5.00, Response Quality 4.33 (out of 5 → 97.8% / 100% / 100% / 86.6%; n=9/10 scored — see note below) |
| Final report | ⬜ Not started |
| Bonus: feedback into memory | ⬜ Not started |

See `TECHNICAL_DETAILS.md` for the detailed reasoning behind each completed stage, including the
RAGAS investigation.

---

## Known limitations

- **No follow-up query rewriting.** `retrieve()` embeds only the literal current question. A
  follow-up like "what about its encoder?" won't be resolved against prior turns before
  retrieval, so the vector search may miss what the user actually means — even though the *LLM*
  still sees the conversation history when generating the answer.
- **No relevance-score filtering on retrieval.** All top-`k` results are sent to the LLM
  regardless of how low their similarity score is; there's no threshold (e.g. "drop anything
  below 0.5") to exclude genuinely unrelated chunks from the prompt.
- **Requires Ollama running locally** with both models pulled — the bot will fail its HTTP calls
  otherwise (see `src/vectorstore.py: embed_texts()` and `src/bot.py: call_llama()` for the retry/
  error behavior).
- **LLM-as-judge self-bias.** `evaluate.py` uses `llama3.2` to judge answers generated by the same
  `llama3.2`, which can be more lenient toward its own phrasing/reasoning style than an
  independent judge (e.g. a larger or different model) would be. The near-ceiling Accuracy/
  Contextual Awareness scores (5.00/100% avg) should be read with that in mind — worth naming
  explicitly as a methodology caveat in the report, not treated as proof the bot is flawless.
- **One judge response (question 10) didn't parse even after the brace-repair fallback** — stored
  in `results.json` as `{"parse_error": true, "raw_response": ...}` rather than a score, so
  `n=9/10` for the printed averages. This is a known, occasional failure mode of a small local
  model not always following the "respond with ONLY JSON" instruction strictly.
- **RAGAS, on this local CPU + `llama3.2` setup, was not practical to run at full scale.** A real
  attempt confirmed individual chained LLM calls taking multiple minutes each, and even after
  fixing timeout/concurrency configuration, some metrics returned entirely unscored (`nan`)
  results — so the attempt was removed from the codebase. Full details and the reasoning trail
  are in `TECHNICAL_DETAILS.md`'s "RAGAS investigation" section — this is documented as a genuine
  challenge encountered, not a gap that was ignored.
