# RAG Assignment — Implementation Plan

This document tracks the design and reasoning behind the RAG pipeline being built for the
Nagarro "Generative AI Fundamentals" assignment. It's written stage-by-stage — each stage
explains **why** the approach was chosen (not just what to type), so it can also feed
directly into the Final Report's "Technical Implementation" section later.

## Assignment recap

Build a RAG application over 5 arXiv papers (Transformer, BERT, GPT-3, RoBERTa, T5), using
FAISS for vector storage and Ollama (`llama3.2` for generation, `nomic-embed-text` for
embeddings) running locally in Docker. The bot must remember the last 4 conversation turns,
answer 10 predefined test questions, be evaluated against a documented metric, and ship with
a final PDF report.

## Current status

| Stage | Status |
|---|---|
| PDF extraction + noise cleaning | Done (`src/ingestion.py`) |
| Chunking | Not started |
| Embeddings | Not started |
| FAISS vector store | Not started |
| Retrieval + LLM integration | Not started |
| Conversational memory | Not started |
| Test harness (10 questions) | Not started |
| Evaluation scoring | Not started |
| Final report | Not started |
| Bonus: feedback into memory | Not started |

## Two key decisions already made

**Evaluation approach: self-devised metric, not RAGAS/Trulens**

*Why:* the assignment explicitly allows "RAGAS, Trulens, or a self-devised metric." RAGAS and
Trulens are real frameworks, but they pull in a heavy dependency chain (LangChain, `datasets`,
pinned `pydantic` versions, etc.) and their default metrics generally assume an OpenAI LLM/
embedding model unless manually rewired — which means either a paid API key or extra glue
code to point them at local Ollama models. Given the whole project is intentionally
local-only and zero-cost (Docker + Ollama, no cloud API), a self-devised metric avoids that
entire dependency/cost surface while still meeting the requirement, **as long as the
methodology is clearly documented** — which this plan does, and the final report will repeat.

*How:* two complementary techniques, both using only what's already installed:
1. **Embedding-similarity heuristics** — cosine similarity between an answer and its
   retrieved context (a proxy for "is the answer grounded in the source material," often
   called *faithfulness*), and between an answer and the original question (a proxy for
   *relevance*). Fast, deterministic, no extra LLM calls.
2. **LLM-as-judge using `llama3.2` itself** — for each question, ask the model to score the
   answer 1-5 on the four dimensions the assignment names (relevance, accuracy, contextual
   awareness, response quality), with a one-line justification, run at `temperature=0` for
   repeatability.

**Execution/testing approach: you run scripts, not automated invocation**

*Why:* while building the cleaning stage, running `python src/ingestion.py` through automated
WSL invocation hung for several minutes with no clear cause — versus running things directly
in your own WSL/VS Code terminal, which has worked reliably throughout this project. Rather
than lose time fighting flaky automation, each stage below is written/explained here, then
you run it yourself and share the output for review before moving to the next stage.

---

## Stage 0 — Prep

**Why:** two small structural issues need fixing before other files can safely import from
`ingestion.py`, and before constants get duplicated across files.

1. `ingestion.py`'s driver loop (the `for pdf_file in PDF_FOLDER.glob(...)` block at the
   bottom) currently runs the moment the file is *imported*, not just when it's run directly.
   Once `chunking.py` does `from ingestion import extract_pages, clean_pages`, that loop would
   fire as an unwanted side effect. Fix: wrap it in `if __name__ == "__main__":`, the standard
   Python idiom for "only run this when the file is executed directly" — closest C# analogy
   is guarding code so it only runs from `Main()`, not when the assembly is merely referenced.
2. Add `src/config.py` holding shared constants (`PDF_FOLDER`, output file paths, the Ollama
   base URL, model names). Once 4-5 files need the same values, keeping one source of truth
   avoids the classic bug of updating a path/URL in one file and forgetting another.

## Stage 1 — Chunking (`src/chunking.py`)

**Why paragraph-aware, not fixed-character chunks:** discussed earlier in the project —
blind fixed-size chunking can slice a sentence or idea in half, which hurts both embedding
quality (a chunk should represent one coherent idea) and the final answer's accuracy (a
half-sentence retrieved as "context" is often useless or misleading). Paragraphs are the
natural semantic unit in a research paper.

**Why combine paragraphs up to a target size instead of one paragraph = one chunk:** research
paper paragraphs vary wildly in length — some are one sentence, some are a dense derivation
half a page long. Chunks that are too small lack enough context to be useful; chunks that are
too large dilute the embedding (a very long chunk's vector represents an "average" of many
ideas, making it less precise for retrieval). Combining paragraphs up to ~250 words (roughly
~325 tokens) balances the two.

**Why page-boundary-tolerant:** PDF text extraction cuts text at physical page breaks, which
frequently falls mid-paragraph. If chunking were done independently per page, a paragraph
split across pages 3 and 4 would get truncated. Instead, paragraphs are gathered into one
ordered list across the *whole document* first (each still tagged with its page number), so a
paragraph that continues across a page break stays intact.

**Why ~40-50 word overlap between chunks:** without overlap, a sentence that happens to fall
right at a chunk boundary loses surrounding context on retrieval. A small tail-overlap (the
end of one chunk repeated at the start of the next) means a retrieved chunk is less likely to
start or end awkwardly mid-thought.

**How:**
- Iterate the already-cleaned pages from `ingestion.py`, split each page's text into blocks on
  blank lines (`"\n\n"`, real paragraph breaks when present), and build one ordered
  `(page_number, segment_text)` list per document.
- **Fixed during implementation:** `pypdf` doesn't reliably preserve blank lines between
  paragraphs for every PDF, so a naive blank-line split alone sometimes produced huge merged
  blocks (one file's max chunk hit 4926 words before this fix). `split_into_segments()` now
  self-corrects: any block still over `MAX_WORDS` (400) gets broken down into sentences, and
  even a single oversized sentence (rare) gets hard-sliced by word count as a last resort —
  guaranteeing every segment handed to `build_chunks()` is ≤ 400 words, regardless of how a
  given PDF's text extraction turned out.
- Greedily accumulate segments into a chunk until ~250 words is reached; a segment already
  close to 400 words on its own becomes its own chunk rather than being merged.
- Carry the last ~40-50 words of a closed chunk forward as the start of the next (so a chunk
  can land slightly over 400 words when overlap + a near-max segment combine — expected).
- Best-effort section detection (matching numbered headings / common section names like
  "Introduction", "Method", "Results") — tags each chunk with a `section`, useful metadata
  for citations later, but not something to over-engineer.
- Each chunk records: `chunk_id`, `doc_id`, `source_file`, `page_start`, `page_end`,
  `section`, `text`, `word_count`.
- **Why persist to one combined `data/processed/chunks.json`:** every downstream stage
  (embeddings, FAISS) needs to process the same chunks repeatedly without re-extracting and
  re-cleaning PDFs each time — that would be slow and wasteful. This file's row order becomes
  the fixed reference order that embeddings and the FAISS index line up against.

**Quick-reference checklist (chunking logic):**

| Constant | Value | Controls |
|---|---|---|
| `TARGET_WORDS` | 250 | Soft goal — stop combining once you'd cross this |
| `MAX_WORDS` | 400 | Hard cap — no single segment may exceed this |
| `OVERLAP_WORDS` | 45 | Tail of a closed chunk repeated at the start of the next |

Processing order: extract → clean → **segment** (blank-line split, sentence fallback,
hard-slice last resort — guarantees every segment ≤ 400 words) → **combine** greedily toward
250 words → **carry overlap** → **tag metadata** → **save** to `chunks.json`.

Known open item (not yet fixed): the final leftover chunk in a document has no enforced
minimum size — one run produced a 60-word trailing chunk. Low priority (affects at most one
chunk per document, 5 total across the corpus) but worth revisiting if retrieval quality on
short answers looks weak later.

## Stage 2 — Embeddings (`src/embeddings.py`)

**Why batch the calls to Ollama:** `/api/embed` accepts a list of strings and returns a list
of vectors in one HTTP round trip. Calling it once per chunk (a few hundred separate HTTP
requests) would be far slower than batching ~32 chunks per call.

**Why normalize every embedding to unit length:** this is what makes "cosine similarity" —
the standard way to measure how semantically similar two pieces of text are — equivalent to a
plain dot product. Normalizing once at storage time (and identically at query time) lets the
FAISS index use fast, simple inner-product math instead of a more expensive cosine
computation on every search.

**How:** `embed_texts(texts, batch_size=32)` → `POST /api/embed` per batch → normalize each
returned vector → stack into one NumPy array of shape `(num_chunks, 768)` (768 = this model's
embedding dimension) → save as `data/processed/embeddings.npy`. Row `i` must correspond to
`chunks.json[i]` — this positional link is the single most important invariant in the whole
pipeline.

## Stage 3 — FAISS vector store (`src/vectorstore.py`)

**Why `IndexFlatIP` and not a fancier index type:** FAISS offers approximate/compressed index
types (IVF, PQ) designed for millions of vectors, but they need a training step and introduce
approximation error. This corpus is only ~5 papers, likely a few hundred chunks — an exact
brute-force search (`IndexFlatIP`, "IP" = inner product) is effectively instant at that scale
and gives exact results, so there's no reason to trade accuracy for a speed benefit that isn't
needed yet.

**Why no separate metadata file:** FAISS only stores vectors and returns row numbers on
search — it has no concept of "this vector belongs to this chunk." Since `chunks.json` is
already in the same fixed order the index was built from, it doubles as the metadata lookup:
`chunks[row_index]` gives you the full chunk (text, page, section) for a search hit.

**How:** build the index from `embeddings.npy`, persist with `faiss.write_index`, reload with
`faiss.read_index` (no recomputation needed on reload — the whole point of persisting it).

## Stage 4 — Retrieval + LLM integration (`src/bot.py`)

**Why this prompt structure:** the system message (fixed instructions: answer only from the
given context, admit when context is insufficient, cite source/page) stays constant across
turns, while the context block (which chunks were retrieved) changes every question — keeping
retrieved context in the *user* message rather than mutating the system prompt each time is a
cleaner mental model and avoids the system prompt growing unbounded.

**How:** embed the user's question the same way chunks were embedded (same normalization) →
FAISS top-k search → format retrieved chunks as labeled context blocks → send to `llama3.2`
via `/api/chat`.

## Stage 5 — Conversational memory

**Why `deque(maxlen=4)`:** the assignment requires remembering exactly the last 4 turns. A
bounded double-ended queue is a standard-library structure that automatically drops the
oldest entry once a 5th is added — no manual "if list length > 4, remove first item" logic to
get wrong.

**How:** a small `ChatSession` class holds the deque of `{user, assistant}` turn pairs; each
call to `ask()` expands stored turns into the `/api/chat` messages list ahead of the current
question, then appends the new turn (auto-evicting the oldest if now at 5).

## Stage 6 — Test harness (`src/evaluate.py`, part 1)

**Why a fresh session per question by default, with optional grouping:** most of the 10
required questions are independent factual checks and should be tested cleanly, without
memory from an unrelated prior question leaking in. But 2-3 questions can be deliberately
grouped as a follow-up conversation (sharing one `ChatSession`) specifically to exercise the
"contextual awareness" dimension the rubric names — e.g. "What loss does BERT use?" then "How
does that compare to GPT-3's?", where "that" only resolves correctly if memory is working.

**How:** `data/eval/questions.json` holds the 10 questions (with an optional `group` field);
the harness runs each, logs the retrieved chunks, the answer, and latency to
`data/eval/results.json`.

## Stage 7 — Evaluation scoring (`src/evaluate.py`, part 2)

Covered above under "Two key decisions." Output: a scored version of `results.json` with the
embedding-similarity numbers and the LLM-judge's 1-5 ratings + justifications per question.

## Stage 8 — Final report

**Why written last (mostly):** the "Results & Discussion" section needs real evaluation
numbers to discuss — writing it before Stage 7 exists would mean guessing. The
architecture/technical-implementation sections, however, can be drafted earlier since the
design decisions (this document) don't change once a stage is validated.

**How:** Overview & architecture → Technical implementation (per stage, referencing this
document's "why") → Evaluation methodology (the full self-devised rubric/prompts) → Results &
discussion (the 10-question table, a couple of concrete examples, honest limitations) →
Conclusion. Exported to PDF from Markdown/Word — no new tooling needed.

## Stage 9 — Bonus: real-time feedback into memory (optional, last)

**Why:** the assignment's bonus asks for user feedback to dynamically influence future
context. **How:** after each answer, prompt for quick feedback (`input()`), store it alongside
that turn in the same deque; if a turn in the current memory window carries negative feedback,
prepend a short steering instruction to the next prompt (e.g. "prioritize precision and cite
sources").

---

## File layout

```
src/
  config.py       # shared constants (new)
  ingestion.py     # PDF extraction + noise cleaning (done, needs __main__ guard)
  chunking.py      # Stage 1
  embeddings.py    # Stage 2
  vectorstore.py   # Stage 3
  bot.py           # Stages 4, 5, 9 (retrieval, chat, memory, bonus feedback)
  evaluate.py       # Stages 6, 7 (test harness + scoring)
data/
  pdfs/            # the 5 source PDFs (already present)
  processed/       # chunks.json, embeddings.npy, faiss.index
  eval/            # questions.json, results.json
```

**Why one file per stage:** mirrors the pattern `ingestion.py` already set, keeps each file
small and independently testable (matching the stage-by-stage working style used throughout
this project), and maps cleanly onto the assignment's own rubric sections when writing the
final report or when a grader reads the code.

## New dependencies

None of the stages above need anything beyond what's already installed — `pypdf`, `requests`,
`faiss-cpu`, `numpy`, plus Python's standard library (`re`, `json`, `collections`). A
`requirements.txt` will be generated once the core pipeline (Stages 1-6) is working.
