# RAG Assignment — Conversational PDF Q&A Bot (LangChain branch)

A Retrieval-Augmented Generation (RAG) application that answers questions about 5 NLP research
papers (Transformer, BERT, RoBERTa, T5, GPT-3), remembers the last 4 turns of conversation, and
runs entirely locally using [Ollama](https://ollama.ai) + [FAISS](https://github.com/facebookresearch/faiss) — no cloud API, no API key.

> **This is the `UseLangchain` branch.** It is a rebuild of the same pipeline on top of
> [LangChain](https://python.langchain.com), using the dedicated **`langchain-ollama`** package to
> talk to the local Ollama models. The `main` branch holds the original, framework-free
> implementation where every RAG primitive is hand-written. Both produce comparable results; the
> difference is how much of the plumbing is written here versus configured. See
> [What LangChain changed](#what-langchain-changed) for the file-by-file comparison.

All LangChain usage is funnelled through a single module, [`src/langchain_service.py`](src/langchain_service.py),
so every other file (`ingestion`, `chunking`, `vectorstore`, `bot`, `evaluate`) imports from that
one place rather than importing LangChain directly.

---

## Table of contents

- [How it works, in one picture](#how-it-works-in-one-picture)
- [What LangChain changed](#what-langchain-changed)
- [The service layer](#the-service-layer)
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
    │  ingestion.py: PyPDFLoader → Documents; custom cleaning strips
    │                headers/footers/page numbers
    ▼
  cleaned page Documents
    │  chunking.py: RecursiveCharacterTextSplitter (250-word chunks,
    │               45-word overlap) + custom section tagging
    ▼
  527 chunk Documents  (metadata: chunk_id, doc_id, page, section, word_count)
    │  vectorstore.py: FAISS.from_documents() — OllamaEmbeddings
    │                  (nomic-embed-text) embeds them in one batched call
    ▼
  data/processed/faiss_store/   (index.faiss + index.pkl — vectors AND
                                 documents stored together)


QUERY (every time the user asks a question — src/bot.py)
────────────────────────────────────────────────────────────────────────
  "How does BERT's masked language modeling work?"
    │  store.similarity_search_with_score(question, k=5)
    │    → embeds the question with the same model, searches, and returns
    │      Documents that already carry their own text + metadata
    ▼
  top-5 (Document, score) pairs
    │  format_context() — stitches the 5 chunks into one labeled text block
    ▼
  ChatPromptTemplate:  system prompt
                     + MessagesPlaceholder("history")  ← last 4 turns
                     + "Context: … Question: …"
    │  LCEL chain:  prompt | ChatOllama(llama3.2) | StrOutputParser()
    ▼
  final answer (single string) — also appended to memory for the next turn
```

The key thing this diagram is meant to make obvious: **the embedding model and the LLM are two
different models used for two different jobs**, and they never talk to each other directly.
The embedding model turns text into vectors so similar text can be *found*. The LLM reads
whatever text was found and *reasons* over it to produce an answer. FAISS itself runs no model at
all — it's pure numeric similarity search.

---

## What LangChain changed

Same pipeline, same models, same behaviour — the difference is how much is written by hand versus
configured. Compared to the `main` branch:

| Stage | `main` (hand-written) | This branch (LangChain) |
|---|---|---|
| PDF loading | `pypdf.PdfReader` loop over pages | `PyPDFLoader` → `Document` per page |
| Noise cleaning | `find_repeated_lines()` / `clean_pages()` | **unchanged — LangChain has no equivalent** |
| Chunking | ~80 lines: paragraph accumulation, sentence fallback, manual overlap-tail carrying | one configured `RecursiveCharacterTextSplitter` |
| Section tagging | `detect_section()` | **unchanged — LangChain has no equivalent** |
| Embeddings | hand-written batching (32 at a time) + retry/backoff loop | `OllamaEmbeddings`, still batched 32 at a time — see note below |
| Vector store | `faiss.IndexFlatIP` + `normalize()`, **plus** a `chunks.json` sidecar kept in sync by row position | `FAISS.from_documents(...)` — owns vectors *and* documents together |
| Retrieval | embed → `index.search()` → map row number back into `chunks.json` | `store.similarity_search_with_score()` returns Documents directly |
| Prompt building | manual `messages` list assembly | `ChatPromptTemplate` + `MessagesPlaceholder` |
| LLM call | `requests.post()` + manual timeout/retry | `ChatOllama` from `langchain-ollama` |
| Composition | call `retrieve()` → `format_context()` → `call_llama()` in the right order, every time | `prompt \| llm \| parser` composed once (LCEL) |
| Memory | `collections.deque(maxlen=4)` | **unchanged — a bounded deque already says exactly what the assignment asks for** |

**The single most meaningful structural win** is the vector store. On `main`, `faiss.index` and
`chunks.json` are two separate files linked *only* by matching row position — an invariant the code
had to protect by hand, and one that would silently corrupt every citation if chunks were ever
reordered. LangChain's FAISS object holds the index and the document store together, so a search
returns the text and metadata directly. There is no positional link left to break.

**What LangChain did not replace** is just as informative: the header/footer detection and the
section tagging are bespoke logic with no framework equivalent, so they stayed exactly as they
were. A framework removes boilerplate, not the parts specific to your problem.

**And one place the framework needed help rather than providing it:** `OllamaEmbeddings` sends
every text it's handed in a *single* HTTP request, so the obvious one-liner
(`FAISS.from_documents(all_527_chunks, ...)`) died with a `ReadTimeout`. Batching had to be
reintroduced by hand — the same 32-at-a-time loop `main` already had. Convenience abstractions
still leak the constraints of what they're calling.

---

## The service layer

Every LangChain import in this project lives in one file: [`src/langchain_service.py`](src/langchain_service.py).
Other modules call functions from it and never import LangChain themselves — so if a class is
renamed or deprecated (which happened repeatedly while building this), exactly one file changes.

It is responsible for three things:

**1. Cached model access.** `get_llm()` and `get_embeddings()` build their clients once and reuse
them, keyed by settings — a run making dozens of calls doesn't reconstruct a client each time.

**2. Efficient local-Ollama settings**, using the dedicated `langchain-ollama` package rather than
the generic community wrappers, because it exposes Ollama's own options as first-class parameters:

| Setting | Why it matters here |
|---|---|
| `keep_alive="30m"` | Holds `llama3.2` in memory between calls. A full evaluation makes ~20 calls; without this, Ollama can unload and reload the model repeatedly. |
| `num_ctx=8192` | Ollama's default context window is **2048 tokens**. This project's prompts are estimated at **~2,100 tokens** once memory is full (see below) — i.e. right at the limit. Leaving the default silently truncates the prompt, with no error. |
| `format="json"` (judge only) | Ollama's native JSON mode constrains decoding so the judge's reply is valid JSON *by construction* — this eliminates the parse failure that left one question unscored on `main`. |
| `temperature=0`, `seed=0` | Repeatable scoring across runs. |
| `client_kwargs={"timeout": 300}` | Local CPU inference on long prompts is measured in minutes, not seconds. |

**3. The prompts and chains** — the RAG answer prompt and the judge rubric, as
`ChatPromptTemplate`s composed with LCEL.

> **A finding worth flagging, about `num_ctx`.** Measuring the actual corpus: chunks average 207
> words, so 5 retrieved chunks ≈ 1,035 words ≈ **1,400 tokens**. Add the system prompt (~60), the
> question (~25), and 4 turns of conversation memory (~600) and a full prompt lands around
> **2,100 tokens** — just over Ollama's 2048 default. That means the `main` branch, which never
> set `num_ctx`, was very likely having its prompts silently truncated once the memory window
> filled: the oldest content (the earliest memory turns) would simply fall out of the model's view
> with no error raised anywhere.
>
> Setting `num_ctx=8192` here fixes that, and it is the honest explanation for why this branch's
> answers take longer to generate — the model is now processing the *whole* prompt rather than a
> truncated one. Slower, but correct. (The token figures above are estimates from word counts at
> ~1.35 tokens/word for technical English, not exact tokenizer output.)

> **Deliberate design choice:** the evaluation judge stays a **single LLM call scoring all four
> criteria at once**, exactly as on `main`. Frameworks like RAGAS decompose each metric into
> chained sub-calls — cheap against a cloud API, impractically slow against local CPU inference
> (an earlier attempt took ~55 minutes for 2 questions and still returned unscored metrics). Using
> LangChain for *composition* while keeping the one-call design is what makes this branch produce
> like-for-like results at like-for-like speed.

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
| **System prompt** | The fixed instruction given to the LLM on every call, telling it how to behave (answer only from context, cite sources, admit when it doesn't know). See `langchain_service.py: RAG_SYSTEM_PROMPT`. |
| **Positional join** | The design the `main` branch used: FAISS row `i` always corresponds to `chunks[i]` in `chunks.json`, with no ID-based lookup — just matching array position. **Eliminated on this branch**, since LangChain's FAISS store keeps documents and vectors together. |
| **`Document`** | LangChain's unit of text: a `page_content` string plus a `metadata` dict. Everything flows through the pipeline as Documents here — a page is a Document, a chunk is a Document, a search result is a Document. |
| **LCEL** | LangChain Expression Language — composing steps with the `|` operator (`prompt \| llm \| parser`) into one runnable you call with `.invoke()`. Replaces calling three functions in the right order manually. |
| **`MessagesPlaceholder`** | A slot in a `ChatPromptTemplate` where a list of prior conversation messages gets injected at call time — how the 4-turn memory reaches the prompt. |
| **`keep_alive`** | An Ollama setting for how long a model stays resident in memory after a call. Set to 30m here so a long run doesn't repeatedly reload `llama3.2` from disk. |
| **`num_ctx`** | Ollama's context-window size in tokens. Its default (2048) is smaller than this project's prompts, so it's raised to 8192 to avoid silent truncation. |
| **JSON mode** | `format="json"` — Ollama constrains decoding so output is always valid JSON. Used for the evaluation judge, where a malformed reply would otherwise cost a score. |

---

## Project structure

```
rag-assignment/
├── README.md                    # this file
├── TECHNICAL_DETAILS.md         # full function-by-function deep dive + design reasoning
├── PYTHON_KNOWLEDGEBASE.md      # Python syntax reference (for readers coming from C#/.NET)
├── data/
│   ├── pdfs/                    # the 5 source papers (Transformer, BERT, RoBERTa, T5, GPT-3)
│   ├── processed/
│   │   ├── chunks.json           # human-readable chunk dump (for inspection/reporting only)
│   │   └── faiss_store/           # the LangChain FAISS store — index.faiss + index.pkl
│   └── eval/
│       ├── questions.json        # the 10 test questions
│       └── results.json           # generated by evaluate.py — LLM-as-judge scores
└── src/
    ├── config.py                 # shared paths, model names, and tuning knobs
    ├── langchain_service.py       # ★ ALL LangChain usage lives here — models, loaders,
    │                              #   splitter, vector store, prompts, chains
    ├── ingestion.py               # PyPDFLoader → Documents + custom header/footer cleaning
    ├── chunking.py                # RecursiveCharacterTextSplitter + custom section tagging
    ├── vectorstore.py             # builds and persists the FAISS store
    ├── bot.py                     # the conversational bot: retrieval + LCEL chain + 4-turn memory
    └── evaluate.py                # self-devised LLM-as-judge, 4 criteria → results.json
```

Note that `chunks.json` is no longer load-bearing on this branch — the FAISS store carries its own
copy of the text and metadata, so nothing downstream depends on that file's row order. It is
written purely for inspection and for the report.

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
source .venv/bin/activate       # macOS/Linux/WSL
# .venv\Scripts\activate        # Windows

pip install "langchain<0.3" "langchain-community<0.3" "langchain-ollama<0.2" faiss-cpu pypdf
```

> **On the version pins:** LangChain's 1.x line restructured several integration packages, and
> `langchain-ollama` ≥1.0 requires `langchain-core` ≥1.0, which the 0.2.x `langchain-community`
> used here does not accept. Pinning all three to the same generation keeps the import paths in
> `langchain_service.py` valid. This ecosystem moves fast — if you install unpinned and get an
> `ImportError` from inside a LangChain module, a version mismatch is the first thing to check.

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

# 2. Build embeddings + FAISS store (re-chunks internally, so it's self-contained)
python src/vectorstore.py
#    → writes data/processed/faiss_store/ (index.faiss + index.pkl)
#    → also runs a reload check and a sample query as sanity checks

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
- `load_pdf_pages()` (in `langchain_service`): uses LangChain's `PyPDFLoader` to produce one
  `Document` per page, then normalizes page numbers to 1-based and tags each with `doc_id` /
  `source_file`.
- `find_repeated_lines()` / `clean_documents()`: PDFs extract with recurring noise — running
  headers, footers, standalone page numbers. This is detected **structurally**, not by keyword: any
  line that repeats on ≥60% of a document's pages is treated as noise and stripped, regardless of
  what it actually says. Standalone page numbers are caught with a regex. Runs of 3+ blank lines
  are collapsed. **This logic is unchanged from `main`** — LangChain ships no equivalent, so it
  simply operates on `Document.page_content` instead of plain strings.

### 2. Chunking — `src/chunking.py`
- `RecursiveCharacterTextSplitter` (configured in `langchain_service.get_text_splitter()`) replaces
  the hand-written paragraph-accumulation algorithm. Its `length_function` counts **words** rather
  than characters, so `chunk_size=250` / `chunk_overlap=45` stay in the same units the original
  used. Its separator list (`["\n\n", "\n", ". ", " ", ""]`) reproduces the original's
  "paragraphs first, sentences as a fallback" behaviour: it only drops to a lower-priority
  separator when a piece is still over-length.
- `detect_section()` / `tag_chunk_metadata()`: best-effort tagging of which paper section
  (Introduction, Method, etc.) a chunk falls in, plus a stable `chunk_id` and word count. **Also
  unchanged from `main`** — bespoke metadata with no framework equivalent.
- Output: 527 chunks. `chunks.json` is still written, but only for inspection — the FAISS store
  carries its own copy, so nothing depends on this file's row order any more.

### 3. Embeddings + vector store — `src/vectorstore.py`
- `FAISS.from_documents(chunks, OllamaEmbeddings(...))` does in one call what previously took a
  hand-written batching loop, a retry/backoff wrapper, an L2-normalize step, an
  `IndexFlatIP` construction, and a `chunks.json` sidecar. `OllamaEmbeddings.embed_documents()`
  sends every chunk to `/api/embed` in a single batched request.
- `distance_strategy=MAX_INNER_PRODUCT` reproduces the original `IndexFlatIP` setup, so scores
  remain directly comparable to `main`. No normalization step is needed: Ollama's `/api/embed`
  already returns unit-length vectors (measured L2 norm: exactly `1.000000`), and for unit vectors
  an inner product *is* cosine similarity — which also means `main`'s explicit `normalize()` was
  defensive rather than load-bearing.
- Embedding runs in batches of 32. `OllamaEmbeddings.embed_documents()` sends everything it's given
  in one HTTP request, and all 527 chunks at once overran the client timeout — so the first batch
  creates the store and the rest are added to it, with progress printed as it goes.
- Persisted with `save_local()` to `data/processed/faiss_store/` as two files: `index.faiss` (the
  vectors) and `index.pkl` (the document store). Reloaded with `load_local()`.
- **The metadata sidecar is gone.** Where `main` needed `chunks.json[row]` to interpret FAISS row
  `row` — a hand-maintained positional link — the store now returns `Document` objects carrying
  their own text and metadata, so there is no invariant left to protect.

### 4. Conversational bot with memory — `src/bot.py`
- `retrieve()`: one call to `similarity_search_with_score()` returns the top `TOP_K=5` chunks as
  `(Document, score)` pairs — the embed → search → map-row-number-back-to-text sequence collapses
  into a single method. **All 5 results are returned unconditionally — there is still no
  relevance-score filtering.**
- `format_context()`: concatenates the 5 chunks into one labeled text block
  (`[Source: doc_id, page X]` + text), which becomes part of a single prompt.
- `ChatSession`: owns a `deque(maxlen=4)` of past `(question, answer)` pairs — **kept from `main`
  deliberately**, since a bounded deque states "the last 4 interactions" more directly than
  LangChain's memory abstractions would. `to_history_messages()` converts those turns into the
  `HumanMessage`/`AIMessage` objects the prompt's `MessagesPlaceholder` expects.
- The chain itself is composed once in `build_answer_chain()` as `prompt | llm | parser`, so
  `ask()` is a single `self.chain.invoke({...})` rather than three functions called in the right
  order.
- The LLM still makes **one call per question**, receiving the 5 chunks already merged into a
  single block of text — it has no awareness that "5 separate documents were retrieved".

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
  RAGAS/Trulens. **This is a like-for-like port of `main`'s judge** — same rubric text, same four
  criteria, same 1–5 scale, same single call per judgment, same `results.json` shape. Only the
  plumbing differs (`ChatPromptTemplate` + LCEL instead of a hand-built messages list).
- **The judge runs in Ollama's native JSON mode** (`format="json"`), which constrains decoding so
  the reply is valid JSON by construction. On `main`, one of the ten questions came back as prose-
  wrapped JSON and could not be scored even after a repair fallback; this removes that failure mode
  at the source rather than patching it afterwards.
- The 10 questions (`data/eval/questions.json`) cover all 5 papers, and several are deliberately
  written as pronoun-based follow-ups ("it"/"that" referring to a prior question's topic) so
  answering them correctly requires the bot to actually resolve the reference from
  `ChatSession.history` — not just from the retrieved text.
- Output: `data/eval/results.json` (every question, answer, retrieved chunk, and judge score) plus
  a printed summary of average scores per criterion.

### Why not RAGAS — a tested conclusion, not an assumption

RAGAS was implemented properly on this branch and then removed, because it demonstrably cannot
produce usable scores on this hardware. The evidence is worth keeping:

A full implementation was built against the four metrics the instructor named (Faithfulness >90%,
Answer Correctness >80%, Context Recall >85%, Context Precision >80%), including ten hand-written
reference answers, and run with **every** configuration issue from an earlier attempt fixed and
verified reaching Ollama:

| Setting | Failure it addressed | Verified? |
|---|---|---|
| `keep_alive="30m"` | Model unloading/reloading between calls | ✅ |
| `num_ctx=16384` | Long few-shot metric prompts truncated at Ollama's 2048 default | ✅ (`ctx: 16384` in `/api/ps`) |
| `format="json"` | Malformed, prose-wrapped output | ✅ (`format='json'` in params) |
| `max_workers=1` | Concurrency pile-up against a single-threaded local server | ✅ |
| `timeout=900` | Premature 180s cutoffs | ✅ |

**The result on a 2-question test run:**

```
faithfulness         no valid scores (all nan)
answer_correctness   no valid scores (all nan)
context_recall       no valid scores (all nan)
context_precision    100.0%  (need >80%)  PASS
```

Three of four metrics failed with `OutputParserException` — `llama3.2` emitting `{` and stopping,
or echoing RAGAS's prompt back rather than answering it. One (`answer_correctness`) exceeded even
the 900-second timeout. The run took ~25 minutes for two questions, projecting to 4–6 hours for
ten.

**The telling detail:** the only metric that worked, `context_precision`, is also the only one of
the four that doesn't require the model to *generate* complex structured JSON. That isolates the
root cause precisely — this is a **structured-output capability limit of a 3B model**, not a
configuration problem. Every configuration hypothesis was tested and eliminated.

The untested lever is a larger model (7–8B), which would very likely handle RAGAS's prompts;
that's a genuine avenue, not a dead end. But on this setup, the self-devised metric above is the
only mechanism that produces a complete, defensible set of numbers — and it covers all four
evaluation aspects the assignment brief itself names.

---

## Design decisions

- **All LangChain usage behind one module.** `langchain_service.py` is the only file that imports
  LangChain; everything else imports from it. LangChain's package layout churns (this project hit
  three separate version/deprecation breakages while being built), and this keeps the blast radius
  of any future change to exactly one file.
- **The dedicated `langchain-ollama` package, not the generic `langchain_community` wrappers.**
  It talks to the local server through Ollama's own Python client and exposes Ollama-specific
  options — `keep_alive`, `num_ctx`, `seed`, native JSON mode — as first-class parameters, which is
  what makes the efficiency settings in [The service layer](#the-service-layer) possible.
- **Bespoke logic stays hand-written.** Header/footer detection and section tagging have no
  LangChain equivalent and were left exactly as they are on `main`. A framework removes
  boilerplate, not the parts specific to your problem — and rewriting working, graded code for its
  own sake adds risk without adding capability.
- **The 4-turn memory stays a `collections.deque(maxlen=4)`.** LangChain's memory abstractions
  would add indirection without changing behaviour, and the assignment asks specifically for the
  last 4 interactions — a bounded deque says that directly.
- **Exact FAISS search (`MAX_INNER_PRODUCT` + `normalize_L2`) over approximate indexes** — at ~500
  vectors, brute-force search is already fast; approximate methods would trade accuracy for speed
  the project doesn't need. This configuration also matches `main`'s `IndexFlatIP` + `normalize()`
  exactly, so similarity scores stay comparable across branches.
- **Retrieval always returns exactly `k` chunks, with no score threshold.** This is a known,
  intentional simplification — see [Known limitations](#known-limitations).
- **LLM-as-judge for evaluation, not manual scoring** — and deliberately a *single* call per
  judgment rather than RAGAS-style chained sub-calls. See the note at the end of
  [The service layer](#the-service-layer) for why that distinction decides whether the evaluation
  finishes in minutes or hours on this hardware.

---

## Current status

| Stage | Status |
|---|---|
| PDF extraction + noise cleaning | ✅ Done — `PyPDFLoader` + custom cleaning |
| Chunking | ✅ Done — `RecursiveCharacterTextSplitter`, 527 chunks |
| Embeddings | ✅ Done — `OllamaEmbeddings`, batched 32 at a time |
| FAISS vector store | ✅ Done — LangChain `FAISS`, vectors + documents stored together |
| Retrieval + LLM integration | ✅ Done — LCEL chain via `langchain-ollama` |
| Conversational memory (4 turns) | ✅ Done — `deque(maxlen=4)` + `MessagesPlaceholder` |
| Test harness (10 questions) | ✅ Done (`data/eval/questions.json`) |
| Evaluation scoring (self-devised) | ✅ Done — see [Evaluation results](#evaluation-results) below |
| Final report | ⬜ Not started |
| Bonus: feedback into memory | ⬜ Not started |

See `TECHNICAL_DETAILS.md` for the detailed reasoning behind each completed stage, including the
RAGAS investigation.

---

## Evaluation results

Scored by the self-devised LLM-as-judge rubric described above — 1–5 per criterion, averaged
across the 10 test questions. Both branches ran the same rubric on the same questions, so the
columns are directly comparable.

| Criterion | `main` (hand-written) | This branch (LangChain) |
|---|---|---|
| Relevance | 4.89 (97.8%) | **5.00 (100%)** |
| Accuracy | 5.00 (100%) | **5.00 (100%)** |
| Contextual Awareness | 5.00 (100%) | **5.00 (100%)** |
| Response Quality | 4.33 (86.6%) | **4.90 (98.0%)** |
| Questions scored | 9 / 10 | **10 / 10** |

**The `n` row is the most meaningful difference.** On `main`, one judge reply came back as
prose-wrapped JSON and could not be parsed even after a brace-repair fallback, so question 10 went
unscored. Here, Ollama's native JSON mode (`format="json"`) constrains decoding at the grammar
level, so every reply parsed on the first attempt and no repair path was needed.

The score improvements should be read with more caution than the `n` improvement. Two things
changed at once besides the framework: `num_ctx=8192` means the model now sees the full,
untruncated prompt (see the note above), and answers came out noticeably longer — which plausibly
lifts Response Quality in particular. This is a single run of a non-deterministic system, not a
controlled A/B test, and the judge self-bias caveat below applies to both columns equally.

Full per-question detail — every answer, its retrieved chunks with similarity scores, the memory
available at that turn, and the judge's four scores with justifications — is in
`data/eval/results.json`.

---

## Known limitations

- **No follow-up query rewriting.** `retrieve()` embeds only the literal current question. A
  follow-up like "what about its encoder?" won't be resolved against prior turns before
  retrieval, so the vector search may miss what the user actually means — even though the *LLM*
  still sees the conversation history when generating the answer.
- **No relevance-score filtering on retrieval.** All top-`k` results are sent to the LLM
  regardless of how low their similarity score is; there's no threshold (e.g. "drop anything
  below 0.5") to exclude genuinely unrelated chunks from the prompt.
- **Requires Ollama running locally** with both models pulled — every embedding and generation call
  is a live request to `localhost:11434`, with no offline fallback.
- **LLM-as-judge self-bias.** `evaluate.py` uses `llama3.2` to judge answers generated by the same
  `llama3.2`, which can be more lenient toward its own phrasing/reasoning style than an
  independent judge (e.g. a larger or different model) would be. Near-ceiling scores should be read
  with that in mind — worth naming explicitly as a methodology caveat in the report, not treated as
  proof the bot is flawless. Unchanged from `main`; LangChain does not affect this either way.
- **The evaluation cannot measure retrieval quality on its own.** The judge only ever sees the
  final answer, so a bad answer caused by *bad retrieval* and one caused by *bad reasoning over
  good retrieval* score identically. RAGAS's `context_recall` / `context_precision` exist to
  separate exactly those, and that capability is genuinely absent here.
- **LangChain's version churn is a real maintenance cost.** Building this branch required pinning
  `langchain`, `langchain-community`, and `langchain-ollama` to a mutually-compatible generation,
  and working around several deprecated import paths. Confining every import to
  `langchain_service.py` limits the damage but does not remove it — this is the concrete trade-off
  for the code the framework saves.
- **RAGAS remains impractical on this hardware, LangChain or not.** An earlier attempt confirmed
  chained per-metric LLM calls taking minutes each, with some metrics returning entirely unscored
  (`nan`) results even after timeout/concurrency fixes. That finding is about RAGAS's multi-call
  design meeting local CPU inference, and is unchanged by this branch — see
  `TECHNICAL_DETAILS.md`'s "RAGAS investigation" section.
