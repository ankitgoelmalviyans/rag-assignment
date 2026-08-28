# Technical Deep Dive — RAG Pipeline, File by File

This document is a complete, detailed walkthrough of every file in this project, every
function inside them, and the full data flow end to end — including the evaluation stage.
It's written to double as source material for the assignment's Final Report
("Technical Implementation" and "Evaluation" sections), so it goes deeper than `README.md`,
which is the quick-reference version of the same material.

> ### ⚠ Branch note — read this first
>
> **Sections 4–8 below document the hand-written implementation on the `main` branch**, where
> every RAG primitive is built from scratch in plain Python. They are kept intact because the
> *reasoning* in them (why chunks overlap, why vectors are normalized, why memory is a bounded
> deque, what the judge rubric checks) applies to both branches and is the substance the Final
> Report draws on.
>
> **On the `UseLangchain` branch, the plumbing described in those sections has been replaced by
> LangChain**, with every framework import confined to a single module. See
> **[Section 3.5 — `langchain_service.py`](#35-langchain_servicepy--the-langchain-layer-uselangchain-branch)**
> for what that layer does, and the **"What LangChain changed"** table in `README.md` for the
> file-by-file before/after comparison. The behaviour, models, prompts, and evaluation rubric are
> the same on both branches; what differs is how much is written by hand versus configured.

---

## Table of contents

1. [Architecture pattern](#1-architecture-pattern)
2. [Complete end-to-end flow diagram](#2-complete-end-to-end-flow-diagram)
3. [`config.py` — shared configuration](#3-configpy--shared-configuration)
3.5. [`langchain_service.py` — the LangChain layer](#35-langchain_servicepy--the-langchain-layer-uselangchain-branch)
4. [`ingestion.py` — PDF extraction & cleaning](#4-ingestionpy--pdf-extraction--cleaning)
5. [`chunking.py` — chunking](#5-chunkingpy--chunking)
6. [`vectorstore.py` — embeddings & FAISS index](#6-vectorstorepy--embeddings--faiss-index)
7. [`bot.py` — retrieval, prompting, memory, generation](#7-botpy--retrieval-prompting-memory-generation)
8. [`evaluate.py` — the evaluation stage](#8-evaluatepy--the-evaluation-stage)
9. [Data schemas (exact JSON shapes)](#9-data-schemas-exact-json-shapes)
10. [Design decisions and why](#10-design-decisions-and-why)
11. [The RAGAS investigation — a documented challenge](#11-the-ragas-investigation--a-documented-challenge)
12. [Known limitations and honest caveats](#12-known-limitations-and-honest-caveats)
13. [Reproduction guide](#13-reproduction-guide)

---

## 1. Architecture pattern

This project implements **RAG — Retrieval-Augmented Generation**. The core problem RAG
solves: an LLM on its own can only answer from what it memorized during training — it has
no access to your specific documents, and it will confidently make things up
("hallucinate") if asked about something it doesn't actually know. RAG fixes this by
inserting a retrieval step before generation:

```
   Traditional LLM chat:          RAG:

   Question ──► LLM ──► Answer    Question ──► [search your documents] ──► relevant text
                                       │                                         │
                                       └──────────────► LLM (given the text) ──► Answer
```

The LLM in RAG never "knows" your documents — every single time it answers, it's handed
the relevant text fresh, as part of the prompt, and asked to answer *from that text*. This
project's specific instantiation of that pattern:

- **Retrieval mechanism:** FAISS vector similarity search over 768-dimensional embeddings
- **Embedding model:** `nomic-embed-text` (via Ollama, local)
- **Generation model:** `llama3.2` (via Ollama, local)
- **No framework** (no LangChain/LlamaIndex) — every piece below is hand-written

---

## 2. Complete end-to-end flow diagram

There are three distinct flows in this project, running at three different times. They
share code (the same `embed_texts()`, the same `retrieve()`, the same `call_llama()`) but
never run in the same moment.

### 2.1 Offline indexing flow (run once, whenever the source PDFs change)

```
┌─────────────┐
│  5 PDFs      │  data/pdfs/1706.03762.pdf, 1810.04805.pdf, 1907.11692.pdf,
│              │  1910.10683.pdf, 2005.14165.pdf
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ ingestion.py                                                   │
│  extract_pages(pdf) ──► list[str], one raw string per page     │
│  find_repeated_lines(pages_lines) ──► set of header/footer text│
│  clean_pages(pages_text) ──► (cleaned_pages, repeated_lines)   │
└──────┬───────────────────────────────────────────────────────┘
       │  cleaned per-page text (in memory, not saved to disk)
       ▼
┌──────────────────────────────────────────────────────────────┐
│ chunking.py                                                     │
│  split_into_segments(pages_text) ──► list[(page_num, text)]    │
│  detect_section(text, current_section) ──► section label       │
│  build_chunks(segments, doc_id, source_file) ──► list[chunk]   │
└──────┬───────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│ chunks.json   │  398 chunk dicts: {chunk_id, doc_id, source_file,
│               │  page_start, page_end, section, text, word_count}
└──────┬───────┘
       │  texts extracted: [c["text"] for c in chunks]
       ▼
┌──────────────────────────────────────────────────────────────┐
│ vectorstore.py                                                  │
│  embed_texts(texts) ──► calls Ollama /api/embed in batches of 32│
│  normalize(vectors)  ──► L2-normalize so inner product = cosine │
│  faiss.IndexFlatIP(dim); index.add(embeddings)                  │
└──────┬───────────────────────────────────────────────────────┘
       │
       ▼
┌───────────────┐   ┌──────────────────┐
│ faiss.index    │   │ embeddings.npy    │
│ (768-dim       │   │ (raw numpy array, │
│  vectors, one  │   │  same data, kept  │
│  per chunk)    │   │  for inspection)  │
└───────────────┘   └──────────────────┘
```

**The invariant this whole flow depends on:** FAISS row `i` == `chunks[i]` in `chunks.json`.
There is no ID-based lookup between the two — just matching list position ("the positional
join"). If `chunks.json` were ever regenerated in a different order without also rebuilding
`faiss.index`, every search result would silently point at the wrong text.

### 2.2 Online query flow (runs every single time a question is asked)

```
User types: "How does BERT's masked language modeling work?"
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ bot.py : retrieve(query, chunks, index, k=5)                    │
│                                                                   │
│  1. query_vector = embed_texts([query])                          │
│       → ONE call to Ollama /api/embed, same model as indexing    │
│       → shape (1, 768) — a single row of 768 numbers             │
│                                                                   │
│  2. scores, row_indices = index.search(query_vector, k=5)        │
│       → PURE MATH, no model call. FAISS compares the query        │
│         vector's direction against all 398 stored vectors and    │
│         returns the 5 closest by inner product (= cosine sim,    │
│         since both sides are normalized).                        │
│       → scores       = [[0.87, 0.84, 0.81, 0.79, 0.75]]          │
│       → row_indices  = [[142, 145, 391, 12, 288]]                │
│                                                                   │
│  3. for score, row in zip(scores[0], row_indices[0]):            │
│         chunk = chunks[row]        ← the positional join in use  │
│         chunk["score"] = score                                   │
│       → returns 5 full chunk dicts (text + metadata + score)     │
└──────┬───────────────────────────────────────────────────────┘
       │  5 chunk dicts
       ▼
┌──────────────────────────────────────────────────────────────┐
│ bot.py : format_context(retrieved_chunks)                        │
│    "[Source: BERT, page 4]\n<chunk text>\n\n                     │
│     [Source: BERT, page 5]\n<chunk text>\n\n                     │
│     [Source: RoBERTa, page 3]\n<chunk text>\n\n..."               │
│    → ONE plain string, all 5 chunks merged                        │
└──────┬───────────────────────────────────────────────────────┘
       │  context string
       ▼
┌──────────────────────────────────────────────────────────────┐
│ bot.py : ChatSession.ask(question)                                │
│                                                                    │
│  messages = [{"role": "system", "content": SYSTEM_PROMPT}]        │
│  for turn in self.history:               ← up to 4 prior turns   │
│      messages.append({"role": "user", "content": turn.question})  │
│      messages.append({"role": "assistant", "content": turn.answer})│
│  messages.append({"role": "user",                                 │
│      "content": f"Context:\n{context}\n\nQuestion: {question}"})  │
│                                                                    │
│  answer = call_llama(messages)      ← ONE call to Ollama /api/chat│
│  self.history.append({"question": question, "answer": answer})   │
│  → deque(maxlen=4) automatically drops the oldest turn once       │
│    a 5th is added                                                  │
└──────┬───────────────────────────────────────────────────────┘
       │
       ▼
   Final answer string, printed to the user
```

**Key point made visible by this diagram:** the embedding model runs exactly once per
question (step 1), FAISS runs zero model calls (step 2, pure math), and the LLM runs
exactly once per question (inside `ChatSession.ask`). The 5 retrieved chunks are merged
into a *single* prompt — the LLM never "sees" them as 5 separate things, and it never makes
5 separate answers.

### 2.3 Evaluation flow (`evaluate.py`, run once per evaluation pass)

```
data/eval/questions.json  (10 questions, several pronoun-based memory follow-ups)
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ evaluate.py : run_interactions(chunks, index, questions)         │
│                                                                    │
│  session = ChatSession(chunks, index)   ← ONE session, all 10 Qs │
│  for item in questions:                                          │
│      history_snapshot = list(session.history)  ← BEFORE asking   │
│      retrieved = retrieve(item.question, chunks, index)          │
│      answer = session.ask(item.question)  ← same bot.py flow     │
│                                            ← as section 2.2      │
│      records.append({question, retrieved, history_snapshot,      │
│                        answer, ...})                              │
└──────┬───────────────────────────────────────────────────────┘
       │  10 records (question + context + answer + memory state)
       ▼
┌──────────────────────────────────────────────────────────────┐
│ evaluate.py : judge_answer(record)   — for EACH of the 10 records │
│                                                                     │
│  messages = build_judge_messages(record)                           │
│    → JUDGE_SYSTEM_PROMPT (the rubric)                               │
│    → + retrieved context, + conversation history available,        │
│      + the question, + the candidate answer                        │
│                                                                     │
│  raw = call_llama(messages, timeout=300)  ← a SECOND, SEPARATE     │
│                                              LLM call — "the judge" │
│                                                                     │
│  scores = json.loads(raw)   (with truncation-repair fallback)      │
│    → {"relevance": {"score": 5, "justification": "..."},           │
│       "accuracy": {...}, "contextual_awareness": {...},            │
│       "response_quality": {...}}                                    │
└──────┬───────────────────────────────────────────────────────┘
       │  10 records, each now with a "judge" field attached
       ▼
┌───────────────────┐
│ results.json        │  full record per question: question, doc_focus,
│                      │  history_available, retrieved chunks (with scores),
│                      │  answer, judge scores
└───────────────────┘
       │
       ▼
  summarize(records) — prints average score per criterion to console
```

---

## 3. `config.py` — shared configuration

The single source of truth for every path and model name used across all other files —
written once so no stage duplicates or drifts from another's settings.

```python
PDF_FOLDER = Path("data/pdfs")
PROCESSED_DIR = Path("data/processed")
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"
EMBEDDINGS_FILE = PROCESSED_DIR / "embeddings.npy"
INDEX_FILE = PROCESSED_DIR / "faiss.index"

EVAL_DIR = Path("data/eval")
QUESTIONS_FILE = EVAL_DIR / "questions.json"
RESULTS_FILE = EVAL_DIR / "results.json"

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"
```

**Important operational detail:** all the `Path(...)` values here are **relative paths**.
They resolve against whatever the *current working directory* is when a script runs — not
against where `config.py` itself lives. This is why every script in this project must be
run from the **project root** (e.g. `python src/bot.py`), not from inside `src/`
(`cd src && python bot.py` would break, since `data/pdfs` doesn't exist relative to `src/`).

On the `UseLangchain` branch `config.py` also carries the tuning knobs the LangChain layer
reads — `OLLAMA_KEEP_ALIVE`, `OLLAMA_NUM_CTX`, `OLLAMA_TIMEOUT`, `EMBED_BATCH_SIZE`,
`TARGET_WORDS`, `OVERLAP_WORDS`, `TOP_K`, `MEMORY_TURNS` — and `INDEX_FILE`/`EMBEDDINGS_FILE`
are replaced by a single `VECTORSTORE_DIR`, since LangChain's FAISS wrapper persists a folder
(`index.faiss` + `index.pkl`) rather than a bare index file.

---

## 3.5. `langchain_service.py` — the LangChain layer (`UseLangchain` branch)

**Purpose:** be the *only* file in the project that imports LangChain. Every other module
calls functions from here, so a renamed or deprecated framework class means editing exactly
one file. This is not hypothetical caution — building this branch hit three separate version
and deprecation breakages in the LangChain ecosystem.

### Model access, cached

```python
_LLM_CACHE: dict[tuple, ChatOllama] = {}

def get_llm(temperature: float = 0.0, json_mode: bool = False) -> ChatOllama:
    key = (temperature, json_mode)
    if key not in _LLM_CACHE:
        kwargs = {
            "model": CHAT_MODEL,
            "base_url": OLLAMA_BASE_URL,
            "temperature": temperature,
            "num_ctx": OLLAMA_NUM_CTX,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "seed": 0,
            "client_kwargs": {"timeout": OLLAMA_TIMEOUT},
        }
        if json_mode:
            kwargs["format"] = "json"
        _LLM_CACHE[key] = ChatOllama(**kwargs)
    return _LLM_CACHE[key]
```

Instances are cached by their settings, so a run making dozens of calls reuses one configured
client rather than constructing a new one per call.

**Why the dedicated `langchain-ollama` package and not `langchain_community`'s wrappers:** it
talks to the local server through Ollama's own Python client and exposes Ollama-specific
options as real constructor parameters. Each one below solves a concrete problem observed on
this hardware:

| Setting | What it does | Why it matters here |
|---|---|---|
| `keep_alive="30m"` | Holds the model resident in Ollama between calls | A full evaluation makes ~20 calls; without this Ollama can unload and reload `llama3.2` (a 2 GB model) repeatedly |
| `num_ctx=8192` | Raises the context window | Ollama's default is **2048 tokens**, and this project's full prompt is ~2,100 (see below) — the default silently truncates, with no error |
| `format="json"` | Ollama's native JSON mode (judge only) | Constrains decoding so the reply is valid JSON *by construction* — removes the prose-wrapped-JSON parse failure that left one question unscored on `main` |
| `temperature=0`, `seed=0` | Deterministic decoding | Repeatable scores across runs, which is what an evaluation judge wants |
| `client_kwargs={"timeout": 300}` | Per-request timeout | Local CPU inference on long prompts is measured in minutes; the client default is far too short |

#### Why `num_ctx` turned out to matter more than expected

Measured against the actual corpus: chunks average **207 words**, so the 5 retrieved chunks come
to roughly 1,035 words ≈ **1,400 tokens** (at ~1.35 tokens/word for technical English). Adding
the system prompt (~60 tokens), the question (~25), and four turns of conversation memory
(~600) puts a full prompt at approximately **2,100 tokens**.

Ollama's default context window is **2048**. The `main` branch never set `num_ctx`, which means
that once the 4-turn memory window filled, its prompts were sitting at or just past that limit —
and Ollama truncates silently rather than erroring. The oldest content in the prompt (the
earliest memory turns) would simply have been dropped before the model ever saw it.

This has two consequences worth stating plainly in the report:
1. It is a **correctness** issue on `main`, not just a tuning detail — the conversational memory
   the assignment specifically asks for may have been partially discarded on later questions.
2. It is the honest explanation for why this branch generates **more slowly**: the model is now
   processing the entire prompt instead of a truncated one. That is a real cost for a real
   correctness gain, not a regression.

(These are estimates derived from word counts, not exact tokenizer output — but the margin is
narrow enough that the conclusion holds either way.)

### Batched vector store construction

```python
first, rest = documents[:batch_size], documents[batch_size:]
store = FAISS.from_documents(first, get_embeddings(),
                             distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
for start in range(0, len(rest), batch_size):
    store.add_documents(rest[start:start + batch_size])
```

**Why batching is required, not optional:** `OllamaEmbeddings.embed_documents()` sends every
text it is handed in a *single* HTTP request. A first attempt passed all 527 chunks at once
and died with `httpx.ReadTimeout` after 300 seconds. Building the store from the first batch
and adding the rest in `EMBED_BATCH_SIZE` groups keeps each request survivable and prints
progress during what is otherwise a silent multi-minute wait — the same reasoning behind the
hand-written batching on `main`.

**Why there is no normalization step**, unlike `main`'s `normalize()`: Ollama's `/api/embed`
returns already-unit-length vectors from `nomic-embed-text` — measured directly, the L2 norm
of an embedding is exactly `1.000000` — and for unit vectors an inner product *is* cosine
similarity. Passing `normalize_L2=True` would also be ignored here: LangChain applies that
flag only to `EUCLIDEAN_DISTANCE` stores and emits a warning otherwise. The implication worth
noting is that `main`'s explicit normalize step was defensive rather than load-bearing; both
branches are doing genuine cosine-similarity search.

### Prompts and chains

The RAG prompt is a `ChatPromptTemplate` with a `MessagesPlaceholder` for conversation
history; the judge prompt is a second template carrying the same rubric text as `main`. Both
are composed with LCEL — LangChain Expression Language — where `|` chains runnables:

```python
def build_answer_chain(temperature: float = 0.0):
    return build_rag_prompt() | get_llm(temperature=temperature) | StrOutputParser()

def build_judge_chain():
    return build_judge_prompt() | get_llm(temperature=0.0, json_mode=True) | StrOutputParser()
```

Invoking either with a dict of variables formats the prompt, calls Ollama, and returns a
string. This is what replaces `main`'s "call `retrieve()`, then `format_context()`, then
`call_llama()`, in that order, every time" — the sequencing becomes structural rather than
something re-typed at each call site.

> **The one thing deliberately *not* changed:** the judge remains a **single LLM call scoring
> all four criteria at once**. Frameworks like RAGAS decompose each metric into chained
> sub-calls — cheap against a cloud API, impractical against local CPU inference (see
> [section 11](#11-the-ragas-investigation--a-documented-challenge)). Using LangChain for
> composition while keeping the one-call design is what makes this branch produce like-for-like
> results at like-for-like speed.

---

## 4. `ingestion.py` — PDF extraction & cleaning

**Purpose:** turn a PDF file into clean, page-by-page text, with structural extraction
noise removed.

### `extract_pages(pdf_path) -> list[str]`
```python
reader = PdfReader(pdf_path)
return [page.extract_text() or "" for page in reader.pages]
```
Uses `pypdf` to pull text out of every page. One string per page, in page order. The
`or ""` guards against pages that extract to `None` (e.g. an image-only page).

### `find_repeated_lines(pages_lines, threshold=0.6) -> set[str]`
For **one document**, counts how many distinct pages each line appears on, and returns the
set of lines appearing on 60%+ of pages. The insight: a paper's running header ("Attention
Is All You Need") or footer literally repeats near-identically on almost every page — no
normal body-text sentence does that. This is **content-agnostic**: it doesn't need a list
of "known header patterns," it detects noise purely from its repetition behavior, so it
works the same way regardless of which paper it's run on.

```python
line_counts = Counter()
for lines in pages_lines:
    unique_lines_on_page = set(line.strip() for line in lines if line.strip())
    for line in unique_lines_on_page:
        line_counts[line] += 1
# a line repeated twice ON THE SAME PAGE only counts once, via the set()
return {line for line, count in line_counts.items() if count / total_pages >= threshold}
```

### `clean_pages(pages_text) -> (cleaned_pages, repeated_lines)`
Applies three cleaning rules per page:
1. Drop any line that's in the `repeated_lines` set (headers/footers)
2. Drop any line matching `^(page\s+)?\d+(\s+of\s+\d+)?$` (standalone page numbers like
   "12" or "Page 12 of 20")
3. Collapse 3+ consecutive blank lines down to 2 (`\n{3,}` → `\n\n`)

**Example, concretely:** if a raw extracted page looked like —
```
Attention Is All You Need


We propose a new simple network architecture...


12
```
— after cleaning it becomes:
```
We propose a new simple network architecture...
```
Both the running title and the trailing page number are gone; only real body text remains.

---

## 5. `chunking.py` — chunking

**Purpose:** turn a full document's cleaned text into a list of ~250-word chunks small
enough to embed meaningfully, each carrying the metadata needed to cite it later.

### `split_into_sentences(text) -> list[str]`
```python
SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
```
Splits on whitespace that comes right after `.`/`!`/`?` and right before a capital letter —
a lightweight heuristic sentence boundary (not a full NLP sentence tokenizer, but sufficient
for the fallback role it plays below).

### `split_into_segments(pages_text) -> list[(page_number, text)]`
1. Splits each page's text on blank lines (`\n\n`) — real paragraph breaks, when the PDF
   preserved them.
2. **Any block still over `MAX_WORDS=400`** gets broken down further: first by sentence, and
   if a single sentence is *still* over 400 words (pathological case), by a hard word-count
   slice. This guarantees no unit handed downstream is ever oversized, no matter how a given
   PDF happened to format its paragraphs.

### `detect_section(paragraph_text, current_section) -> str | None`
```python
SECTION_PATTERN = re.compile(
    r"^(abstract|introduction|related work|background|method|model architecture|"
    r"experiments|results|discussion|conclusion|references|\d+(\.\d+)*\s+[A-Z])",
    re.IGNORECASE,
)
```
If a segment's first line looks like a section heading (short, matches the pattern above —
either a known section name or a numbered heading like "3.2 Model Architecture"), that
becomes the "current section." Otherwise the previous value is carried forward. This is
best-effort metadata for citations/debugging — it does not affect retrieval at all.

### `build_chunks(segments, doc_id, source_file) -> list[chunk dict]`
The core greedy algorithm:
```python
for page_number, segment_text in segments:
    current_section = detect_section(segment_text, current_section)
    segment_words = len(segment_text.split())

    if current_segments and current_words + segment_words > TARGET_WORDS:
        # closing the current chunk: emit it, then carry the tail forward
        chunk = make_chunk(current_segments, doc_id, source_file, current_section, overlap_text, len(chunks))
        chunks.append(chunk)
        words = chunk["text"].split()
        overlap_text = " ".join(words[-OVERLAP_WORDS:]) + " "
        current_segments, current_words = [], 0

    current_segments.append((page_number, segment_text))
    current_words += segment_words
```
In words: keep adding segments to the current chunk until adding one more would push it
over 250 words. Then close the chunk, save its **last 45 words** as `overlap_text`, and
start the next chunk by *prepending* that overlap text.

**Why the overlap matters, concretely:** imagine a key sentence lands right at a chunk
boundary — "The base model uses 8 attention heads, each with dimension 64." split so that
"8 attention heads" ends chunk N and "each with dimension 64" starts chunk N+1. Without
overlap, neither chunk alone fully contains the fact. With a 45-word overlap, chunk N+1
actually starts with the tail of chunk N, so the full sentence appears intact in at least
one chunk.

### `make_chunk(...) -> dict`
Assembles the final chunk record:
```python
{
    "chunk_id": f"{doc_id}-{chunk_index:04d}",   # e.g. "1706.03762-0012"
    "doc_id": doc_id,                             # e.g. "1706.03762" (the PDF's filename stem)
    "source_file": source_file,                   # e.g. "1706.03762.pdf"
    "page_start": min(page_numbers),
    "page_end": max(page_numbers),
    "section": section,
    "text": overlap_text + " ".join(s for _, s in segments_in_chunk),
    "word_count": len(text.split()),
}
```

**Result across all 5 papers:** 398 chunks written to `data/processed/chunks.json`, in the
order they were generated — an order that `vectorstore.py` will treat as permanent (see the
positional join, explained in section 6).

---

## 6. `vectorstore.py` — embeddings & FAISS index

**Purpose:** turn every chunk's text into a vector, and build a searchable index over those
vectors.

### `embed_texts(texts, model="nomic-embed-text", batch_size=32) -> np.ndarray`
```python
for start in range(0, len(texts), batch_size):
    batch = texts[start:start + batch_size]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(f"{OLLAMA_BASE_URL}/api/embed",
                                      json={"model": model, "input": batch}, timeout=120)
            response.raise_for_status()
            batch_vectors = response.json()["embeddings"]
            break
        except requests.exceptions.RequestException as error:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2 ** attempt)   # 2s, 4s, 8s backoff
    all_vectors.extend(batch_vectors)
return normalize(np.array(all_vectors, dtype="float32"))
```
Batches of 32 keep the number of HTTP round-trips manageable (398 chunks → 13 requests
instead of 398). Retries with exponential backoff handle transient failures (e.g. Ollama
briefly busy) without crashing the whole run over one bad request.

**This exact function is reused, unmodified, by `bot.py` to embed the user's live
question.** That reuse is what guarantees the question's vector lands in the same
768-dimensional space as the indexed chunks — if the question were embedded with a
*different* model or process, comparing it against the chunk vectors would be meaningless
(like comparing distances measured in miles against distances measured in kilometers).

### `normalize(vectors) -> np.ndarray`
```python
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
return vectors / norms
```
Scales every vector to length 1 (unit length). This is what makes FAISS's `IndexFlatIP`
(inner product search) mathematically equivalent to cosine similarity search — inner
product of two unit vectors *is* their cosine similarity.

### Index construction (in the `__main__` block)
```python
dimension = embeddings.shape[1]           # 768, from nomic-embed-text
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)                     # adds all 398 vectors, in order
faiss.write_index(index, str(INDEX_FILE))
```
`IndexFlatIP` is an **exact** (brute-force) index — every search compares the query against
*all* stored vectors, no approximation. At ~400 vectors this is fast (sub-millisecond);
approximate index types (like IVF or HNSW, which trade a small accuracy loss for speed at
millions-of-vectors scale) would add complexity this project's corpus size doesn't need.

### The positional join, made explicit
```python
reloaded_index = faiss.read_index(str(INDEX_FILE))
assert reloaded_index.ntotal == len(chunks), "index size doesn't match chunk count!"
```
This assertion exists specifically to catch drift in the one invariant the whole retrieval
mechanism depends on: **FAISS row `i` must equal `chunks[i]`**. There's no ID field linking
them — a search result is just a row number, and that row number is used directly as a
Python list index into `chunks.json`. If chunks were ever re-ordered, filtered, or
regenerated without also rebuilding the index in lockstep, this link would silently break —
searches would keep returning row numbers, but those numbers would now point at the *wrong*
chunk text, with no error thrown anywhere.

---

## 7. `bot.py` — retrieval, prompting, memory, generation

**Purpose:** the live conversational interface — takes a question, returns an answer,
remembers recent turns.

### `load_chunks_and_index()`
Reads `chunks.json` back into a Python list and `faiss.index` back into a FAISS index
object — the two artifacts `vectorstore.py` produced, now loaded fresh for a chat session.

### `retrieve(query, chunks, index, k=5) -> list[dict]`
```python
query_vector = embed_texts([query])
scores, row_indices = index.search(query_vector, k)

results = []
for score, row in zip(scores[0], row_indices[0]):
    chunk_with_score = dict(chunks[row])      # copy, so we don't mutate the shared list
    chunk_with_score["score"] = float(score)
    results.append(chunk_with_score)
return results
```
Embeds the question (one call), searches (pure math, zero model calls), and maps the `k`
result row numbers back to full chunk dicts via the positional join, tagging each with its
similarity score. **Always returns exactly `k` results — there's no relevance threshold.**
Even a weak match (e.g. score 0.3) is returned and passed downstream unfiltered; nothing in
this function decides "this isn't actually relevant enough."

### `format_context(retrieved_chunks) -> str`
```python
for chunk in retrieved_chunks:
    label = f"[Source: {chunk['doc_id']}, page {chunk['page_start']}]"
    blocks.append(f"{label}\n{chunk['text']}")
return "\n\n".join(blocks)
```
Turns the list of chunk dicts into one plain-text block, each chunk prefixed with a
human-readable source label. This is what gets inserted into the LLM's prompt — by the time
the LLM sees it, it's indistinguishable from a person having pasted 5 excerpts into the chat.

### `call_llama(messages, timeout=180) -> str`
```python
response = requests.post(f"{OLLAMA_BASE_URL}/api/chat",
                          json={"model": CHAT_MODEL, "messages": messages, "stream": False},
                          timeout=timeout)
response.raise_for_status()
return response.json()["message"]["content"]
```
One HTTP call to Ollama's chat endpoint. `messages` follows the standard chat-format list of
`{"role": ..., "content": ...}` dicts (`system`/`user`/`assistant`) — the same shape used by
OpenAI's API and most chat-model APIs. `stream: False` means we wait for the full response
rather than receiving it token-by-token. The `timeout` parameter was made configurable
(default 180s) specifically because the evaluation stage needed a longer allowance for its
longer judge prompts — see section 8 and section 11.

### `ChatSession` class — the memory mechanism
```python
class ChatSession:
    def __init__(self, chunks, index):
        self.chunks = chunks
        self.index = index
        self.history = deque(maxlen=MEMORY_TURNS)   # MEMORY_TURNS = 4

    def ask(self, question: str) -> str:
        retrieved = retrieve(question, self.chunks, self.index)
        context = format_context(retrieved)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in self.history:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
        messages.append({"role": "user",
                          "content": f"Context:\n{context}\n\nQuestion: {question}"})

        answer = call_llama(messages)
        self.history.append({"question": question, "answer": answer})
        return answer
```
`collections.deque(maxlen=4)` is a Python list-like structure that **automatically drops the
oldest item once you append past its max length** — that one line is the entire memory
implementation. No manual "if len > 4: pop the first one" logic needed; the data structure
does it natively. Each `.ask()` call: retrieves *fresh* context for the current question
(memory does not affect retrieval — see the limitation on this in section 11), replays up to
4 prior turns as alternating user/assistant messages (so the LLM sees the actual back-and-
forth, not a summary), appends the new turn, sends one combined prompt, then stores the new
(question, answer) pair — which may push the oldest turn out if the deque was already full.

### `SYSTEM_PROMPT`
```
You are a helpful assistant answering questions about a set of research papers
(Transformer, BERT, GPT-3, RoBERTa, T5). Answer ONLY using the provided context.
If the context doesn't contain enough information to answer, say so explicitly
instead of guessing. When possible, cite the source as [doc_id, page X].
```
This is the instruction that tries to keep the LLM grounded in retrieved text rather than
its own pretrained knowledge, and asks it to admit uncertainty rather than hallucinate. It's
sent on **every** call, as the first message — Ollama's chat models treat the `system` role
as standing instructions that apply for the whole conversation.

---

## 8. `evaluate.py` — the evaluation stage

**Purpose:** run the bot against 10 predefined questions, then score each answer against a
documented rubric, without a framework (RAGAS/Trulens) — this project's "self-devised
metric," as explicitly permitted by the assignment.

### `load_questions() -> list[dict]`
Reads `data/eval/questions.json` — 10 entries, each `{question, doc_focus}`. `doc_focus` is
pure bookkeeping (which paper the question targets, for coverage-tracking) — it's not read by
any scoring logic. Several questions are deliberately written using a pronoun ("it"/"that")
referring back to a *previous* question's topic, so answering correctly requires the bot to
actually resolve that reference from `ChatSession.history`, not just from the current
question's own wording — but this is baked into the question *text* itself now, not tracked
as a separate flag in the data (an earlier version had explicit `id`/`tests_memory`/
`memory_note` fields; they were trimmed as unnecessary metadata once the questions themselves
already encode the same intent).

### `run_interactions(chunks, index, questions) -> list[dict]`
```python
session = ChatSession(chunks, index)     # ONE session for all 10 questions, in order
for item in questions:
    history_snapshot = [dict(turn) for turn in session.history]   # BEFORE asking
    retrieved = retrieve(item["question"], chunks, index)
    answer = session.ask(item["question"])
    records.append({..., "history_available": history_snapshot,
                     "retrieved": [...], "answer": answer})
```
Using one continuous session (rather than a fresh session per question) is deliberate: it's
what makes the 4-turn memory window fill and evict *exactly* as it would in a real
conversation, so by question 9 and 10, questions 1–5's turns have genuinely fallen out of
memory — the same way they would for a real user. `history_snapshot` captures what memory
looked like at the *moment* each question was asked (before that question's own answer gets
appended), because the judge needs to know what the bot actually had access to, not what it
has access to by the time evaluation finishes.

### `build_judge_messages(record) -> list[dict]`
Assembles the judge's input: the retrieved context (as the factual ground truth to check
against), the conversation history available at that turn, the question, and the candidate
answer — formatted into one user message, paired with `JUDGE_SYSTEM_PROMPT` as the system
message.

### `JUDGE_SYSTEM_PROMPT` — the rubric, in full
```
Score the CANDIDATE ANSWER on four criteria, each from 1 (poor) to 5 (excellent):

- relevance: Does the answer directly address what was actually asked?
- accuracy: Is the answer factually consistent with the RETRIEVED CONTEXT
  provided below? Judge against that context only, not your own general
  knowledge...
- contextual_awareness: If the question depended on earlier conversation turns...
  did the answer correctly use the CONVERSATION HISTORY to resolve it?...
- response_quality: Is the answer clearly written, well-organized, and
  appropriately detailed...

Respond with ONLY a JSON object... {relevance: {score, justification}, ...}
```
These four criteria are taken directly from the assignment brief's own "Evaluation Aspects
to Consider" (Relevance, Accuracy, Contextual Awareness, Response Quality) — the rubric
doesn't invent new criteria, it operationalizes the ones the assignment already specifies
into something an LLM can score consistently.

### `judge_answer(record) -> dict`
```python
raw = call_llama(build_judge_messages(record))   # timeout+retry handled inside call_llama itself

cleaned = raw.strip()
if cleaned.startswith("```"):                            # strip markdown fences if present
    cleaned = cleaned.strip("`")
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:]
    cleaned = cleaned.strip()

try:
    return json.loads(cleaned)
except json.JSONDecodeError:
    pass

missing_braces = cleaned.count("{") - cleaned.count("}")   # repair a truncated response
if missing_braces > 0:
    try:
        return json.loads(cleaned + "}" * missing_braces)
    except json.JSONDecodeError:
        pass

return {"parse_error": True, "raw_response": raw}          # give up cleanly, don't crash
```
This function evolved directly from real failures during testing: an early run crashed
entirely when one judge call exceeded the (then-hardcoded) 180-second timeout — fixed by
raising `bot.py`'s `call_llama()` default timeout to 300s and adding a built-in retry there
(so every caller benefits, not just the judge). A separate failure — one response got cut
off by the model one token early, missing its closing `}` — led to the brace-repair fallback
here. If parsing still fails after all of that, the record is marked `parse_error: true` and
the run continues rather than crashing over one bad response.

### `summarize(records)`
Averages each of the four criteria's scores across all successfully-parsed records, printed
to the console as the evaluation's headline result.

**Actual results from real runs, both branches, same rubric and same questions:**

| Criterion | `main` (hand-written) | `UseLangchain` |
|---|---|---|
| Relevance | 4.89 (97.8%) | 5.00 (100%) |
| Accuracy | 5.00 (100%) | 5.00 (100%) |
| Contextual Awareness | 5.00 (100%) | 5.00 (100%) |
| Response Quality | 4.33 (86.6%) | 4.90 (98.0%) |
| Questions scored | 9 / 10 | 10 / 10 |

On `main`, question 10's judge response failed to parse even after the brace-repair fallback, so
it went unscored. On the LangChain branch, Ollama's native JSON mode constrains decoding to valid
JSON, so all ten parsed on the first attempt — that `n` difference is the clearest, most
attributable improvement.

The score differences deserve more caution: two variables changed alongside the framework
(`num_ctx` now prevents prompt truncation, and answers came out longer), and this is one run of a
non-deterministic system rather than a controlled comparison.

---

## 9. Data schemas (exact JSON shapes)

### `chunks.json` — one entry per chunk
```json
{
  "chunk_id": "1810.04805-0042",
  "doc_id": "1810.04805",
  "source_file": "1810.04805.pdf",
  "page_start": 4,
  "page_end": 4,
  "section": "3.3.1 Task #1: Masked LM",
  "text": "In order to train a deep bidirectional representation, we simply mask...",
  "word_count": 213
}
```

### `questions.json` — one entry per test question
```json
{
  "question": "How many attention heads does the base version of that model use?",
  "doc_focus": "Transformer"
}
```

### `results.json` — one entry per evaluated question
```json
{
  "question": "What is the 'text-to-text' framework introduced in the T5 paper...",
  "doc_focus": "T5",
  "history_available": [],
  "retrieved": [
    {"chunk_id": "...", "doc_id": "...", "page_start": 8, "score": 0.8123, "text": "..."},
    "... 4 more ..."
  ],
  "answer": "The \"text-to-text\" framework introduced in the T5 paper is...",
  "judge": {
    "relevance": {"score": 5, "justification": "..."},
    "accuracy": {"score": 5, "justification": "..."},
    "contextual_awareness": {"score": 5, "justification": "..."},
    "response_quality": {"score": 5, "justification": "..."}
  }
}
```

---

## 10. Design decisions and why

| Decision | Alternative considered | Why this choice |
|---|---|---|
| No LangChain/LlamaIndex (`main`) | Use a RAG framework | Not required by the assignment; hand-writing every piece keeps the mechanics (chunk boundaries, the positional join, prompt construction, memory eviction) visible instead of hidden behind abstractions. The `UseLangchain` branch explores the opposite choice — see section 3.5 |
| Self-devised evaluation metric | RAGAS or Trulens | Tested, not assumed: RAGAS was fully implemented twice and could not produce scores for 3 of its 4 required metrics on this local model (section 11). A self-devised metric is explicitly allowed by the brief, covers all four aspects the brief names, and completes a 10-question run in minutes |
| LLM-as-judge scoring | Manual human scoring | Automated and repeatable, and it's the same underlying technique RAGAS itself uses internally — just implemented directly instead of as a dependency |
| `faiss.IndexFlatIP` (exact search) | Approximate index (IVF/HNSW) | At ~400 vectors, brute-force search is already fast; approximate methods trade accuracy for speed this project's scale doesn't need |
| Structural header/footer detection (repetition-based) | Keyword/pattern blocklist | Works identically across any PDF regardless of its actual title/footer text — no per-document tuning needed |
| Chunk overlap (45 words) | No overlap | Prevents a sentence that lands on a chunk boundary from losing meaning in both halves |
| Positional join (`chunks[i]` == FAISS row `i`) | A separate ID-keyed metadata store | Simpler to implement, though it creates a single invariant that must be maintained carefully (see limitations) |

---

## 11. The RAGAS investigation — a documented challenge

This section exists because the assignment's own Final Report structure explicitly asks for
"challenges faced" — and this was a real one, worth documenting honestly rather than quietly
dropping.

**Where it came from:** beyond the assignment brief's own allowance for a self-devised metric,
the instructor separately specified four exact RAGAS metrics with minimum thresholds —
Faithfulness (>90%), Answer Correctness (>80%), Context Recall (>85%), Context Precision
(>80%). Two of those (Answer Correctness, Context Recall) require a ground-truth reference
answer per question, which the self-devised metric never needed.

**What was actually tried:** a separate script (`evaluate_ragas.py`, since removed) used the
real `ragas` package with `langchain_community`'s `ChatOllama`/`OllamaEmbeddings` wrapped in
`ragas.llms.LangchainLLMWrapper`/`ragas.embeddings.LangchainEmbeddingsWrapper`, targeting
`llama3.2`/`nomic-embed-text` — the same models the rest of this project already uses. A set
of 10 hand-written, source-grounded reference answers was drafted to support the ground-truth-
dependent metrics.

**What went wrong, in order:**
1. **A version conflict on install.** `ragas` (0.4.3) internally imports
   `langchain_community.chat_models.vertexai.ChatVertexAI` unconditionally, but the latest
   `langchain-community` (which the newest `langchain` 1.x line pulls in) had already
   restructured that integration out. Fixed by pinning `langchain`/`langchain-community` to an
   older, mutually-compatible line and dropping the separate `langchain-ollama` package in
   favor of `langchain_community`'s own (older) Ollama classes.
2. **Concurrency overload.** RAGAS's `evaluate()` fires many requests in parallel by default —
   fine against a cloud API, but a local single-process Ollama instance can only really handle
   one generation at a time. The result was every job timing out while queued, not because any
   individual call was slow. Fixed with `RunConfig(max_workers=1, timeout=300)`.
3. **Per-call latency, even sequential.** Even with concurrency fixed, individual calls were
   observed taking close to or over 300 seconds each on this CPU-only local setup — RAGAS's
   metrics aren't single calls, they're multi-step chains (e.g. `faithfulness` decomposes the
   answer into claims, then verifies each one separately; `answer_correctness` does that for
   *both* the answer and the reference). A 2-question test run took roughly 55 minutes.
4. **Unreliable structured output from a small local model.** Several jobs failed with
   `OutputParserException` — `llama3.2` (3B parameters) sometimes wrapped its JSON response in
   explanatory prose instead of returning bare JSON, which RAGAS's strict parser rejected. This
   is a model-capability limitation, not a config issue.
5. **Even after every fix, the metrics that mattered most came back unscored.** A 2-question
   run with corrected timeout/concurrency settings completed in ~55 minutes but returned
   `{'faithfulness': 0.625, 'answer_relevancy': 0.697, 'contextual_awareness': nan,
   'response_quality': nan}` — the two custom-defined metrics were entirely unscored.

### The second attempt (on the `UseLangchain` branch) — and why it settled the question

The first attempt had a legitimate weakness as evidence: it used a bare `ChatOllama` with **no
Ollama tuning whatsoever**. It was therefore reloading the model between calls, and — more
importantly — running at Ollama's 2048-token default context, which would truncate RAGAS's long
few-shot metric prompts mid-example. Those are plausible root causes for the malformed output,
and they had never been ruled out.

So RAGAS was reimplemented on top of this branch's service layer, targeting the instructor's
exact four metrics with ten hand-written reference answers, and with every one of those
hypotheses fixed **and verified as actually reaching Ollama**:

| Setting | Hypothesis it tested | Verified |
|---|---|---|
| `keep_alive="30m"` | Model reloading between calls | ✅ |
| `num_ctx=16384` | Prompts truncated at the 2048 default | ✅ `/api/ps` reported `ctx: 16384` |
| `format="json"` | Model emitting prose instead of JSON | ✅ `format='json'` present in params |
| `max_workers=1` | Concurrency pile-up | ✅ |
| `timeout=900` | Premature cutoffs at 180s | ✅ |

**Result of the 2-question run:**

```
faithfulness         no valid scores (all nan)
answer_correctness   no valid scores (all nan)
context_recall       no valid scores (all nan)
context_precision    100.0%  (need >80%)  PASS
```

Failures were `OutputParserException` on jobs 0, 1, 4, 5 (the model emitting `{` and stopping,
or echoing RAGAS's instructions back verbatim) and one `TimeoutError` on `answer_correctness`
that exceeded even the 900-second ceiling. Total runtime ~25 minutes for two questions,
projecting to 4–6 hours for ten.

**The diagnostic detail that settles it:** the *only* metric that produced a score,
`context_precision`, is also the only one of the four that does not require the model to
**generate** complex structured JSON. Every metric that asked `llama3.2` to decompose text into
schema-conforming JSON statements failed; the one that didn't, succeeded — and scored 100%.

That isolates the root cause to the **structured-output capability of a 3B model**, with every
configuration explanation tested and eliminated. It is not a dependency problem, a timeout
problem, a concurrency problem, or a truncation problem.

**What remains untested** is a larger model. A 7–8B model (`llama3.1:8b`, `qwen2.5:7b`) follows
structured-output instructions substantially better, and is the one lever that could plausibly
make RAGAS viable here. That is an honest "future work" item, not a closed door.

**The conclusion:** RAGAS's design assumes a fast, highly capable, cloud-hosted LLM (it was
built and is documented around OpenAI models) — multi-step chained calls and few-shot-heavy
internal prompts are cheap against that kind of backend and expensive against a local,
CPU-bound, small open-source model. This project's own hand-written `bot.py`/`evaluate.py`
avoid the same trap by design: one simple call per step, no chaining, no few-shot-bloated
prompts — which is exactly why they run in minutes on the same hardware where RAGAS did not.
Given that three of the four required metrics cannot produce a score at all on this setup — not
a low score, *no* score — and given the assignment's own text explicitly permits a self-devised
metric as a full alternative (*"Alternatively, you may design your own evaluation metric, but
ensure you clearly document the criteria and methodology"*), the decision was made to keep the
self-devised `evaluate.py` (section 8) as the project's primary, submitted evaluation, and to
remove the RAGAS code from the codebase rather than leave a non-functional script in the
repository. This section is that documentation.

**A distinction worth being precise about in the report:** the two sets of four criteria are not
the same things.

| The assignment brief's four aspects | The instructor's four RAGAS metrics |
|---|---|
| Relevance | Faithfulness (>90%) |
| Accuracy | Answer Correctness (>80%) |
| Contextual Awareness | Context Recall (>85%) |
| Response Quality | Context Precision (>80%) |

`evaluate.py` covers the **left** column completely — those are the criteria the assignment
itself names, and it scores all four. It does not cover the right column, and no claim is made
that it does. The genuine capability gap is that `context_recall` / `context_precision` measure
**retrieval quality independently of the answer**, which an answer-only judge structurally
cannot do — it cannot distinguish "the model reasoned poorly" from "retrieval handed it the
wrong chunks."

---

## 12. Known limitations and honest caveats

- **No follow-up query rewriting.** `retrieve()` embeds only the literal current question
  text. A follow-up like "what about its encoder?" is not rewritten against prior turns
  before searching — so retrieval might miss the intended topic even though the *LLM*
  itself still sees the conversation history when generating the final answer.
- **No relevance-score threshold on retrieval.** All `k=5` results are sent to the LLM
  regardless of how low their similarity score is — there's no cutoff (e.g. "drop anything
  below 0.5") to exclude genuinely unrelated chunks from the prompt.
- **LLM-as-judge self-bias.** The same model (`llama3.2`) both generates and judges the
  answers, which can make it more lenient toward its own reasoning/phrasing style than an
  independent judge would be. The near-ceiling Accuracy/Contextual Awareness averages (5.00,
  100%) should be read with this in mind, not taken as proof of flawless performance.
- **One judge response didn't parse (question 10), even after the brace-repair fallback** —
  stored as `{"parse_error": true, "raw_response": ...}` rather than a score, so the
  reported averages are n=9/10. A small local model occasionally doesn't strictly follow the
  "respond with ONLY JSON" instruction — the same class of issue that affected RAGAS far more
  severely (section 11).
- **Question 10 tests retrieval more than memory.** It was designed to verify the bot does
  *not* have access to question 1 (which falls outside the 4-turn window by then), but
  because the question names "T5" explicitly, retrieval alone finds the right chunks
  regardless of what's in conversation memory — so it answered correctly via retrieval, not
  recall. This is a genuine finding (retrieval and memory are independent mechanisms in this
  architecture), but it means question 10 isn't the clean memory-boundary test it was
  intended to be.
- **RAGAS was attempted twice and is not part of the final submission** — see section 11 for the
  full investigation, including the second attempt with every configuration hypothesis fixed and
  verified. Three of its four required metrics could not produce a score at all on `llama3.2`;
  the self-devised metric (section 8) is the project's actual evaluation methodology.
- **The self-devised metric cannot measure retrieval quality on its own.** Its judge only ever
  sees the final answer, so a bad answer caused by *bad retrieval* and one caused by *bad
  reasoning over good retrieval* score the same way. RAGAS's `context_recall` /
  `context_precision` exist precisely to separate those, and that separation is genuinely absent
  here — worth naming as a limitation rather than glossing over.
- **Requires Ollama running locally**, with both `nomic-embed-text` and `llama3.2` pulled —
  every retrieval and generation call is a live HTTP request to `localhost:11434`, with no
  offline fallback.
- **Scripts must be run from the project root**, not from inside `src/` — the relative paths
  in `config.py` resolve against the current working directory, not the script's location.

---

## 13. Reproduction guide

Run each command from the **project root**:

```bash
# 1. Ingest, clean, and chunk all 5 PDFs
python src/chunking.py
#    → data/processed/chunks.json (527 chunks on UseLangchain, 398 on main)

# 2. Embed every chunk and build the vector store
python src/vectorstore.py
#    → UseLangchain: data/processed/faiss_store/ (index.faiss + index.pkl)
#    → main:         data/processed/embeddings.npy + faiss.index
#    → runs a reload check and a sample query as sanity checks

# 3. Interactive chat (manual testing)
python src/bot.py
#    → type questions, "quit" to exit

# 4. Full 10-question evaluation with LLM-as-judge scoring
python src/evaluate.py
#    → data/eval/results.json
#    → prints per-criterion average scores to the console
```

Prerequisites:
- Ollama running locally with both models pulled — `ollama pull nomic-embed-text` and
  `ollama pull llama3.2`.
- On `main`: `pip install pypdf faiss-cpu numpy requests`
- On `UseLangchain`: `pip install "langchain<0.3" "langchain-community<0.3" "langchain-ollama<0.2" faiss-cpu pypdf`
  — the version pins matter; LangChain's 1.x line moved integration packages around, and mixing
  generations produces `ImportError`s from inside LangChain itself.

Note that step 2 re-runs chunking internally on the `UseLangchain` branch, so it is
self-contained; step 1 is only needed if you want the human-readable `chunks.json` dump.
