"""
Centralized LangChain layer for the whole RAG pipeline.

Every other module (ingestion, chunking, vectorstore, bot, evaluate) calls
into this file instead of importing LangChain directly. That keeps all
framework-specific code in one place: if a LangChain class is renamed,
deprecated, or reconfigured, exactly one file changes.

Three things this module is responsible for:

1. **Model access.** ChatOllama / OllamaEmbeddings instances are built once
   and cached (see _LLM_CACHE), so a run that makes dozens of calls reuses a
   single configured client rather than constructing a new one per call.
2. **Efficient local Ollama settings.** keep_alive holds the model in memory
   between calls, num_ctx is raised above Ollama's 2048-token default so long
   prompts aren't silently truncated, and the judge runs in Ollama's native
   JSON mode so its output is valid JSON by construction.
3. **The prompts and chains themselves** -- the RAG answer prompt and the
   evaluation judge rubric, expressed as LangChain ChatPromptTemplates.

Design note: the judge deliberately stays a SINGLE LLM call scoring all four
criteria at once, exactly like the non-LangChain implementation on `main`.
Frameworks like RAGAS decompose each metric into chained sub-calls, which is
cheap against a cloud API and impractically slow against local CPU inference.
Using LangChain for composition while keeping the one-call design is what
makes this branch produce like-for-like results at like-for-like speed.
"""

import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# The dedicated Ollama integration package -- not the generic
# langchain_community wrappers. langchain-ollama talks to the local Ollama
# server through Ollama's own Python client, and exposes Ollama-specific
# options (keep_alive, num_ctx, seed, native JSON mode) as first-class
# parameters.
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHAT_MODEL,
    EMBED_BATCH_SIZE,
    EMBED_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_NUM_CTX,
    OLLAMA_TIMEOUT,
    OVERLAP_WORDS,
    TARGET_WORDS,
    TOP_K,
    VECTORSTORE_DIR,
)

# ---------------------------------------------------------------------------
# 1. Model access (cached)
# ---------------------------------------------------------------------------

_LLM_CACHE: dict[tuple, ChatOllama] = {}
_EMBEDDINGS: OllamaEmbeddings | None = None


def get_llm(
    temperature: float = 0.0,
    json_mode: bool = False,
    num_ctx: int | None = None,
    timeout: int | None = None,
) -> ChatOllama:
    """
    Return a configured ChatOllama, reusing an existing instance when the same
    settings are requested again.

    Args:
        temperature: 0.0 gives the most repeatable output, which is what an
            evaluation judge wants. Raise it for more varied chat answers.
        json_mode: when True, uses Ollama's native JSON mode (`format="json"`),
            which constrains decoding so the reply is always parseable JSON.
            This removes the "model wrapped its JSON in prose" failure mode
            entirely, rather than repairing it after the fact.

    Ollama-specific settings applied to every instance:
      keep_alive     -- hold the model in memory between calls, so a long run
                        doesn't pay the model-load cost on every request
      num_ctx        -- raise the context window above Ollama's 2048 default
      seed           -- fixed seed; with temperature=0 this makes repeated
                        runs reproducible
      client_kwargs  -- passed straight through to Ollama's own Python client,
                        which is where the per-request timeout lives
    """
    num_ctx = num_ctx if num_ctx is not None else OLLAMA_NUM_CTX
    timeout = timeout if timeout is not None else OLLAMA_TIMEOUT

    key = (temperature, json_mode, num_ctx, timeout)
    if key not in _LLM_CACHE:
        kwargs = {
            "model": CHAT_MODEL,
            "base_url": OLLAMA_BASE_URL,
            "temperature": temperature,
            "num_ctx": num_ctx,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "seed": 0,
            "client_kwargs": {"timeout": timeout},
        }
        if json_mode:
            kwargs["format"] = "json"
        _LLM_CACHE[key] = ChatOllama(**kwargs)
    return _LLM_CACHE[key]


def get_embeddings() -> OllamaEmbeddings:
    """Return the shared OllamaEmbeddings instance (built once per process)."""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = OllamaEmbeddings(
            model=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
            client_kwargs={"timeout": OLLAMA_TIMEOUT},
        )
    return _EMBEDDINGS


# ---------------------------------------------------------------------------
# 2. Document loading
# ---------------------------------------------------------------------------


def load_pdf_pages(pdf_path: Path) -> list[Document]:
    """
    Load one PDF as a list of LangChain Documents, one per page.

    PyPDFLoader already attaches {"source": <path>, "page": <0-based index>}
    to each Document's metadata. Page numbers are normalized to 1-based here
    so they read naturally in a citation.
    """
    pages = PyPDFLoader(str(pdf_path)).load()
    for page in pages:
        page.metadata["page"] = page.metadata.get("page", 0) + 1
        page.metadata["doc_id"] = pdf_path.stem
        page.metadata["source_file"] = pdf_path.name
    return pages


# ---------------------------------------------------------------------------
# 3. Splitting
# ---------------------------------------------------------------------------


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Build the splitter used for chunking.

    `length_function` counts WORDS rather than characters, so chunk_size and
    chunk_overlap stay in the same units the hand-written implementation used
    (TARGET_WORDS / OVERLAP_WORDS) and produce comparable chunks.

    The separator list is ordered most- to least-preferred: split on paragraph
    breaks first, then single newlines, then sentence ends, then whitespace.
    The splitter only falls to a lower-priority separator when a piece is
    still over-length -- which is the same "paragraph first, sentences as a
    fallback" behaviour the original build_chunks() implemented by hand.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=TARGET_WORDS,
        chunk_overlap=OVERLAP_WORDS,
        length_function=lambda text: len(text.split()),
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def split_documents(documents: list[Document]) -> list[Document]:
    """Split page-level Documents into chunk-level Documents."""
    return get_text_splitter().split_documents(documents)


# ---------------------------------------------------------------------------
# 4. Vector store
# ---------------------------------------------------------------------------


def build_vectorstore(documents: list[Document], batch_size: int = EMBED_BATCH_SIZE) -> FAISS:
    """
    Embed every Document and build a FAISS store over the result.

    MAX_INNER_PRODUCT reproduces the hand-written setup (faiss.IndexFlatIP),
    so similarity scores stay directly comparable to the non-LangChain
    implementation.

    No explicit normalization step is needed: Ollama's /api/embed returns
    already-unit-length vectors from nomic-embed-text (measured: L2 norm
    1.000000), and for unit vectors an inner product IS cosine similarity.
    Passing normalize_L2=True here would additionally be ignored -- LangChain
    only applies it to EUCLIDEAN_DISTANCE stores and warns otherwise. The
    normalize() call in the hand-written version was therefore defensive
    rather than load-bearing.

    Built in batches rather than one FAISS.from_documents(all_documents)
    call: OllamaEmbeddings sends everything it is handed in a single HTTP
    request, and ~500 chunks at once overruns the client timeout on CPU-only
    inference. The first batch creates the store; the rest are added to it.

    Unlike the hand-written version, the resulting object owns the vectors
    AND the document text/metadata together, so there is no separate
    chunks.json to keep positionally in sync with the index.
    """
    if not documents:
        raise ValueError("build_vectorstore() needs at least one document")

    first, rest = documents[:batch_size], documents[batch_size:]
    store = FAISS.from_documents(
        first,
        get_embeddings(),
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    print(f"  embedded {len(first)}/{len(documents)}")

    for start in range(0, len(rest), batch_size):
        batch = rest[start:start + batch_size]
        store.add_documents(batch)
        print(f"  embedded {len(first) + start + len(batch)}/{len(documents)}")

    return store


def save_vectorstore(store: FAISS, folder: Path = VECTORSTORE_DIR) -> None:
    """Persist the store (writes index.faiss + index.pkl into the folder)."""
    folder.mkdir(parents=True, exist_ok=True)
    store.save_local(str(folder))


def load_vectorstore(folder: Path = VECTORSTORE_DIR) -> FAISS:
    """
    Load a previously saved store.

    allow_dangerous_deserialization is required because the docstore half is
    a pickle; it is safe here because this process wrote that file itself.
    """
    return FAISS.load_local(
        str(folder),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def search_with_scores(store: FAISS, query: str, k: int = TOP_K):
    """
    Return the k most similar chunks as (Document, score) pairs.

    The Document carries its own text and metadata, so unlike the row-number
    lookup the hand-written version needed, nothing has to be mapped back to
    a separate metadata file.
    """
    return store.similarity_search_with_score(query, k=k)


# ---------------------------------------------------------------------------
# 5. Prompts
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a set of research papers "
    "(Transformer, BERT, GPT-3, RoBERTa, T5). Answer ONLY using the provided context. "
    "If the context doesn't contain enough information to answer, say so explicitly "
    "instead of guessing. When possible, cite the source as [doc_id, page X]."
)

JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator scoring answers from a RAG (Retrieval-Augmented
Generation) chatbot that answers questions about NLP research papers using only
retrieved excerpts as its source of truth.

Score the CANDIDATE ANSWER on four criteria, each from 1 (poor) to 5 (excellent):

- relevance: Does the answer directly address what was actually asked?
- accuracy: Is the answer factually consistent with the RETRIEVED CONTEXT
  provided below? Judge against that context only, not your own general
  knowledge -- an answer should be marked down if it states something the
  retrieved context does not support, even if it happens to be true in general.
- contextual_awareness: If the question depended on earlier conversation turns
  (e.g. used a pronoun like "it"/"that" referring to a prior topic), did the
  answer correctly use the CONVERSATION HISTORY to resolve it? If the question
  was standalone, score this on whether the answer stayed consistent with
  earlier turns and did not contradict them.
- response_quality: Is the answer clearly written, well-organized, and
  appropriately detailed (not needlessly verbose, not too terse)?

Respond with ONLY a JSON object, no markdown fences, no extra commentary, in
exactly this shape:

{{
  "relevance": {{"score": <1-5>, "justification": "<one sentence>"}},
  "accuracy": {{"score": <1-5>, "justification": "<one sentence>"}},
  "contextual_awareness": {{"score": <1-5>, "justification": "<one sentence>"}},
  "response_quality": {{"score": <1-5>, "justification": "<one sentence>"}}
}}
"""


def build_rag_prompt() -> ChatPromptTemplate:
    """
    The chat prompt: fixed system instructions, then up to MEMORY_TURNS of
    prior turns replayed via MessagesPlaceholder, then the current question
    with its retrieved context.
    """
    return ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])


def build_judge_prompt() -> ChatPromptTemplate:
    """The evaluation rubric prompt -- one call, all four criteria."""
    return ChatPromptTemplate.from_messages([
        ("system", JUDGE_SYSTEM_PROMPT),
        ("human",
         "RETRIEVED CONTEXT:\n{context}\n\n"
         "CONVERSATION HISTORY AVAILABLE TO THE BOT AT THIS TURN:\n{history}\n\n"
         "QUESTION: {question}\n\n"
         "CANDIDATE ANSWER:\n{answer}"),
    ])


# ---------------------------------------------------------------------------
# 6. Chains
# ---------------------------------------------------------------------------


def build_answer_chain(temperature: float = 0.0):
    """
    prompt | llm | parser -- LangChain Expression Language ("LCEL").

    The `|` operator composes the three steps into one runnable: invoking it
    with {"context", "question", "history"} formats the prompt, sends it to
    Ollama, and returns the reply as a plain string.
    """
    return build_rag_prompt() | get_llm(temperature=temperature) | StrOutputParser()


def build_judge_chain():
    """
    The judge chain. temperature=0 for repeatable scoring, json_mode=True so
    Ollama constrains decoding to valid JSON -- the parse-repair fallback the
    hand-written version needed is unnecessary here.
    """
    return build_judge_prompt() | get_llm(temperature=0.0, json_mode=True) | StrOutputParser()


def to_history_messages(turns) -> list:
    """
    Convert stored (question, answer) turns into the alternating
    Human/AI message objects MessagesPlaceholder expects.
    """
    messages = []
    for turn in turns:
        messages.append(HumanMessage(content=turn["question"]))
        messages.append(AIMessage(content=turn["answer"]))
    return messages


def parse_judge_json(raw: str) -> dict:
    """
    Parse the judge's reply.

    With json_mode enabled this should always succeed on the first attempt;
    the fence-stripping and brace-repair fallbacks are kept as cheap
    insurance, and a failure is reported rather than crashing the run.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    missing_braces = cleaned.count("{") - cleaned.count("}")
    if missing_braces > 0:
        try:
            return json.loads(cleaned + "}" * missing_braces)
        except json.JSONDecodeError:
            pass

    return {"parse_error": True, "raw_response": raw}


def format_context(scored_documents) -> str:
    """
    Turn (Document, score) pairs into the labeled context block the prompt
    expects -- same shape as the hand-written format_context().
    """
    blocks = []
    for document, _score in scored_documents:
        label = f"[Source: {document.metadata.get('doc_id', '?')}, page {document.metadata.get('page', '?')}]"
        blocks.append(f"{label}\n{document.page_content}")
    return "\n\n".join(blocks)
