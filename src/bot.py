"""
Retrieval + conversational chat stage of the RAG pipeline.

Combines FAISS-retrieved chunks with the last MEMORY_TURNS turns of
conversation memory into a prompt sent to llama3.2, and returns the
generated answer. ChatSession owns the memory; everything else here is
a stateless helper function.
"""

import json
import time
from collections import deque

import faiss
import requests

from config import (
    CHAT_MODEL,
    CHUNKS_FILE,
    INDEX_FILE,
    KEEP_ALIVE,
    NUM_CTX,
    OLLAMA_BASE_URL,
)
from vectorstore import embed_texts

TOP_K = 5
MEMORY_TURNS = 4

# Seconds to wait before retrying a failed call -- see call_llama() for why a
# retry must not fire immediately.
RETRY_BACKOFF = 30

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a set of research papers "
    "(Transformer, BERT, GPT-3, RoBERTa, T5). Answer ONLY using the provided context. "
    "If the context doesn't contain enough information to answer, say so explicitly "
    "instead of guessing. When possible, cite the source as [doc_id, page X]."
)


def load_chunks_and_index():
    """Load the chunk metadata and FAISS index built by vectorstore.py."""
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    index = faiss.read_index(str(INDEX_FILE))
    return chunks, index


def retrieve(query: str, chunks, index, k: int = TOP_K):
    """Embed a query and return the top-k most similar chunks with their scores."""
    query_vector = embed_texts([query])
    scores, row_indices = index.search(query_vector, k)

    results = []
    for score, row in zip(scores[0], row_indices[0]):
        chunk_with_score = dict(chunks[row])
        chunk_with_score["score"] = float(score)
        results.append(chunk_with_score)
    return results


def format_context(retrieved_chunks) -> str:
    """Turn retrieved chunks into a labeled context block for the prompt."""
    blocks = []
    for chunk in retrieved_chunks:
        label = f"[Source: {chunk['doc_id']}, page {chunk['page_start']}]"
        blocks.append(f"{label}\n{chunk['text']}")
    return "\n\n".join(blocks)


def call_llama(messages, timeout: int = 300, max_retries: int = 2, json_mode: bool = False) -> str:
    """Send a messages list to llama3.2 via Ollama's /api/chat and return the reply text.

    Retries once on a read timeout -- later turns carry a larger prompt (full
    memory history + retrieved context), which can occasionally exceed a
    single attempt's timeout on a slow/loaded local machine.

    Args:
        json_mode: when True, asks Ollama to constrain decoding so the reply is
            always valid JSON. Used by the evaluation judge, whose reply has to
            be machine-parsed; normal chat answers are prose and leave it off.

    Two Ollama options are set explicitly on every call:

      num_ctx     Ollama's default context window is 2048 tokens. A prompt here
                  carries 5 retrieved chunks (~1,400 tokens) plus the system
                  prompt, the question, and up to 4 turns of memory -- roughly
                  2,100 tokens once memory fills, i.e. over the default. Ollama
                  truncates silently rather than erroring, so leaving this unset
                  would quietly discard the oldest part of the prompt (the
                  earliest memory turns) on later questions.
      keep_alive  How long the model stays resident in memory between calls.
                  An evaluation run makes ~20 calls; without this, Ollama can
                  unload and reload llama3.2 repeatedly.
    """
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": NUM_CTX},
        "keep_alive": KEEP_ALIVE,
    }
    if json_mode:
        payload["format"] = "json"

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except (requests.exceptions.ReadTimeout, requests.exceptions.HTTPError) as error:
            if attempt == max_retries:
                raise
            # A read timeout is client-side only: Ollama keeps working on the
            # request we gave up waiting for. Retrying immediately therefore
            # puts a SECOND generation in flight alongside the first, and two
            # concurrent num_ctx-sized KV caches is enough to make the server
            # return a 500. Pausing before the retry lets the original finish
            # and the memory be released first.
            print(f"  call_llama failed (attempt {attempt}/{max_retries}): "
                  f"{type(error).__name__} -- waiting {RETRY_BACKOFF}s before retry")
            time.sleep(RETRY_BACKOFF)


class ChatSession:
    """
    Holds one conversation's memory: the last MEMORY_TURNS (question, answer)
    pairs. Each ask() call retrieves fresh context for the new question,
    builds a prompt including prior turns, sends it to llama3.2, then
    appends the new turn to memory -- automatically dropping the oldest
    turn once more than MEMORY_TURNS have accumulated.
    """

    def __init__(self, chunks, index):
        self.chunks = chunks
        self.index = index
        self.history = deque(maxlen=MEMORY_TURNS)

    def ask(self, question: str) -> str:
        retrieved = retrieve(question, self.chunks, self.index)
        context = format_context(retrieved)

        print(f"[memory: {len(self.history)} turn(s) currently held]")
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for turn in self.history:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})

        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        })

        answer = call_llama(messages)
        self.history.append({"question": question, "answer": answer})
        return answer


if __name__ == "__main__":
    chunks, index = load_chunks_and_index()
    session = ChatSession(chunks, index)

    print("RAG chat ready. Type a question (or 'quit' to exit).\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue
        answer = session.ask(question)
        print(f"\nBot: {answer}\n")
