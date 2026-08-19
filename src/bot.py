"""
Retrieval + conversational chat stage of the RAG pipeline.

Combines FAISS-retrieved chunks with the last MEMORY_TURNS turns of
conversation memory into a prompt sent to llama3.2, and returns the
generated answer. ChatSession owns the memory; everything else here is
a stateless helper function.
"""

import json
from collections import deque

import faiss
import requests

from config import CHAT_MODEL, CHUNKS_FILE, INDEX_FILE, OLLAMA_BASE_URL
from vectorstore import embed_texts

TOP_K = 5
MEMORY_TURNS = 4

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


def call_llama(messages) -> str:
    """Send a messages list to llama3.2 via Ollama's /api/chat and return the reply text."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": CHAT_MODEL, "messages": messages, "stream": False},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


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
