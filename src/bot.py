"""
Retrieval + conversational chat stage of the RAG pipeline (LangChain branch).

Combines FAISS-retrieved chunks with the last MEMORY_TURNS turns of
conversation memory into a prompt sent to llama3.2, and returns the
generated answer. ChatSession owns the memory; everything else here is a
thin wrapper over langchain_service.

What LangChain replaced compared to `main`:
  - the manual embed -> index.search -> map-row-number-back-to-chunk dance is
    now one similarity_search_with_score() call returning Documents that
    carry their own text and metadata
  - the hand-built messages list is now a ChatPromptTemplate with a
    MessagesPlaceholder for history
  - the raw requests.post + retry loop is now the ChatOllama client

What stayed the same on purpose: the memory window is still a
collections.deque(maxlen=4). LangChain's memory abstractions add indirection
without changing behaviour here, and the assignment asks specifically for the
last 4 interactions -- a bounded deque states that directly.
"""

from collections import deque

from config import MEMORY_TURNS, TOP_K, VECTORSTORE_DIR
from langchain_service import (
    build_answer_chain,
    format_context,
    load_vectorstore,
    search_with_scores,
    to_history_messages,
)


def load_store():
    """Load the FAISS store built by vectorstore.py."""
    return load_vectorstore(VECTORSTORE_DIR)


def retrieve(query: str, store, k: int = TOP_K):
    """
    Return the top-k most similar chunks as (Document, score) pairs.

    Always returns exactly k results -- there is no relevance threshold, so a
    weak match is still passed to the LLM. Kept as-is to match `main`.
    """
    return search_with_scores(store, query, k=k)


def retrieved_to_records(scored_documents) -> list[dict]:
    """Flatten (Document, score) pairs into plain dicts for results.json."""
    return [
        {
            "chunk_id": document.metadata.get("chunk_id"),
            "doc_id": document.metadata.get("doc_id"),
            "page": document.metadata.get("page"),
            "score": round(float(score), 4),
            "text": document.page_content,
        }
        for document, score in scored_documents
    ]


class ChatSession:
    """
    Holds one conversation's memory: the last MEMORY_TURNS (question, answer)
    pairs. Each ask() call retrieves fresh context for the new question,
    builds a prompt including prior turns, sends it to llama3.2, then appends
    the new turn to memory -- automatically dropping the oldest turn once more
    than MEMORY_TURNS have accumulated.
    """

    def __init__(self, store, temperature: float = 0.0):
        self.store = store
        self.history = deque(maxlen=MEMORY_TURNS)
        self.chain = build_answer_chain(temperature=temperature)

    def ask(self, question: str) -> str:
        retrieved = retrieve(question, self.store)
        context = format_context(retrieved)

        print(f"[memory: {len(self.history)} turn(s) currently held]")

        answer = self.chain.invoke({
            "context": context,
            "question": question,
            "history": to_history_messages(self.history),
        })

        self.history.append({"question": question, "answer": answer})
        return answer


if __name__ == "__main__":
    store = load_store()
    session = ChatSession(store)

    print("RAG chat ready. Type a question (or 'quit' to exit).\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue
        answer = session.ask(question)
        print(f"\nBot: {answer}\n")
