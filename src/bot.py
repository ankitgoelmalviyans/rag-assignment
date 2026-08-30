"""
Retrieval + conversational chat stage of the RAG pipeline.

Combines FAISS-retrieved chunks with the last MEMORY_TURNS turns of
conversation memory into a prompt sent to llama3.2, and returns the
generated answer. ChatSession owns the memory; everything else here is
a stateless helper function.

WHAT HAPPENS WHEN SOMEONE ASKS A QUESTION (the whole file in one picture):

    question
      -> retrieve()        embed the question, ask FAISS for the 5 closest
                           chunks. Uses the EMBEDDING model + pure maths.
      -> format_context()  glue those 5 chunks into one labelled text block
      -> ChatSession.ask() build the prompt: system instructions + up to 4
                           previous turns + context + the new question
      -> call_llama()      ONE HTTP request to Ollama. Uses the CHAT model.
      -> answer

Two different models are involved and they never talk to each other:
  * nomic-embed-text turns text into numbers so similar text can be FOUND
  * llama3.2 reads the text that was found and WRITES the answer
FAISS itself runs no model at all -- it is pure numeric comparison.
"""

# --- standard library (ships with Python, nothing to install) ---
import json          # read/write JSON files, like System.Text.Json in .NET
import time          # time.sleep(seconds) -- used for the retry pause below
from collections import deque   # a list that can have a maximum length

# --- third-party packages (installed with pip) ---
import faiss         # Facebook AI Similarity Search: the vector index
import requests      # makes HTTP calls, roughly .NET's HttpClient

# --- our own modules (the other .py files in this folder) ---
from config import (
    CHAT_MODEL,
    CHUNKS_FILE,
    INDEX_FILE,
    KEEP_ALIVE,
    NUM_CTX,
    OLLAMA_BASE_URL,
)
from vectorstore import embed_texts   # reuse the SAME embedding function
                                       # the chunks were indexed with

# How many chunks to retrieve per question. Purely our choice -- nothing in
# FAISS or the model requires 5.
TOP_K = 5

# How many past (question, answer) pairs the bot remembers. The assignment
# asks for the last 4 interactions.
MEMORY_TURNS = 4

# Seconds to wait before retrying a failed call -- see call_llama() for why a
# retry must not fire immediately.
RETRY_BACKOFF = 30

# Standing instructions sent to the model on EVERY call. This is how you tell
# a chat model how to behave: answer only from what we give it, admit when it
# doesn't know (rather than inventing an answer), and cite its source.
SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a set of research papers "
    "(Transformer, BERT, GPT-3, RoBERTa, T5). Answer ONLY using the provided context. "
    "If the context doesn't contain enough information to answer, say so explicitly "
    "instead of guessing. When possible, cite the source as [doc_id, page X]."
)


def load_chunks_and_index():
    """Load the chunk metadata and FAISS index built by vectorstore.py.

    Returns TWO values at once (Python can do that; C# would need a tuple or
    out-parameters). Callers unpack them:  chunks, index = load_chunks_and_index()
    """
    # "with open(...) as f" auto-closes the file at the end of the block,
    # exactly like C#'s "using (var f = ...)".
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)          # JSON text -> Python list of dicts

    index = faiss.read_index(str(INDEX_FILE))   # rebuild the search index
                                                 # from the saved file
    return chunks, index


def retrieve(query: str, chunks, index, k: int = TOP_K):
    """Embed a query and return the top-k most similar chunks with their scores.

    This is the "R" in RAG. Note there is NO language model here -- finding
    the right chunks is a maths problem, not a reading-comprehension one.
    """
    # 1. Turn the question into a vector (a list of 768 numbers). We pass a
    #    list containing one string, because embed_texts() handles batches.
    query_vector = embed_texts([query])

    # 2. Ask FAISS for the k closest stored vectors. It returns two arrays:
    #       scores      -> how similar each match is (higher = closer)
    #       row_indices -> WHICH ROW of the index matched, e.g. 142
    #    Both are "2D" because you could search several questions at once;
    #    we only ever search one, so we read element [0] of each below.
    scores, row_indices = index.search(query_vector, k)

    # 3. FAISS only knows row numbers -- it never stores the text. Look each
    #    row number up in chunks to get the actual chunk back. This works
    #    ONLY because chunks[i] and index row i were built in the same order.
    results = []
    for score, row in zip(scores[0], row_indices[0]):   # zip pairs them up
        chunk_with_score = dict(chunks[row])   # dict(...) makes a COPY, so we
                                                # don't modify the shared list
        chunk_with_score["score"] = float(score)
        results.append(chunk_with_score)       # .append is C#'s List.Add

    return results


def format_context(retrieved_chunks) -> str:
    """Turn retrieved chunks into a labeled context block for the prompt.

    Output looks like:

        [Source: 1706.03762, page 4]
        <chunk text>

        [Source: 1810.04805, page 5]
        <chunk text>

    By the time the model sees this it is just text -- it has no idea five
    separate chunks were retrieved.
    """
    blocks = []
    for chunk in retrieved_chunks:
        # An f-string inserts values into text, like C#'s $"...{value}..."
        label = f"[Source: {chunk['doc_id']}, page {chunk['page_start']}]"
        blocks.append(f"{label}\n{chunk['text']}")     # \n is a line break

    # "separator".join(list) glues a list of strings together with that
    # separator between them -- C#'s string.Join("\n\n", blocks).
    return "\n\n".join(blocks)


def call_llama(messages, timeout: int = 300, max_retries: int = 2, json_mode: bool = False) -> str:
    """Send a messages list to llama3.2 via Ollama's /api/chat and return the reply text.

    THIS IS THE ONLY PLACE THE PROJECT TALKS TO THE LANGUAGE MODEL.

    There is no magic to "calling an LLM": Ollama runs a small web server on
    your machine (localhost:11434), and we POST some JSON to it and read the
    JSON that comes back. It is an ordinary HTTP call.

    `messages` is the standard chat format used by almost every chat-model
    API -- a list of dicts, each with a role and some text:

        [{"role": "system",    "content": "you are a helpful assistant"},
         {"role": "user",      "content": "what is BERT?"},
         {"role": "assistant", "content": "BERT is..."},      <- past reply
         {"role": "user",      "content": "how is it trained?"}]

    The model reads the whole list every time. It has no memory of its own:
    "conversation memory" just means we resend the earlier turns.

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
    # The request body. requests turns this dict into JSON for us.
    payload = {
        "model": CHAT_MODEL,                  # which model Ollama should use
        "messages": messages,                 # the conversation so far
        "stream": False,                      # wait for the whole reply
                                              # instead of token-by-token
        "options": {"num_ctx": NUM_CTX},      # model settings live in here
        "keep_alive": KEEP_ALIVE,
    }
    if json_mode:
        payload["format"] = "json"            # add a key to an existing dict

    # Try up to max_retries times. range(1, 3) gives 1, 2 -- the upper bound
    # is excluded in Python.
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,        # `json=` sets the body AND the
                                     # Content-Type header for us
                timeout=timeout,     # seconds to wait before giving up
            )
            response.raise_for_status()   # throw if the status is 4xx/5xx,
                                          # like EnsureSuccessStatusCode()

            # Ollama replies with a JSON object shaped like:
            #   {"message": {"role": "assistant", "content": "the answer"}, ...}
            # so we dig out the text we actually want and return it.
            return response.json()["message"]["content"]

        # "except X as error" is Python's "catch (X error)".
        except (requests.exceptions.ReadTimeout, requests.exceptions.HTTPError) as error:
            if attempt == max_retries:
                raise            # out of attempts: let the caller deal with it

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

    Why a class rather than plain functions: a conversation has STATE (the
    history), and a class keeps that state together with the method that
    uses it. Same reasoning as any C# class with a private field.
    """

    def __init__(self, chunks, index):
        """The constructor. Runs when you write ChatSession(chunks, index).

        `self` is this object -- Python's `this`, except it must be written
        out as the first parameter of every method.
        """
        self.chunks = chunks
        self.index = index

        # deque(maxlen=4) is the entire memory implementation. It behaves like
        # a list, except that appending a 5th item automatically discards the
        # oldest one. No manual "if too many, remove the first" code needed.
        self.history = deque(maxlen=MEMORY_TURNS)

    def ask(self, question: str) -> str:
        """Answer one question, then remember it."""
        # 1. Find relevant chunks for THIS question. Note memory plays no part
        #    in retrieval -- we search using the question's own words only.
        retrieved = retrieve(question, self.chunks, self.index)
        context = format_context(retrieved)

        # A debug line so the memory window is visible while chatting, rather
        # than having to trust the model's own account of what it remembers.
        print(f"[memory: {len(self.history)} turn(s) currently held]")

        # 2. Build the messages list. Standing instructions go first.
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 3. Replay past turns as real user/assistant messages, so the model
        #    sees the actual back-and-forth rather than a summary of it.
        for turn in self.history:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})

        # 4. Finally the new question, with its retrieved context attached.
        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        })

        # 5. ONE call to the model for the whole thing.
        answer = call_llama(messages)

        # 6. Remember this turn. If history already held 4, the oldest is
        #    silently dropped here by the deque.
        self.history.append({"question": question, "answer": answer})
        return answer


# This block runs ONLY when you execute "python src/bot.py" directly. If
# another file does "from bot import call_llama", it is skipped. C# separates
# these with a dedicated Main(); Python does it with this if-statement.
if __name__ == "__main__":
    chunks, index = load_chunks_and_index()
    session = ChatSession(chunks, index)      # one session = one conversation

    print("RAG chat ready. Type a question (or 'quit' to exit).\n")

    while True:                                # loop until we break out
        question = input("You: ").strip()       # input() waits for typing;
                                                # .strip() trims whitespace
        if question.lower() in ("quit", "exit"):
            break                               # leave the loop -> program ends
        if not question:                        # empty line: an empty string
            continue                            # is falsy in Python
        answer = session.ask(question)
        print(f"\nBot: {answer}\n")
