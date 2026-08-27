"""
Evaluation stage of the RAG pipeline -- self-devised metric.

Runs the 10 predefined test questions through the bot as ONE continuous
conversation (so the 4-turn memory window fills/evicts exactly as it would
in real use), then scores each answer with an LLM-as-judge: a second call
to llama3.2, given the question, the retrieved source chunks, the answer,
and the conversation history actually available at that turn, asked to
rate the answer on the four criteria the assignment specifies (Relevance,
Accuracy, Contextual Awareness, Response Quality) on a 1-5 scale with a
short justification each.

This is a self-devised metric (the assignment's documented alternative to
RAGAS/Trulens) -- the rubric and methodology live in JUDGE_SYSTEM_PROMPT
below rather than being delegated to an external framework. A real attempt
was made to use RAGAS first; it was abandoned after RAGAS's chained,
few-shot-heavy internal prompts proved unworkable against local CPU
inference (roughly an hour for just 2 questions, and the two custom
AspectCritic metrics standing in for Contextual Awareness/Response Quality
came back entirely unscored ("nan") even after fixing timeout/concurrency
config) -- documented as a challenge encountered in TECHNICAL_DETAILS.md.
"""

import json

from bot import ChatSession, call_llama, load_chunks_and_index, retrieve
from config import QUESTIONS_FILE, RESULTS_FILE

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

{
  "relevance": {"score": <1-5>, "justification": "<one sentence>"},
  "accuracy": {"score": <1-5>, "justification": "<one sentence>"},
  "contextual_awareness": {"score": <1-5>, "justification": "<one sentence>"},
  "response_quality": {"score": <1-5>, "justification": "<one sentence>"}
}
"""


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def run_interactions(chunks, index, questions):
    """
    Ask all 10 questions through ONE ChatSession, in order, so the 4-turn
    memory window fills and evicts exactly as it would in a real
    conversation. Records the retrieved context and the exact memory
    snapshot available at the moment each question was asked, since the
    judge needs both to score contextual_awareness fairly.
    """
    session = ChatSession(chunks, index)
    records = []

    for item in questions:
        history_snapshot = [dict(turn) for turn in session.history]
        retrieved = retrieve(item["question"], chunks, index)
        answer = session.ask(item["question"])

        records.append({
            "question": item["question"],
            "doc_focus": item.get("doc_focus", ""),
            "history_available": history_snapshot,
            "retrieved": [
                {
                    "chunk_id": r["chunk_id"],
                    "doc_id": r["doc_id"],
                    "page_start": r["page_start"],
                    "score": round(r["score"], 4),
                    "text": r["text"],
                }
                for r in retrieved
            ],
            "answer": answer,
        })
        print(f"  asked: {item['question'][:60]}... -- answer length {len(answer)} chars")

    return records


def build_judge_messages(record):
    context_block = "\n\n".join(
        f"[{c['doc_id']}, page {c['page_start']}]\n{c['text']}"
        for c in record["retrieved"]
    )
    history_block = "\n".join(
        f"Q: {turn['question']}\nA: {turn['answer']}"
        for turn in record["history_available"]
    ) or "(no prior turns -- this was asked with an empty memory window)"

    user_content = (
        f"RETRIEVED CONTEXT:\n{context_block}\n\n"
        f"CONVERSATION HISTORY AVAILABLE TO THE BOT AT THIS TURN:\n{history_block}\n\n"
        f"QUESTION: {record['question']}\n\n"
        f"CANDIDATE ANSWER:\n{record['answer']}"
    )

    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def judge_answer(record):
    """Call the LLM as an impartial judge and parse its JSON scoring."""
    raw = call_llama(build_judge_messages(record))

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

    # small local models occasionally stop one token early and omit the
    # final closing brace(s) -- repair that specific case before giving up
    missing_braces = cleaned.count("{") - cleaned.count("}")
    if missing_braces > 0:
        try:
            return json.loads(cleaned + "}" * missing_braces)
        except json.JSONDecodeError:
            pass

    print(f"  [warn] could not parse judge output for: {record['question'][:60]}...")
    return {"parse_error": True, "raw_response": raw}


def summarize(records):
    criteria = ["relevance", "accuracy", "contextual_awareness", "response_quality"]
    totals = {c: [] for c in criteria}

    for record in records:
        judge = record["judge"]
        if judge.get("parse_error"):
            continue
        for c in criteria:
            totals[c].append(judge[c]["score"])

    print("\n--- Evaluation summary ---")
    for c in criteria:
        scores = totals[c]
        avg = sum(scores) / len(scores) if scores else float("nan")
        print(f"  {c:22s} avg = {avg:.2f}  (n={len(scores)})")


if __name__ == "__main__":
    chunks, index = load_chunks_and_index()
    questions = load_questions()

    print(f"Running {len(questions)} questions through one continuous ChatSession...")
    records = run_interactions(chunks, index, questions)

    print("\nScoring answers with LLM-as-judge...")
    for record in records:
        record["judge"] = judge_answer(record)
        print(f"  scored: {record['question'][:60]}...")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\nWrote results to {RESULTS_FILE}")

    summarize(records)
