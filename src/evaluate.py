"""
Evaluation stage of the RAG pipeline (LangChain branch) -- self-devised metric.

Runs the 10 predefined test questions through the bot as ONE continuous
conversation (so the 4-turn memory window fills/evicts exactly as it would in
real use), then scores each answer with an LLM-as-judge: a second call to
llama3.2, given the question, the retrieved source chunks, the answer, and
the conversation history actually available at that turn, asked to rate the
answer on the four criteria the assignment specifies (Relevance, Accuracy,
Contextual Awareness, Response Quality) on a 1-5 scale with a short
justification each.

This is deliberately a LIKE-FOR-LIKE port of the non-LangChain implementation
on `main`: same rubric text, same four criteria, same 1-5 scale, same
single-call-per-judgment design, same results.json shape. What changed is
only the plumbing -- the prompt is a ChatPromptTemplate, the call goes
through ChatOllama from the dedicated langchain-ollama package, and the two
are composed with LCEL (`prompt | llm | parser`).

Two things the LangChain layer buys us over the hand-written version:
  - Ollama's native JSON mode constrains the judge's decoding to valid JSON,
    which removes the parse failure that left one question unscored on `main`
  - keep_alive holds the model in memory across the ~20 calls a full run
    makes, instead of paying the load cost repeatedly
"""

import json

from bot import ChatSession, load_store, retrieve, retrieved_to_records
from config import QUESTIONS_FILE, RESULTS_FILE
from langchain_service import build_judge_chain, format_context, parse_judge_json

CRITERIA = ["relevance", "accuracy", "contextual_awareness", "response_quality"]


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def run_interactions(store, questions):
    """
    Ask all 10 questions through ONE ChatSession, in order, so the 4-turn
    memory window fills and evicts exactly as it would in a real
    conversation. Records the retrieved context and the exact memory snapshot
    available at the moment each question was asked, since the judge needs
    both to score contextual_awareness fairly.
    """
    session = ChatSession(store)
    records = []

    for item in questions:
        history_snapshot = [dict(turn) for turn in session.history]
        retrieved = retrieve(item["question"], store)
        answer = session.ask(item["question"])

        records.append({
            "question": item["question"],
            "doc_focus": item.get("doc_focus", ""),
            "history_available": history_snapshot,
            "retrieved": retrieved_to_records(retrieved),
            "answer": answer,
        })
        print(f"  asked: {item['question'][:60]}... -- answer length {len(answer)} chars")

    return records


def judge_answer(record, judge_chain):
    """Call the LLM as an impartial judge and parse its JSON scoring."""
    history_block = "\n".join(
        f"Q: {turn['question']}\nA: {turn['answer']}"
        for turn in record["history_available"]
    ) or "(no prior turns -- this was asked with an empty memory window)"

    context_block = "\n\n".join(
        f"[{c['doc_id']}, page {c['page']}]\n{c['text']}"
        for c in record["retrieved"]
    )

    raw = judge_chain.invoke({
        "context": context_block,
        "history": history_block,
        "question": record["question"],
        "answer": record["answer"],
    })

    parsed = parse_judge_json(raw)
    if parsed.get("parse_error"):
        print(f"  [warn] could not parse judge output for: {record['question'][:60]}...")
    return parsed


def summarize(records):
    totals = {c: [] for c in CRITERIA}

    for record in records:
        judge = record["judge"]
        if judge.get("parse_error"):
            continue
        for c in CRITERIA:
            if c in judge and isinstance(judge[c], dict) and "score" in judge[c]:
                totals[c].append(judge[c]["score"])

    print("\n--- Evaluation summary ---")
    for c in CRITERIA:
        scores = totals[c]
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  {c:22s} avg = {avg:.2f}/5  ({avg / 5 * 100:.1f}%)  (n={len(scores)})")
        else:
            print(f"  {c:22s} avg = n/a  (n=0)")


if __name__ == "__main__":
    store = load_store()
    questions = load_questions()

    print(f"Running {len(questions)} questions through one continuous ChatSession...")
    records = run_interactions(store, questions)

    print("\nScoring answers with LLM-as-judge...")
    judge_chain = build_judge_chain()
    for record in records:
        record["judge"] = judge_answer(record, judge_chain)
        print(f"  scored: {record['question'][:60]}...")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\nWrote results to {RESULTS_FILE}")

    summarize(records)
