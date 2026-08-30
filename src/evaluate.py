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

HOW "LLM-AS-JUDGE" WORKS, IN PLAIN TERMS

There is no automatic way to check whether an answer written in English is
"good" -- you cannot compare it to an expected string, because a dozen
different wordings could all be correct. So we ask a language model to grade
it, the way a teacher marks an essay against a rubric.

Concretely, this file makes TWO separate model calls per question:

    call 1 (in bot.py)  the bot ANSWERS the question
    call 2 (here)       the model GRADES that answer, given the question,
                        the source chunks, and the memory available

Same model, two different jobs. Call 2 gets a completely different prompt --
JUDGE_SYSTEM_PROMPT below, which is the rubric written out in full.

Important honesty note: llama3.2 grading answers produced by llama3.2 is not
an independent check, and it may be gentler on its own writing style than a
different model would be. That caveat is recorded in the docs rather than
hidden.
"""

import json

import requests

# Reuse the real bot rather than reimplementing it -- these are the exact
# same functions the interactive chat uses, so we are evaluating the actual
# system and not a copy of it that might drift.
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
    """Read the 10 test questions from data/eval/questions.json."""
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def run_interactions(chunks, index, questions):
    """
    Ask all 10 questions through ONE ChatSession, in order, so the 4-turn
    memory window fills and evicts exactly as it would in a real
    conversation. Records the retrieved context and the exact memory
    snapshot available at the moment each question was asked, since the
    judge needs both to score contextual_awareness fairly.

    Creating the session OUTSIDE the loop is the important detail. One shared
    session means question 4 can still see questions 1-3, which is what makes
    the follow-up questions ("how many heads does THAT model use?") a real
    test of memory. A fresh session per question would start blank every time
    and the memory feature would never be exercised at all.
    """
    session = ChatSession(chunks, index)
    records = []

    for item in questions:
        # Snapshot what memory holds BEFORE asking, because session.ask()
        # will append this turn and change it. The judge needs to know what
        # the bot could actually see at the time, not what it holds now.
        # dict(turn) copies each turn so later changes cannot affect it.
        history_snapshot = [dict(turn) for turn in session.history]

        # Retrieve separately just so we can RECORD which chunks were used.
        # session.ask() retrieves again internally; retrieval depends only on
        # the question text, so both calls return the same chunks. A small
        # duplicated cost in exchange for being able to inspect the evidence.
        retrieved = retrieve(item["question"], chunks, index)

        answer = session.ask(item["question"])

        # Save everything needed to judge this answer later AND to show a
        # human what happened. This dict becomes one entry in results.json.
        records.append({
            "question": item["question"],
            # .get(key, default) returns the default instead of crashing when
            # the key is missing -- safer than item["doc_focus"] here.
            "doc_focus": item.get("doc_focus", ""),
            "history_available": history_snapshot,
            "retrieved": [
                # Keep only the fields worth reading back; drop the rest.
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
        # [:60] takes the first 60 characters, so the log stays readable.
        print(f"  asked: {item['question'][:60]}... -- answer length {len(answer)} chars")

    return records


def build_judge_messages(record):
    """Assemble the prompt that asks the model to grade one answer.

    The judge gets four things, clearly labelled: the source chunks (its
    ground truth), the memory the bot had, the question, and the answer to
    grade. Same messages format as any other chat call -- a system message
    holding the rubric, then one user message holding the material.
    """
    # Stitch the 5 retrieved chunks into one labelled block. The bit inside
    # join(...) is a generator expression: it produces one formatted string
    # per chunk, and join glues them together with a blank line between.
    context_block = "\n\n".join(
        f"[{c['doc_id']}, page {c['page_start']}]\n{c['text']}"
        for c in record["retrieved"]
    )

    # Same idea for the conversation history. The trailing "or ..." is a
    # Python idiom: an empty string is falsy, so if there were no prior turns
    # the fallback text is used instead. Roughly C#'s ?? for empty strings.
    history_block = "\n".join(
        f"Q: {turn['question']}\nA: {turn['answer']}"
        for turn in record["history_available"]
    ) or "(no prior turns -- this was asked with an empty memory window)"

    # Adjacent strings in brackets are automatically joined in Python -- this
    # is one long string, split across lines purely for readability.
    user_content = (
        f"RETRIEVED CONTEXT:\n{context_block}\n\n"
        f"CONVERSATION HISTORY AVAILABLE TO THE BOT AT THIS TURN:\n{history_block}\n\n"
        f"QUESTION: {record['question']}\n\n"
        f"CANDIDATE ANSWER:\n{record['answer']}"
    )

    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},   # the rubric
        {"role": "user", "content": user_content},            # the material
    ]


def judge_answer(record):
    """Call the LLM as an impartial judge and parse its JSON scoring.

    json_mode=True asks Ollama to constrain decoding to valid JSON, so the
    reply is parseable by construction. Without it, the model occasionally
    wrapped its JSON in explanatory prose or stopped a token early, which cost
    one question its score in an earlier run. The cleanup and repair steps
    below are kept as cheap insurance rather than the primary defence.

    A failed call is recorded and skipped rather than raised. An earlier run
    lost nine completed judgements because the tenth call returned a 500 and
    took the whole process down with it -- roughly twenty minutes of work
    thrown away over one bad response. One unscored question is a footnote in
    the results; an aborted run is a lost afternoon.
    """
    # The judge call itself. Everything after this is defending against the
    # model returning something we cannot read.
    try:
        raw = call_llama(build_judge_messages(record), json_mode=True)
    except requests.exceptions.RequestException as error:
        # Record the failure and carry on. Do NOT re-raise: one bad call
        # must not destroy the other nine judgements (that happened once).
        print(f"  [warn] judge call failed ({type(error).__name__}) for: "
              f"{record['question'][:60]}...")
        return {"parse_error": True, "raw_response": f"request failed: {error}"}

    # The model returns TEXT. We asked for JSON, but text is all it can emit,
    # so we still have to parse it into a real Python dict.
    cleaned = raw.strip()

    # Models often wrap JSON in a markdown code fence (```json ... ```).
    # Strip that if present, since it would break the parser.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]          # drop the leading word "json"
        cleaned = cleaned.strip()

    try:
        # json.loads() = string -> Python objects (the reverse of json.dumps).
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass                                # not valid yet; try the repair below

    # small local models occasionally stop one token early and omit the
    # final closing brace(s) -- repair that specific case before giving up.
    # "}" * 2 repeats the string, giving "}}".
    missing_braces = cleaned.count("{") - cleaned.count("}")
    if missing_braces > 0:
        try:
            return json.loads(cleaned + "}" * missing_braces)
        except json.JSONDecodeError:
            pass

    # Everything failed. Keep the raw text so it can be inspected afterwards,
    # and mark the record so summarize() knows to skip it.
    print(f"  [warn] could not parse judge output for: {record['question'][:60]}...")
    return {"parse_error": True, "raw_response": raw}


def summarize(records):
    """Average each criterion across all successfully-scored questions."""
    criteria = ["relevance", "accuracy", "contextual_awareness", "response_quality"]

    # A dict comprehension: build {"relevance": [], "accuracy": [], ...} --
    # one empty list per criterion, ready to collect scores into.
    totals = {c: [] for c in criteria}

    for record in records:
        judge = record["judge"]
        if judge.get("parse_error"):
            continue                    # skip unscored questions entirely
        for c in criteria:
            totals[c].append(judge[c]["score"])

    print("\n--- Evaluation summary ---")
    for c in criteria:
        scores = totals[c]
        if scores:                       # a non-empty list is truthy
            avg = sum(scores) / len(scores)
            # Format specifiers: {c:22s} pads the name to 22 characters so the
            # columns line up; {avg:.2f} rounds to 2 decimal places.
            print(f"  {c:22s} avg = {avg:.2f}/5  ({avg / 5 * 100:.1f}%)  (n={len(scores)})")
        else:
            print(f"  {c:22s} no valid scores")

    # Count how many questions failed to score, and say so plainly. Reporting
    # an average over 9 of 10 questions without mentioning it would be
    # quietly misleading.
    unscored = sum(1 for r in records if r["judge"].get("parse_error"))
    if unscored:
        print(f"\n  note: {unscored} of {len(records)} question(s) could not be scored "
              f"(see parse_error entries in the results file)")


# Runs only when executed directly: python src/evaluate.py
if __name__ == "__main__":
    # PHASE 1 -- get answers. Loads the prebuilt index, then asks all 10
    # questions through one conversation.
    chunks, index = load_chunks_and_index()
    questions = load_questions()

    print(f"Running {len(questions)} questions through one continuous ChatSession...")
    records = run_interactions(chunks, index, questions)

    # PHASE 2 -- grade those answers. A second model call per question,
    # completely separate from the answering above.
    print("\nScoring answers with LLM-as-judge...")
    for record in records:
        # Add a "judge" key to each record dict as we go.
        record["judge"] = judge_answer(record)
        print(f"  scored: {record['question'][:60]}...")

    # Save everything: questions, answers, retrieved chunks, memory state and
    # scores. indent=2 pretty-prints it so the file is readable by a human.
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\nWrote results to {RESULTS_FILE}")

    summarize(records)
