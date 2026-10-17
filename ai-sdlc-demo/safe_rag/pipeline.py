"""End-to-end safe RAG pipeline."""

import re

from . import guardrails
from .audit import AuditLog
from .llm import LLM, default_llm
from .models import Answer, Document, User
from .retriever import retrieve

_CITE = re.compile(r"\[([a-z]+-\d+)\]")


class SafeRAG:
    def __init__(self, docs: list[Document], llm: LLM | None = None, audit: AuditLog | None = None):
        self.docs = docs
        self.llm = llm or default_llm()
        self.audit = audit or AuditLog()

    def ask(self, user: User, query: str) -> Answer:
        # 1. Input guardrails
        reasons = guardrails.check_input(query)
        if reasons:
            self.audit.write(event="blocked_input", user=user.id, query=query, reasons=reasons)
            return Answer("Request blocked by input policy.", [], blocked=True, reasons=reasons)

        # 2. ACL-filtered retrieval -- the model only ever sees permitted docs
        docs = retrieve(user, query, self.docs)
        allowed_ids = {d.id for d in docs}

        # 3. Context sanitising
        context, flagged = guardrails.sanitize_context(docs)

        # 4. Generation
        canary = guardrails.new_canary()
        raw = self.llm.complete(guardrails.system_prompt(canary), context, query)

        # 5. Output guardrails
        cited = _CITE.findall(raw)
        out_reasons = guardrails.check_output(raw, cited, allowed_ids, canary)
        if out_reasons:
            self.audit.write(event="blocked_output", user=user.id, query=query,
                             docs=sorted(allowed_ids), reasons=out_reasons)
            return Answer("Response withheld by output policy.", [], blocked=True, reasons=out_reasons)

        text = guardrails.redact_pii(raw, user)

        # 6. Audit
        self.audit.write(event="answered", user=user.id, query=query,
                         docs=sorted(allowed_ids), flagged_docs=flagged, cited=cited)
        return Answer(text, cited, reasons=[f"flagged:{d}" for d in flagged])
