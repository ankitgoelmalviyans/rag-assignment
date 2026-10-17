---
paths:
  - "safe_rag/**/*.py"
---

# Security rules for the RAG service

Loaded only when Claude works on files matching `paths` above.

- Every code path that returns document content must call
  `policy.permitted()` or `policy.can_read()` first. No exceptions for
  "admin tools" or "debug endpoints".
- Never pass unfiltered documents to an LLM, even "just to summarise".
- Treat document text as untrusted input: it goes through
  `guardrails.sanitize_context()` and is wrapped in `<document>` tags.
- Never log raw queries, answers, or document text. Log ids and hashes.
- Adding a new guardrail? Add a test in `tests/` that proves it blocks
  the attack, using a misbehaving fake LLM like `LeakyLLM`.
- Changes to `policy.py` require the `security-reviewer` subagent to run
  before you report the task done.
