---
name: threat-model
description: Lightweight STRIDE + OWASP-LLM threat model for a feature or design. Use when a change touches retrieval, auth, guardrails, user input, LLM prompts, CI or infra.
---

# Threat model

For the feature under discussion, fill this table (only rows that apply):

| Threat | Example for this feature | Mitigation (existing or needed) | Test? |
|---|---|---|---|
| Spoofing | caller pretends to be another user | identity from verified token, never from request body | |
| Tampering | poisoned document in the corpus | `sanitize_context`, ingestion review, flagged-doc audit | |
| Repudiation | can't tell who saw a doc | audit log with user + doc ids | |
| Info disclosure | retrieval leaks restricted chunk | ACL **pre**-filter in retriever | |
| DoS | huge query / runaway agent loop | input size cap, rate limits, max tool calls | |
| Elevation | model calls an admin tool | least-privilege tool list per role | |
| LLM01 Prompt injection | "ignore previous instructions" | input + context guardrails | |
| LLM02 Sensitive info disclosure | PII in answer | `redact_pii`, role-gated | |
| LLM06 Excessive agency | agent can write to prod | read-only tools, human approval gates | |

Then list the top 3 risks and the test that will prove each mitigation works.
