# CLAUDE.md — project memory for AI agents

Read automatically by Claude Code at session start. Keep it short, factual,
and enforceable; put long guidance in `docs/` and path-specific rules in
`.claude/rules/`.

## What this is
A privilege-aware RAG service (`safe_rag/`) plus DevOps templates (`devops/`).
Python 3.11+, stdlib only at runtime, pytest for tests.

## Commands
- Test: `python -m pytest -q`
- Run:  `python -m safe_rag.cli --user intern "what are the salary bands?"`

## Non-negotiable rules
1. **Plan before code** for anything touching more than one file: use the
   `plan-feature` skill and wait for approval.
2. **Tests first or together.** No behaviour change merges without a test.
   Never delete, skip, or weaken a test to make CI green.
3. **Access control is code, never prompt.** Any retrieval path must go
   through `policy.permitted()` *before* scoring. See `.claude/rules/security.md`.
4. **No secrets** in code, prompts, logs, or test fixtures. Use env vars.
5. **No new dependencies** without saying why in the PR description.
6. Small diffs. One concern per PR.

## Architecture (one line each)
- `policy.py` — deterministic ACL (clearance, department, role allow-list)
- `retriever.py` — ACL pre-filter, then similarity
- `guardrails.py` — input / context / output checks, PII redaction
- `pipeline.py` — orchestrates the above and writes the audit log

## When unsure
Ask. Don't guess at security-relevant behaviour.
