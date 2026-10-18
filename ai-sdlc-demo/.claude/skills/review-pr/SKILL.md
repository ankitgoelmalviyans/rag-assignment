---
name: review-pr
description: Run a layered AI review of the current branch or a PR - code quality, security and tests - and produce a single merged report. Use before opening a PR or when asked to review.
---

# Review a PR

1. Get the diff: `git diff main...HEAD --stat`, then the full diff.
2. Run in parallel:
   - `code-reviewer` subagent on the full diff
   - `security-reviewer` subagent if the diff touches `safe_rag/policy.py`,
     `safe_rag/guardrails.py`, `devops/`, `.github/`, or any auth/secret handling
3. Run `python -m pytest -q` and include the result.
4. Merge findings, de-duplicate, sort by severity. Drop anything without a file:line.
5. End with a verdict: **ready**, **ready after nits**, or **needs changes**.

AI review is an extra reviewer, not a replacement: a human still approves the merge.
