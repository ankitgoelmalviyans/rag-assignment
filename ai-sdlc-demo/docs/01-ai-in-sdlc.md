# 1. Using AI safely across the SDLC

## The mental model

Treat an AI coding agent like a **fast, tireless, junior-to-mid engineer with no memory
and occasional overconfidence**. It's very productive, but:

- it only knows what you put in its context → **rules & memory** (`CLAUDE.md`, `.claude/rules/`)
- it does what it's allowed to do → **permissions** (`.claude/settings.json`)
- it can be talked into things by text it reads → **guardrails & hooks**
- it's checked by the same gates as humans → **tests, CI, CODEOWNERS, human approval**

The golden rule: **AI proposes, deterministic gates verify, humans approve.**

## The toolbox, and what each piece is for

| Feature | What it is | Use it for | In this repo |
|---|---|---|---|
| **CLAUDE.md** | Project memory, loaded every session | Commands, architecture, non-negotiable rules | `CLAUDE.md` |
| **Rules** | Markdown loaded only for matching `paths:` | Domain rules that would bloat CLAUDE.md | `.claude/rules/security.md`, `testing.md` |
| **Permissions** | allow / ask / deny lists for tools | Least privilege: tests auto-allowed, push = ask, secrets = deny | `.claude/settings.json` |
| **Hooks** | Your scripts run on agent events; exit 2 blocks | Hard guarantees the model can't argue with | `.claude/hooks/*` |
| **Subagents** | Specialised agents with own prompt, tools, model | Separation of duties: planner can't write, reviewer can't edit | `.claude/agents/*` |
| **Skills** | Reusable playbooks invoked by name or auto-matched | Repeatable workflows: plan, test, review, threat-model | `.claude/skills/*` |
| **MCP servers** | Connect tools/data (GitHub, Jira, DBs) | Give *read-only* access first; scope tokens tightly | (add `.mcp.json`) |
| **Plan mode** | Agent researches & proposes, can't edit | Any non-trivial change | Shift+Tab in Claude Code |
| **Headless / CI** | `claude -p` / `claude-code-action` | AI review, triage, docs in pipelines | `devops/github-workflows/` |

**Prompts are guidance, hooks and permissions are enforcement.** If something must
*never* happen (push to main, read `.env`, edit CI config), make it a deny rule or a hook —
don't just write "please don't" in CLAUDE.md.

## Phase by phase

### Plan
- Start in **plan mode** or the `plan-feature` skill. The `planner` subagent is read-only
  (`tools: Read, Grep, Glob`), so planning can't accidentally change code.
- Output: goal, non-goals, acceptance criteria, affected files, risks, test plan, steps.
- **Human checkpoint:** approve the plan before any code is written. This is where you catch
  the misunderstanding that would otherwise cost a 500-line wrong diff.
- Good AI uses: turn a vague ticket into acceptance criteria; find every call site affected;
  spot missing requirements ("what happens for a user with no department?").

### Design
- Ask for 2–3 options with trade-offs, not one answer. Pick yourself.
- Record the decision in a short design note / ADR (`docs/templates/design-note.md`) with a
  mermaid diagram — AI is great at drafting these; you own the decision.
- Run the `threat-model` skill for anything touching auth, data, user input, prompts, CI, infra.

### Build
- Small steps. One concern per PR. The `PostToolUse` hook re-runs tests after each edit, so the
  agent sees breakage immediately and fixes it in the same loop.
- `PreToolUse` hooks block dangerous commands (force push, `rm -rf`, `terraform apply
  -auto-approve`, pipe-to-shell) and writes to protected paths (CI config, `.env`, the agent's own
  `settings.json` / hooks — **an agent must not be able to loosen its own guardrails**).
- Secrets: deny-list reads of `.env`/`~/.aws`, scan prompts (`UserPromptSubmit` hook) and
  content (`guard_files.py`), plus `gitleaks` in pre-commit and CI.
- New dependency? Agent must justify it (supply-chain risk: typosquatted / hallucinated packages
  are real). Pin versions, run `pip-audit` / Dependabot.

### Unit tests
- `write-unit-tests` skill → `test-writer` subagent, which may **only edit `tests/`**, so it
  can't "fix" production code to make a bad test pass.
- Ask for behaviours, negatives, boundaries and adversarial cases — not line coverage.
- Prefer **invariant tests** for security (see `test_retrieval_never_returns_unauthorized_docs`:
  loops every user × query).
- Use fakes for the LLM, including *malicious* ones (`LeakyLLM`) to prove guardrails work.
- Red flags to reject in review: `skip`, `xfail`, loosened asserts, tests that pass against the bug.

### Review
- Layered: `code-reviewer` + `security-reviewer` subagents in parallel (`review-pr` skill), then
  CI's AI review comment, then **a human approval via branch protection + CODEOWNERS**.
- AI reviewers must cite `file:line` and severity; drop vague findings.
- A deterministic `risk_gate.py` decides *who* must review (policy/CI/infra = high = security
  team), independent of what any model thinks.

### Release & operate
- Same gates as humans: green CI, required reviews, environments with approvers for prod.
- Track AI-assisted PRs (label) and measure: lead time, change-failure rate, revert rate.
  If change-failure rate climbs, tighten gates.

## Anti-patterns
- "Vibe coding" to prod: accepting large diffs you didn't read.
- Giving the agent `bypassPermissions` on your laptop with prod credentials loaded.
- Letting AI approve or merge PRs.
- Long-lived cloud keys in CI (use OIDC).
- One giant CLAUDE.md nobody maintains — keep it short, move detail to rules/skills.
- Pasting customer data / secrets into prompts.

## Team rollout checklist
- [ ] `CLAUDE.md` with commands + 5–10 hard rules
- [ ] Shared `.claude/settings.json` committed (deny secrets, ask for push/infra)
- [ ] Hooks for the 3–5 things that must never happen
- [ ] Subagents with least-privilege tool lists
- [ ] Skills for your top repeated workflows
- [ ] Branch protection: required CI + human review + CODEOWNERS
- [ ] Secret scanning (pre-commit + CI) and dependency scanning
- [ ] Enterprise/zero-data-retention terms for the AI provider; approved-tools list
- [ ] Metrics: DORA + AI-PR revert rate
