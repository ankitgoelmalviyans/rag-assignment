# 2. AI in DevOps, infrastructure and pipelines

## Principle: AI is advisory, policy is authoritative

```
 PR opened
   │
   ├─▶ deterministic gates (HARD)          ├─▶ AI (ADVISORY)
   │     lint, tests, gitleaks,            │     PR review comment
   │     conftest/OPA, risk_gate.py        │     plan summary / blast radius
   │                                       │     CI failure diagnosis
   ▼                                       ▼
 human approval (CODEOWNERS, environments) ──▶ deploy
```

The AI can explain, summarise, suggest and draft. It does not approve, merge, apply or deploy.

## High-value, low-risk uses

| Use case | How | Template |
|---|---|---|
| AI PR review | `claude-code-action` with read-only tools, comments only | `devops/github-workflows/ai-pr-review.yml` |
| CI failure triage | Feed redacted failed logs, get root cause + suggested patch | `ai-ci-failure-triage.yml` |
| Terraform plan explainer | Summarise `plan.json`: destroys, IAM, cost, blast radius | `ai-infra-plan-review.yml` |
| Risk-based reviewer routing | Deterministic path rules → required reviewers | `devops/scripts/risk_gate.py` |
| Writing IaC / pipelines | Agent drafts, **OPA policy** gates the result | `devops/policy/terraform.rego` |
| Incident support | Summarise alerts/logs/runbooks, draft postmortem timeline | read-only observability MCP |
| Dependency upgrades | Agent reads changelogs, bumps, runs tests, opens PR | Dependabot + AI summary |
| Docs / runbooks | Generate from code and pipeline definitions | `claude -p` in a scheduled job |

## Pipeline security rules for AI jobs

1. **Use `pull_request`, not `pull_request_target`**, for AI jobs on untrusted code. Fork PRs
   then run without secrets, so a prompt injection in a PR can't exfiltrate your API key.
2. **Minimal `permissions:`** per job (`contents: read`, `pull-requests: write`). Never `write-all`.
3. **Restrict the agent's tools** in CI (`--allowedTools "Read,Grep,Glob"`) and cap `--max-turns`
   and `timeout-minutes`.
4. **Treat PR text, commit messages, issues and logs as untrusted input** — say so in the prompt,
   and don't give the agent tools that could act on injected instructions.
5. **Redact logs** before sending to a model (secrets, tokens, customer data).
6. **OIDC to the cloud**, no long-lived keys. Separate **plan-only** (read) and **apply** roles;
   AI jobs only ever get plan-only.
7. **Apply/deploy behind GitHub Environments** with required human reviewers.
8. **Pin actions** (ideally by SHA) and keep an allow-list of third-party actions.

## Infra guardrails, in layers

1. **Agent level:** hooks/deny-rules block `terraform apply/destroy`, `kubectl delete`, etc.
2. **Commit level:** pre-commit (`terraform fmt`, `tflint`, `gitleaks`).
3. **PR level:** `terraform plan` + `conftest` with org policy (tags, no public buckets, no
   wildcard IAM, no deleting confidential resources) + `checkov`/`tfsec`.
4. **Deploy level:** environment approvals, change windows, automatic rollback on SLO breach.
5. **Runtime level:** cloud guardrails (SCPs, Azure Policy, GCP Org Policy) — the last line,
   which even a perfect-looking plan can't bypass.

## Agentic ops (going further, carefully)

Autonomous agents that *act* in production (restart pods, scale, roll back) should:
- have a narrow, allow-listed action set (e.g. "rollback to previous release", "scale within 2–10")
- require human confirmation for anything outside it
- log every action with reasoning to an immutable audit trail
- run with their own identity and least-privilege role, never a human's credentials
- start in "suggest" mode for weeks before "act" mode
