---
name: plan-feature
description: Plan a feature or bugfix end-to-end (requirements, design, tests, steps) before writing code. Use when the user asks to build, add, change or fix something non-trivial.
---

# Plan a feature

1. **Clarify.** Restate the request in one sentence. List assumptions. If any
   assumption is security-relevant or changes the public API, ask before continuing.
2. **Explore.** Delegate to the `planner` subagent with the request and any ticket text.
3. **Design check.** If the change adds a component or data flow, add a short
   design note to the plan: a mermaid diagram plus the trade-off you chose and why.
   Use `docs/templates/design-note.md` as the shape.
4. **Threat model** — if the change touches retrieval, auth, guardrails, or
   anything user-facing, run the `threat-model` skill and fold results into Risks.
5. **Present the plan** and STOP. Do not write code until the user approves.
6. After approval: implement step by step, running tests after each step
   (the PostToolUse hook does this automatically), then run `code-reviewer`.
