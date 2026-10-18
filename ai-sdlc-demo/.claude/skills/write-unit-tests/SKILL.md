---
name: write-unit-tests
description: Generate meaningful pytest unit tests for a module, function, or recent change, including negative and adversarial cases. Use when the user asks for tests or coverage.
---

# Write unit tests

1. Identify the target: a path, a function, or `git diff` of the current branch.
2. List behaviours (not lines) the target must guarantee. Show the list.
3. Delegate to the `test-writer` subagent with that list.
4. Run `python -m pytest -q`. Then sanity-check the tests are real:
   temporarily break one guarded behaviour in your head (or describe the mutation)
   and confirm at least one test would fail. A test that can't fail is noise.
5. Report: behaviours covered, behaviours still uncovered, suspected bugs.

See `.claude/rules/testing.md` for style rules.
