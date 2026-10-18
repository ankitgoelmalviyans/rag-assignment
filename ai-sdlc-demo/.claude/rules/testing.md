---
paths:
  - "tests/**/*.py"
---

# Testing rules

- One behaviour per test; name it after the behaviour (`test_intern_cannot_get_salaries`).
- Prefer invariant tests (loop over every user x query) for security properties.
- Use fakes (`ExtractiveLLM`, `LeakyLLM`) — tests must not call a real model or network.
- Do not use `pytest.skip`, `xfail`, or loosen an assertion to get green. If a
  test is wrong, explain why in the PR and fix it properly.
