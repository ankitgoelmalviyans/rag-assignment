import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "risk_gate", Path(__file__).parent.parent / "devops/scripts/risk_gate.py")
risk_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(risk_gate)


@pytest.mark.parametrize("files,expected", [
    (["README.md"], "low"),
    (["tests/test_x.py", "docs/a.md"], "low"),
    (["safe_rag/retriever.py"], "medium"),
    (["something/new.go"], "medium"),             # unknown defaults to medium
    (["docs/a.md", "safe_rag/policy.py"], "high"),  # highest file wins
    ([".github/workflows/ci.yml"], "high"),
    ([".claude/settings.json"], "high"),            # loosening agent guardrails
])
def test_classify(files, expected):
    assert risk_gate.classify(files) == expected
