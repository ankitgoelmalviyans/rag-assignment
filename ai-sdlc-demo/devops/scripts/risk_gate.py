#!/usr/bin/env python3
"""Classify a change's risk from the files it touches.

Deterministic on purpose: the decision "does this PR need a senior human
reviewer?" should not depend on what an LLM thinks of the diff. Output is
`level=<low|medium|high>` so it can be appended to $GITHUB_OUTPUT.

    python devops/scripts/risk_gate.py --base origin/main
    python devops/scripts/risk_gate.py --files safe_rag/policy.py README.md
"""

import argparse
import fnmatch
import subprocess
import sys

RULES = [
    ("high", ["safe_rag/policy.py", "safe_rag/guardrails.py", ".github/workflows/*",
              "devops/terraform/*", "devops/policy/*", ".claude/settings.json",
              ".claude/hooks/*", "CODEOWNERS", "*requirements*.txt"]),
    ("medium", ["safe_rag/*", "devops/*"]),
    ("low", ["docs/*", "*.md", "tests/*"]),
]
ORDER = {"low": 0, "medium": 1, "high": 2}


def classify(files: list[str]) -> str:
    level = "low"
    for f in files:
        file_level = "medium"  # unknown paths default to medium, not low
        for rule_level, patterns in RULES:
            if any(fnmatch.fnmatch(f, p) for p in patterns):
                file_level = rule_level
                break
        if ORDER[file_level] > ORDER[level]:
            level = file_level
    return level


def changed_files(base: str) -> list[str]:
    out = subprocess.run(["git", "diff", "--name-only", f"{base}...HEAD"],
                         capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--base")
    g.add_argument("--files", nargs="+")
    args = p.parse_args(argv)
    files = args.files or changed_files(args.base)
    print(f"level={classify(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
