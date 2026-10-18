#!/usr/bin/env python3
"""PreToolUse hook for Edit/Write: protect sensitive paths and block secrets in content."""

import json
import re
import sys

PROTECTED = [
    r"(^|/)\.env",
    r"(^|/)secrets/",
    r"(^|/)\.github/workflows/",   # CI changes need a human
    r"(^|/)CODEOWNERS$",
    r"(^|/)\.claude/settings\.json$",  # the agent must not loosen its own guardrails
    r"(^|/)\.claude/hooks/",
]

SECRET_PATTERNS = [
    r"AKIA[0-9A-Z]{16}",                       # AWS access key id
    r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"sk-ant-[A-Za-z0-9_-]{20,}",              # Anthropic key
    r"ghp_[A-Za-z0-9]{36}",                    # GitHub PAT
    r"xox[baprs]-[A-Za-z0-9-]{10,}",           # Slack token
]

tool_input = json.load(sys.stdin).get("tool_input", {})
path = tool_input.get("file_path", "")
content = tool_input.get("content", "") + tool_input.get("new_string", "")
for e in tool_input.get("edits", []):
    content += e.get("new_string", "")

for p in PROTECTED:
    if re.search(p, path):
        print(f"Blocked: {path} is a protected path. Propose the change to the user instead.", file=sys.stderr)
        sys.exit(2)

for p in SECRET_PATTERNS:
    if re.search(p, content):
        print("Blocked: content looks like it contains a secret. Use an env var reference.", file=sys.stderr)
        sys.exit(2)

sys.exit(0)
