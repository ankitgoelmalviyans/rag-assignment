#!/usr/bin/env python3
"""PreToolUse hook for Bash: block dangerous commands the permission list can't express.

Claude Code sends the tool call as JSON on stdin. Exit code 2 blocks the
call and feeds stderr back to Claude so it can choose a safer approach.
Exit 0 lets it through (normal permission rules still apply).
"""

import json
import re
import sys

BLOCKED = [
    (r"\brm\s+-[a-z]*r[a-z]*f?\s+(/|~|\$HOME|\.)(\s|$)", "recursive delete of a root/home/project dir"),
    (r"git\s+push\s+.*(--force|-f)\b", "force push"),
    (r"git\s+push\s+\S+\s+(main|master)\b", "direct push to main"),
    (r"\b(terraform|tofu)\s+(apply|destroy)\b.*-auto-approve", "unattended infra change"),
    (r"kubectl\s+delete\s+(ns|namespace)\b", "namespace deletion"),
    (r"(curl|wget)[^|]*\|\s*(ba|z)?sh", "pipe-to-shell install"),
    (r"\b(printenv|env)\b\s*($|\|)", "dumping environment (may expose secrets)"),
    (r"cat\s+.*(\.env|id_rsa|credentials)", "reading secret files"),
]

cmd = json.load(sys.stdin).get("tool_input", {}).get("command", "")
for pattern, why in BLOCKED:
    if re.search(pattern, cmd):
        print(f"Blocked by guard_bash hook: {why}. Ask the user if this is really needed.", file=sys.stderr)
        sys.exit(2)
sys.exit(0)
