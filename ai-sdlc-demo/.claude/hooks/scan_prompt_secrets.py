#!/usr/bin/env python3
"""UserPromptSubmit hook: stop a pasted secret from being sent to the model.

Exit 2 blocks the prompt and shows stderr to the user (not to the model).
"""

import json
import re
import sys

PATTERNS = [
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"sk-ant-[A-Za-z0-9_-]{20,}",
    r"ghp_[A-Za-z0-9]{36}",
    r"(?i)password\s*[:=]\s*\S{6,}",
]

prompt = json.load(sys.stdin).get("prompt", "")
if any(re.search(p, prompt) for p in PATTERNS):
    print("Your prompt looks like it contains a secret. Remove it and use an env var name instead.",
          file=sys.stderr)
    sys.exit(2)
sys.exit(0)
