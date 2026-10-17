"""Try it:  python -m safe_rag.cli --user intern "what are the salary bands?"
"""

import argparse
from pathlib import Path

from .audit import AuditLog
from .corpus import DOCUMENTS, USERS
from .pipeline import SafeRAG


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--user", choices=sorted(USERS), required=True)
    p.add_argument("question")
    args = p.parse_args()

    rag = SafeRAG(DOCUMENTS, audit=AuditLog(Path("audit.jsonl")))
    ans = rag.ask(USERS[args.user], args.question)
    print(ans.text)
    if ans.reasons:
        print(f"  (policy notes: {', '.join(ans.reasons)})")


if __name__ == "__main__":
    main()
