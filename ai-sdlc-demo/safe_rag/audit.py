"""Append-only audit log (JSON lines).

Log enough to answer "who saw which document, when, and was anything
blocked?" -- but never log the raw query or answer text, which may contain
PII. A hash lets you correlate repeats without storing content.
"""

import hashlib
import json
import time
from pathlib import Path


class AuditLog:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.records: list[dict] = []

    def write(self, **record) -> None:
        record = {"ts": time.time(), **record}
        if "query" in record:
            record["query_sha256"] = hashlib.sha256(record.pop("query").encode()).hexdigest()[:16]
        self.records.append(record)
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(record) + "\n")
