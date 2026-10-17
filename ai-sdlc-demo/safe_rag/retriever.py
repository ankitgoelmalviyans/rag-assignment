"""Tiny dependency-free retriever (bag-of-words cosine).

Swap for FAISS / pgvector / OpenSearch in real life -- the important part
is not the scoring, it is that the ACL filter runs BEFORE scoring, so an
unauthorized chunk can never even be a candidate.
"""

import math
import re
from collections import Counter

from .models import Document, User
from .policy import permitted

_WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = {"the", "a", "an", "is", "of", "to", "and", "for", "in", "on", "what", "how", "do", "i"}


def _vector(text: str) -> Counter:
    return Counter(w for w in _WORD.findall(text.lower()) if w not in STOPWORDS)


def _cosine(a: Counter, b: Counter) -> float:
    dot = sum(a[k] * b[k] for k in a.keys() & b.keys())
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


def retrieve(user: User, query: str, docs: list[Document], k: int = 3) -> list[Document]:
    q = _vector(query)
    candidates = permitted(user, docs)  # <-- the security boundary
    scored = [(d, _cosine(q, _vector(d.title + " " + d.text))) for d in candidates]
    scored = [s for s in scored if s[1] > 0]
    scored.sort(key=lambda s: s[1], reverse=True)
    return [d for d, _ in scored[:k]]
