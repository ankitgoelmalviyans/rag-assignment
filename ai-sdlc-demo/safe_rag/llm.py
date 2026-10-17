"""LLM backends.

`ExtractiveLLM` is a deterministic stand-in so the demo and the tests run
offline with zero API keys. `ClaudeLLM` shows where a real model plugs in;
it is only used when ANTHROPIC_API_KEY is set and the `anthropic` package
is installed.
"""

import os
import re
from typing import Protocol

from .retriever import STOPWORDS

_DOC = re.compile(r'<document id="([^"]+)"[^>]*>\n(.*?)\n</document>', re.DOTALL)
_WORD = re.compile(r"[a-z0-9]+")


class LLM(Protocol):
    def complete(self, system: str, context: str, question: str) -> str: ...


class ExtractiveLLM:
    """Picks the context sentence with the most word overlap and cites it."""

    def complete(self, system: str, context: str, question: str) -> str:
        q = set(_WORD.findall(question.lower())) - STOPWORDS
        best, best_id, best_score = None, None, 0
        for doc_id, body in _DOC.findall(context):
            for sentence in re.split(r"(?<=[.!?])\s+", body):
                score = len(q & set(_WORD.findall(sentence.lower())))
                if score > best_score:
                    best, best_id, best_score = sentence, doc_id, score
        if best is None:
            return "I don't know based on the documents you can access."
        return f"{best} [{best_id}]"


class ClaudeLLM:
    def __init__(self, model: str = "claude-sonnet-5-5"):
        import anthropic  # optional dependency

        self._client = anthropic.Anthropic()
        self._model = model

    def complete(self, system: str, context: str, question: str) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": f"{context}\n\nQuestion: {question}"}],
        )
        return msg.content[0].text


def default_llm() -> LLM:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return ClaudeLLM()
        except ImportError:
            pass
    return ExtractiveLLM()
