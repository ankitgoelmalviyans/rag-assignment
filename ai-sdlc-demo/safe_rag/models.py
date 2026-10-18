"""Plain data types shared across the pipeline."""

from dataclasses import dataclass, field
from enum import IntEnum


class Classification(IntEnum):
    """Sensitivity levels. Higher number = more sensitive.

    IntEnum so clearance checks are a simple `>=` comparison.
    """

    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    classification: Classification
    # None means "any department"; otherwise only that department (plus admins).
    department: str | None = None
    # Optional extra gate: if non-empty, the user needs at least one of these roles.
    allowed_roles: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class User:
    id: str
    department: str
    clearance: Classification
    roles: frozenset[str] = field(default_factory=frozenset)


@dataclass
class Answer:
    text: str
    citations: list[str]
    blocked: bool = False
    reasons: list[str] = field(default_factory=list)
