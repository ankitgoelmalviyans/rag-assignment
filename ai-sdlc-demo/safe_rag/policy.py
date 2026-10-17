"""Authorization: decides which documents a user may read.

This is deliberately boring, deterministic code. Access control is never
delegated to the LLM ("please don't reveal salaries to interns") -- a prompt
is a suggestion, a filter is a guarantee.
"""

from .models import Document, User

ADMIN_ROLE = "admin"


def can_read(user: User, doc: Document) -> bool:
    """Return True only if every gate passes (deny by default)."""
    # Gate 1: clearance must meet the document's classification.
    if user.clearance < doc.classification:
        return False

    # Gate 2: department scoping (admins bypass).
    if doc.department is not None and ADMIN_ROLE not in user.roles:
        if user.department != doc.department:
            return False

    # Gate 3: optional role allow-list.
    if doc.allowed_roles and not (doc.allowed_roles & user.roles):
        return False

    return True


def permitted(user: User, docs: list[Document]) -> list[Document]:
    return [d for d in docs if can_read(user, d)]
