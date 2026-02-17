"""Validation helpers for scoped memory requests."""

from __future__ import annotations

from .typing import MemoryScopeLike


class ScopeValidationError(ValueError):
    """Raised when request fields violate scope constraints."""


def validate_scope_fields(
    *,
    scope: MemoryScopeLike,
    project_id: str | None,
    agent_id: str | None,
    session_id: str | None,
) -> None:
    scope_value = str(scope)

    if scope_value == "MemoryScope.PROJECT" or scope_value == "project":
        if not project_id:
            raise ScopeValidationError("project_id is required for PROJECT scope")
        return

    if scope_value == "MemoryScope.AGENT" or scope_value == "agent":
        if not project_id:
            raise ScopeValidationError("project_id is required for AGENT scope")
        if not agent_id:
            raise ScopeValidationError("agent_id is required for AGENT scope")
        return

    if scope_value == "MemoryScope.USER_PREFERENCE" or scope_value == "user_preference":
        if project_id:
            raise ScopeValidationError("project_id is not allowed for USER_PREFERENCE scope")
        if agent_id:
            raise ScopeValidationError("agent_id is not allowed for USER_PREFERENCE scope")
        return

    if scope_value == "MemoryScope.SESSION" or scope_value == "session":
        if not session_id:
            raise ScopeValidationError("session_id is required for SESSION scope")
        return

    raise ScopeValidationError(f"Unsupported scope: {scope}")
