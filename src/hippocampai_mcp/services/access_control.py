"""Access-control rules for scoped memory operations."""

from __future__ import annotations

from hippocampai_mcp.domain.models import MemoryScope
from hippocampai_mcp.domain.validation import ScopeValidationError, validate_scope_fields


class AccessControlError(ValueError):
    """Raised when scope/ownership rules are violated."""


class AccessController:
    VALID_VISIBILITY = {"private", "shared", "public"}

    def parse_scope(self, scope: str | MemoryScope | None) -> MemoryScope | None:
        if scope is None:
            return None
        if isinstance(scope, MemoryScope):
            return scope
        try:
            return MemoryScope(scope)
        except ValueError as exc:
            raise AccessControlError(f"Unsupported scope: {scope}") from exc

    def enforce_update_delete(self, *, user_id: str | None, action: str) -> None:
        if not user_id:
            raise AccessControlError(f"user_id is required for {action}")

    def normalize_agent_visibility(self, visibility: str | None) -> str:
        value = (visibility or "private").lower()
        if value not in self.VALID_VISIBILITY:
            raise AccessControlError("visibility must be one of: private, shared, public")
        return value

    def enforce_scope_fields(
        self,
        *,
        scope: MemoryScope,
        project_id: str | None,
        agent_id: str | None,
        session_id: str | None,
    ) -> None:
        try:
            validate_scope_fields(
                scope=scope,
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
            )
        except ScopeValidationError as exc:
            raise AccessControlError(str(exc)) from exc

    def enforce_recall_scope(
        self,
        *,
        scope: str | MemoryScope | None,
        project_id: str | None,
        agent_id: str | None,
        session_id: str | None,
        include_cross_scope: bool,
    ) -> MemoryScope | None:
        parsed_scope = self.parse_scope(scope)
        if parsed_scope is None:
            if include_cross_scope:
                return None
            raise AccessControlError(
                "scope is required unless include_cross_scope is explicitly true"
            )
        self.enforce_scope_fields(
            scope=parsed_scope,
            project_id=project_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        return parsed_scope
