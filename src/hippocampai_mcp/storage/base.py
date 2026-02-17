"""Storage interfaces for memory backends."""

from __future__ import annotations

from typing import Any, Protocol

from hippocampai_mcp.domain.models import MemoryScope


class MemoryStore(Protocol):
    def remember(
        self,
        *,
        text: str,
        user_id: str,
        session_id: str | None = None,
        memory_type: str = "context",
        importance: float | None = None,
        tags: list[str] | None = None,
        agent_id: str | None = None,
        project_id: str | None = None,
        ttl_days: int | None = None,
        scope: MemoryScope | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any: ...

    def recall(
        self,
        *,
        query: str,
        user_id: str,
        session_id: str | None = None,
        k: int = 5,
        min_importance: float | None = None,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        agent_id: str | None = None,
        project_id: str | None = None,
        scope: MemoryScope | None = None,
    ) -> list[Any]: ...

    def update(
        self,
        *,
        memory_id: str,
        text: str | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
    ) -> Any: ...

    def delete(self, *, memory_id: str, user_id: str | None = None) -> bool: ...

    def list(
        self,
        *,
        user_id: str,
        memory_type: Any | None = None,
        tags: list[str] | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        project_id: str | None = None,
        scope: MemoryScope | None = None,
        limit: int = 50,
        sort_by: str = "created_at",
        order: str = "desc",
    ) -> list[Any]: ...

    def stats(self, *, user_id: str) -> dict[str, Any]: ...
