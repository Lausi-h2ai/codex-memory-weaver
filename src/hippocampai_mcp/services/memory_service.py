"""Service layer for memory tool behavior and scope handling."""

from __future__ import annotations

from typing import Any

from hippocampai_mcp.domain.models import MemoryScope
from hippocampai_mcp.services.access_control import AccessControlError, AccessController


def _attr(obj: Any, *names: str) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _iso_attr(obj: Any, *names: str) -> str | None:
    value = _attr(obj, *names)
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


class MemoryService:
    def __init__(self, store: Any, access: AccessController | None = None) -> None:
        self.store = store
        self.access = access or AccessController()

    def _infer_scope(
        self,
        *,
        scope: str | MemoryScope | None,
        project_id: str | None,
        agent_id: str | None,
        session_id: str | None,
    ) -> MemoryScope:
        parsed = self.access.parse_scope(scope)
        if parsed is not None:
            return parsed
        if agent_id:
            return MemoryScope.AGENT
        if project_id:
            return MemoryScope.PROJECT
        if session_id:
            return MemoryScope.SESSION
        return MemoryScope.USER_PREFERENCE

    def remember(
        self,
        *,
        text: str,
        user_id: str,
        scope: str | MemoryScope | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        memory_type: str = "context",
        importance: float | None = None,
        tags: list[str] | None = None,
        ttl_days: int | None = None,
        visibility: str | None = None,
        run_id: str | None = None,
        project: str | None = None,
    ) -> dict[str, Any]:
        project_id = project_id or project
        scope_value = self._infer_scope(
            scope=scope,
            project_id=project_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        self.access.enforce_scope_fields(
            scope=scope_value,
            project_id=project_id,
            agent_id=agent_id,
            session_id=session_id,
        )

        metadata: dict[str, Any] = {}
        if run_id:
            metadata["run_id"] = run_id
        if scope_value == MemoryScope.AGENT:
            metadata["visibility"] = self.access.normalize_agent_visibility(visibility)

        memory = self.store.remember(
            text=text,
            user_id=user_id,
            session_id=session_id,
            memory_type=memory_type,
            importance=importance,
            tags=tags,
            agent_id=agent_id,
            project_id=project_id,
            ttl_days=ttl_days,
            scope=scope_value,
            metadata=metadata or None,
        )
        return {
            "id": memory.id,
            "text": _attr(memory, "text"),
            "type": _attr(memory, "type"),
            "importance": _attr(memory, "importance"),
            "tags": _attr(memory, "tags") or [],
            "extracted_facts": _attr(memory, "extracted_facts", "extractedfacts"),
            "created_at": _iso_attr(memory, "created_at", "createdat"),
        }

    def remember_project_memory(self, *, text: str, user_id: str, project_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.remember(
            text=text,
            user_id=user_id,
            scope=MemoryScope.PROJECT,
            project_id=project_id,
            **kwargs,
        )

    def remember_agent_memory(
        self,
        *,
        text: str,
        user_id: str,
        project_id: str,
        agent_id: str,
        visibility: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self.remember(
            text=text,
            user_id=user_id,
            scope=MemoryScope.AGENT,
            project_id=project_id,
            agent_id=agent_id,
            visibility=visibility,
            **kwargs,
        )

    def remember_user_preference(self, *, text: str, user_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.remember(
            text=text,
            user_id=user_id,
            scope=MemoryScope.USER_PREFERENCE,
            **kwargs,
        )

    def recall(
        self,
        *,
        query: str,
        user_id: str,
        scope: str | MemoryScope | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        k: int = 5,
        min_importance: float | None = None,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        include_cross_scope: bool = False,
    ) -> dict[str, Any]:
        scope_value = self.access.enforce_recall_scope(
            scope=scope,
            project_id=project_id,
            agent_id=agent_id,
            session_id=session_id,
            include_cross_scope=include_cross_scope,
        )

        results = self.store.recall(
            query=query,
            user_id=user_id,
            session_id=session_id,
            k=k,
            min_importance=min_importance,
            memory_type=memory_type,
            tags=tags,
            agent_id=agent_id,
            project_id=project_id,
            scope=scope_value,
        )
        return {
            "query": query,
            "count": len(results),
            "results": [
                {
                    "memory_id": r.memory.id,
                    "text": r.memory.text,
                    "score": r.score,
                    "type": _attr(r.memory, "type"),
                    "importance": _attr(r.memory, "importance"),
                    "tags": _attr(r.memory, "tags") or [],
                    "session_id": _attr(r.memory, "session_id", "sessionid"),
                    "agent_id": _attr(r.memory, "agent_id", "agentid"),
                    "created_at": _iso_attr(r.memory, "created_at", "createdat"),
                }
                for r in results
            ],
        }

    def recall_project_context(self, *, query: str, user_id: str, project_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.recall(
            query=query,
            user_id=user_id,
            scope=MemoryScope.PROJECT,
            project_id=project_id,
            **kwargs,
        )

    def recall_agent_context(
        self,
        *,
        query: str,
        user_id: str,
        project_id: str,
        agent_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self.recall(
            query=query,
            user_id=user_id,
            scope=MemoryScope.AGENT,
            project_id=project_id,
            agent_id=agent_id,
            **kwargs,
        )

    def recall_user_preferences(self, *, query: str, user_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.recall(
            query=query,
            user_id=user_id,
            scope=MemoryScope.USER_PREFERENCE,
            **kwargs,
        )

    def list_memories(
        self,
        *,
        user_id: str,
        scope: str | MemoryScope | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        sort_by: str = "created_at",
        order: str = "desc",
    ) -> dict[str, Any]:
        scope_value = self.access.parse_scope(scope)
        if scope_value is not None:
            self.access.enforce_scope_fields(
                scope=scope_value,
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
            )

        memories = self.store.list(
            user_id=user_id,
            memory_type=memory_type,
            tags=tags,
            session_id=session_id,
            agent_id=agent_id,
            project_id=project_id,
            scope=scope_value,
            limit=limit,
            sort_by=sort_by,
            order=order,
        )
        return {
            "count": len(memories),
            "memories": [
                {
                    "id": m.id,
                    "text": m.text,
                    "type": _attr(m, "type"),
                    "importance": _attr(m, "importance"),
                    "tags": _attr(m, "tags") or [],
                    "session_id": _attr(m, "session_id", "sessionid"),
                    "created_at": _iso_attr(m, "created_at", "createdat"),
                }
                for m in memories
            ],
        }

    def update_memory(
        self,
        *,
        memory_id: str,
        user_id: str | None,
        text: str | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        self.access.enforce_update_delete(user_id=user_id, action="update")
        updated = self.store.update(
            memory_id=memory_id,
            text=text,
            importance=importance,
            tags=tags,
            user_id=user_id,
        )
        if not updated:
            raise AccessControlError("memory not found")
        return {
            "id": updated.id,
            "text": _attr(updated, "text"),
            "importance": _attr(updated, "importance"),
            "tags": _attr(updated, "tags") or [],
        }

    def delete_memory(self, *, memory_id: str, user_id: str | None) -> dict[str, Any]:
        self.access.enforce_update_delete(user_id=user_id, action="delete")
        success = self.store.delete(memory_id=memory_id, user_id=user_id)
        return {"success": bool(success)}

    def get_memory_statistics(self, *, user_id: str) -> dict[str, Any]:
        return self.store.stats(user_id=user_id)
