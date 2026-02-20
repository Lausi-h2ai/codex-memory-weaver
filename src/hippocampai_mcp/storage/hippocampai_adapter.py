"""HippocampAI-backed storage adapter."""

from __future__ import annotations

from typing import Any

from hippocampai_mcp.domain.models import MemoryScope


class HippocampAIAdapter:
    def __init__(self, client: Any) -> None:
        self._client = client

    @staticmethod
    def _encode_tags(
        *,
        scope: MemoryScope | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        tags: list[str] | None = None,
    ) -> list[str]:
        merged_tags: list[str] = list(tags or [])

        if scope is not None:
            merged_tags.append(f"scope:{scope.value}")
        if project_id:
            merged_tags.append(f"project:{project_id}")
        if agent_id:
            merged_tags.append(f"agent:{agent_id}")

        deduped: list[str] = []
        seen: set[str] = set()
        for tag in merged_tags:
            if tag not in seen:
                deduped.append(tag)
                seen.add(tag)
        return deduped

    @staticmethod
    def _encode_metadata(
        *,
        scope: MemoryScope | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        merged = dict(metadata or {})
        if scope is not None:
            merged["scope"] = scope.value
        if project_id:
            merged["project_id"] = project_id
        if agent_id:
            merged["agent_id"] = agent_id
        return merged

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
    ) -> Any:
        payload = dict(
            text=text,
            user_id=user_id,
            session_id=session_id,
            type=memory_type,
            importance=importance,
            tags=self._encode_tags(scope=scope, project_id=project_id, agent_id=agent_id, tags=tags),
            agent_id=agent_id,
            ttl_days=ttl_days,
        )
        encoded_metadata = self._encode_metadata(
            scope=scope, project_id=project_id, agent_id=agent_id, metadata=metadata
        )
        if encoded_metadata:
            payload["metadata"] = encoded_metadata

        try:
            return self._client.remember(**payload)
        except TypeError as exc:
            # Backward compatibility for HippocampAI clients that do not accept metadata.
            if "metadata" not in str(exc):
                raise
            payload.pop("metadata", None)
            return self._client.remember(**payload)

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
    ) -> list[Any]:
        filters: dict[str, Any] = {}
        encoded_tags = self._encode_tags(scope=scope, project_id=project_id, agent_id=agent_id, tags=tags)

        if min_importance is not None:
            filters["min_importance"] = min_importance
        if memory_type:
            filters["type"] = memory_type
        if encoded_tags:
            filters["tags"] = encoded_tags

        return self._client.recall(
            query=query,
            user_id=user_id,
            session_id=session_id,
            k=k,
            filters=filters or None,
        )

    def update(
        self,
        *,
        memory_id: str,
        text: str | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
    ) -> Any:
        return self._client.update_memory(
            memory_id=memory_id,
            text=text,
            importance=importance,
            tags=tags,
            user_id=user_id,
        )

    def delete(self, *, memory_id: str, user_id: str | None = None) -> bool:
        return self._client.delete_memory(memory_id=memory_id, user_id=user_id)

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
    ) -> list[Any]:
        encoded_tags = self._encode_tags(
            scope=scope, project_id=project_id, agent_id=agent_id, tags=tags
        )
        return self._client.get_memories(
            user_id=user_id,
            type=memory_type,
            tags=encoded_tags or None,
            session_id=session_id,
            agent_id=agent_id,
            limit=limit,
            sort_by=sort_by,
            order=order,
        )

    def stats(self, *, user_id: str) -> dict[str, Any]:
        return self._client.get_memory_statistics(user_id=user_id)
