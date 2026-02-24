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
        search_mode: str | None = None,
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
        if search_mode:
            filters["search_mode"] = search_mode
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
        # user_id is enforced at service layer; HippocampAI v0.5.0 update_memory does not accept it.
        _ = user_id
        return self._client.update_memory(
            memory_id=memory_id,
            text=text,
            importance=importance,
            tags=tags,
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
        filters: dict[str, Any] = {}
        if memory_type is not None:
            filters["type"] = memory_type
        if encoded_tags:
            filters["tags"] = encoded_tags
        if session_id is not None:
            filters["session_id"] = session_id
        if agent_id is not None:
            filters["agent_id"] = agent_id

        return self._client.get_memories(
            user_id=user_id,
            filters=filters or None,
            limit=limit,
        )

    def stats(self, *, user_id: str) -> dict[str, Any]:
        return self._client.get_memory_statistics(user_id=user_id)

    def add_relationship(
        self,
        *,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
    ) -> bool:
        payload = {
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": relation_type,
            "weight": weight,
        }
        try:
            return bool(self._client.add_relationship(**payload))
        except TypeError as exc:
            if "weight" not in str(exc):
                raise
            payload.pop("weight", None)
            return bool(self._client.add_relationship(**payload))

    def get_related_memories(
        self,
        *,
        memory_id: str,
        relation_types: list[str] | None = None,
        max_depth: int = 1,
    ) -> list[Any]:
        payload: dict[str, Any] = {
            "memory_id": memory_id,
            "relation_types": relation_types,
            "max_depth": max_depth,
        }
        try:
            return self._client.get_related_memories(**payload)
        except TypeError as exc:
            if "relation_types" not in str(exc):
                raise

        fallback_payload: dict[str, Any] = {"memory_id": memory_id, "max_depth": max_depth}
        if relation_types and len(relation_types) == 1:
            fallback_payload["relation_type"] = relation_types[0]
        return self._client.get_related_memories(**fallback_payload)
