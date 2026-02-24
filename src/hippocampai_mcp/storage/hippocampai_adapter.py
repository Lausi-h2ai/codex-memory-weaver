"""HippocampAI-backed storage adapter."""

from __future__ import annotations

from typing import Any

from hippocampai_mcp.domain.models import MemoryScope


class HippocampAIAdapter:
    _relation_type_enum: Any | None = None
    _relation_type_enum_resolved: bool = False

    def __init__(self, client: Any) -> None:
        self._client = client

    @classmethod
    def _resolve_relation_type_enum(cls) -> Any | None:
        if cls._relation_type_enum_resolved:
            return cls._relation_type_enum
        cls._relation_type_enum_resolved = True

        try:
            from hippocampai.graph.memory_graph import RelationType as relation_type_enum

            cls._relation_type_enum = relation_type_enum
            return cls._relation_type_enum
        except Exception:
            pass

        try:
            from hippocampai.client import RelationType as relation_type_enum

            cls._relation_type_enum = relation_type_enum
            return cls._relation_type_enum
        except Exception:
            cls._relation_type_enum = None
            return None

    @classmethod
    def _coerce_relation_type(cls, relation_type: str) -> Any:
        relation_type_enum = cls._resolve_relation_type_enum()
        if relation_type_enum is None:
            return relation_type
        try:
            return relation_type_enum(relation_type)
        except Exception:
            enum_member = getattr(relation_type_enum, relation_type.upper(), None)
            return enum_member if enum_member is not None else relation_type

    @classmethod
    def _coerce_relation_types(cls, relation_types: list[str] | None) -> list[Any] | None:
        if relation_types is None:
            return None
        return [cls._coerce_relation_type(relation_type) for relation_type in relation_types]

    @staticmethod
    def _relationship_result_to_success(result: Any) -> bool:
        if isinstance(result, bool):
            return result
        if result is None:
            return True
        if isinstance(result, dict):
            if "success" in result:
                return bool(result["success"])
            return True
        success_attr = getattr(result, "success", None)
        if success_attr is not None:
            return bool(success_attr)
        return bool(result)

    def _ensure_graph_nodes(self, memory_ids: list[str]) -> bool:
        graph = getattr(self._client, "graph", None)
        add_memory = getattr(graph, "add_memory", None)
        get_memory = getattr(self._client, "get_memory", None)
        if not callable(add_memory) or not callable(get_memory):
            return False

        added_any = False
        graph_nodes = getattr(graph, "graph", None)

        for memory_id in memory_ids:
            try:
                if graph_nodes is not None and memory_id in graph_nodes:
                    continue
            except Exception:
                pass

            try:
                memory = get_memory(memory_id)
            except Exception:
                continue

            user_id = getattr(memory, "user_id", None)
            if memory is None or not user_id:
                continue

            metadata: dict[str, Any] = {}
            memory_type = getattr(memory, "type", None)
            if memory_type is not None:
                metadata["type"] = getattr(memory_type, "value", memory_type)
            created_at = getattr(memory, "created_at", None)
            if created_at is not None:
                metadata["created_at"] = (
                    created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
                )
            add_memory(memory_id, user_id, metadata or None)
            added_any = True
        return added_any

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
            "relation_type": self._coerce_relation_type(relation_type),
            "weight": weight,
        }
        try:
            result = self._client.add_relationship(**payload)
        except TypeError as exc:
            if "weight" not in str(exc):
                raise
            payload.pop("weight", None)
            result = self._client.add_relationship(**payload)

        success = self._relationship_result_to_success(result)
        if success:
            return True

        if self._ensure_graph_nodes([source_id, target_id]):
            retry_result = self._client.add_relationship(**payload)
            return self._relationship_result_to_success(retry_result)
        return False

    def get_related_memories(
        self,
        *,
        memory_id: str,
        relation_types: list[str] | None = None,
        max_depth: int = 1,
    ) -> list[Any]:
        payload: dict[str, Any] = {
            "memory_id": memory_id,
            "relation_types": self._coerce_relation_types(relation_types),
            "max_depth": max_depth,
        }
        try:
            return self._client.get_related_memories(**payload)
        except TypeError as exc:
            if "relation_types" not in str(exc):
                raise

        fallback_payload: dict[str, Any] = {"memory_id": memory_id, "max_depth": max_depth}
        if relation_types and len(relation_types) == 1:
            fallback_payload["relation_type"] = self._coerce_relation_type(relation_types[0])
        return self._client.get_related_memories(**fallback_payload)

    def submit_memory_feedback(
        self,
        *,
        memory_id: str,
        user_id: str,
        feedback_type: str,
        query: str | None = None,
    ) -> dict[str, Any]:
        if not hasattr(self._client, "submit_memory_feedback"):
            raise NotImplementedError("submit_memory_feedback is not available on this client")
        return self._client.submit_memory_feedback(
            memory_id=memory_id,
            user_id=user_id,
            feedback_type=feedback_type,
            query=query,
        )

    def get_memory_feedback(self, *, memory_id: str) -> dict[str, Any]:
        if not hasattr(self._client, "get_memory_feedback"):
            raise NotImplementedError("get_memory_feedback is not available on this client")
        return self._client.get_memory_feedback(memory_id=memory_id)

    def get_feedback_stats(self, *, user_id: str) -> dict[str, Any]:
        if not hasattr(self._client, "get_feedback_stats"):
            raise NotImplementedError("get_feedback_stats is not available on this client")
        return self._client.get_feedback_stats(user_id=user_id)

    def get_memory_clusters(self, *, user_id: str) -> list[Any]:
        if not hasattr(self._client, "get_memory_clusters"):
            raise NotImplementedError("get_memory_clusters is not available on this client")
        return self._client.get_memory_clusters(user_id=user_id)

    def get_knowledge_subgraph(
        self,
        *,
        center_id: str,
        radius: int = 2,
        include_types: list[str] | None = None,
    ) -> dict[str, Any]:
        if not hasattr(self._client, "get_knowledge_subgraph"):
            raise NotImplementedError("get_knowledge_subgraph is not available on this client")
        return self._client.get_knowledge_subgraph(
            center_id=center_id,
            radius=radius,
            include_types=include_types,
        )

    def extract_relationships(self, *, text: str) -> list[Any]:
        if not hasattr(self._client, "extract_relationships"):
            raise NotImplementedError("extract_relationships is not available on this client")
        return self._client.extract_relationships(text=text)
