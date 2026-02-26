"""Service layer for memory tool behavior and scope handling."""

from __future__ import annotations

import math
from datetime import datetime, timezone
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


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _recency_decay_factor(*, now: datetime, created_at: datetime | None, half_life_days: float) -> float:
    if created_at is None or half_life_days <= 0:
        return 1.0
    age_days = max((now - created_at).total_seconds(), 0.0) / 86_400.0
    return math.exp(-math.log(2.0) * age_days / half_life_days)


def _normalize_usage_signal(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, numeric))


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
        search_mode: str | None = None,
        tags: list[str] | None = None,
        include_cross_scope: bool = False,
        recency_half_life_days: float | None = None,
        recency_weight: float = 0.25,
        usage_weight: float = 0.35,
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
            search_mode=search_mode,
            tags=tags,
            agent_id=agent_id,
            project_id=project_id,
            scope=scope_value,
        )
        now = datetime.now(timezone.utc)
        recency_enabled = recency_half_life_days is not None and recency_half_life_days > 0
        bounded_recency_weight = max(0.0, min(1.0, recency_weight))
        bounded_usage_weight = max(0.0, min(1.0, usage_weight))

        response_results: list[dict[str, Any]] = []
        for r in results:
            memory = r.memory
            base_score = float(r.score)
            created_dt = _parse_datetime(_attr(memory, "created_at", "createdat"))
            recency_factor = (
                _recency_decay_factor(
                    now=now,
                    created_at=created_dt,
                    half_life_days=recency_half_life_days or 0.0,
                )
                if recency_enabled
                else 1.0
            )
            recency_multiplier = (1.0 - bounded_recency_weight) + (bounded_recency_weight * recency_factor)

            metadata = _attr(memory, "metadata") or {}
            usage_raw_candidates = [
                _attr(memory, "usage_signal"),
                _attr(memory, "feedback_score"),
                (metadata.get("usage_signal") if isinstance(metadata, dict) else None),
                (metadata.get("feedback_score") if isinstance(metadata, dict) else None),
            ]
            usage_raw = next((candidate for candidate in usage_raw_candidates if candidate is not None), None)
            usage_signal = _normalize_usage_signal(usage_raw)
            usage_multiplier = 1.0
            if usage_signal is not None:
                usage_multiplier = 1.0 + (bounded_usage_weight * usage_signal)

            adjusted_score = base_score * recency_multiplier * usage_multiplier

            response_results.append(
                {
                    "memory_id": memory.id,
                    "text": memory.text,
                    "score": adjusted_score,
                    "base_score": base_score,
                    "type": _attr(memory, "type"),
                    "importance": _attr(memory, "importance"),
                    "tags": _attr(memory, "tags") or [],
                    "session_id": _attr(memory, "session_id", "sessionid"),
                    "agent_id": _attr(memory, "agent_id", "agentid"),
                    "created_at": _iso_attr(memory, "created_at", "createdat"),
                    "recency_factor": recency_factor,
                    "usage_signal": usage_signal,
                }
            )

        response_results.sort(key=lambda item: item["score"], reverse=True)

        return {
            "query": query,
            "count": len(results),
            "results": response_results,
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

    def submit_memory_feedback(
        self,
        *,
        memory_id: str,
        user_id: str,
        feedback_type: str,
        query: str | None = None,
    ) -> dict[str, Any]:
        return self.store.submit_memory_feedback(
            memory_id=memory_id,
            user_id=user_id,
            feedback_type=feedback_type,
            query=query,
        )

    def get_memory_feedback(self, *, memory_id: str) -> dict[str, Any]:
        return self.store.get_memory_feedback(memory_id=memory_id)

    def get_feedback_stats(self, *, user_id: str) -> dict[str, Any]:
        return self.store.get_feedback_stats(user_id=user_id)

    def get_memory_clusters(self, *, user_id: str) -> dict[str, Any]:
        clusters = self.store.get_memory_clusters(user_id=user_id)
        normalized: list[dict[str, Any]] = []
        for cluster in clusters:
            if isinstance(cluster, set):
                memory_ids = sorted(cluster)
                normalized.append({"memory_ids": memory_ids, "memory_count": len(memory_ids)})
                continue
            if isinstance(cluster, dict):
                cluster_dict = dict(cluster)
                if "memory_ids" in cluster_dict and "memory_count" not in cluster_dict:
                    cluster_dict["memory_count"] = len(cluster_dict["memory_ids"])
                normalized.append(cluster_dict)
                continue
            memories = _attr(cluster, "memories")
            if memories is not None:
                memory_ids = [_attr(m, "id") for m in memories]
                normalized.append(
                    {
                        "topic": _attr(cluster, "topic"),
                        "memory_ids": memory_ids,
                        "memory_count": len(memory_ids),
                    }
                )
                continue
            normalized.append({"value": cluster})
        return {"count": len(normalized), "clusters": normalized}

    def get_knowledge_subgraph(
        self,
        *,
        center_id: str,
        radius: int = 2,
        include_types: list[str] | None = None,
    ) -> dict[str, Any]:
        subgraph = self.store.get_knowledge_subgraph(
            center_id=center_id,
            radius=radius,
            include_types=include_types,
        )
        return {"center_id": center_id, "subgraph": subgraph}

    def extract_relationships(self, *, text: str) -> dict[str, Any]:
        relationships = self.store.extract_relationships(text=text)
        normalized = []
        for rel in relationships:
            if isinstance(rel, dict):
                normalized.append(rel)
            else:
                normalized.append(
                    {
                        "source": _attr(rel, "source", "source_id"),
                        "target": _attr(rel, "target", "target_id"),
                        "relation_type": _attr(rel, "relation_type", "type"),
                        "confidence": _attr(rel, "confidence"),
                    }
                )
        return {"count": len(normalized), "relationships": normalized}

    def add_relationship(
        self,
        *,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
    ) -> bool:
        return self.store.add_relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
        )

    def get_related_memories(
        self,
        *,
        memory_id: str,
        relation_types: list[str] | None = None,
        max_depth: int = 1,
    ) -> dict[str, Any]:
        related = self.store.get_related_memories(
            memory_id=memory_id,
            relation_types=relation_types,
            max_depth=max_depth,
        )
        normalized: list[dict[str, Any]] = []
        for item in related:
            if isinstance(item, tuple):
                target_id = item[0] if len(item) > 0 else None
                relation_type = item[1] if len(item) > 1 else None
                weight = item[2] if len(item) > 2 else None
            else:
                target_id = _attr(item, "memory_id", "target_id")
                relation_type = _attr(item, "relation_type", "type")
                weight = _attr(item, "weight")
            normalized.append(
                {
                    "memory_id": target_id,
                    "relation_type": relation_type,
                    "weight": weight,
                }
            )
        return {
            "memory_id": memory_id,
            "count": len(normalized),
            "related": normalized,
        }
