from types import SimpleNamespace

from hippocampai_mcp.domain.models import MemoryScope
from hippocampai_mcp.storage.hippocampai_adapter import HippocampAIAdapter


class StrictV05Client:
    def __init__(self) -> None:
        self.last_call = {}

    def remember(
        self,
        *,
        text,
        user_id,
        session_id=None,
        type="context",
        importance=None,
        tags=None,
        agent_id=None,
        ttl_days=None,
        metadata=None,
    ):
        self.last_call["remember"] = {
            "text": text,
            "user_id": user_id,
            "session_id": session_id,
            "type": type,
            "importance": importance,
            "tags": tags,
            "agent_id": agent_id,
            "ttl_days": ttl_days,
            "metadata": metadata,
        }
        return SimpleNamespace(
            id="m1",
            text=text,
            type=type,
            importance=importance,
            tags=tags or [],
            created_at=None,
            extracted_facts=[],
            session_id=session_id,
            agent_id=agent_id,
        )

    def recall(self, *, query, user_id, session_id=None, k=5, filters=None):
        self.last_call["recall"] = {
            "query": query,
            "user_id": user_id,
            "session_id": session_id,
            "k": k,
            "filters": filters,
        }
        return []

    def update_memory(self, *, memory_id, text=None, importance=None, tags=None):
        self.last_call["update_memory"] = {
            "memory_id": memory_id,
            "text": text,
            "importance": importance,
            "tags": tags,
        }
        return SimpleNamespace(id=memory_id)

    def delete_memory(self, *, memory_id, user_id=None):
        self.last_call["delete_memory"] = {
            "memory_id": memory_id,
            "user_id": user_id,
        }
        return True

    def get_memories(
        self,
        *,
        user_id,
        filters=None,
        limit=50,
    ):
        self.last_call["get_memories"] = {
            "user_id": user_id,
            "filters": filters,
            "limit": limit,
        }
        return []

    def get_memory_statistics(self, *, user_id):
        self.last_call["get_memory_statistics"] = {"user_id": user_id}
        return {"total": 0}

    def add_relationship(self, *, source_id, target_id, relation_type, weight=1.0):
        self.last_call["add_relationship"] = {
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": relation_type,
            "weight": weight,
        }
        return True

    def get_related_memories(self, *, memory_id, relation_types=None, max_depth=1):
        self.last_call["get_related_memories"] = {
            "memory_id": memory_id,
            "relation_types": relation_types,
            "max_depth": max_depth,
        }
        return [("m2", "related_to", 0.8)]


def test_adapter_uses_v05_keyword_names_for_all_core_methods() -> None:
    client = StrictV05Client()
    adapter = HippocampAIAdapter(client)

    adapter.remember(
        text="note",
        user_id="u1",
        scope=MemoryScope.PROJECT,
        project_id="proj-1",
    )
    assert "remember" in client.last_call

    adapter.list(
        user_id="u1",
        memory_type="context",
        scope=MemoryScope.PROJECT,
        project_id="proj-1",
    )
    assert "get_memories" in client.last_call

    adapter.recall(
        query="note",
        user_id="u1",
        session_id="s1",
        k=3,
        memory_type="context",
        search_mode="graph_hybrid",
        scope=MemoryScope.PROJECT,
        project_id="proj-1",
        created_after_iso="2026-01-01T00:00:00+00:00",
        created_before_iso="2026-02-01T00:00:00+00:00",
    )
    assert "recall" in client.last_call
    assert client.last_call["recall"]["filters"]["search_mode"] == "graph_hybrid"
    assert client.last_call["recall"]["filters"]["created_after"] == "2026-01-01T00:00:00+00:00"
    assert client.last_call["recall"]["filters"]["created_before"] == "2026-02-01T00:00:00+00:00"

    adapter.update(
        memory_id="m1",
        text="updated",
        user_id="u1",
    )
    assert "update_memory" in client.last_call

    assert adapter.delete(memory_id="m1", user_id="u1") is True
    assert "delete_memory" in client.last_call

    stats = adapter.stats(user_id="u1")
    assert stats == {"total": 0}
    assert "get_memory_statistics" in client.last_call

    assert adapter.add_relationship(
        source_id="m1",
        target_id="m2",
        relation_type="related_to",
        weight=0.9,
    ) is True
    assert "add_relationship" in client.last_call

    related = adapter.get_related_memories(memory_id="m1", max_depth=2)
    assert related == [("m2", "related_to", 0.8)]
    assert "get_related_memories" in client.last_call
