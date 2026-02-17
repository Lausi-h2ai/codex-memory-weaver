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

    def update_memory(self, *, memory_id, text=None, importance=None, tags=None, user_id=None):
        self.last_call["update_memory"] = {
            "memory_id": memory_id,
            "text": text,
            "importance": importance,
            "tags": tags,
            "user_id": user_id,
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
        type=None,
        tags=None,
        session_id=None,
        agent_id=None,
        limit=50,
        sort_by="created_at",
        order="desc",
    ):
        self.last_call["get_memories"] = {
            "user_id": user_id,
            "type": type,
            "tags": tags,
            "session_id": session_id,
            "agent_id": agent_id,
            "limit": limit,
            "sort_by": sort_by,
            "order": order,
        }
        return []

    def get_memory_statistics(self, *, user_id):
        self.last_call["get_memory_statistics"] = {"user_id": user_id}
        return {"total": 0}


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
        scope=MemoryScope.PROJECT,
        project_id="proj-1",
    )
    assert "recall" in client.last_call

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
