from types import SimpleNamespace

from hippocampai_mcp.domain.models import MemoryScope
from hippocampai_mcp.storage.hippocampai_adapter import HippocampAIAdapter


class DummyClient:
    def __init__(self) -> None:
        self.calls: dict[str, dict] = {}

    def remember(self, **kwargs):
        self.calls["remember"] = kwargs
        return SimpleNamespace(id="m1", type="context", importance=1.0, tags=kwargs.get("tags", []))

    def recall(self, **kwargs):
        self.calls["recall"] = kwargs
        return []

    def update_memory(self, **kwargs):
        self.calls["update_memory"] = kwargs
        return SimpleNamespace(id=kwargs["memory_id"], text=kwargs.get("text", ""), tags=kwargs.get("tags", []))

    def delete_memory(self, **kwargs):
        self.calls["delete_memory"] = kwargs
        return True

    def get_memories(self, **kwargs):
        self.calls["get_memories"] = kwargs
        return []

    def get_memory_statistics(self, **kwargs):
        self.calls["get_memory_statistics"] = kwargs
        return {"total": 0}

    def add_relationship(self, **kwargs):
        self.calls["add_relationship"] = kwargs
        return True

    def get_related_memories(self, **kwargs):
        self.calls["get_related_memories"] = kwargs
        return [("m2", "related_to", 0.8)]


def test_remember_encodes_scope_tags_and_metadata() -> None:
    client = DummyClient()
    adapter = HippocampAIAdapter(client)

    adapter.remember(
        text="hello",
        user_id="u1",
        scope=MemoryScope.AGENT,
        project_id="proj-1",
        agent_id="agent-1",
        tags=["custom"],
        metadata={"source": "tool"},
    )

    payload = client.calls["remember"]
    assert payload["user_id"] == "u1"
    assert "scope:agent" in payload["tags"]
    assert "project:proj-1" in payload["tags"]
    assert "agent:agent-1" in payload["tags"]
    assert "custom" in payload["tags"]
    assert payload["metadata"]["scope"] == "agent"
    assert payload["metadata"]["project_id"] == "proj-1"
    assert payload["metadata"]["agent_id"] == "agent-1"


def test_recall_builds_scope_filters() -> None:
    client = DummyClient()
    adapter = HippocampAIAdapter(client)

    adapter.recall(
        query="auth",
        user_id="u1",
        k=4,
        scope=MemoryScope.PROJECT,
        project_id="proj-2",
        min_importance=0.6,
        tags=["architecture"],
    )

    payload = client.calls["recall"]
    assert payload["user_id"] == "u1"
    assert payload["k"] == 4
    assert payload["filters"]["min_importance"] == 0.6
    assert "scope:project" in payload["filters"]["tags"]
    assert "project:proj-2" in payload["filters"]["tags"]
    assert "architecture" in payload["filters"]["tags"]


def test_recall_includes_search_mode_filter_for_graph_hybrid() -> None:
    client = DummyClient()
    adapter = HippocampAIAdapter(client)

    adapter.recall(
        query="auth",
        user_id="u1",
        search_mode="graph_hybrid",
    )

    payload = client.calls["recall"]
    assert payload["filters"]["search_mode"] == "graph_hybrid"


def test_list_uses_scope_tags_for_filter_generation() -> None:
    client = DummyClient()
    adapter = HippocampAIAdapter(client)

    adapter.list(
        user_id="u1",
        scope=MemoryScope.USER_PREFERENCE,
        limit=10,
    )

    payload = client.calls["get_memories"]
    assert payload["user_id"] == "u1"
    assert payload["limit"] == 10
    assert payload["filters"]["tags"] == ["scope:user_preference"]


def test_add_relationship_maps_to_client_signature() -> None:
    client = DummyClient()
    adapter = HippocampAIAdapter(client)

    created = adapter.add_relationship(
        source_id="m1",
        target_id="m2",
        relation_type="related_to",
        weight=0.9,
    )

    assert created is True
    payload = client.calls["add_relationship"]
    assert payload["source_id"] == "m1"
    assert payload["target_id"] == "m2"
    assert payload["relation_type"] == "related_to"
    assert payload["weight"] == 0.9


def test_get_related_memories_maps_depth_and_returns_client_result() -> None:
    client = DummyClient()
    adapter = HippocampAIAdapter(client)

    related = adapter.get_related_memories(memory_id="m1", max_depth=2)

    assert related == [("m2", "related_to", 0.8)]
    payload = client.calls["get_related_memories"]
    assert payload["memory_id"] == "m1"
    assert payload["max_depth"] == 2
