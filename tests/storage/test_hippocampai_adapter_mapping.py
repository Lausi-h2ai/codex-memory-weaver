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


class EnumStrictRelationshipClient(DummyClient):
    def add_relationship(self, **kwargs):
        # Mimic HippocampAI client behavior that expects enum-like relation types.
        _ = kwargs["relation_type"].value
        return super().add_relationship(**kwargs)

    def get_related_memories(self, **kwargs):
        relation_types = kwargs.get("relation_types") or []
        for relation_type in relation_types:
            _ = relation_type.value
        return super().get_related_memories(**kwargs)


class NoneReturningRelationshipClient(DummyClient):
    def add_relationship(self, **kwargs):
        self.calls["add_relationship"] = kwargs
        return None


class _GraphStub:
    def __init__(self) -> None:
        self.nodes: set[str] = set()

    def add_memory(self, memory_id: str, user_id: str, metadata=None) -> None:
        _ = user_id
        _ = metadata
        self.nodes.add(memory_id)


class GraphHydrationRelationshipClient(DummyClient):
    def __init__(self) -> None:
        super().__init__()
        self.graph = _GraphStub()
        self._memories = {
            "m1": SimpleNamespace(id="m1", user_id="u1", type="fact", created_at=None),
            "m2": SimpleNamespace(id="m2", user_id="u1", type="fact", created_at=None),
        }
        self.calls["add_relationship_attempts"] = []

    def get_memory(self, memory_id: str):
        return self._memories.get(memory_id)

    def add_relationship(self, **kwargs):
        self.calls["add_relationship_attempts"].append(dict(kwargs))
        if kwargs["source_id"] not in self.graph.nodes or kwargs["target_id"] not in self.graph.nodes:
            return False
        self.calls["add_relationship"] = kwargs
        return True


class FeedbackClient(DummyClient):
    def submit_memory_feedback(self, **kwargs):
        self.calls["submit_memory_feedback"] = kwargs
        return {"memory_id": kwargs["memory_id"], "feedback_type": kwargs["feedback_type"]}

    def get_memory_feedback(self, **kwargs):
        self.calls["get_memory_feedback"] = kwargs
        return {"memory_id": kwargs["memory_id"], "score": 0.75}

    def get_feedback_stats(self, **kwargs):
        self.calls["get_feedback_stats"] = kwargs
        return {"user_id": kwargs["user_id"], "stats": {"relevant": 2}}

    def get_memory_clusters(self, **kwargs):
        self.calls["get_memory_clusters"] = kwargs
        return [{"cluster_id": "c1", "memory_ids": ["m1", "m2"]}]

    def get_knowledge_subgraph(self, **kwargs):
        self.calls["get_knowledge_subgraph"] = kwargs
        return {"nodes": [{"id": "m1"}], "edges": [{"source": "m1", "target": "m2"}]}

    def extract_relationships(self, **kwargs):
        self.calls["extract_relationships"] = kwargs
        return [{"source": "alice", "target": "qdrant", "relation_type": "uses"}]


def _as_relation_value(value):
    return getattr(value, "value", value)


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
    assert _as_relation_value(payload["relation_type"]) == "related_to"
    assert payload["weight"] == 0.9


def test_add_relationship_coerces_string_relation_type_for_enum_clients() -> None:
    client = EnumStrictRelationshipClient()
    adapter = HippocampAIAdapter(client)

    created = adapter.add_relationship(
        source_id="m1",
        target_id="m2",
        relation_type="related_to",
        weight=0.9,
    )

    assert created is True
    payload = client.calls["add_relationship"]
    assert payload["relation_type"].value == "related_to"


def test_add_relationship_treats_none_result_as_success() -> None:
    client = NoneReturningRelationshipClient()
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


def test_add_relationship_retries_after_graph_node_hydration() -> None:
    client = GraphHydrationRelationshipClient()
    adapter = HippocampAIAdapter(client)

    created = adapter.add_relationship(
        source_id="m1",
        target_id="m2",
        relation_type="related_to",
        weight=0.9,
    )

    assert created is True
    assert client.graph.nodes == {"m1", "m2"}
    assert len(client.calls["add_relationship_attempts"]) == 2


def test_get_related_memories_maps_depth_and_returns_client_result() -> None:
    client = DummyClient()
    adapter = HippocampAIAdapter(client)

    related = adapter.get_related_memories(memory_id="m1", max_depth=2)

    assert related == [("m2", "related_to", 0.8)]
    payload = client.calls["get_related_memories"]
    assert payload["memory_id"] == "m1"
    assert payload["max_depth"] == 2


def test_get_related_memories_coerces_relation_types_for_enum_clients() -> None:
    client = EnumStrictRelationshipClient()
    adapter = HippocampAIAdapter(client)

    related = adapter.get_related_memories(
        memory_id="m1",
        relation_types=["related_to"],
        max_depth=2,
    )

    assert related == [("m2", "related_to", 0.8)]
    payload = client.calls["get_related_memories"]
    assert _as_relation_value(payload["relation_types"][0]) == "related_to"


def test_feedback_methods_passthrough_to_client() -> None:
    client = FeedbackClient()
    adapter = HippocampAIAdapter(client)

    submitted = adapter.submit_memory_feedback(
        memory_id="m1",
        user_id="u1",
        feedback_type="relevant",
        query="preferences",
    )
    score = adapter.get_memory_feedback(memory_id="m1")
    stats = adapter.get_feedback_stats(user_id="u1")

    assert submitted["feedback_type"] == "relevant"
    assert client.calls["submit_memory_feedback"]["query"] == "preferences"
    assert score["memory_id"] == "m1"
    assert stats["user_id"] == "u1"


def test_feedback_methods_raise_not_supported_when_backend_missing() -> None:
    client = DummyClient()
    adapter = HippocampAIAdapter(client)

    try:
        adapter.submit_memory_feedback(memory_id="m1", user_id="u1", feedback_type="relevant")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("expected NotImplementedError for submit_memory_feedback")

    try:
        adapter.get_memory_feedback(memory_id="m1")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("expected NotImplementedError for get_memory_feedback")

    try:
        adapter.get_feedback_stats(user_id="u1")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("expected NotImplementedError for get_feedback_stats")


def test_graph_extras_methods_passthrough_to_client() -> None:
    client = FeedbackClient()
    adapter = HippocampAIAdapter(client)

    clusters = adapter.get_memory_clusters(user_id="u1")
    subgraph = adapter.get_knowledge_subgraph(center_id="m1", radius=2, include_types=["memory"])
    relationships = adapter.extract_relationships(text="Alice uses Qdrant")

    assert clusters[0]["cluster_id"] == "c1"
    assert client.calls["get_memory_clusters"]["user_id"] == "u1"
    assert subgraph["nodes"][0]["id"] == "m1"
    assert client.calls["get_knowledge_subgraph"]["center_id"] == "m1"
    assert relationships[0]["relation_type"] == "uses"
    assert client.calls["extract_relationships"]["text"] == "Alice uses Qdrant"


def test_graph_extras_methods_raise_not_supported_when_backend_missing() -> None:
    client = DummyClient()
    adapter = HippocampAIAdapter(client)

    try:
        adapter.get_memory_clusters(user_id="u1")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("expected NotImplementedError for get_memory_clusters")

    try:
        adapter.get_knowledge_subgraph(center_id="m1", radius=2, include_types=None)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("expected NotImplementedError for get_knowledge_subgraph")

    try:
        adapter.extract_relationships(text="Alice uses Qdrant")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("expected NotImplementedError for extract_relationships")
