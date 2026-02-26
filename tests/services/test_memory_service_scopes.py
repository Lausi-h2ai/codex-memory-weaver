from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from hippocampai_mcp.services.memory_service import MemoryService


class StubStore:
    def __init__(self) -> None:
        self.calls = {}

    def remember(self, **kwargs):
        self.calls["remember"] = kwargs
        return SimpleNamespace(
            id="mem-1",
            text=kwargs["text"],
            type=kwargs["memory_type"],
            importance=kwargs.get("importance"),
            tags=kwargs.get("tags") or [],
            created_at=None,
        )

    def recall(self, **kwargs):
        self.calls["recall"] = kwargs
        return [
            SimpleNamespace(
                memory=SimpleNamespace(
                    id="m1",
                    text="remember me",
                    type="context",
                    importance=0.7,
                    tags=["scope:project"],
                    session_id="s1",
                    agent_id=kwargs.get("agent_id"),
                    created_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
                ),
                score=0.9,
            )
        ]

    def list(self, **kwargs):
        self.calls["list"] = kwargs
        return []

    def update(self, **kwargs):
        self.calls["update"] = kwargs
        return SimpleNamespace(id=kwargs["memory_id"], text=kwargs.get("text") or "", importance=None, tags=[])

    def delete(self, **kwargs):
        self.calls["delete"] = kwargs
        return True

    def stats(self, **kwargs):
        self.calls["stats"] = kwargs
        return {"total": 0}

    def add_relationship(self, **kwargs):
        self.calls["add_relationship"] = kwargs
        return True

    def get_related_memories(self, **kwargs):
        self.calls["get_related_memories"] = kwargs
        return [("m2", "related_to", 0.8)]

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


def test_remember_project_memory_scopes_to_project() -> None:
    store = StubStore()
    service = MemoryService(store)

    payload = service.remember_project_memory(
        text="Use pydantic v2",
        user_id="u1",
        project_id="proj-1",
    )

    assert payload["id"] == "mem-1"
    assert store.calls["remember"]["scope"].value == "project"
    assert store.calls["remember"]["project_id"] == "proj-1"


def test_remember_agent_memory_includes_visibility_metadata() -> None:
    store = StubStore()
    service = MemoryService(store)

    service.remember_agent_memory(
        text="Agent-specific workaround",
        user_id="u1",
        project_id="proj-1",
        agent_id="agent-1",
        visibility="shared",
    )

    call = store.calls["remember"]
    assert call["scope"].value == "agent"
    assert call["metadata"]["visibility"] == "shared"


def test_recall_project_context_is_scope_deterministic() -> None:
    store = StubStore()
    service = MemoryService(store)

    payload = service.recall_project_context(
        query="pydantic",
        user_id="u1",
        project_id="proj-1",
    )

    assert payload["count"] == 1
    assert store.calls["recall"]["scope"].value == "project"
    assert store.calls["recall"]["project_id"] == "proj-1"


def test_recall_passes_graph_search_mode() -> None:
    store = StubStore()
    service = MemoryService(store)

    service.recall(
        query="pydantic",
        user_id="u1",
        scope="project",
        project_id="proj-1",
        search_mode="graph_hybrid",
    )

    assert store.calls["recall"]["search_mode"] == "graph_hybrid"


def test_legacy_aliases_are_supported() -> None:
    store = StubStore()
    service = MemoryService(store)

    service.remember(
        text="legacy",
        user_id="u1",
        project="proj-legacy",
    )

    assert store.calls["remember"]["project_id"] == "proj-legacy"


def test_service_responses_use_snake_case() -> None:
    store = StubStore()
    service = MemoryService(store)

    payload = service.recall(
        query="ctx",
        user_id="u1",
        scope="user_preference",
    )

    result = payload["results"][0]
    assert "memory_id" in result
    assert "session_id" in result
    assert "agent_id" in result
    assert "created_at" in result


def test_user_preference_recall_disallows_project_filter() -> None:
    store = StubStore()
    service = MemoryService(store)

    with pytest.raises(ValueError):
        service.recall(
            query="pref",
            user_id="u1",
            scope="user_preference",
            project_id="proj-1",
        )


def test_unscoped_list_memories_does_not_force_user_preference_scope() -> None:
    store = StubStore()
    service = MemoryService(store)

    service.list_memories(user_id="u1", scope=None)

    assert store.calls["list"]["scope"] is None


def test_add_relationship_passthrough() -> None:
    store = StubStore()
    service = MemoryService(store)

    created = service.add_relationship(
        source_id="m1",
        target_id="m2",
        relation_type="related_to",
        weight=0.9,
    )

    assert created is True
    assert store.calls["add_relationship"]["source_id"] == "m1"
    assert store.calls["add_relationship"]["target_id"] == "m2"
    assert store.calls["add_relationship"]["relation_type"] == "related_to"
    assert store.calls["add_relationship"]["weight"] == 0.9


def test_get_related_memories_response_shape() -> None:
    store = StubStore()
    service = MemoryService(store)

    payload = service.get_related_memories(memory_id="m1", max_depth=2)

    assert payload["memory_id"] == "m1"
    assert payload["count"] == 1
    assert payload["related"][0]["memory_id"] == "m2"
    assert payload["related"][0]["relation_type"] == "related_to"
    assert payload["related"][0]["weight"] == 0.8
    assert store.calls["get_related_memories"]["max_depth"] == 2


def test_feedback_methods_passthrough() -> None:
    store = StubStore()
    service = MemoryService(store)

    submitted = service.submit_memory_feedback(
        memory_id="m1",
        user_id="u1",
        feedback_type="relevant",
        query="auth",
    )
    score = service.get_memory_feedback(memory_id="m1")
    stats = service.get_feedback_stats(user_id="u1")

    assert submitted["feedback_type"] == "relevant"
    assert score["memory_id"] == "m1"
    assert stats["user_id"] == "u1"
    assert store.calls["submit_memory_feedback"]["query"] == "auth"


def test_graph_extras_methods_passthrough() -> None:
    store = StubStore()
    service = MemoryService(store)

    clusters = service.get_memory_clusters(user_id="u1")
    subgraph = service.get_knowledge_subgraph(center_id="m1", radius=2, include_types=["memory"])
    extracted = service.extract_relationships(text="Alice uses Qdrant")

    assert clusters["count"] == 1
    assert clusters["clusters"][0]["cluster_id"] == "c1"
    assert subgraph["center_id"] == "m1"
    assert subgraph["subgraph"]["nodes"][0]["id"] == "m1"
    assert extracted["count"] == 1
    assert extracted["relationships"][0]["relation_type"] == "uses"


def test_recall_passes_temporal_filters_to_store_and_filters_response() -> None:
    store = StubStore()
    service = MemoryService(store)

    payload = service.recall(
        query="ctx",
        user_id="u1",
        scope="project",
        project_id="proj-1",
        created_after_iso="2026-01-10T00:00:00+00:00",
        created_before_iso="2026-01-20T00:00:00+00:00",
    )

    assert payload["count"] == 1
    assert store.calls["recall"]["created_after_iso"] == "2026-01-10T00:00:00+00:00"
    assert store.calls["recall"]["created_before_iso"] == "2026-01-20T00:00:00+00:00"


def test_recall_temporal_post_filter_drops_out_of_window_results() -> None:
    store = StubStore()

    def _recall(**kwargs):
        store.calls["recall"] = kwargs
        return [
            SimpleNamespace(
                memory=SimpleNamespace(
                    id="m1",
                    text="old",
                    type="context",
                    importance=0.7,
                    tags=["scope:project"],
                    session_id="s1",
                    agent_id=None,
                    created_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
                ),
                score=0.9,
            )
        ]

    store.recall = _recall
    service = MemoryService(store)

    payload = service.recall(
        query="ctx",
        user_id="u1",
        scope="project",
        project_id="proj-1",
        created_after_iso="2026-01-10T00:00:00+00:00",
    )

    assert payload["count"] == 0
    assert payload["results"] == []
