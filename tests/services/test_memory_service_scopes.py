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
                    created_at=None,
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
