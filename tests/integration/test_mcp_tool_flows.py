import inspect
from types import SimpleNamespace

import pytest

from hippocampai_mcp.services.memory_service import MemoryService


class InMemoryStore:
    def __init__(self) -> None:
        self.items = []
        self.next_id = 1

    def remember(self, **kwargs):
        item = {
            "id": f"m{self.next_id}",
            "text": kwargs["text"],
            "user_id": kwargs["user_id"],
            "project_id": kwargs.get("project_id"),
            "agent_id": kwargs.get("agent_id"),
            "session_id": kwargs.get("session_id"),
            "scope": kwargs.get("scope").value if kwargs.get("scope") else None,
            "type": kwargs.get("memory_type", "context"),
            "importance": kwargs.get("importance"),
            "tags": kwargs.get("tags") or [],
            "metadata": kwargs.get("metadata") or {},
            "created_at": None,
        }
        self.next_id += 1
        self.items.append(item)
        return SimpleNamespace(**item)

    def _matches_scope(self, item, scope, project_id, agent_id):
        if scope is None:
            return True
        if scope.value == "project":
            return item["scope"] == "project" and item["project_id"] == project_id
        if scope.value == "agent":
            return (
                item["scope"] == "agent"
                and item["project_id"] == project_id
                and item["agent_id"] == agent_id
            )
        if scope.value == "user_preference":
            return item["scope"] == "user_preference"
        return True

    def recall(self, **kwargs):
        scope = kwargs.get("scope")
        project_id = kwargs.get("project_id")
        agent_id = kwargs.get("agent_id")
        user_id = kwargs.get("user_id")
        query = kwargs.get("query", "").lower()

        results = []
        for item in self.items:
            if item["user_id"] != user_id:
                continue
            if query and query not in item["text"].lower():
                continue
            if not self._matches_scope(item, scope, project_id, agent_id):
                continue
            results.append(SimpleNamespace(memory=SimpleNamespace(**item), score=1.0))
        return results

    def list(self, **kwargs):
        scope = kwargs.get("scope")
        project_id = kwargs.get("project_id")
        agent_id = kwargs.get("agent_id")
        user_id = kwargs.get("user_id")
        results = []
        for item in self.items:
            if item["user_id"] != user_id:
                continue
            if not self._matches_scope(item, scope, project_id, agent_id):
                continue
            results.append(SimpleNamespace(**item))
        return results

    def update(self, **kwargs):
        return None

    def delete(self, **kwargs):
        return False

    def stats(self, **kwargs):
        return {"total": len(self.items)}


def test_scope_isolation_across_projects() -> None:
    service = MemoryService(InMemoryStore())
    service.remember_project_memory(text="Alpha auth", user_id="u1", project_id="alpha")
    service.remember_project_memory(text="Beta auth", user_id="u1", project_id="beta")

    alpha = service.recall_project_context(query="auth", user_id="u1", project_id="alpha")
    beta = service.recall_project_context(query="auth", user_id="u1", project_id="beta")

    assert alpha["count"] == 1
    assert alpha["results"][0]["text"] == "Alpha auth"
    assert beta["count"] == 1
    assert beta["results"][0]["text"] == "Beta auth"


def test_user_preference_recall_cross_project() -> None:
    service = MemoryService(InMemoryStore())
    service.remember_user_preference(text="Prefer pytest -q", user_id="u1")
    service.remember_project_memory(text="Alpha-only detail", user_id="u1", project_id="alpha")

    prefs = service.recall_user_preferences(query="Prefer", user_id="u1")

    assert prefs["count"] == 1
    assert prefs["results"][0]["text"] == "Prefer pytest -q"


def test_agent_scope_isolation() -> None:
    service = MemoryService(InMemoryStore())
    service.remember_agent_memory(
        text="Agent A workaround",
        user_id="u1",
        project_id="alpha",
        agent_id="agent-a",
    )
    service.remember_agent_memory(
        text="Agent B workaround",
        user_id="u1",
        project_id="alpha",
        agent_id="agent-b",
    )

    a = service.recall_agent_context(
        query="workaround", user_id="u1", project_id="alpha", agent_id="agent-a"
    )
    b = service.recall_agent_context(
        query="workaround", user_id="u1", project_id="alpha", agent_id="agent-b"
    )

    assert a["count"] == 1
    assert a["results"][0]["text"] == "Agent A workaround"
    assert b["count"] == 1
    assert b["results"][0]["text"] == "Agent B workaround"


def test_hippocampai_v05_signature_expectations() -> None:
    hp = pytest.importorskip("hippocampai")
    try:
        MemoryClient = hp.MemoryClient
    except ModuleNotFoundError as exc:
        pytest.skip(f"HippocampAI optional client deps unavailable: {exc}")
    remember_params = inspect.signature(MemoryClient.remember).parameters
    get_memories_params = inspect.signature(MemoryClient.get_memories).parameters

    assert "user_id" in remember_params
    assert "session_id" in remember_params
    assert "type" in remember_params
    assert "agent_id" in remember_params

    assert "user_id" in get_memories_params
    assert "type" in get_memories_params
    assert "session_id" in get_memories_params
    assert "agent_id" in get_memories_params
