#!/usr/bin/env python
"""Lightweight smoke runner for scoped memory flows via service layer."""

from __future__ import annotations

from types import SimpleNamespace

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
            "scope": kwargs.get("scope").value if kwargs.get("scope") else None,
            "type": kwargs.get("memory_type", "context"),
            "importance": kwargs.get("importance"),
            "tags": kwargs.get("tags") or [],
            "created_at": None,
        }
        self.items.append(item)
        self.next_id += 1
        return SimpleNamespace(**item)

    def recall(self, **kwargs):
        scope = kwargs.get("scope")
        project_id = kwargs.get("project_id")
        agent_id = kwargs.get("agent_id")
        user_id = kwargs.get("user_id")
        query = kwargs.get("query", "").lower()

        out = []
        for item in self.items:
            if item["user_id"] != user_id:
                continue
            if query and query not in item["text"].lower():
                continue
            if scope is not None:
                if scope.value == "project" and not (
                    item["scope"] == "project" and item["project_id"] == project_id
                ):
                    continue
                if scope.value == "agent" and not (
                    item["scope"] == "agent"
                    and item["project_id"] == project_id
                    and item["agent_id"] == agent_id
                ):
                    continue
                if scope.value == "user_preference" and item["scope"] != "user_preference":
                    continue
            out.append(SimpleNamespace(memory=SimpleNamespace(**item), score=1.0))
        return out

    def list(self, **kwargs):
        return []

    def update(self, **kwargs):
        return None

    def delete(self, **kwargs):
        return False

    def stats(self, **kwargs):
        return {"total": len(self.items)}


def main() -> int:
    service = MemoryService(InMemoryStore())

    service.remember_project_memory(text="Alpha architecture", user_id="u1", project_id="alpha")
    service.remember_project_memory(text="Beta architecture", user_id="u1", project_id="beta")
    service.remember_agent_memory(
        text="Agent A detail",
        user_id="u1",
        project_id="alpha",
        agent_id="agent-a",
    )
    service.remember_user_preference(text="Use snake_case in tool payloads", user_id="u1")

    alpha = service.recall_project_context(query="architecture", user_id="u1", project_id="alpha")
    beta = service.recall_project_context(query="architecture", user_id="u1", project_id="beta")
    agent = service.recall_agent_context(
        query="detail", user_id="u1", project_id="alpha", agent_id="agent-a"
    )
    prefs = service.recall_user_preferences(query="snake_case", user_id="u1")

    print(f"project alpha recall: {alpha['count']}")
    print(f"project beta recall: {beta['count']}")
    print(f"agent recall: {agent['count']}")
    print(f"preference recall: {prefs['count']}")

    assert alpha["count"] == 1
    assert beta["count"] == 1
    assert agent["count"] == 1
    assert prefs["count"] == 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
