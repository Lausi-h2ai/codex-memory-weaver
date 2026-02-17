"""Domain models for scoped memory operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .validation import validate_scope_fields


class MemoryScope(str, Enum):
    PROJECT = "project"
    AGENT = "agent"
    USER_PREFERENCE = "user_preference"
    SESSION = "session"


class MemoryType(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    GOAL = "goal"
    HABIT = "habit"
    EVENT = "event"
    CONTEXT = "context"


class Visibility(str, Enum):
    PRIVATE = "private"
    SHARED = "shared"
    PUBLIC = "public"


@dataclass(slots=True)
class WriteMemoryRequest:
    text: str
    user_id: str
    scope: MemoryScope
    memory_type: MemoryType = MemoryType.CONTEXT
    project_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    tags: list[str] = field(default_factory=list)
    importance: float | None = None
    visibility: Visibility = Visibility.PRIVATE
    run_id: str | None = None

    def __post_init__(self) -> None:
        validate_scope_fields(
            scope=self.scope,
            project_id=self.project_id,
            agent_id=self.agent_id,
            session_id=self.session_id,
        )


@dataclass(slots=True)
class ReadMemoryRequest:
    user_id: str
    scope: MemoryScope | None = None
    query: str | None = None
    project_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    limit: int = 10
    min_importance: float | None = None
    include_cross_scope: bool = False
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.scope is None:
            return
        validate_scope_fields(
            scope=self.scope,
            project_id=self.project_id,
            agent_id=self.agent_id,
            session_id=self.session_id,
        )
