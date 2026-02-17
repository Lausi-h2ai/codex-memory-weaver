"""Typing helpers to avoid circular imports in domain validation."""

from __future__ import annotations

from typing import Protocol


class MemoryScopeLike(Protocol):
    def __str__(self) -> str: ...
