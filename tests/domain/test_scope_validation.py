import pytest

from hippocampai_mcp.domain.models import MemoryScope, ReadMemoryRequest, Visibility, WriteMemoryRequest
from hippocampai_mcp.domain.validation import ScopeValidationError


def test_project_scope_requires_project_id() -> None:
    with pytest.raises(ScopeValidationError, match="project_id"):
        WriteMemoryRequest(text="a", user_id="u1", scope=MemoryScope.PROJECT)


def test_project_scope_accepts_optional_agent_id() -> None:
    request = WriteMemoryRequest(
        text="a",
        user_id="u1",
        scope=MemoryScope.PROJECT,
        project_id="proj-1",
        agent_id="agent-1",
    )
    assert request.project_id == "proj-1"
    assert request.agent_id == "agent-1"


def test_agent_scope_requires_project_id_and_agent_id() -> None:
    with pytest.raises(ScopeValidationError, match="project_id"):
        WriteMemoryRequest(text="a", user_id="u1", scope=MemoryScope.AGENT, agent_id="a1")

    with pytest.raises(ScopeValidationError, match="agent_id"):
        WriteMemoryRequest(text="a", user_id="u1", scope=MemoryScope.AGENT, project_id="p1")


def test_user_preference_scope_rejects_project_and_agent() -> None:
    with pytest.raises(ScopeValidationError, match="project_id"):
        WriteMemoryRequest(
            text="a",
            user_id="u1",
            scope=MemoryScope.USER_PREFERENCE,
            project_id="p1",
        )

    with pytest.raises(ScopeValidationError, match="agent_id"):
        WriteMemoryRequest(
            text="a",
            user_id="u1",
            scope=MemoryScope.USER_PREFERENCE,
            agent_id="a1",
        )


def test_session_scope_requires_session_id() -> None:
    with pytest.raises(ScopeValidationError, match="session_id"):
        WriteMemoryRequest(text="a", user_id="u1", scope=MemoryScope.SESSION)


def test_domain_model_includes_visibility_and_run_id() -> None:
    request = WriteMemoryRequest(
        text="a",
        user_id="u1",
        scope=MemoryScope.AGENT,
        project_id="p1",
        agent_id="a1",
        visibility=Visibility.SHARED,
        run_id="run-123",
    )
    assert request.visibility == Visibility.SHARED
    assert request.run_id == "run-123"


def test_read_request_scope_validation_applies() -> None:
    with pytest.raises(ScopeValidationError):
        ReadMemoryRequest(user_id="u1", scope=MemoryScope.PROJECT)

    read_request = ReadMemoryRequest(
        user_id="u1",
        scope=MemoryScope.USER_PREFERENCE,
    )
    assert read_request.scope is MemoryScope.USER_PREFERENCE
