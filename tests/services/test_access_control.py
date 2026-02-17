import pytest

from hippocampai_mcp.services.access_control import AccessControlError, AccessController


def test_update_and_delete_require_user_id() -> None:
    access = AccessController()

    with pytest.raises(AccessControlError):
        access.enforce_update_delete(user_id=None, action="update")

    with pytest.raises(AccessControlError):
        access.enforce_update_delete(user_id="", action="delete")


def test_cross_scope_recall_blocked_by_default() -> None:
    access = AccessController()

    with pytest.raises(AccessControlError):
        access.enforce_recall_scope(
            scope="project",
            project_id=None,
            agent_id=None,
            session_id=None,
            include_cross_scope=False,
        )


def test_cross_scope_recall_allowed_when_explicit() -> None:
    access = AccessController()

    # no exception when include_cross_scope=True
    access.enforce_recall_scope(
        scope=None,
        project_id=None,
        agent_id=None,
        session_id=None,
        include_cross_scope=True,
    )


def test_agent_visibility_validation() -> None:
    access = AccessController()

    assert access.normalize_agent_visibility("private") == "private"
    assert access.normalize_agent_visibility("shared") == "shared"
    assert access.normalize_agent_visibility("public") == "public"

    with pytest.raises(AccessControlError):
        access.normalize_agent_visibility("team")
