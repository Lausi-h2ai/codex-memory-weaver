from hippocampai_mcp import server


class _FeedbackClient:
    def submit_memory_feedback(self, *, memory_id, user_id, feedback_type, query=None):
        return {
            "memory_id": memory_id,
            "feedback_type": feedback_type,
            "score": 0.75,
            "query": query,
            "user_id": user_id,
        }

    def get_memory_feedback(self, *, memory_id):
        return {"memory_id": memory_id, "score": 0.75, "event_count": 2}

    def get_feedback_stats(self, *, user_id):
        return {"user_id": user_id, "stats": {"relevant": 4, "not_relevant": 1}}


def test_submit_memory_feedback_returns_not_supported_without_backend_method(monkeypatch) -> None:
    monkeypatch.setattr(server, "memory_client", object())

    payload = server.submit_memory_feedback(
        memory_id="m1",
        user_id="u1",
        feedback_type="relevant",
    )

    assert payload["code"] == "not_supported"


def test_feedback_tools_return_backend_values_when_supported(monkeypatch) -> None:
    monkeypatch.setattr(server, "memory_client", _FeedbackClient())

    submitted = server.submit_memory_feedback(
        memory_id="m1",
        user_id="u1",
        feedback_type="relevant",
        query="auth preferences",
    )
    assert submitted["memory_id"] == "m1"
    assert submitted["feedback_type"] == "relevant"
    assert submitted["score"] == 0.75

    score = server.get_memory_feedback(memory_id="m1")
    assert score["memory_id"] == "m1"
    assert score["event_count"] == 2

    stats = server.get_feedback_stats(user_id="u1")
    assert stats["user_id"] == "u1"
    assert stats["stats"]["relevant"] == 4
