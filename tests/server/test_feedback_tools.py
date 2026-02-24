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


class _ExplodingFeedbackService:
    def submit_memory_feedback(self, **kwargs):
        _ = kwargs
        raise RuntimeError("submit boom")

    def get_memory_feedback(self, **kwargs):
        _ = kwargs
        raise RuntimeError("fetch boom")

    def get_feedback_stats(self, **kwargs):
        _ = kwargs
        raise RuntimeError("stats boom")


class _NotSupportedFeedbackService:
    def submit_memory_feedback(self, **kwargs):
        _ = kwargs
        raise NotImplementedError("no submit")

    def get_memory_feedback(self, **kwargs):
        _ = kwargs
        raise NotImplementedError("no get")

    def get_feedback_stats(self, **kwargs):
        _ = kwargs
        raise NotImplementedError("no stats")


class _AttributeErrorFeedbackService:
    def submit_memory_feedback(self, **kwargs):
        _ = kwargs
        raise AttributeError("submit missing field")

    def get_memory_feedback(self, **kwargs):
        _ = kwargs
        raise AttributeError("fetch missing field")

    def get_feedback_stats(self, **kwargs):
        _ = kwargs
        raise AttributeError("stats missing field")


def test_submit_memory_feedback_returns_not_supported_without_backend_method(monkeypatch) -> None:
    monkeypatch.setattr(server, "memory_service", _NotSupportedFeedbackService())

    payload = server.submit_memory_feedback(
        memory_id="m1",
        user_id="u1",
        feedback_type="relevant",
    )

    assert payload["code"] == "not_supported"
    assert "correlation_id" in payload


def test_feedback_tools_return_backend_values_when_supported(monkeypatch) -> None:
    monkeypatch.setattr(server, "memory_service", _FeedbackClient())

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


def test_submit_memory_feedback_rejects_invalid_feedback_type(monkeypatch) -> None:
    monkeypatch.setattr(server, "memory_service", _FeedbackClient())

    payload = server.submit_memory_feedback(
        memory_id="m1",
        user_id="u1",
        feedback_type="thumbs_up",
    )

    assert payload["code"] == "validation_error"


def test_submit_memory_feedback_normalizes_feedback_type(monkeypatch) -> None:
    monkeypatch.setattr(server, "memory_service", _FeedbackClient())

    payload = server.submit_memory_feedback(
        memory_id="m1",
        user_id="u1",
        feedback_type="  ReLeVaNt  ",
    )

    assert payload["feedback_type"] == "relevant"


def test_get_feedback_tools_return_not_supported_without_backend_methods(monkeypatch) -> None:
    monkeypatch.setattr(server, "memory_service", _NotSupportedFeedbackService())

    score = server.get_memory_feedback(memory_id="m1")
    stats = server.get_feedback_stats(user_id="u1")

    assert score["code"] == "not_supported"
    assert stats["code"] == "not_supported"


def test_feedback_tools_map_runtime_errors_to_stable_codes(monkeypatch) -> None:
    monkeypatch.setattr(server, "memory_service", _ExplodingFeedbackService())

    submitted = server.submit_memory_feedback(
        memory_id="m1",
        user_id="u1",
        feedback_type="relevant",
    )
    score = server.get_memory_feedback(memory_id="m1")
    stats = server.get_feedback_stats(user_id="u1")

    assert submitted["code"] == "feedback_submit_failed"
    assert score["code"] == "feedback_fetch_failed"
    assert stats["code"] == "feedback_stats_failed"


def test_feedback_tools_map_attribute_errors_to_failure_codes(monkeypatch) -> None:
    monkeypatch.setattr(server, "memory_service", _AttributeErrorFeedbackService())

    submitted = server.submit_memory_feedback(
        memory_id="m1",
        user_id="u1",
        feedback_type="relevant",
    )
    score = server.get_memory_feedback(memory_id="m1")
    stats = server.get_feedback_stats(user_id="u1")

    assert submitted["code"] == "feedback_submit_failed"
    assert score["code"] == "feedback_fetch_failed"
    assert stats["code"] == "feedback_stats_failed"
