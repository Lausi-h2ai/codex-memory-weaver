from types import SimpleNamespace

from hippocampai_mcp import server


class _InsightsClient:
    def detect_patterns(self, *, user_id, session_ids=None):
        _ = user_id
        _ = session_ids
        return [
            SimpleNamespace(
                id="p1",
                pattern_type="recurring",
                description="Daily standup",
                confidence=0.9,
                occurrences=5,
            )
        ]

    def track_behavior_changes(self, *, user_id, comparison_days=30):
        _ = user_id
        _ = comparison_days
        return [SimpleNamespace(id="c1", change_type="PREFERENCE_SHIFT", confidence=0.8)]

    def analyze_preference_drift(self, *, user_id, category=None):
        _ = user_id
        _ = category
        return [SimpleNamespace(id="d1", category="editor", drift_score=0.5)]

    def detect_habits(self, *, user_id, min_occurrences=5):
        _ = user_id
        _ = min_occurrences
        return [SimpleNamespace(id="h1", behavior="daily review", habit_score=0.7)]

    def analyze_trends(self, *, user_id, window_days=30):
        _ = user_id
        _ = window_days
        return [SimpleNamespace(id="t1", category="productivity", trend_type="increasing")]


class _ExplodingInsightsClient(_InsightsClient):
    def detect_patterns(self, **kwargs):
        _ = kwargs
        raise RuntimeError("pattern boom")

    def track_behavior_changes(self, **kwargs):
        _ = kwargs
        raise RuntimeError("changes boom")

    def analyze_preference_drift(self, **kwargs):
        _ = kwargs
        raise RuntimeError("drift boom")

    def detect_habits(self, **kwargs):
        _ = kwargs
        raise RuntimeError("habit boom")

    def analyze_trends(self, **kwargs):
        _ = kwargs
        raise RuntimeError("trend boom")


def test_cross_session_insights_success_shapes(monkeypatch) -> None:
    monkeypatch.setattr(server, "memory_client", _InsightsClient())

    patterns = server.detect_patterns(user_id="u1")
    changes = server.track_behavior_changes(user_id="u1")
    drifts = server.analyze_preference_drift(user_id="u1")
    habits = server.detect_habits(user_id="u1")
    trends = server.analyze_trends(user_id="u1")

    assert patterns["count"] == 1
    assert patterns["patterns"][0]["id"] == "p1"
    assert changes["count"] == 1
    assert changes["changes"][0]["id"] == "c1"
    assert drifts["count"] == 1
    assert drifts["drifts"][0]["id"] == "d1"
    assert habits["count"] == 1
    assert habits["habits"][0]["id"] == "h1"
    assert trends["count"] == 1
    assert trends["trends"][0]["id"] == "t1"


def test_cross_session_insights_not_supported_without_backend_methods(monkeypatch) -> None:
    monkeypatch.setattr(server, "memory_client", object())

    patterns = server.detect_patterns(user_id="u1")
    changes = server.track_behavior_changes(user_id="u1")
    drifts = server.analyze_preference_drift(user_id="u1")
    habits = server.detect_habits(user_id="u1")
    trends = server.analyze_trends(user_id="u1")

    assert patterns["code"] == "not_supported"
    assert changes["code"] == "not_supported"
    assert drifts["code"] == "not_supported"
    assert habits["code"] == "not_supported"
    assert trends["code"] == "not_supported"
    assert "correlation_id" in patterns
    assert "correlation_id" in changes
    assert "correlation_id" in drifts
    assert "correlation_id" in habits
    assert "correlation_id" in trends


def test_cross_session_insights_map_runtime_errors_to_stable_codes(monkeypatch) -> None:
    monkeypatch.setattr(server, "memory_client", _ExplodingInsightsClient())

    patterns = server.detect_patterns(user_id="u1")
    changes = server.track_behavior_changes(user_id="u1")
    drifts = server.analyze_preference_drift(user_id="u1")
    habits = server.detect_habits(user_id="u1")
    trends = server.analyze_trends(user_id="u1")

    assert patterns["code"] == "detect_patterns_failed"
    assert changes["code"] == "track_behavior_changes_failed"
    assert drifts["code"] == "analyze_preference_drift_failed"
    assert habits["code"] == "detect_habits_failed"
    assert trends["code"] == "analyze_trends_failed"
    assert "message" in patterns and "details" in patterns
    assert "message" in changes and "details" in changes
    assert "message" in drifts and "details" in drifts
    assert "message" in habits and "details" in habits
    assert "message" in trends and "details" in trends
