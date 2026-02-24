import json

from hippocampai_mcp import server


class _FakeHTTPResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self.status = status
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        _ = exc_type
        _ = exc
        _ = tb
        return False


def test_procedural_tools_return_not_supported_without_base_url(monkeypatch) -> None:
    monkeypatch.delenv("HIPPOCAMPAI_API_BASE_URL", raising=False)

    listed = server.list_procedural_rules(user_id="u1")
    extracted = server.extract_procedural_rules(user_id="u1", interactions=[{"role": "user", "content": "x"}])
    injected = server.inject_procedural_rules(user_id="u1", prompt="Hello")
    updated = server.update_procedural_rule_feedback(rule_id="r1", effectiveness=0.9, user_id="u1")
    consolidated = server.consolidate_procedural_rules(user_id="u1")

    assert listed["code"] == "not_supported"
    assert extracted["code"] == "not_supported"
    assert injected["code"] == "not_supported"
    assert updated["code"] == "not_supported"
    assert consolidated["code"] == "not_supported"
    assert "correlation_id" in listed
    assert "correlation_id" in extracted
    assert "correlation_id" in injected
    assert "correlation_id" in updated
    assert "correlation_id" in consolidated


def test_procedural_tools_use_http_bridge_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("HIPPOCAMPAI_API_BASE_URL", "http://localhost:8000")

    def _fake_urlopen(req, timeout=0):
        _ = timeout
        url = req.full_url
        method = req.get_method()
        if url.endswith("/v1/procedural/extract") and method == "POST":
            return _FakeHTTPResponse({"rules": [{"id": "rule-1"}]})
        if url.endswith("/v1/procedural/inject") and method == "POST":
            return _FakeHTTPResponse({"rules_injected": 1, "enhanced_prompt": "..."})
        if "/v1/procedural/rules/" in url and method == "PUT":
            return _FakeHTTPResponse({"rule_id": "rule-1", "effectiveness": 0.9})
        if url.startswith("http://localhost:8000/v1/procedural/rules?") and method == "GET":
            return _FakeHTTPResponse({"rules": [{"id": "rule-1"}]})
        if url.startswith("http://localhost:8000/v1/procedural/consolidate?") and method == "POST":
            return _FakeHTTPResponse({"merged_rules": 2})
        return _FakeHTTPResponse({"ok": True})

    monkeypatch.setattr(server.urllib_request, "urlopen", _fake_urlopen)

    listed = server.list_procedural_rules(user_id="u1")
    extracted = server.extract_procedural_rules(user_id="u1", interactions=[{"role": "user", "content": "x"}])
    injected = server.inject_procedural_rules(user_id="u1", prompt="Hello")
    updated = server.update_procedural_rule_feedback(rule_id="rule-1", effectiveness=0.9, user_id="u1")
    consolidated = server.consolidate_procedural_rules(user_id="u1")

    assert listed["rules"][0]["id"] == "rule-1"
    assert extracted["rules"][0]["id"] == "rule-1"
    assert injected["rules_injected"] == 1
    assert updated["rule_id"] == "rule-1"
    assert consolidated["merged_rules"] == 2


def test_procedural_tools_map_http_failures_to_stable_codes(monkeypatch) -> None:
    monkeypatch.setenv("HIPPOCAMPAI_API_BASE_URL", "http://localhost:8000")

    def _boom(*args, **kwargs):
        _ = args
        _ = kwargs
        raise RuntimeError("http boom")

    monkeypatch.setattr(server.urllib_request, "urlopen", _boom)

    listed = server.list_procedural_rules(user_id="u1")
    extracted = server.extract_procedural_rules(user_id="u1", interactions=[{"role": "user", "content": "x"}])
    injected = server.inject_procedural_rules(user_id="u1", prompt="Hello")
    updated = server.update_procedural_rule_feedback(rule_id="r1", effectiveness=0.9, user_id="u1")
    consolidated = server.consolidate_procedural_rules(user_id="u1")

    assert listed["code"] == "list_procedural_rules_failed"
    assert extracted["code"] == "extract_procedural_rules_failed"
    assert injected["code"] == "inject_procedural_rules_failed"
    assert updated["code"] == "update_procedural_rule_feedback_failed"
    assert consolidated["code"] == "consolidate_procedural_rules_failed"
    assert "message" in listed and "details" in listed
    assert "message" in extracted and "details" in extracted
    assert "message" in injected and "details" in injected
    assert "message" in updated and "details" in updated
    assert "message" in consolidated and "details" in consolidated
