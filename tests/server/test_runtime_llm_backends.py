import json
from types import SimpleNamespace

from hippocampai_mcp import server


def _reset_runtime(monkeypatch) -> None:
    monkeypatch.setattr(server, "memory_client", None)
    monkeypatch.setattr(server, "memory_store", None)
    monkeypatch.setattr(server, "memory_service", None)


def _stub_runtime_construction(monkeypatch, captured_kwargs: dict) -> None:
    def _memory_client_factory(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(config=SimpleNamespace(graph_persistence_path=None))

    monkeypatch.setattr(server, "MemoryClient", _memory_client_factory)
    monkeypatch.setattr(server, "HippocampAIAdapter", lambda c: SimpleNamespace(client=c))
    monkeypatch.setattr(server, "MemoryService", lambda s: SimpleNamespace(store=s))


def test_initialize_runtime_clients_defaults_to_ollama_when_llm_env_is_absent(monkeypatch) -> None:
    _reset_runtime(monkeypatch)
    captured = {}
    _stub_runtime_construction(monkeypatch, captured)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    server._initialize_runtime_clients()

    assert captured["llm_provider"] == "ollama"
    assert captured["llm_model"] == "qwen2.5:7b-instruct"
    assert server.os.getenv("LLM_PROVIDER") == "ollama"
    assert server.os.getenv("LLM_BASE_URL") == "http://localhost:11434"
    assert server.os.getenv("LLM_MODEL") == "qwen2.5:7b-instruct"


def test_initialize_runtime_clients_uses_generic_llm_env_values(monkeypatch) -> None:
    _reset_runtime(monkeypatch)
    captured = {}
    _stub_runtime_construction(monkeypatch, captured)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

    server._initialize_runtime_clients()

    assert captured["llm_provider"] == "openai"
    assert captured["llm_model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert server.os.getenv("LLM_BASE_URL") == "http://localhost:8000/v1"
    assert server.os.getenv("LLM_MODEL") == "Qwen/Qwen2.5-7B-Instruct"


def test_initialize_runtime_clients_backfills_llm_env_from_legacy_ollama(monkeypatch) -> None:
    _reset_runtime(monkeypatch)
    captured = {}
    _stub_runtime_construction(monkeypatch, captured)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://legacy-ollama:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "legacy-model")

    server._initialize_runtime_clients()

    assert captured["llm_provider"] == "ollama"
    assert captured["llm_model"] == "legacy-model"
    assert server.os.getenv("LLM_BASE_URL") == "http://legacy-ollama:11434"
    assert server.os.getenv("LLM_MODEL") == "legacy-model"


def test_require_memory_service_returns_structured_error_for_unsupported_provider(monkeypatch) -> None:
    _reset_runtime(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "vllm")

    payload = server._require_memory_service(correlation_id="cid-1")

    assert payload["code"] == "memory_service_uninitialized"
    assert payload["correlation_id"] == "cid-1"
    assert "unsupported llm provider" in payload["details"]["error"].lower()


def test_health_and_config_report_generic_llm_runtime(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    monkeypatch.setattr(
        server,
        "_check_tcp_dependency",
        lambda _url, _port: {"status": "ok", "host": "localhost", "port": 8000},
    )

    health = json.loads(server.health_check())
    config = server.get_config()

    assert "llm" in health["dependencies"]
    assert "ollama" not in health["dependencies"]
    assert "LLM Provider: openai" in config
    assert "LLM Base URL: http://localhost:8000/v1" in config
    assert "LLM Model: Qwen/Qwen2.5-7B-Instruct" in config
