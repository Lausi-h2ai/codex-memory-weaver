from types import SimpleNamespace

from hippocampai_mcp import server


class _DummyClient:
    def __init__(self, graph_path: str) -> None:
        self.config = SimpleNamespace(graph_persistence_path=graph_path)
        self.graph = "initial-graph"
        self.graph_retriever = SimpleNamespace(graph="initial-graph")


def test_initialize_runtime_clients_autoloads_persisted_graph(monkeypatch) -> None:
    client = _DummyClient(graph_path="data/knowledge_graph.json")
    loaded_graph = SimpleNamespace(graph=SimpleNamespace(number_of_nodes=lambda: 3, number_of_edges=lambda: 2))

    monkeypatch.setattr(server, "memory_client", None)
    monkeypatch.setattr(server, "memory_store", None)
    monkeypatch.setattr(server, "memory_service", None)
    monkeypatch.setattr(server, "MemoryClient", lambda **kwargs: client)
    monkeypatch.setattr(server, "HippocampAIAdapter", lambda c: SimpleNamespace(client=c))
    monkeypatch.setattr(server, "MemoryService", lambda s: SimpleNamespace(store=s))
    monkeypatch.setattr(server.os.path, "exists", lambda p: True)
    monkeypatch.setattr(server, "_load_persisted_graph", lambda p: loaded_graph)

    server._initialize_runtime_clients()

    assert server.memory_client is client
    assert client.graph is loaded_graph
    assert client.graph_retriever.graph is loaded_graph


def test_initialize_runtime_clients_skips_graph_load_when_file_missing(monkeypatch) -> None:
    client = _DummyClient(graph_path="data/knowledge_graph.json")

    monkeypatch.setattr(server, "memory_client", None)
    monkeypatch.setattr(server, "memory_store", None)
    monkeypatch.setattr(server, "memory_service", None)
    monkeypatch.setattr(server, "MemoryClient", lambda **kwargs: client)
    monkeypatch.setattr(server, "HippocampAIAdapter", lambda c: SimpleNamespace(client=c))
    monkeypatch.setattr(server, "MemoryService", lambda s: SimpleNamespace(store=s))
    monkeypatch.setattr(server.os.path, "exists", lambda p: False)

    server._initialize_runtime_clients()

    assert client.graph == "initial-graph"
    assert client.graph_retriever.graph == "initial-graph"


def test_initialize_runtime_clients_survives_graph_load_error(monkeypatch) -> None:
    client = _DummyClient(graph_path="data/knowledge_graph.json")

    monkeypatch.setattr(server, "memory_client", None)
    monkeypatch.setattr(server, "memory_store", None)
    monkeypatch.setattr(server, "memory_service", None)
    monkeypatch.setattr(server, "MemoryClient", lambda **kwargs: client)
    monkeypatch.setattr(server, "HippocampAIAdapter", lambda c: SimpleNamespace(client=c))
    monkeypatch.setattr(server, "MemoryService", lambda s: SimpleNamespace(store=s))
    monkeypatch.setattr(server.os.path, "exists", lambda p: True)

    def _raise(_: str):
        raise RuntimeError("bad graph file")

    monkeypatch.setattr(server, "_load_persisted_graph", _raise)

    server._initialize_runtime_clients()

    assert server.memory_client is client
    assert client.graph == "initial-graph"
