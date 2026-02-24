from types import SimpleNamespace

from hippocampai_mcp import server


class _GraphExtrasService:
    def get_memory_clusters(self, *, user_id):
        _ = user_id
        return {"count": 1, "clusters": [{"cluster_id": "c1", "memory_ids": ["m1", "m2"]}]}

    def get_knowledge_subgraph(self, *, center_id, radius=2, include_types=None):
        _ = center_id
        _ = radius
        _ = include_types
        return {
            "center_id": "m1",
            "subgraph": {"nodes": [{"id": "m1"}], "edges": [{"source": "m1", "target": "m2"}]},
        }

    def extract_relationships(self, *, text, entities=None):
        _ = text
        _ = entities
        return {
            "count": 1,
            "relationships": [
                {"source": "alice", "target": "qdrant", "relation_type": "uses", "confidence": 0.9}
            ],
        }


class _ExplodingGraphExtrasService(_GraphExtrasService):
    def get_memory_clusters(self, **kwargs):
        _ = kwargs
        raise RuntimeError("clusters boom")

    def get_knowledge_subgraph(self, **kwargs):
        _ = kwargs
        raise RuntimeError("subgraph boom")

    def extract_relationships(self, **kwargs):
        _ = kwargs
        raise RuntimeError("extract boom")


class _NotSupportedGraphExtrasService:
    def get_memory_clusters(self, **kwargs):
        _ = kwargs
        raise NotImplementedError("no clusters")

    def get_knowledge_subgraph(self, **kwargs):
        _ = kwargs
        raise NotImplementedError("no subgraph")

    def extract_relationships(self, **kwargs):
        _ = kwargs
        raise NotImplementedError("no extraction")


def test_graph_extras_success_shapes(monkeypatch) -> None:
    monkeypatch.setattr(server, "memory_service", _GraphExtrasService())

    clusters = server.get_memory_clusters(user_id="u1")
    subgraph = server.get_knowledge_subgraph(center_id="m1", radius=2)
    extracted = server.extract_relationships(text="Alice uses Qdrant")

    assert clusters["count"] == 1
    assert clusters["clusters"][0]["cluster_id"] == "c1"
    assert subgraph["center_id"] == "m1"
    assert subgraph["subgraph"]["nodes"][0]["id"] == "m1"
    assert extracted["count"] == 1
    assert extracted["relationships"][0]["relation_type"] == "uses"


def test_graph_extras_not_supported_without_backend_methods(monkeypatch) -> None:
    monkeypatch.setattr(server, "memory_service", _NotSupportedGraphExtrasService())

    clusters = server.get_memory_clusters(user_id="u1")
    subgraph = server.get_knowledge_subgraph(center_id="m1", radius=2)
    extracted = server.extract_relationships(text="Alice uses Qdrant")

    assert clusters["code"] == "not_supported"
    assert subgraph["code"] == "not_supported"
    assert extracted["code"] == "not_supported"
    assert "correlation_id" in clusters
    assert "correlation_id" in subgraph
    assert "correlation_id" in extracted


def test_graph_extras_map_runtime_errors_to_stable_codes(monkeypatch) -> None:
    monkeypatch.setattr(server, "memory_service", _ExplodingGraphExtrasService())

    clusters = server.get_memory_clusters(user_id="u1")
    subgraph = server.get_knowledge_subgraph(center_id="m1", radius=2)
    extracted = server.extract_relationships(text="Alice uses Qdrant")

    assert clusters["code"] == "get_memory_clusters_failed"
    assert subgraph["code"] == "get_knowledge_subgraph_failed"
    assert extracted["code"] == "extract_relationships_failed"
    assert "message" in clusters and "details" in clusters
    assert "message" in subgraph and "details" in subgraph
    assert "message" in extracted and "details" in extracted
