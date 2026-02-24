# Graph Extras Tools

MVP tools:

- `get_memory_clusters(user_id)`
- `get_knowledge_subgraph(center_id, radius=2, include_types=None)`
- `extract_relationships(text)`

These are exposed through `server -> MemoryService -> HippocampAIAdapter`.

Example success:

```json
{
  "count": 1,
  "clusters": [{ "cluster_id": "c1", "memory_ids": ["m1", "m2"] }]
}
```

Subgraph success:

```json
{
  "center_id": "m1",
  "subgraph": {
    "nodes": [{ "id": "m1" }],
    "edges": [{ "source": "m1", "target": "m2" }]
  }
}
```

Failure shape:

```json
{
  "code": "not_supported",
  "message": "...",
  "details": {},
  "correlation_id": "..."
}
```

If backend methods are unavailable, adapter raises `NotImplementedError` and server returns `not_supported`.
