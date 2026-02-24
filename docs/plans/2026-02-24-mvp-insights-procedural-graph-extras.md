# MVP Cross-Session Insights + Procedural + Graph Extras Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship MVP tool surfaces for cross-session insights, procedural-memory rule operations, and graph extras beyond relation traversal.

**Architecture:** Extend the existing layered shape for client-backed features (`server -> MemoryService -> HippocampAIAdapter`) and add a minimal HTTP bridge helper in `server.py` only for procedural endpoints that are REST-only in HippocampAI docs. Keep strict backward compatibility: if a capability is unavailable, return deterministic `not_supported` payloads rather than crashing. Normalize responses into stable snake_case dicts so MCP clients get predictable schemas.

**Tech Stack:** Python 3.10+, FastMCP (`mcp`), HippocampAI v0.5.x `MemoryClient`, stdlib `urllib`/`json` for procedural HTTP bridge, pytest.

---

## Scope (MVP Only)

Implement these new MCP tools:

1. Cross-session insights (client-backed):
- `detect_patterns(user_id, session_ids=None)`
- `track_behavior_changes(user_id, comparison_days=30)`
- `analyze_preference_drift(user_id, category=None)`
- `detect_habits(user_id, min_occurrences=5)`
- `analyze_trends(user_id, window_days=30)`

2. Graph extras (client-backed):
- `get_memory_clusters(user_id)`
- `get_knowledge_subgraph(center_id, radius=2, include_types=None)`
- `extract_relationships(text)`

3. Procedural memory/rule self-optimization (HTTP bridge):
- `list_procedural_rules(user_id)`
- `extract_procedural_rules(user_id, interactions)`
- `inject_procedural_rules(user_id, prompt, max_rules=3)`
- `update_procedural_rule_feedback(rule_id, effectiveness, user_id=None)`
- `consolidate_procedural_rules(user_id)`

Non-goals for this pass:
- Full typed domain models for new payloads
- Advanced orchestration/workflow composition across tools
- Rule conflict resolution in MCP layer (delegate to backend)

## Task 1: Cross-Session Insights Tool Contracts (TDD)

**Files:**
- Modify: `tests/server/test_cross_session_insights_tools.py` (create if missing)
- Modify: `src/hippocampai_mcp/server.py`

**Step 1: Write failing tests**
- Add tool-level tests that patch `server.memory_client` with stubs implementing:
  - `detect_patterns`
  - `track_behavior_changes`
  - `analyze_preference_drift`
  - `detect_habits`
  - `analyze_trends`
- Add tests for:
  - success shape (`count` + list payload)
  - backend missing method -> `not_supported`
  - runtime exception -> stable `*_failed` code

**Step 2: Run test to verify RED**
- Run: `python -m pytest tests/server/test_cross_session_insights_tools.py -v`
- Expected: FAIL for missing tool methods.

**Step 3: Write minimal implementation**
- Add 5 MCP tools in `src/hippocampai_mcp/server.py`.
- Follow existing patterns:
  - `_new_correlation_id()`
  - `_require_memory_client()`
  - `emit_tool_log(...)`
  - `_error_payload(...)`

**Step 4: Run test to verify GREEN**
- Run: `python -m pytest tests/server/test_cross_session_insights_tools.py -v`
- Expected: PASS.

**Step 5: Commit**
```bash
git add tests/server/test_cross_session_insights_tools.py src/hippocampai_mcp/server.py
git commit -m "feat: add MVP cross-session insights MCP tools"
```

## Task 2: Graph Extras Beyond Relations (TDD)

**Files:**
- Modify: `tests/storage/test_hippocampai_adapter_mapping.py`
- Modify: `tests/services/test_memory_service_scopes.py`
- Modify: `tests/server/test_graph_extras_tools.py` (create if missing)
- Modify: `src/hippocampai_mcp/storage/hippocampai_adapter.py`
- Modify: `src/hippocampai_mcp/services/memory_service.py`
- Modify: `src/hippocampai_mcp/server.py`

**Step 1: Write failing tests**
- Adapter tests:
  - `get_memory_clusters(user_id)`
  - `get_knowledge_subgraph(center_id, radius, include_types)`
  - `extract_relationships(text)`
- Service tests:
  - pass-through behavior and stable dict/list normalization
- Server tests:
  - successful payload shapes
  - missing backend methods -> `not_supported`

**Step 2: Run tests to verify RED**
- Run:
  - `python -m pytest tests/storage/test_hippocampai_adapter_mapping.py -k "clusters or subgraph or extract_relationships" -v`
  - `python -m pytest tests/services/test_memory_service_scopes.py -k "clusters or subgraph or extract_relationships" -v`
  - `python -m pytest tests/server/test_graph_extras_tools.py -v`
- Expected: FAIL for missing methods/tools.

**Step 3: Write minimal implementation**
- Adapter:
  - add `get_memory_clusters`, `get_knowledge_subgraph`, `extract_relationships`
  - raise `NotImplementedError` when absent
- Service:
  - add pass-through wrappers with small normalization helpers
- Server:
  - add MCP tools with logging/error handling

**Step 4: Run tests to verify GREEN**
- Re-run same commands; expect PASS.

**Step 5: Commit**
```bash
git add tests/storage/test_hippocampai_adapter_mapping.py tests/services/test_memory_service_scopes.py tests/server/test_graph_extras_tools.py src/hippocampai_mcp/storage/hippocampai_adapter.py src/hippocampai_mcp/services/memory_service.py src/hippocampai_mcp/server.py
git commit -m "feat: add graph extras MCP surfaces (clusters/subgraph/relationship extraction)"
```

## Task 3: Procedural Memory HTTP Bridge Skeleton (TDD)

**Files:**
- Modify: `tests/server/test_procedural_memory_tools.py` (create)
- Modify: `src/hippocampai_mcp/server.py`
- Modify: `.env.example`

**Step 1: Write failing tests**
- Add tests for helper behavior:
  - missing `HIPPOCAMPAI_API_BASE_URL` env -> `not_supported`
  - HTTP 200 returns parsed JSON payload
  - HTTP non-2xx maps to deterministic error code per tool
- Add tool tests for:
  - `list_procedural_rules`
  - `extract_procedural_rules`
  - `inject_procedural_rules`
  - `update_procedural_rule_feedback`
  - `consolidate_procedural_rules`

**Step 2: Run tests to verify RED**
- Run: `python -m pytest tests/server/test_procedural_memory_tools.py -v`
- Expected: FAIL for missing bridge/tools.

**Step 3: Write minimal implementation**
- In `server.py` add private helper:
  - `_procedural_api_request(method, path, body=None, correlation_id=None) -> dict[str, Any] | _error_payload`
- Use stdlib:
  - `urllib.request.Request`
  - `json.dumps/loads`
- Env/config:
  - `HIPPOCAMPAI_API_BASE_URL` (example: `http://localhost:8000`)
- Map endpoints:
  - `GET /v1/procedural/rules?user_id=...`
  - `POST /v1/procedural/extract`
  - `POST /v1/procedural/inject`
  - `PUT /v1/procedural/rules/{rule_id}/feedback`
  - `POST /v1/procedural/consolidate?user_id=...`

**Step 4: Run tests to verify GREEN**
- Run: `python -m pytest tests/server/test_procedural_memory_tools.py -v`
- Expected: PASS.

**Step 5: Commit**
```bash
git add tests/server/test_procedural_memory_tools.py src/hippocampai_mcp/server.py .env.example
git commit -m "feat: add procedural-memory HTTP bridge MCP tools"
```

## Task 4: Response Schema Normalization + Error Code Stability

**Files:**
- Modify: `tests/server/test_cross_session_insights_tools.py`
- Modify: `tests/server/test_graph_extras_tools.py`
- Modify: `tests/server/test_procedural_memory_tools.py`
- Modify: `src/hippocampai_mcp/server.py`

**Step 1: Write failing tests**
- Assert each new tool returns stable top-level schema:
  - either `{count, results|items|rules}` on success
  - or `{code, message, correlation_id, details?}` on failure
- Assert all `not_supported` paths include `correlation_id`.

**Step 2: Run tests to verify RED**
- Run: `python -m pytest tests/server/test_cross_session_insights_tools.py tests/server/test_graph_extras_tools.py tests/server/test_procedural_memory_tools.py -v`

**Step 3: Write minimal implementation**
- Add small normalization wrappers in server tools.
- Ensure no raw model objects leak (convert via `_attr` and primitive dicts where needed).

**Step 4: Run tests to verify GREEN**
- Re-run same command and verify PASS.

**Step 5: Commit**
```bash
git add tests/server/test_cross_session_insights_tools.py tests/server/test_graph_extras_tools.py tests/server/test_procedural_memory_tools.py src/hippocampai_mcp/server.py
git commit -m "feat: stabilize schemas and error codes for new MVP tools"
```

## Task 5: Documentation + Discovery Surface

**Files:**
- Modify: `README.md`
- Create: `docs/cross-session-insights.md`
- Create: `docs/procedural-memory.md`
- Create: `docs/graph-extras.md`

**Step 1: Write docs checks (manual checklist)**
- New tools listed in README feature list.
- Procedural env var documented (`HIPPOCAMPAI_API_BASE_URL`).
- Each doc page has one success example + one failure/not_supported note.

**Step 2: Implement docs**
- README: add tool descriptions and one-liners.
- Add docs pages with request/response examples.

**Step 3: Run verification commands**
- Run:
  - `python -m pytest tests/server/test_cross_session_insights_tools.py -v`
  - `python -m pytest tests/server/test_graph_extras_tools.py -v`
  - `python -m pytest tests/server/test_procedural_memory_tools.py -v`
  - `python -m pytest tests/storage/test_hippocampai_adapter_mapping.py -v`
  - `python -m pytest tests/services/test_memory_service_scopes.py -v`

**Step 4: Commit**
```bash
git add README.md docs/cross-session-insights.md docs/procedural-memory.md docs/graph-extras.md
git commit -m "docs: document MVP insights procedural bridge and graph extras tools"
```

## Final Verification Gate

Run full targeted suite:
```bash
python -m pytest tests/server/test_feedback_tools.py -v
python -m pytest tests/server/test_cross_session_insights_tools.py -v
python -m pytest tests/server/test_graph_extras_tools.py -v
python -m pytest tests/server/test_procedural_memory_tools.py -v
python -m pytest tests/storage/test_hippocampai_adapter_mapping.py -v
python -m pytest tests/services/test_memory_service_scopes.py -v
python -m pytest tests/integration/test_mcp_tool_flows.py -v
```

Release criteria:
- New tools are callable and stable
- All unsupported paths return deterministic `not_supported`
- No regressions in existing feedback/graph tooling
- Docs clearly state configuration and fallback behavior
