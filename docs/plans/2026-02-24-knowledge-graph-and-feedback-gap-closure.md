# Knowledge Graph And Feedback Gap Closure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the highest-impact HippocampAI feature gaps in this MCP server by shipping knowledge-graph tooling first (including graph-aware retrieval control), then relevance feedback loop tools.

**Architecture:** Extend the existing layered MCP server (`server` -> `MemoryService` -> `HippocampAIAdapter`) with pass-through wrappers for supported HippocampAI v0.5 methods and explicit MCP contracts. Keep scope-tag semantics unchanged for backward compatibility while adding targeted advanced tools and minimal new service surface. Build incrementally with strict TDD and focused tests per new tool.

**Tech Stack:** Python 3.10+, FastMCP (`mcp`), HippocampAI v0.5.x, pytest.

---

## Gap Summary (from HippocampAI FEATURES.md vs MCP tool surface)

Current MCP server exposes core CRUD, scoped memory tools, sessions, clustering/facts, and basic temporal functions. Missing practical advanced capabilities include:

1. Knowledge graph and relationships
2. Graph-aware retrieval control
3. Relevance feedback loop
4. Version/audit tooling
5. Advanced temporal/insight tools

This plan implements (1) and (2) first, then starts (3), matching project priority.

## Task 1: Knowledge Graph Tool Contracts (TDD)

**Files:**
- Modify: `tests/integration/test_mcp_tool_flows.py`
- Modify: `tests/storage/test_hippocampai_adapter_mapping.py`
- Modify: `tests/storage/test_hippocampai_adapter_strict_signatures.py`

**Step 1: Write failing tests**
- Add tests for adapter/service support of:
  - `add_relationship`
  - `get_related_memories`
- Add server-level tests for MCP response shape and error handling.

**Step 2: Run tests to verify RED**
- Run: `python -m pytest tests/storage/test_hippocampai_adapter_mapping.py tests/storage/test_hippocampai_adapter_strict_signatures.py tests/integration/test_mcp_tool_flows.py -v`
- Expected: failures for missing methods/calls.

**Step 3: Write minimal implementation**
- Add adapter wrappers and service passthrough methods.
- Add MCP tools in `server.py`.

**Step 4: Run tests to verify GREEN**
- Re-run same commands and verify pass.

## Task 2: Graph-Aware Retrieval Mode Support (TDD)

**Files:**
- Modify: `tests/services/test_memory_service_scopes.py`
- Modify: `tests/storage/test_hippocampai_adapter_mapping.py`
- Modify: `src/hippocampai_mcp/storage/hippocampai_adapter.py`
- Modify: `src/hippocampai_mcp/services/memory_service.py`
- Modify: `src/hippocampai_mcp/server.py`

**Step 1: Write failing tests**
- Add tests asserting optional retrieval mode wiring:
  - MCP `recall(..., search_mode="graph_hybrid")`
  - Adapter forwards mode safely using `filters`/kwargs fallback.

**Step 2: Run tests to verify RED**
- Run targeted pytest command for modified tests.
- Expected: failures for unsupported parameter path.

**Step 3: Write minimal implementation**
- Add optional `search_mode` argument in server/service/adapter.
- Ensure backward compatibility when client does not accept mode.

**Step 4: Run tests to verify GREEN**
- Run same tests; ensure no regression in existing scope behavior.

## Task 3: Relevance Feedback Loop MCP Tools (Initial Slice, TDD)

**Files:**
- Modify: `tests/integration/test_mcp_tool_flows.py`
- Modify: `src/hippocampai_mcp/server.py`
- Modify: `README.md`

**Step 1: Write failing tests**
- Add tests for:
  - `submit_memory_feedback(memory_id, user_id, feedback_type, query=None)`
  - `get_memory_feedback(memory_id)`
  - `get_feedback_stats(user_id)`
- Use graceful `not_supported` behavior when HippocampAI backend does not expose feedback methods.

**Step 2: Run tests to verify RED**
- Run targeted integration tests.

**Step 3: Write minimal implementation**
- Add MCP tools that call client methods when available.
- Return structured fallback error payload otherwise.

**Step 4: Run tests to verify GREEN**
- Re-run integration tests and ensure deterministic response schema.

## Task 4: Documentation Updates

**Files:**
- Modify: `README.md`

**Steps:**
1. Add new MCP tools to feature list.
2. Add short examples for knowledge graph and feedback operations.
3. Mark graph-aware retrieval as optional backend capability.

## Verification Checklist

- Run:
  - `python -m pytest tests/storage/test_hippocampai_adapter_mapping.py -v`
  - `python -m pytest tests/storage/test_hippocampai_adapter_strict_signatures.py -v`
  - `python -m pytest tests/services/test_memory_service_scopes.py -v`
  - `python -m pytest tests/integration/test_mcp_tool_flows.py -v`
- Confirm:
  - New tools return stable snake_case payloads.
  - Existing scoped memory behavior is unchanged.
  - Unsupported advanced backend paths return structured errors (not crashes).
