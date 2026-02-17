# Memory Management MCP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a local, Codex-connectable MCP memory server with clear memory scopes: project-specific, agent-specific, and cross-project user preferences.

**Architecture:** Start from the existing single-file MCP server and refactor into a layered package (`domain`, `storage`, `mcp_tools`). Keep HippocampAI as the first backend to ship quickly, but add a storage abstraction so we can later swap to native SQLite/vector backends without rewriting tools. Encode scope as first-class metadata + validation rules so read/write APIs are deterministic.

**Tech Stack:** Python 3.10+, FastMCP (`mcp`), HippocampAI, Ollama, Qdrant, Redis, pytest, ruff.

---

## Current State Summary (2026-02-17)

- Existing server already implements core memory tools in `hippocampai-mcp-server.py`.
- Scope behavior (project vs agent vs global user preferences) is partially implicit via tags/filters, not enforced as a domain model.
- No test suite is present.
- Packaging has a script mismatch: `pyproject.toml` points to `hippocampai_mcp_server:main` while file is `hippocampai-mcp-server.py`.
- API method naming in the server does not consistently match HippocampAI v0.5.0 reference names (`snake_case`), creating upgrade/compatibility risk.

## HippocampAI Capability Notes (from API_REFERENCE.md + GETTING_STARTED.md)

- Primary stable interface for this MCP should be `MemoryClient` (sync), not `AsyncMemoryClient` or `UnifiedMemoryClient`, because advanced features are sync-only.
- Prefer documented v0.5.0 methods and fields:
  - `remember`, `recall`, `update_memory`, `delete_memory`, `get_memories`
  - `extract_from_conversation`, `create_session`, `summarize_session`, `get_session_memories`
  - `get_agent_memories`, `cluster_user_memories`, `get_memories_by_time_range`, `schedule_memory`
- Use documented filter keys/metadata keys (`agent_id`, `min_importance`, `created_at` naming family) instead of mixed legacy names.
- Treat feature groups as phased:
  - Phase 1: Core memory + scope isolation + sessions + agent support
  - Phase 2: Feedback/triggers/procedural memory + telemetry + background jobs

## Recommended Approach

1. Keep existing backend dependencies for fast delivery.
2. Introduce explicit scope model and validation layer first (lowest risk, highest value).
3. Split monolith into package modules for maintainability and testability.
4. Add tests before changing behavior (TDD for each scope rule).

## Scope Model (Target)

- `PROJECT`: memory belongs to `(user_id, project_id)` and optional `agent_id`.
- `AGENT`: memory belongs to `(user_id, project_id, agent_id)` and is isolated unless explicitly shared.
- `USER_PREFERENCE`: memory belongs to `user_id` only, globally recallable across projects.
- `SESSION`: short-lived working memory tied to session_id (optional but supported).

## Tool Contract Changes (Target)

- Add `scope` parameter with enum validation to write/update tools.
- Require `project_id` for `PROJECT` and `AGENT` scope.
- Require `agent_id` for `AGENT` scope.
- For `USER_PREFERENCE`, ignore/reject `project_id` and `agent_id` to avoid accidental leakage.
- Add dedicated convenience tools:
  - `remember_project_memory`
  - `remember_agent_memory`
  - `remember_user_preference`
  - `recall_project_context`
  - `recall_agent_context`
  - `recall_user_preferences`

## Execution Tasks

### Task 1: Baseline and Packaging Fix

**Files:**
- Modify: `pyproject.toml`
- Create: `src/hippocampai_mcp/__init__.py`
- Create: `src/hippocampai_mcp/server.py`
- Modify: `README.md`

**Steps:**
1. Move server entrypoint into importable module path (`src/hippocampai_mcp/server.py`).
2. Update script entrypoint to `hippocampai_mcp.server:main`.
3. Keep temporary compatibility shim at repo root if needed.
4. Update run instructions in `README.md`.
5. Add explicit dependency pin/range note for HippocampAI v0.5.x APIs used by this server.

**Verification:**
- Run: `uv run python -m hippocampai_mcp.server`
- Expected: server starts without import/path errors.

### Task 2: Define Memory Domain Schema

**Files:**
- Create: `src/hippocampai_mcp/domain/models.py`
- Create: `src/hippocampai_mcp/domain/validation.py`
- Create: `tests/domain/test_scope_validation.py`

**Steps:**
1. Add enums for `MemoryScope` and `MemoryType` mapping.
2. Add normalized request model for write/read operations.
3. Implement validation rules for required/forbidden fields per scope.
4. Write tests first for all scope combinations and error cases.
5. Include `visibility` and optional `run_id` in domain model to align with v0.5.0 multi-agent and execution tracking support.

**Verification:**
- Run: `uv run pytest tests/domain/test_scope_validation.py -v`
- Expected: all scope rules pass.

### Task 3: API Alignment and Storage Adapter Boundary

**Files:**
- Create: `src/hippocampai_mcp/storage/base.py`
- Create: `src/hippocampai_mcp/storage/hippocampai_adapter.py`
- Create: `tests/storage/test_hippocampai_adapter_mapping.py`
- Modify: `src/hippocampai_mcp/server.py`

**Steps:**
1. Define `MemoryStore` interface (`remember`, `recall`, `update`, `delete`, `list`, `stats`).
2. Implement HippocampAI adapter using official v0.5.0 method names (`snake_case`) and parameter names.
3. Centralize tag/metadata encoding (`scope:*`, `project:*`, `agent:*`).
4. Replace any server-layer calls that rely on non-reference method names.
5. Add tests for mapping and filter generation.

**Verification:**
- Run: `uv run pytest tests/storage/test_hippocampai_adapter_mapping.py -v`
- Expected: adapter emits correct tags and filters.

### Task 4: Refactor MCP Tools to Service Layer

**Files:**
- Create: `src/hippocampai_mcp/services/memory_service.py`
- Modify: `src/hippocampai_mcp/server.py`
- Create: `tests/services/test_memory_service_scopes.py`

**Steps:**
1. Move business logic from tool handlers into service methods.
2. Keep tool handlers thin: parse args, call service, return structured response.
3. Add dedicated scoped tools for project, agent, and user-preference memories.
4. Ensure backward-compatible aliases for current tools where possible.
5. Normalize all response payload fields to one style (`snake_case`) for easier Codex tool consumption.

**Verification:**
- Run: `uv run pytest tests/services/test_memory_service_scopes.py -v`
- Expected: scope-specific behavior is deterministic.

### Task 5: Security and Isolation Rules

**Files:**
- Create: `src/hippocampai_mcp/services/access_control.py`
- Modify: `src/hippocampai_mcp/services/memory_service.py`
- Create: `tests/services/test_access_control.py`

**Steps:**
1. Enforce user-level ownership checks on update/delete/list.
2. Block cross-scope recalls unless explicitly requested.
3. Add optional `visibility` handling for agent memories mapped to HippocampAI values (`private`, `shared`, `public`).
4. Add tests for leakage and isolation regressions.

**Verification:**
- Run: `uv run pytest tests/services/test_access_control.py -v`
- Expected: no cross-user/cross-scope leakage.

### Task 6: Reliability and Observability

**Files:**
- Modify: `src/hippocampai_mcp/server.py`
- Create: `src/hippocampai_mcp/telemetry/logging.py`
- Modify: `README.md`

**Steps:**
1. Standardize error payloads (`code`, `message`, `details`).
2. Add correlation IDs and structured logs for each tool call.
3. Add health/dependency readiness checks for Ollama/Qdrant/Redis.
4. Document troubleshooting paths for local hosting.
5. Add optional wrappers for `get_telemetry_metrics` and `get_recent_operations` when available.

**Verification:**
- Run: `uv run python -m hippocampai_mcp.server`
- Expected: health resource returns dependency status.

### Task 7: Integration Tests and Local E2E

**Files:**
- Create: `tests/integration/test_mcp_tool_flows.py`
- Create: `scripts/smoke_test_mcp.py`
- Modify: `README.md`

**Steps:**
1. Write end-to-end tests for three memory scopes.
2. Add a smoke script that simulates Codex-like calls to remember/recall.
3. Test project isolation with two sample project IDs.
4. Test global preference recall across projects.
5. Add one compatibility test that validates expected behavior against HippocampAI v0.5.x API signatures.

**Verification:**
- Run: `uv run pytest tests/integration/test_mcp_tool_flows.py -v`
- Expected: all flows pass with local dependencies up.

### Task 8: Codex Connection and Developer UX

**Files:**
- Modify: `README.md`
- Create: `docs/codex-config-examples.md`
- Create: `.env.example` (if missing)

**Steps:**
1. Add Codex MCP config examples for stdio usage on Windows.
2. Provide quick-start scripts/commands to boot dependencies and server.
3. Document memory naming conventions (`user_id`, `project_id`, `agent_id`).
4. Add migration notes from old tool names to scoped tool names.

**Verification:**
- Run local connection with Codex MCP config.
- Expected: tools discoverable and scoped recalls behave correctly.

## Milestones

- M1 (Day 1): Packaging fixed + domain schema + validation tests.
- M2 (Day 2): Storage adapter + scoped service layer.
- M3 (Day 3): Isolation/security + observability.
- M4 (Day 4): Integration tests + Codex docs + rollout.

## Optional Phase 2 (Post-MVP)

- Add MCP tools that expose HippocampAI v0.5.0 advanced capabilities where they clearly benefit Codex workflows:
  - Feedback endpoints (`/v1/memories/{id}/feedback`) as MCP tools
  - Trigger management (`/v1/triggers`)
  - Procedural memory helpers (if `ENABLE_PROCEDURAL_MEMORY=true`)
  - Scheduler controls (`start_scheduler`, `get_scheduler_status`)
- Gate these behind feature flags to keep base deployment simple and stable.

## Risks and Mitigations

- Risk: HippocampAI API differences from assumptions.
  - Mitigation: adapter contract tests before service migration.
- Risk: scope leakage through permissive tag filters.
  - Mitigation: strict scope filter builder + negative tests.
- Risk: local dependency instability (Ollama/Qdrant/Redis).
  - Mitigation: health/readiness checks and smoke test script.

## Definition of Done

- Scoped memory model implemented and enforced.
- All scope validation + integration tests passing.
- Codex can connect locally and successfully run remember/recall flows for:
  - project memories,
  - agent memories,
  - cross-project user preferences.
- README includes setup, run, and troubleshooting for local-hosted usage.
- Server methods and adapter mapping are aligned with HippocampAI v0.5.x documented API names.
