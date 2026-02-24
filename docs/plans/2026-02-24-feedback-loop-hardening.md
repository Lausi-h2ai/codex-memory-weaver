# Feedback Loop Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete phase-2 implementation of the HippocampAI relevance feedback loop in this MCP by hardening validation, layering, response contracts, and end-to-end verification.

**Architecture:** Keep MCP tools as the external contract, but route feedback operations through `MemoryService` and `HippocampAIAdapter` for consistent behavior with the rest of the stack. Normalize feedback input (`relevant`, `not_relevant`, `partially_relevant`, `outdated`), preserve graceful `not_supported` handling for older backends, and add deterministic response shapes and tests. This aligns with HippocampAI v0.5 feedback-loop behavior (weighted feedback in retrieval scoring and feedback stats endpoints).

**Tech Stack:** Python 3.10+, FastMCP (`mcp`), HippocampAI v0.5.x, pytest.

---

## Source Requirements (HippocampAI docs)

From `docs/FEATURES.md` in HippocampAI:
- Feedback types: `relevant`, `not_relevant`, `partially_relevant`, `outdated`
- Endpoints:
  - `POST /v1/memories/{memory_id}/feedback`
  - `GET /v1/memories/{memory_id}/feedback`
  - `GET /v1/feedback/stats`
- Retrieval score includes feedback component (`WEIGHT_FEEDBACK`, default `0.1`)
- Feedback events are windowed (`FEEDBACK_WINDOW_DAYS`, default `90`)

This plan treats MCP feedback tools as wrappers over those backend capabilities.

## Task 1: Contract Validation At MCP Boundary (TDD)

**Files:**
- Modify: `tests/server/test_feedback_tools.py`
- Modify: `src/hippocampai_mcp/server.py`

**Step 1: Write failing tests**
- Add invalid feedback type test:

```python
def test_submit_memory_feedback_rejects_invalid_feedback_type(monkeypatch):
    monkeypatch.setattr(server, "memory_client", object())
    payload = server.submit_memory_feedback(
        memory_id="m1",
        user_id="u1",
        feedback_type="thumbs_up",
    )
    assert payload["code"] == "validation_error"
```

- Add normalization test (`RELEVANT` -> `relevant`) and whitespace trimming.

**Step 2: Run tests to verify RED**
- Run: `python -m pytest tests/server/test_feedback_tools.py -v`
- Expected: FAIL on missing validation/normalization.

**Step 3: Write minimal implementation**
- In `server.submit_memory_feedback`:
  - Normalize `feedback_type = feedback_type.strip().lower()`
  - Validate against allowed set
  - Return `_error_payload(code="validation_error", ...)` for invalid values

**Step 4: Run tests to verify GREEN**
- Run: `python -m pytest tests/server/test_feedback_tools.py -v`
- Expected: PASS.

**Step 5: Commit**

```bash
git add tests/server/test_feedback_tools.py src/hippocampai_mcp/server.py
git commit -m "feat: validate and normalize feedback_type in MCP tool"
```

## Task 2: Move Feedback Operations Into Service + Adapter Layers (TDD)

**Files:**
- Modify: `tests/storage/test_hippocampai_adapter_mapping.py`
- Modify: `tests/services/test_memory_service_scopes.py`
- Modify: `src/hippocampai_mcp/storage/hippocampai_adapter.py`
- Modify: `src/hippocampai_mcp/services/memory_service.py`
- Modify: `src/hippocampai_mcp/server.py`

**Step 1: Write failing tests**
- Adapter tests:
  - `submit_memory_feedback` forwards `memory_id`, `user_id`, `feedback_type`, `query`
  - `get_memory_feedback` forwards `memory_id`
  - `get_feedback_stats` forwards `user_id`
  - Missing backend methods return explicit `not_supported` sentinel or raise typed error expected by service/server
- Service tests:
  - Service methods call adapter methods
  - Service returns canonical dict shape
  - Service preserves graceful fallback behavior

**Step 2: Run tests to verify RED**
- Run:
  - `python -m pytest tests/storage/test_hippocampai_adapter_mapping.py -v`
  - `python -m pytest tests/services/test_memory_service_scopes.py -v`
- Expected: FAIL for missing feedback methods.

**Step 3: Write minimal implementation**
- Add to adapter:
  - `submit_memory_feedback(...)`
  - `get_memory_feedback(...)`
  - `get_feedback_stats(...)`
- Add to service:
  - matching pass-through methods
  - normalize output keys when backend returns model objects
- Update server feedback tools to call `memory_service` instead of raw `memory_client` when available.

**Step 4: Run tests to verify GREEN**
- Re-run same test commands; all new tests should pass.

**Step 5: Commit**

```bash
git add tests/storage/test_hippocampai_adapter_mapping.py tests/services/test_memory_service_scopes.py src/hippocampai_mcp/storage/hippocampai_adapter.py src/hippocampai_mcp/services/memory_service.py src/hippocampai_mcp/server.py
git commit -m "refactor: route feedback tools through service and adapter layers"
```

## Task 3: Canonical Response Shapes + Error Semantics (TDD)

**Files:**
- Modify: `tests/server/test_feedback_tools.py`
- Modify: `tests/integration/test_mcp_tool_flows.py`
- Modify: `src/hippocampai_mcp/server.py`

**Step 1: Write failing tests**
- Add response-contract tests:
  - submit returns keys like `memory_id`, `feedback_type`, and backend score/ack fields
  - memory feedback response always includes `memory_id` and aggregate fields when available
  - stats response always includes `user_id`
  - all fallback paths use `code="not_supported"` with correlation ID
- Add error tests for backend exceptions mapping to:
  - `feedback_submit_failed`
  - `feedback_fetch_failed`
  - `feedback_stats_failed`

**Step 2: Run tests to verify RED**
- Run:
  - `python -m pytest tests/server/test_feedback_tools.py -v`
  - `python -m pytest tests/integration/test_mcp_tool_flows.py -k feedback -v`
- Expected: FAIL on inconsistent contracts.

**Step 3: Write minimal implementation**
- Normalize response payloads in server (or service) before returning.
- Ensure each fallback/error payload includes deterministic code/message/details shape.

**Step 4: Run tests to verify GREEN**
- Re-run the same commands and confirm pass.

**Step 5: Commit**

```bash
git add tests/server/test_feedback_tools.py tests/integration/test_mcp_tool_flows.py src/hippocampai_mcp/server.py
git commit -m "feat: stabilize feedback response contracts and error semantics"
```

## Task 4: End-to-End Feedback Influence Smoke Test (TDD)

**Files:**
- Modify: `tests/integration/test_mcp_tool_flows.py`
- Optional Modify: `tests/server/test_feedback_tools.py`

**Step 1: Write failing test**
- Add a backend-stub integration test flow:
  1. recall returns candidate memory
  2. submit `relevant` feedback for that memory with query
  3. subsequent recall reflects feedback-aware behavior in stubbed scoring metadata

```python
def test_feedback_submission_round_trip_integration(...):
    # arrange fake backend with feedback state
    # act: recall -> submit_feedback -> get_feedback -> get_stats
    # assert deterministic updated aggregate
```

**Step 2: Run test to verify RED**
- Run: `python -m pytest tests/integration/test_mcp_tool_flows.py -k "feedback and round_trip" -v`
- Expected: FAIL due missing flow wiring.

**Step 3: Write minimal implementation**
- Implement stub wiring and any missing pass-through fields (especially `query`).
- Keep assertions about deterministic payload changes, not floating score math from real backend.

**Step 4: Run test to verify GREEN**
- Re-run same command and confirm pass.

**Step 5: Commit**

```bash
git add tests/integration/test_mcp_tool_flows.py tests/server/test_feedback_tools.py
git commit -m "test: add feedback loop round-trip integration coverage"
```

## Task 5: Documentation + Operator Guidance

**Files:**
- Modify: `README.md`
- Create: `docs/feedback-loop.md`

**Step 1: Write docs tests/checklist first**
- Add checklist in PR notes:
  - valid feedback types documented
  - fallback behavior documented
  - example tool calls documented

**Step 2: Implement docs**
- Update `README.md` with:
  - feedback tool parameters
  - validation rules for `feedback_type`
  - expected `not_supported` behavior
- Add `docs/feedback-loop.md`:
  - sequence diagram for submit/get/stats
  - troubleshooting for unsupported backend
  - note that backend scoring impact is governed by HippocampAI config (`WEIGHT_FEEDBACK`, `FEEDBACK_WINDOW_DAYS`)

**Step 3: Verification commands**
- Run:
  - `python -m pytest tests/server/test_feedback_tools.py -v`
  - `python -m pytest tests/storage/test_hippocampai_adapter_mapping.py -v`
  - `python -m pytest tests/services/test_memory_service_scopes.py -v`
  - `python -m pytest tests/integration/test_mcp_tool_flows.py -k feedback -v`

**Step 4: Commit**

```bash
git add README.md docs/feedback-loop.md
git commit -m "docs: add feedback loop behavior and troubleshooting guide"
```

## Final Verification Gate

Run full targeted suite:

```bash
python -m pytest tests/server/test_feedback_tools.py -v
python -m pytest tests/storage/test_hippocampai_adapter_mapping.py -v
python -m pytest tests/services/test_memory_service_scopes.py -v
python -m pytest tests/integration/test_mcp_tool_flows.py -v
```

Release criteria:
- All feedback tools validated at input boundary
- Layered architecture is consistent (`server -> service -> adapter`)
- Deterministic error and fallback semantics
- Round-trip feedback behavior is covered by integration tests
- Docs match HippocampAI feedback-loop feature semantics
