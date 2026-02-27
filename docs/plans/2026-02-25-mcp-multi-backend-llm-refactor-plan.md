# MCP Multi-Backend Local LLM Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor MCP runtime initialization so local LLM backend selection is provider-agnostic (Ollama, vLLM OpenAI-compatible, llama.cpp OpenAI-compatible) without breaking existing Ollama behavior.

**Architecture:** Replace Ollama-specific runtime wiring in `server.py` with normalized LLM runtime config (`LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL`) while preserving legacy `OLLAMA_*` compatibility as fallback. Keep this change MCP-only: do not modify HippocampAI package internals in this pass. Update resources/docs/tests so runtime status and configuration reflect the selected backend.

**Tech Stack:** Python 3.10+, FastMCP, HippocampAI v0.5.x, pytest.

---

## Scope Guardrails

- [ ] MCP-side only: no edits under `.tmp_hippocampai/` and no upstream package changes in this plan.
- [ ] Backward compatibility: existing `OLLAMA_BASE_URL` + `OLLAMA_MODEL` users must continue to work unchanged.
- [ ] Security policy: no real API keys in docs/tests/examples.

## Task 1: Add failing tests for provider-agnostic runtime initialization

**Files:**
- Create: `tests/server/test_runtime_llm_backends.py`
- Modify: `src/hippocampai_mcp/server.py`

**Checklist:**
- [ ] Write test: defaults remain Ollama when no new env vars are set.
- [ ] Write test: `LLM_PROVIDER=openai`, `LLM_BASE_URL=http://localhost:8000/v1`, `LLM_MODEL=<model>` is propagated into `MemoryClient` kwargs.
- [ ] Write test: when `LLM_*` present, they take precedence over `OLLAMA_*`.
- [ ] Write test: legacy-only `OLLAMA_*` still maps correctly into `LLM_*` env defaults.
- [ ] Write test: unsupported local provider string returns stable initialization error payload (via `_require_memory_service`).

**Run:**
- [ ] `python -m pytest tests/server/test_runtime_llm_backends.py -v`
- [ ] Confirm new tests fail before implementation.

## Task 2: Implement runtime config normalization in `server.py`

**Files:**
- Modify: `src/hippocampai_mcp/server.py`

**Checklist:**
- [ ] Add helper to resolve effective runtime config:
  - provider: `LLM_PROVIDER` (default `ollama`)
  - base URL/model: `LLM_BASE_URL`/`LLM_MODEL`
  - legacy fallback from `OLLAMA_BASE_URL`/`OLLAMA_MODEL` when `LLM_*` not set.
- [ ] Update `_initialize_runtime_clients()` to use resolved config instead of hard-coded Ollama values.
- [ ] Pass resolved provider/model into `MemoryClient(...)`.
- [ ] Keep env synchronization explicit (set defaults, do not overwrite user-provided `LLM_*`).
- [ ] Add clear warning log when provider is set to a mode likely requiring upstream adapter support (e.g., OpenAI-compatible local endpoint) so operators know where failures originate.

**Run:**
- [ ] `python -m pytest tests/server/test_runtime_llm_backends.py -v`
- [ ] `python -m pytest tests/server/test_graph_autoload.py -v`

## Task 3: Update health/config resources to report active LLM backend

**Files:**
- Modify: `src/hippocampai_mcp/server.py`
- Test: `tests/server/test_runtime_llm_backends.py`

**Checklist:**
- [ ] Refactor `memory://health` dependency key from fixed `ollama` to `llm` while preserving qdrant/redis checks.
- [ ] Health check should probe resolved `LLM_BASE_URL` (with fallback behavior for legacy Ollama env).
- [ ] Update `memory://config` output labels from Ollama-specific to generic LLM provider/base/model.
- [ ] Add tests asserting health/config output under both legacy and generic env settings.

**Run:**
- [ ] `python -m pytest tests/server/test_runtime_llm_backends.py -v`

## Task 4: Update environment templates and documentation

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `docs/codex-config-examples.md`

**Checklist:**
- [ ] Add `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL` to `.env.example` with local-safe defaults.
- [ ] Keep `OLLAMA_*` documented as compatibility aliases/fallback.
- [ ] Add README section: local backend modes:
  - Ollama
  - vLLM (OpenAI-compatible endpoint)
  - llama.cpp server (OpenAI-compatible endpoint)
- [ ] Document known constraint: effective use of `openai` provider with local OpenAI-compatible servers may require HippocampAI adapter behavior support.
- [ ] Ensure docs contain only placeholder keys/values.

**Run:**
- [ ] `rg -n "dev-user|sk-|ghp_|OPENAI_API_KEY=.*[A-Za-z0-9]{10,}" -S README.md docs .env.example`

## Task 5: Regression validation across existing tool surfaces

**Files:**
- Test only (no new file expected)

**Checklist:**
- [ ] Run focused server suites to ensure no regressions:
  - `tests/server/test_feedback_tools.py`
  - `tests/server/test_graph_extras_tools.py`
  - `tests/server/test_cross_session_insights_tools.py`
  - `tests/server/test_procedural_memory_tools.py`
- [ ] Run integration smoke flow:
  - `tests/integration/test_mcp_tool_flows.py`
- [ ] Confirm deterministic error payload shape unchanged for unsupported backends/tools.

**Run:**
- [ ] `python -m pytest tests/server/test_feedback_tools.py tests/server/test_graph_extras_tools.py tests/server/test_cross_session_insights_tools.py tests/server/test_procedural_memory_tools.py -v`
- [ ] `python -m pytest tests/integration/test_mcp_tool_flows.py -v`

## Task 6: Commit sequence (small, reviewable)

**Files:**
- Staged per task

**Checklist:**
- [ ] Commit 1: tests for runtime config (`test_runtime_llm_backends.py`).
- [ ] Commit 2: server runtime/config refactor.
- [ ] Commit 3: docs/env updates.
- [ ] Commit 4: any follow-up fixes from regression tests.

**Suggested commit messages:**
- [ ] `test: add runtime llm backend configuration coverage`
- [ ] `feat: make mcp llm runtime provider-agnostic with ollama fallback`
- [ ] `docs: document local llm backend configuration modes`

## Acceptance Criteria

- [ ] MCP can boot with legacy Ollama-only env unchanged.
- [ ] MCP runtime config supports generic local backend envs (`LLM_*`).
- [ ] `memory://health` and `memory://config` reflect active LLM backend rather than hard-coded Ollama labels.
- [ ] Docs clearly explain Ollama/vLLM/llama.cpp local setup paths and current MCP-vs-upstream boundary.
- [ ] No secret material introduced.
