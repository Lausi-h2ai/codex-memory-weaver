# Feedback Loop Guide

## Scope

This MCP exposes HippocampAI relevance feedback tools:

- `submit_memory_feedback(memory_id, user_id, feedback_type, query=None)`
- `get_memory_feedback(memory_id)`
- `get_feedback_stats(user_id)`

## Validation Rules

- `feedback_type` must be one of:
  - `relevant`
  - `not_relevant`
  - `partially_relevant`
  - `outdated`
- Values are normalized with `strip().lower()` before submission.
- Invalid values return:
  - `code: validation_error`
  - `message` describing allowed values
  - `details.feedback_type` echoing the original input

## Request Flow

1. Client calls `submit_memory_feedback`.
2. Server validates and normalizes `feedback_type`.
3. Server routes to `MemoryService`.
4. `MemoryService` routes to `HippocampAIAdapter`.
5. Adapter calls HippocampAI backend methods.

`get_memory_feedback` and `get_feedback_stats` follow the same `server -> service -> adapter` path.

## Fallback and Error Semantics

- If backend feedback APIs are unavailable:
  - returns `code: not_supported`
  - includes `correlation_id`
- Runtime exceptions are mapped to stable tool codes:
  - `feedback_submit_failed`
  - `feedback_fetch_failed`
  - `feedback_stats_failed`

## Backend Notes

HippocampAI v0.5 feedback loop integrates feedback into retrieval scoring and uses backend configuration values such as:

- `WEIGHT_FEEDBACK` (default `0.1`)
- `FEEDBACK_WINDOW_DAYS` (default `90`)

This MCP forwards feedback operations; scoring math remains owned by HippocampAI backend configuration.
