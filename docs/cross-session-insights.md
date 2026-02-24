# Cross-Session Insights Tools

MVP tools:

- `detect_patterns(user_id, session_ids=None)`
- `track_behavior_changes(user_id, comparison_days=30)`
- `analyze_preference_drift(user_id, category=None)`
- `detect_habits(user_id, min_occurrences=5)`
- `analyze_trends(user_id, window_days=30)`

Success response pattern:

```json
{
  "count": 1,
  "patterns": [{ "id": "p1", "pattern_type": "recurring" }]
}
```

Failure response pattern:

```json
{
  "code": "not_supported",
  "message": "...",
  "details": {},
  "correlation_id": "..."
}
```

Notes:

- These tools call HippocampAI client methods directly.
- If the installed backend/client does not expose a method, the tool returns `not_supported`.
