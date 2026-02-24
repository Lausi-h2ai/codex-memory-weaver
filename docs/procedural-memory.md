# Procedural Memory Tools (HTTP Bridge)

MVP tools:

- `list_procedural_rules(user_id)`
- `extract_procedural_rules(user_id, interactions)`
- `inject_procedural_rules(user_id, prompt, max_rules=3)`
- `update_procedural_rule_feedback(rule_id, effectiveness, user_id=None)`
- `consolidate_procedural_rules(user_id)`

Configuration:

- Set `HIPPOCAMPAI_API_BASE_URL` (example: `http://localhost:8000`).
- Without this env var, tools return `not_supported`.

Endpoint mapping:

- `GET /v1/procedural/rules?user_id=...`
- `POST /v1/procedural/extract`
- `POST /v1/procedural/inject`
- `PUT /v1/procedural/rules/{rule_id}/feedback`
- `POST /v1/procedural/consolidate?user_id=...`

Example success:

```json
{ "rules": [{ "id": "rule-1" }] }
```

Example failure:

```json
{
  "code": "extract_procedural_rules_failed",
  "message": "Failed to extract procedural rules",
  "details": { "error": "..." },
  "correlation_id": "..."
}
```
