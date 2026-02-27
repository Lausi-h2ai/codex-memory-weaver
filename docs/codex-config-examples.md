# Codex MCP Configuration Examples (Windows)

This document provides ready-to-adapt MCP config snippets for Codex-compatible clients on Windows using stdio transport.

## 1) Python module launch (recommended)

Use this when your environment already has dependencies installed.

```json
{
  "mcpServers": {
    "hippocampai": {
      "command": "python",
      "args": ["-m", "hippocampai_mcp.server"],
      "env": {
        "LLM_PROVIDER": "ollama",
        "LLM_BASE_URL": "http://localhost:11434",
        "LLM_MODEL": "qwen2.5:7b-instruct",
        "QDRANT_URL": "http://localhost:6333",
        "REDIS_URL": "redis://localhost:6379"
      }
    }
  }
}
```

## 2) `uv` launch

Use this when your workflow is centered on `uv`.

```json
{
  "mcpServers": {
    "hippocampai": {
      "command": "uv",
      "args": ["run", "python", "-m", "hippocampai_mcp.server"],
      "env": {
        "LLM_PROVIDER": "ollama",
        "LLM_BASE_URL": "http://localhost:11434",
        "LLM_MODEL": "qwen2.5:7b-instruct",
        "QDRANT_URL": "http://localhost:6333",
        "REDIS_URL": "redis://localhost:6379"
      }
    }
  }
}
```

## 3) Full path `venv` Python launch

Use this when the client does not activate your virtual environment.

```json
{
  "mcpServers": {
    "hippocampai": {
      "command": "F:/hippocampai-mcp/venv/Scripts/python.exe",
      "args": ["-m", "hippocampai_mcp.server"],
      "env": {
        "LLM_PROVIDER": "ollama",
        "LLM_BASE_URL": "http://localhost:11434",
        "LLM_MODEL": "qwen2.5:7b-instruct",
        "QDRANT_URL": "http://localhost:6333",
        "REDIS_URL": "redis://localhost:6379"
      }
    }
  }
}
```

## 4) llama.cpp (`llama-server`) with `-hf` model pull

Use this when you run a local OpenAI-compatible `llama-server`, for example:

```bash
llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

Then point MCP to that server:

```json
{
  "mcpServers": {
    "hippocampai": {
      "command": "python",
      "args": ["-m", "hippocampai_mcp.server"],
      "env": {
        "LLM_PROVIDER": "openai",
        "LLM_BASE_URL": "http://localhost:8080/v1",
        "LLM_MODEL": "gemma-3-1b-it",
        "QDRANT_URL": "http://localhost:6333",
        "REDIS_URL": "redis://localhost:6379"
      }
    }
  }
}
```

## Naming conventions

- `user_id`: stable developer or workspace identity (e.g. `dev-user`, `engineer@company.com`).
- `project_id`: stable repository/workspace identifier (e.g. `hippocampai-mcp`).
- `agent_id`: stable tool/agent identity (e.g. `codex-main`, `codex-debugger`).

## Scoped tool migration quick map

- `remember` -> `remember_project_memory`, `remember_agent_memory`, `remember_user_preference`
- `recall` -> `recall_project_context`, `recall_agent_context`, `recall_user_preferences`
- Legacy tools remain available for compatibility, but scoped variants are recommended for deterministic isolation.
