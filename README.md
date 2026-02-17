# HippocampAI MCP Server

A feature-complete Model Context Protocol (MCP) server that provides persistent memory management across sessions, projects, and agents using HippocampAI with local Ollama models.

## Features

### Core Memory Operations
- ✅ **remember**: Store memories with automatic semantic enrichment
- ✅ **recall**: Hybrid semantic search across all memories
- ✅ **extract_from_conversation**: Auto-extract memories from conversations
- ✅ **get_memories**: List and filter memories
- ✅ **update_memory**: Modify existing memories
- ✅ **delete_memory**: Remove memories

### Session Management
- ✅ **create_session**: Track conversation/editing sessions
- ✅ **summarize_session**: Generate session summaries
- ✅ **get_session_memories**: Retrieve session-specific memories

### Multi-Agent Support
- ✅ **get_agent_memories**: Agent-specific memory isolation
- ✅ Cross-agent memory sharing with visibility controls

### Intelligence Features
- ✅ **extract_facts**: Structured fact extraction from text
- ✅ **cluster_memories**: Automatic semantic clustering
- ✅ Pattern detection across memory corpus

### Temporal Features
- ✅ **get_recent_memories**: Time-based memory retrieval
- ✅ **schedule_memory**: Schedule future reminders/follow-ups
- ✅ Automatic importance decay over time

### Statistics & Analytics
- ✅ **get_memory_statistics**: Usage metrics and insights
- ✅ Memory type distribution
- ✅ Tag analysis

## Prerequisites

1. **Python 3.10+** installed
2. **Docker** (for HippocampAI stack) or manual installations:
   - Qdrant vector database
   - Redis cache
   - Ollama with a model

## Quick Start

### Option 1: Docker Stack (Recommended)

```bash
# Clone HippocampAI and start the stack
git clone https://github.com/rexdivakar/HippocampAI.git
cd HippocampAI
docker-compose up -d

# Verify services
curl http://localhost:6333/dashboard  # Qdrant
curl http://localhost:11434/api/tags  # Ollama
```

### Option 2: Manual Setup

```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant:latest

# Start Redis
docker run -p 6379:6379 redis:7-alpine

# Install and run Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b-instruct
```

### Install the MCP Server

```bash
# Create project directory
mkdir hippocampai-mcp && cd hippocampai-mcp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install hippocampai mcp qdrant-client redis

# Copy the server file (hippocampai_mcp_server.py)
# and configuration (.env)
```

### Configure Environment

Create `.env` file:

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
```

### Test the Server

```bash
# Run directly
python hippocampai_mcp_server.py

# Or with uv (recommended)
uv run hippocampai_mcp_server.py
```

## Integration with OpenAI Codex

### Configure Claude for Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "hippocampai": {
      "command": "python",
      "args": ["/absolute/path/to/hippocampai_mcp_server.py"],
      "env": {
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_MODEL": "qwen2.5:7b-instruct",
        "QDRANT_URL": "http://localhost:6333",
        "REDIS_URL": "redis://localhost:6379"
      }
    }
  }
}
```

### Configure for Other MCP Clients

For VS Code Codex, Cursor, or other MCP-compatible tools, use similar configuration pointing to the server script.

## Usage Examples

### Store a memory
```
Remember: "In this project we use FastAPI with async/await patterns throughout. DB models use SQLAlchemy 2.0 style."

User: your-username
Project: my-api-service
Tags: architecture, fastapi, patterns
Importance: 9
```

### Recall relevant context
```
Recall memories about "database patterns and async handling" for project "my-api-service"
```

### Extract from conversation
```
Extract memories from this conversation:
User: Should we use Pydantic v2?
Assistant: Yes, it's more performant
User: Great, let's standardize on that
```

### Get project overview
```
Get all memories for project "my-api-service" sorted by importance
```

### Schedule a reminder
```
Schedule memory: "Review API rate limiting implementation" 
For: 2026-02-25T10:00:00Z
Recurrence: weekly
```

## Memory Organization Best Practices

### User ID Strategy
- Use consistent developer ID: `git config user.name` or OS username
- Example: `alice`, `bob@company.com`

### Session ID Strategy
- Per Codex conversation: `codex-chat-{timestamp}`
- Per editing session: `vscode-{workspace}-{timestamp}`
- Per task/ticket: `task-{ticket-id}`

### Project Tagging
- Always tag with `project:{name}` for project-scoped recall
- Use consistent project identifiers across tools

### Agent ID Strategy
- Per Codex agent: `codex-agent`, `codex-debug-agent`
- Per tool: `linter`, `refactor-tool`, `test-generator`

### Memory Types
- `fact`: Architecture decisions, technical facts
- `preference`: User coding preferences, patterns
- `goal`: Project objectives, TODO items
- `context`: Background information
- `event`: Meetings, discussions, decisions made

### Importance Scoring
- 9-10: Critical architecture decisions, security patterns
- 7-8: Important conventions, frequently needed context
- 5-6: Useful but not critical information
- 1-4: Temporary or low-priority context

## Architecture

```
┌─────────────────────────────────────────────┐
│         OpenAI Codex / MCP Client           │
│    (Claude Desktop, VS Code, Cursor, etc.)  │
└──────────────────┬──────────────────────────┘
                   │ MCP Protocol (stdio/http)
                   ▼
┌─────────────────────────────────────────────┐
│       HippocampAI MCP Server (Python)       │
│  ┌──────────────────────────────────────┐   │
│  │  FastMCP Framework                   │   │
│  │  - Tool handlers                     │   │
│  │  - Resource providers                │   │
│  │  - Prompt templates                  │   │
│  └──────────────┬───────────────────────┘   │
│                 │                            │
│  ┌──────────────▼───────────────────────┐   │
│  │  HippocampAI MemoryClient            │   │
│  │  - Hybrid search (vector + BM25)     │   │
│  │  - Automatic enrichment              │   │
│  │  - Multi-agent coordination          │   │
│  └──────────────┬───────────────────────┘   │
└─────────────────┼───────────────────────────┘
                  │
       ┌──────────┼──────────┐
       ▼          ▼           ▼
  ┌────────┐  ┌──────┐  ┌─────────┐
  │ Ollama │  │Qdrant│  │  Redis  │
  │ (LLM)  │  │(Vec) │  │ (Cache) │
  └────────┘  └──────┘  └─────────┘
```

## Troubleshooting

### Server not connecting
```bash
# Check logs
tail -f ~/Library/Logs/Claude/mcp-server-hippocampai.log

# Test server directly
python hippocampai_mcp_server.py
```

### Ollama not responding
```bash
# Check Ollama
ollama list
ollama run qwen2.5:7b-instruct "Hello"

# Restart Ollama service
# On macOS: brew services restart ollama
# On Linux: systemctl restart ollama
```

### Qdrant connection failed
```bash
# Check Qdrant health
curl http://localhost:6333/healthz

# View logs
docker logs <qdrant-container-id>
```

### Redis connection failed
```bash
# Check Redis
redis-cli ping

# If not running
docker run -d -p 6379:6379 redis:7-alpine
```

## Advanced Configuration

### Custom Ollama Model

```bash
# Use a different model
export OLLAMA_MODEL=llama3:8b

# Or for better performance
export OLLAMA_MODEL=mistral:7b-instruct
```

### Production Deployment

For production use with multiple users:

```bash
# Use remote Qdrant cloud
export QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333

# Use managed Redis
export REDIS_URL=redis://your-redis.cloud:6379

# Consider using Groq/OpenAI for better quality
pip install hippocampai[groq]
export LLM_PROVIDER=groq
export GROQ_API_KEY=your-key
```

## API Reference

### Core Tools

**remember(text, user_id, ...)**
- Stores a new memory with semantic enrichment
- Returns: memory ID, extracted facts, importance

**recall(query, user_id, ...)**
- Hybrid search across memories
- Returns: ranked list of relevant memories

**extract_from_conversation(conversation, user_id, ...)**
- Batch memory extraction from conversation
- Returns: list of extracted and stored memories

### Session Tools

**create_session(user_id, title, project)**
- Creates a new session for grouping memories
- Returns: session ID

**summarize_session(session_id)**
- Generates LLM summary of session memories
- Returns: summary text

### Intelligence Tools

**extract_facts(text, user_id)**
- Extracts structured facts with confidence scores
- Returns: list of categorized facts

**cluster_memories(user_id, max_clusters)**
- Semantic clustering of all user memories
- Returns: clusters with topics and member IDs

### Temporal Tools

**get_recent_memories(user_id, time_window)**
- Time-based memory retrieval
- Windows: LASTHOUR, LASTDAY, LASTWEEK, LASTMONTH

**schedule_memory(text, user_id, scheduled_for_iso, recurrence)**
- Schedule future memory activation
- Recurrence: daily, weekly, monthly

## License

MIT License - see LICENSE file

## Contributing

Contributions welcome! Please submit issues and pull requests.

## Support

- HippocampAI: https://github.com/rexdivakar/HippocampAI
- MCP Protocol: https://modelcontextprotocol.io
- Issues: https://github.com/your-repo/issues
