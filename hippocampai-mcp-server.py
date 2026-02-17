"""
HippocampAI MCP Server
A feature-complete Model Context Protocol server for persistent memory management
across sessions, projects, and agents using HippocampAI with local Ollama models.
"""

import os
import sys
import logging
from typing import Any, Optional
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP
from hippocampai import MemoryClient
from hippocampai.adapters import OllamaLLM
from hippocampai.types import MemoryType, TimeRange

# Configure logging to stderr (required for STDIO transport)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("hippocampai-mcp")

# Initialize MCP server
mcp = FastMCP(
    "HippocampAI Memory Server",
    dependencies=["hippocampai", "qdrant-client", "redis"]
)

# Global memory client (initialized in lifespan)
memory_client: Optional[MemoryClient] = None


@asynccontextmanager
async def lifespan():
    """Initialize and cleanup HippocampAI client."""
    global memory_client
    
    try:
        # Load configuration from environment
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        
        logger.info(f"Initializing HippocampAI with Ollama at {ollama_url}")
        logger.info(f"Using model: {ollama_model}")
        
        # Initialize memory client with Ollama
        memory_client = MemoryClient(
            llmprovider=OllamaLLM(
                baseurl=ollama_url,
                model=ollama_model
            ),
            qdranturl=qdrant_url,
            redisurl=redis_url,
        )
        
        logger.info("HippocampAI Memory Server initialized successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize memory client: {e}")
        raise
    finally:
        # Cleanup if needed
        memory_client = None
        logger.info("HippocampAI Memory Server shutdown")


# Apply lifespan handler
mcp.lifespan_handler = lifespan


# ============================================================================
# CORE MEMORY OPERATIONS
# ============================================================================

@mcp.tool()
def remember(
    text: str,
    user_id: str,
    session_id: Optional[str] = None,
    memory_type: str = "context",
    importance: Optional[float] = None,
    tags: Optional[list[str]] = None,
    agent_id: Optional[str] = None,
    project: Optional[str] = None,
    ttl_days: Optional[int] = None,
) -> dict[str, Any]:
    """
    Store a new memory in HippocampAI with automatic semantic enrichment.
    
    Use this to persist important information across sessions, projects, and agents.
    
    Args:
        text: The content to remember
        user_id: User/developer identifier (e.g., git username, OS user)
        session_id: Optional session/conversation identifier
        memory_type: Type of memory - "fact", "preference", "goal", "habit", "event", "context"
        importance: Importance score 0-10 (auto-calculated if not provided)
        tags: List of tags for categorization
        agent_id: Agent identifier for multi-agent systems
        project: Project name/identifier (stored as tag)
        ttl_days: Time-to-live in days (memory expires after this period)
    
    Returns:
        Dictionary with memory ID, type, importance, and extracted facts
    """
    try:
        # Add project as tag if provided
        if project:
            tags = tags or []
            if f"project:{project}" not in tags:
                tags.append(f"project:{project}")
        
        memory = memory_client.remember(
            text=text,
            userid=user_id,
            sessionid=session_id,
            type=memory_type,
            importance=importance,
            tags=tags,
            agentid=agent_id,
            ttldays=ttl_days,
        )
        
        return {
            "id": memory.id,
            "type": memory.type,
            "importance": memory.importance,
            "tags": memory.tags,
            "extracted_facts": getattr(memory, "extractedfacts", None),
            "created_at": memory.createdat.isoformat() if hasattr(memory, "createdat") else None,
            "message": f"Memory stored successfully with ID: {memory.id}"
        }
        
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        return {"error": str(e), "success": False}


@mcp.tool()
def recall(
    query: str,
    user_id: str,
    session_id: Optional[str] = None,
    k: int = 5,
    min_importance: Optional[float] = None,
    memory_type: Optional[str] = None,
    tags: Optional[list[str]] = None,
    agent_id: Optional[str] = None,
    project: Optional[str] = None,
) -> dict[str, Any]:
    """
    Retrieve relevant memories using hybrid semantic search.
    
    Searches across all stored memories for the user and returns the most relevant results.
    
    Args:
        query: Search query describing what you want to recall
        user_id: User identifier to search memories for
        session_id: Optional filter by specific session
        k: Number of results to return (default: 5)
        min_importance: Minimum importance threshold (0-10)
        memory_type: Filter by memory type
        tags: Filter by tags (AND logic)
        agent_id: Filter by agent
        project: Filter by project name
    
    Returns:
        List of relevant memories with scores and metadata
    """
    try:
        # Build filters
        filters = {}
        if min_importance is not None:
            filters["minimportance"] = min_importance
        if memory_type:
            filters["type"] = memory_type
        if agent_id:
            filters["agentid"] = agent_id
        if project:
            filters["tags"] = [f"project:{project}"]
        elif tags:
            filters["tags"] = tags
        
        results = memory_client.recall(
            query=query,
            userid=user_id,
            sessionid=session_id,
            k=k,
            filters=filters if filters else None,
        )
        
        return {
            "results": [
                {
                    "memory_id": r.memory.id,
                    "text": r.memory.text,
                    "score": r.score,
                    "type": r.memory.type,
                    "importance": getattr(r.memory, "importance", None),
                    "tags": getattr(r.memory, "tags", []),
                    "session_id": getattr(r.memory, "sessionid", None),
                    "agent_id": getattr(r.memory, "agentid", None),
                    "created_at": r.memory.createdat.isoformat() if hasattr(r.memory, "createdat") else None,
                }
                for r in results
            ],
            "count": len(results),
            "query": query
        }
        
    except Exception as e:
        logger.error(f"Error recalling memories: {e}")
        return {"error": str(e), "results": [], "count": 0}


@mcp.tool()
def extract_from_conversation(
    conversation: str,
    user_id: str,
    session_id: Optional[str] = None,
    project: Optional[str] = None,
) -> dict[str, Any]:
    """
    Extract and store multiple memories from a conversation using LLM analysis.
    
    Automatically identifies facts, preferences, goals, and other memorable information
    from conversation text and stores them as separate memories.
    
    Args:
        conversation: Multi-turn conversation text to analyze
        user_id: User identifier
        session_id: Session identifier for tracking
        project: Project name to associate with extracted memories
    
    Returns:
        List of extracted and stored memories
    """
    try:
        metadata = {}
        if project:
            metadata["project"] = project
        
        memories = memory_client.extractfromconversation(
            conversation=conversation,
            userid=user_id,
            sessionid=session_id,
            metadata=metadata,
        )
        
        return {
            "extracted_count": len(memories),
            "memories": [
                {
                    "id": m.id,
                    "text": m.text,
                    "type": m.type,
                    "importance": getattr(m, "importance", None),
                }
                for m in memories
            ],
            "message": f"Extracted and stored {len(memories)} memories from conversation"
        }
        
    except Exception as e:
        logger.error(f"Error extracting from conversation: {e}")
        return {"error": str(e), "extracted_count": 0}


# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

@mcp.tool()
def get_memories(
    user_id: str,
    memory_type: Optional[str] = None,
    tags: Optional[list[str]] = None,
    session_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    project: Optional[str] = None,
    limit: int = 50,
    sort_by: str = "createdat",
    order: str = "desc",
) -> dict[str, Any]:
    """
    List and browse memories with filtering and sorting.
    
    Args:
        user_id: User identifier
        memory_type: Filter by type
        tags: Filter by tags
        session_id: Filter by session
        agent_id: Filter by agent
        project: Filter by project
        limit: Maximum results (default: 50)
        sort_by: Sort field - "createdat", "importance", "accessedat"
        order: Sort order - "asc" or "desc"
    
    Returns:
        List of memories with metadata
    """
    try:
        # Add project filter to tags if provided
        if project:
            tags = tags or []
            tags.append(f"project:{project}")
        
        memories = memory_client.getmemories(
            userid=user_id,
            type=getattr(MemoryType, memory_type.upper(), None) if memory_type else None,
            tags=tags,
            sessionid=session_id,
            agentid=agent_id,
            limit=limit,
            sortby=sort_by,
            order=order,
        )
        
        return {
            "memories": [
                {
                    "id": m.id,
                    "text": m.text,
                    "type": m.type,
                    "importance": getattr(m, "importance", None),
                    "tags": getattr(m, "tags", []),
                    "session_id": getattr(m, "sessionid", None),
                    "created_at": m.createdat.isoformat() if hasattr(m, "createdat") else None,
                }
                for m in memories
            ],
            "count": len(memories),
        }
        
    except Exception as e:
        logger.error(f"Error getting memories: {e}")
        return {"error": str(e), "memories": [], "count": 0}


@mcp.tool()
def update_memory(
    memory_id: str,
    text: Optional[str] = None,
    importance: Optional[float] = None,
    tags: Optional[list[str]] = None,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Update an existing memory's content or metadata.
    
    Args:
        memory_id: Memory ID to update
        text: New memory text (creates new version)
        importance: New importance score (0-10)
        tags: New tags (replaces existing)
        user_id: User ID for validation
    
    Returns:
        Updated memory information
    """
    try:
        updated = memory_client.updatememory(
            memoryid=memory_id,
            text=text,
            importance=importance,
            tags=tags,
            userid=user_id,
        )
        
        if updated:
            return {
                "id": updated.id,
                "text": updated.text,
                "importance": getattr(updated, "importance", None),
                "tags": getattr(updated, "tags", []),
                "message": "Memory updated successfully"
            }
        else:
            return {"error": "Memory not found", "success": False}
            
    except Exception as e:
        logger.error(f"Error updating memory: {e}")
        return {"error": str(e), "success": False}


@mcp.tool()
def delete_memory(
    memory_id: str,
    user_id: Optional[str] = None,
) -> dict[str, bool]:
    """
    Delete a memory by ID.
    
    Args:
        memory_id: Memory ID to delete
        user_id: User ID for validation
    
    Returns:
        Success status
    """
    try:
        success = memory_client.deletememory(
            memoryid=memory_id,
            userid=user_id,
        )
        
        return {
            "success": success,
            "message": "Memory deleted successfully" if success else "Memory not found"
        }
        
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def get_memory_statistics(
    user_id: str,
    project: Optional[str] = None,
) -> dict[str, Any]:
    """
    Get statistics about a user's memories.
    
    Args:
        user_id: User identifier
        project: Optional project filter
    
    Returns:
        Statistics including counts, types, tags, and usage metrics
    """
    try:
        stats = memory_client.getmemorystatistics(userid=user_id)
        
        # If project filter is provided, also get project-specific stats
        if project:
            project_memories = memory_client.getmemories(
                userid=user_id,
                tags=[f"project:{project}"],
                limit=1000,
            )
            stats["project_memory_count"] = len(project_memories)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return {"error": str(e)}


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

@mcp.tool()
def create_session(
    user_id: str,
    title: Optional[str] = None,
    project: Optional[str] = None,
) -> dict[str, Any]:
    """
    Create a new conversation session for tracking related memories.
    
    Sessions help organize memories by conversation, task, or editing session.
    
    Args:
        user_id: User identifier
        title: Session title/description
        project: Project name to associate with this session
    
    Returns:
        Session ID and metadata
    """
    try:
        metadata = {}
        if project:
            metadata["project"] = project
        
        session = memory_client.createsession(
            userid=user_id,
            title=title,
            metadata=metadata,
        )
        
        return {
            "session_id": session.id,
            "title": session.title if hasattr(session, "title") else None,
            "created_at": session.createdat.isoformat() if hasattr(session, "createdat") else None,
            "message": f"Session created with ID: {session.id}"
        }
        
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return {"error": str(e)}


@mcp.tool()
def summarize_session(
    session_id: str,
    force: bool = False,
) -> dict[str, str]:
    """
    Generate or retrieve a summary of a conversation session.
    
    Args:
        session_id: Session identifier
        force: Force regeneration even if summary exists
    
    Returns:
        Session summary text
    """
    try:
        summary = memory_client.summarizesession(
            sessionid=session_id,
            force=force,
        )
        
        return {
            "session_id": session_id,
            "summary": summary,
        }
        
    except Exception as e:
        logger.error(f"Error summarizing session: {e}")
        return {"error": str(e)}


@mcp.tool()
def get_session_memories(
    session_id: str,
    limit: int = 100,
) -> dict[str, Any]:
    """
    Get all memories associated with a specific session.
    
    Args:
        session_id: Session identifier
        limit: Maximum results
    
    Returns:
        List of memories from the session
    """
    try:
        memories = memory_client.getsessionmemories(
            sessionid=session_id,
            limit=limit,
        )
        
        return {
            "memories": [
                {
                    "id": m.id,
                    "text": m.text,
                    "type": m.type,
                    "created_at": m.createdat.isoformat() if hasattr(m, "createdat") else None,
                }
                for m in memories
            ],
            "count": len(memories),
        }
        
    except Exception as e:
        logger.error(f"Error getting session memories: {e}")
        return {"error": str(e), "memories": [], "count": 0}


# ============================================================================
# MULTI-AGENT SUPPORT
# ============================================================================

@mcp.tool()
def get_agent_memories(
    agent_id: str,
    user_id: str,
    limit: int = 100,
) -> dict[str, Any]:
    """
    Get memories owned by a specific agent.
    
    Args:
        agent_id: Agent identifier
        user_id: User identifier
        limit: Maximum results
    
    Returns:
        List of agent's memories
    """
    try:
        memories = memory_client.getagentmemories(
            agentid=agent_id,
            userid=user_id,
            limit=limit,
        )
        
        return {
            "memories": [
                {
                    "id": m.id,
                    "text": m.text,
                    "type": m.type,
                    "created_at": m.createdat.isoformat() if hasattr(m, "createdat") else None,
                }
                for m in memories
            ],
            "count": len(memories),
            "agent_id": agent_id,
        }
        
    except Exception as e:
        logger.error(f"Error getting agent memories: {e}")
        return {"error": str(e), "memories": [], "count": 0}


# ============================================================================
# INTELLIGENCE FEATURES
# ============================================================================

@mcp.tool()
def extract_facts(
    text: str,
    user_id: str,
    confidence_threshold: float = 0.7,
) -> dict[str, Any]:
    """
    Extract structured facts from text using LLM analysis.
    
    Args:
        text: Text to analyze
        user_id: User identifier (for context)
        confidence_threshold: Minimum confidence score (0-1)
    
    Returns:
        List of extracted facts with categories and confidence scores
    """
    try:
        facts = memory_client.extractfacts(
            text=text,
            confidencethreshold=confidence_threshold,
        )
        
        return {
            "facts": [
                {
                    "category": f.category,
                    "fact": f.fact,
                    "confidence": getattr(f, "confidence", None),
                }
                for f in facts
            ],
            "count": len(facts),
        }
        
    except Exception as e:
        logger.error(f"Error extracting facts: {e}")
        return {"error": str(e), "facts": [], "count": 0}


@mcp.tool()
def cluster_memories(
    user_id: str,
    max_clusters: int = 10,
) -> dict[str, Any]:
    """
    Automatically cluster memories by semantic similarity.
    
    Discovers patterns and topics across all memories.
    
    Args:
        user_id: User identifier
        max_clusters: Maximum number of clusters
    
    Returns:
        List of memory clusters with topics and member memories
    """
    try:
        clusters = memory_client.clusterusermemories(
            userid=user_id,
            maxclusters=max_clusters,
        )
        
        return {
            "clusters": [
                {
                    "topic": c.topic if hasattr(c, "topic") else None,
                    "memory_count": len(c.memories) if hasattr(c, "memories") else 0,
                    "memories": [m.id for m in c.memories] if hasattr(c, "memories") else [],
                }
                for c in clusters
            ],
            "cluster_count": len(clusters),
        }
        
    except Exception as e:
        logger.error(f"Error clustering memories: {e}")
        return {"error": str(e), "clusters": [], "cluster_count": 0}


# ============================================================================
# TEMPORAL FEATURES
# ============================================================================

@mcp.tool()
def get_recent_memories(
    user_id: str,
    time_window: str = "LASTDAY",
    project: Optional[str] = None,
) -> dict[str, Any]:
    """
    Get memories from a specific time period.
    
    Args:
        user_id: User identifier
        time_window: Time period - "LASTHOUR", "LASTDAY", "LASTWEEK", "LASTMONTH", "LASTYEAR"
        project: Optional project filter
    
    Returns:
        List of memories from the specified time window
    """
    try:
        time_range = getattr(TimeRange, time_window.upper(), TimeRange.LASTDAY)
        memories = memory_client.getmemoriesbytimerange(
            userid=user_id,
            timerange=time_range,
        )
        
        # Filter by project if provided
        if project:
            memories = [
                m for m in memories
                if f"project:{project}" in getattr(m, "tags", [])
            ]
        
        return {
            "memories": [
                {
                    "id": m.id,
                    "text": m.text,
                    "type": m.type,
                    "created_at": m.createdat.isoformat() if hasattr(m, "createdat") else None,
                }
                for m in memories
            ],
            "count": len(memories),
            "time_window": time_window,
        }
        
    except Exception as e:
        logger.error(f"Error getting recent memories: {e}")
        return {"error": str(e), "memories": [], "count": 0}


@mcp.tool()
def schedule_memory(
    text: str,
    user_id: str,
    scheduled_for_iso: str,
    recurrence: Optional[str] = None,
) -> dict[str, Any]:
    """
    Schedule a memory for future activation (e.g., reminders, follow-ups).
    
    Args:
        text: Memory content
        user_id: User identifier
        scheduled_for_iso: ISO 8601 datetime string (e.g., "2026-02-20T14:00:00Z")
        recurrence: Recurrence pattern - "daily", "weekly", "monthly"
    
    Returns:
        Scheduled memory information
    """
    try:
        scheduled_for = datetime.fromisoformat(scheduled_for_iso.replace("Z", "+00:00"))
        
        memory = memory_client.schedulememory(
            text=text,
            userid=user_id,
            scheduledfor=scheduled_for,
            recurrence=recurrence,
        )
        
        return {
            "id": memory.id,
            "text": memory.text,
            "scheduled_for": scheduled_for_iso,
            "recurrence": recurrence,
            "message": "Memory scheduled successfully"
        }
        
    except Exception as e:
        logger.error(f"Error scheduling memory: {e}")
        return {"error": str(e)}


# ============================================================================
# RESOURCES (Read-only data access)
# ============================================================================

@mcp.resource("memory://health")
def health_check() -> str:
    """Server health check."""
    return "HippocampAI MCP Server is running"


@mcp.resource("memory://config")
def get_config() -> str:
    """Get current server configuration."""
    return f"""
HippocampAI MCP Server Configuration:
- Ollama URL: {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}
- Ollama Model: {os.getenv('OLLAMA_MODEL', 'qwen2.5:7b-instruct')}
- Qdrant URL: {os.getenv('QDRANT_URL', 'http://localhost:6333')}
- Redis URL: {os.getenv('REDIS_URL', 'redis://localhost:6379')}
"""


# ============================================================================
# PROMPTS (Reusable templates)
# ============================================================================

@mcp.prompt()
def memory_workflow_prompt(
    user_id: str,
    project: str,
    task_description: str,
) -> str:
    """
    Generate a prompt for a memory-enhanced workflow.
    
    Args:
        user_id: User identifier
        project: Project name
        task_description: Description of the task
    """
    return f"""
You are working on the project "{project}" for user {user_id}.

Task: {task_description}

Before starting, use the recall tool to check for:
1. Relevant past decisions or architecture choices for this project
2. User preferences that might affect this task
3. Related code patterns or conventions used in this project

As you work:
- Use remember to store important decisions, patterns, or learnings
- Tag memories with "project:{project}" for easy retrieval
- Set appropriate importance levels (8-10 for critical decisions)

After completing the task:
- Summarize key learnings or decisions made
- Store them for future reference
"""


@mcp.prompt()
def project_context_prompt(project: str, user_id: str) -> str:
    """
    Generate a prompt to retrieve full project context.
    
    Args:
        project: Project name
        user_id: User identifier
    """
    return f"""
Retrieve the complete context for project "{project}" by:

1. Call get_memories with project="{project}" to get all project memories
2. Call cluster_memories to see patterns and themes
3. Call get_memory_statistics to understand the memory landscape

Use this context to provide informed answers about the project's:
- Architecture decisions
- Coding patterns and conventions
- Past challenges and solutions
- User preferences for this project
"""


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Run the HippocampAI MCP server."""
    try:
        logger.info("Starting HippocampAI MCP Server...")
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()
