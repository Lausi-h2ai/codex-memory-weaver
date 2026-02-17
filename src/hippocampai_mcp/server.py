"""
HippocampAI MCP Server
A feature-complete Model Context Protocol server for persistent memory management
across sessions, projects, and agents using HippocampAI with local Ollama models.
"""

import os
import sys
import logging
import json
import socket
import uuid
from typing import Any, Optional
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP
from hippocampai import MemoryClient
from hippocampai.adapters import OllamaLLM
from hippocampai.types import TimeRange
from hippocampai_mcp.domain.models import MemoryScope
from hippocampai_mcp.storage import HippocampAIAdapter, MemoryStore
from hippocampai_mcp.services import AccessControlError, MemoryService
from hippocampai_mcp.telemetry.logging import emit_tool_log

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
memory_store: Optional[MemoryStore] = None
memory_service: Optional[MemoryService] = None


def _attr(obj: Any, *names: str) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _iso_attr(obj: Any, *names: str) -> str | None:
    value = _attr(obj, *names)
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _error_payload(
    *,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    payload = {
        "code": code,
        "message": message,
        "details": details or {},
    }
    if correlation_id:
        payload["correlation_id"] = correlation_id
    return payload


def _new_correlation_id() -> str:
    return str(uuid.uuid4())


def _check_tcp_dependency(url: str, default_port: int) -> dict[str, Any]:
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or default_port
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return {"status": "ok", "host": host, "port": port}
    except OSError as exc:
        return {"status": "error", "host": host, "port": port, "error": str(exc)}


@asynccontextmanager
async def lifespan():
    """Initialize and cleanup HippocampAI client."""
    global memory_client, memory_store, memory_service
    
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
            llm_provider=OllamaLLM(
                base_url=ollama_url,
                model=ollama_model
            ),
            qdrant_url=qdrant_url,
            redis_url=redis_url,
        )
        memory_store = HippocampAIAdapter(memory_client)
        memory_service = MemoryService(memory_store)
        
        logger.info("HippocampAI Memory Server initialized successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize memory client: {e}")
        raise
    finally:
        # Cleanup if needed
        memory_client = None
        memory_store = None
        memory_service = None
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
    correlation_id = _new_correlation_id()
    emit_tool_log(logger, event="tool_start", tool="remember", correlation_id=correlation_id)
    try:
        response = memory_service.remember(
            text=text,
            user_id=user_id,
            project=project,
            agent_id=agent_id,
            session_id=session_id,
            memory_type=memory_type,
            importance=importance,
            tags=tags,
            ttl_days=ttl_days,
        )
        response["message"] = f"Memory stored successfully with ID: {response['id']}"
        emit_tool_log(
            logger,
            event="tool_success",
            tool="remember",
            correlation_id=correlation_id,
            memory_id=response["id"],
        )
        return response
    except Exception as e:
        emit_tool_log(
            logger,
            event="tool_error",
            tool="remember",
            correlation_id=correlation_id,
            error=str(e),
            level=logging.ERROR,
        )
        return _error_payload(
            code="memory_store_failed",
            message="Failed to store memory",
            details={"error": str(e)},
            correlation_id=correlation_id,
        )


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
    correlation_id = _new_correlation_id()
    emit_tool_log(logger, event="tool_start", tool="recall", correlation_id=correlation_id)
    try:
        response = memory_service.recall(
            query=query,
            user_id=user_id,
            scope=None,
            project_id=project,
            agent_id=agent_id,
            session_id=session_id,
            k=k,
            min_importance=min_importance,
            memory_type=memory_type,
            tags=tags,
            include_cross_scope=False,
        )
        emit_tool_log(
            logger,
            event="tool_success",
            tool="recall",
            correlation_id=correlation_id,
            result_count=response["count"],
        )
        return response
    except Exception as e:
        emit_tool_log(
            logger,
            event="tool_error",
            tool="recall",
            correlation_id=correlation_id,
            error=str(e),
            level=logging.ERROR,
        )
        return _error_payload(
            code="memory_recall_failed",
            message="Failed to recall memories",
            details={"error": str(e)},
            correlation_id=correlation_id,
        )


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
        
        memories = memory_client.extract_from_conversation(
            conversation=conversation,
            user_id=user_id,
            session_id=session_id,
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
    sort_by: str = "created_at",
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
        sort_by: Sort field - "created_at", "importance", "accessed_at"
        order: Sort order - "asc" or "desc"
    
    Returns:
        List of memories with metadata
    """
    correlation_id = _new_correlation_id()
    emit_tool_log(logger, event="tool_start", tool="get_memories", correlation_id=correlation_id)
    try:
        scope: MemoryScope | None = None
        if session_id:
            scope = MemoryScope.SESSION
        elif agent_id:
            scope = MemoryScope.AGENT
        elif project:
            scope = MemoryScope.PROJECT
        response = memory_service.list_memories(
            user_id=user_id,
            scope=scope,
            project_id=project,
            agent_id=agent_id,
            session_id=session_id,
            memory_type=memory_type,
            tags=tags,
            limit=limit,
            sort_by=sort_by,
            order=order,
        )
        emit_tool_log(
            logger,
            event="tool_success",
            tool="get_memories",
            correlation_id=correlation_id,
            result_count=response["count"],
        )
        return response
    except Exception as e:
        emit_tool_log(
            logger,
            event="tool_error",
            tool="get_memories",
            correlation_id=correlation_id,
            error=str(e),
            level=logging.ERROR,
        )
        return _error_payload(
            code="memory_list_failed",
            message="Failed to list memories",
            details={"error": str(e)},
            correlation_id=correlation_id,
        )


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
    correlation_id = _new_correlation_id()
    emit_tool_log(logger, event="tool_start", tool="update_memory", correlation_id=correlation_id)
    try:
        response = memory_service.update_memory(
            memory_id=memory_id,
            text=text,
            importance=importance,
            tags=tags,
            user_id=user_id,
        )
        response["message"] = "Memory updated successfully"
        emit_tool_log(
            logger,
            event="tool_success",
            tool="update_memory",
            correlation_id=correlation_id,
            memory_id=response["id"],
        )
        return response
    except Exception as e:
        emit_tool_log(
            logger,
            event="tool_error",
            tool="update_memory",
            correlation_id=correlation_id,
            error=str(e),
            level=logging.ERROR,
        )
        return _error_payload(
            code="memory_update_failed",
            message="Failed to update memory",
            details={"error": str(e)},
            correlation_id=correlation_id,
        )


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
    correlation_id = _new_correlation_id()
    emit_tool_log(logger, event="tool_start", tool="delete_memory", correlation_id=correlation_id)
    try:
        response = memory_service.delete_memory(memory_id=memory_id, user_id=user_id)
        response["message"] = (
            "Memory deleted successfully" if response["success"] else "Memory not found"
        )
        emit_tool_log(
            logger,
            event="tool_success",
            tool="delete_memory",
            correlation_id=correlation_id,
            deleted=response["success"],
        )
        return response
    except Exception as e:
        emit_tool_log(
            logger,
            event="tool_error",
            tool="delete_memory",
            correlation_id=correlation_id,
            error=str(e),
            level=logging.ERROR,
        )
        return _error_payload(
            code="memory_delete_failed",
            message="Failed to delete memory",
            details={"error": str(e)},
            correlation_id=correlation_id,
        )


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
    correlation_id = _new_correlation_id()
    emit_tool_log(
        logger, event="tool_start", tool="get_memory_statistics", correlation_id=correlation_id
    )
    try:
        stats = memory_service.get_memory_statistics(user_id=user_id)
        if project:
            project_memories = memory_service.list_memories(
                user_id=user_id,
                scope=MemoryScope.PROJECT,
                project_id=project,
                limit=1000,
            )
            stats["project_memory_count"] = project_memories["count"]
        emit_tool_log(
            logger,
            event="tool_success",
            tool="get_memory_statistics",
            correlation_id=correlation_id,
        )
        return stats
    except Exception as e:
        emit_tool_log(
            logger,
            event="tool_error",
            tool="get_memory_statistics",
            correlation_id=correlation_id,
            error=str(e),
            level=logging.ERROR,
        )
        return _error_payload(
            code="memory_stats_failed",
            message="Failed to fetch memory statistics",
            details={"error": str(e)},
            correlation_id=correlation_id,
        )


# ============================================================================
# SCOPED CONVENIENCE TOOLS (SERVICE LAYER)
# ============================================================================

@mcp.tool()
def remember_project_memory(
    text: str,
    user_id: str,
    project_id: str,
    memory_type: str = "context",
    importance: Optional[float] = None,
    tags: Optional[list[str]] = None,
    ttl_days: Optional[int] = None,
) -> dict[str, Any]:
    correlation_id = _new_correlation_id()
    emit_tool_log(
        logger, event="tool_start", tool="remember_project_memory", correlation_id=correlation_id
    )
    try:
        response = memory_service.remember_project_memory(
            text=text,
            user_id=user_id,
            project_id=project_id,
            memory_type=memory_type,
            importance=importance,
            tags=tags,
            ttl_days=ttl_days,
        )
        emit_tool_log(
            logger,
            event="tool_success",
            tool="remember_project_memory",
            correlation_id=correlation_id,
            memory_id=response["id"],
        )
        return response
    except Exception as e:
        emit_tool_log(
            logger,
            event="tool_error",
            tool="remember_project_memory",
            correlation_id=correlation_id,
            error=str(e),
            level=logging.ERROR,
        )
        return _error_payload(
            code="remember_project_failed",
            message="Failed to store project memory",
            details={"error": str(e)},
            correlation_id=correlation_id,
        )


@mcp.tool()
def remember_agent_memory(
    text: str,
    user_id: str,
    project_id: str,
    agent_id: str,
    visibility: str = "private",
    memory_type: str = "context",
    importance: Optional[float] = None,
    tags: Optional[list[str]] = None,
) -> dict[str, Any]:
    correlation_id = _new_correlation_id()
    emit_tool_log(
        logger, event="tool_start", tool="remember_agent_memory", correlation_id=correlation_id
    )
    try:
        response = memory_service.remember_agent_memory(
            text=text,
            user_id=user_id,
            project_id=project_id,
            agent_id=agent_id,
            visibility=visibility,
            memory_type=memory_type,
            importance=importance,
            tags=tags,
        )
        emit_tool_log(
            logger,
            event="tool_success",
            tool="remember_agent_memory",
            correlation_id=correlation_id,
            memory_id=response["id"],
        )
        return response
    except Exception as e:
        emit_tool_log(
            logger,
            event="tool_error",
            tool="remember_agent_memory",
            correlation_id=correlation_id,
            error=str(e),
            level=logging.ERROR,
        )
        return _error_payload(
            code="remember_agent_failed",
            message="Failed to store agent memory",
            details={"error": str(e)},
            correlation_id=correlation_id,
        )


@mcp.tool()
def remember_user_preference(
    text: str,
    user_id: str,
    memory_type: str = "preference",
    importance: Optional[float] = None,
    tags: Optional[list[str]] = None,
) -> dict[str, Any]:
    correlation_id = _new_correlation_id()
    emit_tool_log(
        logger, event="tool_start", tool="remember_user_preference", correlation_id=correlation_id
    )
    try:
        response = memory_service.remember_user_preference(
            text=text,
            user_id=user_id,
            memory_type=memory_type,
            importance=importance,
            tags=tags,
        )
        emit_tool_log(
            logger,
            event="tool_success",
            tool="remember_user_preference",
            correlation_id=correlation_id,
            memory_id=response["id"],
        )
        return response
    except Exception as e:
        emit_tool_log(
            logger,
            event="tool_error",
            tool="remember_user_preference",
            correlation_id=correlation_id,
            error=str(e),
            level=logging.ERROR,
        )
        return _error_payload(
            code="remember_preference_failed",
            message="Failed to store user preference",
            details={"error": str(e)},
            correlation_id=correlation_id,
        )


@mcp.tool()
def recall_project_context(
    query: str,
    user_id: str,
    project_id: str,
    k: int = 5,
    min_importance: Optional[float] = None,
    memory_type: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> dict[str, Any]:
    try:
        return memory_service.recall_project_context(
            query=query,
            user_id=user_id,
            project_id=project_id,
            k=k,
            min_importance=min_importance,
            memory_type=memory_type,
            tags=tags,
        )
    except Exception as e:
        return _error_payload(
            code="recall_project_failed",
            message="Failed to recall project context",
            details={"error": str(e)},
        )


@mcp.tool()
def recall_agent_context(
    query: str,
    user_id: str,
    project_id: str,
    agent_id: str,
    k: int = 5,
    min_importance: Optional[float] = None,
    memory_type: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> dict[str, Any]:
    try:
        return memory_service.recall_agent_context(
            query=query,
            user_id=user_id,
            project_id=project_id,
            agent_id=agent_id,
            k=k,
            min_importance=min_importance,
            memory_type=memory_type,
            tags=tags,
        )
    except Exception as e:
        return _error_payload(
            code="recall_agent_failed",
            message="Failed to recall agent context",
            details={"error": str(e)},
        )


@mcp.tool()
def recall_user_preferences(
    query: str,
    user_id: str,
    k: int = 5,
    min_importance: Optional[float] = None,
    tags: Optional[list[str]] = None,
) -> dict[str, Any]:
    try:
        return memory_service.recall_user_preferences(
            query=query,
            user_id=user_id,
            k=k,
            min_importance=min_importance,
            tags=tags,
        )
    except Exception as e:
        return _error_payload(
            code="recall_preferences_failed",
            message="Failed to recall user preferences",
            details={"error": str(e)},
        )


@mcp.tool()
def get_telemetry_metrics() -> dict[str, Any]:
    """Return backend telemetry metrics when supported by HippocampAI."""
    if memory_client and hasattr(memory_client, "get_telemetry_metrics"):
        try:
            return memory_client.get_telemetry_metrics()
        except Exception as e:
            return _error_payload(
                code="telemetry_metrics_failed",
                message="Failed to fetch telemetry metrics",
                details={"error": str(e)},
            )
    return _error_payload(
        code="not_supported",
        message="get_telemetry_metrics is not available on this HippocampAI client",
    )


@mcp.tool()
def get_recent_operations(limit: int = 20) -> dict[str, Any]:
    """Return recent backend operations when supported by HippocampAI."""
    if memory_client and hasattr(memory_client, "get_recent_operations"):
        try:
            return memory_client.get_recent_operations(limit=limit)
        except Exception as e:
            return _error_payload(
                code="recent_operations_failed",
                message="Failed to fetch recent operations",
                details={"error": str(e)},
            )
    return _error_payload(
        code="not_supported",
        message="get_recent_operations is not available on this HippocampAI client",
    )


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
        
        session = memory_client.create_session(
            user_id=user_id,
            title=title,
            metadata=metadata,
        )
        session_id_value = _attr(session, "session_id", "id")
        
        return {
            "session_id": session_id_value,
            "title": session.title if hasattr(session, "title") else None,
            "created_at": _iso_attr(session, "created_at", "createdat"),
            "message": f"Session created with ID: {session_id_value}"
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
        summary = memory_client.summarize_session(
            session_id=session_id,
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
        memories = memory_client.get_session_memories(
            session_id=session_id,
            limit=limit,
        )
        
        return {
            "memories": [
                {
                    "id": m.id,
                    "text": m.text,
                    "type": m.type,
                    "created_at": _iso_attr(m, "created_at", "createdat"),
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
        memories = memory_client.get_agent_memories(
            agent_id=agent_id,
            user_id=user_id,
            limit=limit,
        )
        
        return {
            "memories": [
                {
                    "id": m.id,
                    "text": m.text,
                    "type": m.type,
                    "created_at": _iso_attr(m, "created_at", "createdat"),
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
        facts = memory_client.extract_facts(
            text=text,
            confidence_threshold=confidence_threshold,
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
        clusters = memory_client.cluster_user_memories(
            user_id=user_id,
            max_clusters=max_clusters,
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
    time_window: str = "LAST_DAY",
    project: Optional[str] = None,
) -> dict[str, Any]:
    """
    Get memories from a specific time period.
    
    Args:
        user_id: User identifier
        time_window: Time period - "LAST_HOUR", "LAST_DAY", "LAST_WEEK", "LAST_MONTH", "LAST_YEAR"
        project: Optional project filter
    
    Returns:
        List of memories from the specified time window
    """
    try:
        normalized_window = time_window.upper().replace("-", "_")
        time_range = getattr(TimeRange, normalized_window, TimeRange.LAST_DAY)
        memories = memory_client.get_memories_by_time_range(
            user_id=user_id,
            time_range=time_range,
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
                    "created_at": _iso_attr(m, "created_at", "createdat"),
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
        
        memory = memory_client.schedule_memory(
            text=text,
            user_id=user_id,
            scheduled_for=scheduled_for,
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
    """Server and dependency health check."""
    ollama = _check_tcp_dependency(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), 11434)
    qdrant = _check_tcp_dependency(os.getenv("QDRANT_URL", "http://localhost:6333"), 6333)
    redis = _check_tcp_dependency(os.getenv("REDIS_URL", "redis://localhost:6379"), 6379)
    dependencies = {
        "ollama": ollama,
        "qdrant": qdrant,
        "redis": redis,
    }
    overall = "ok" if all(d["status"] == "ok" for d in dependencies.values()) else "degraded"
    return json.dumps({"status": overall, "dependencies": dependencies})


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
