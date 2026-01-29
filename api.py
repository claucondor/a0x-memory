"""
SimpleMem HTTP API - Multi-tenant memory service

Single shared LanceDB, tenant isolation via agent_id + user_id filtering.
Compatible with Zep-like memory operations.
"""
import os
import uuid
import asyncio
import threading
import queue
import time
import schedule
import psutil
from collections import OrderedDict
from typing import Dict, List, Optional, Any, Callable
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
from main import SimpleMemSystem
from models.memory_entry import MemoryEntry
from models.user_profile import UserProfile
from models.group_memory import GroupProfile, UserInGroupProfile
from database import VectorStore, UserProfileStore
from database.group_profile_store import GroupProfileStore
from utils.embedding import EmbeddingModel
from utils.llm_client import LLMClient


# ============================================================================
# Request/Response Models
# ============================================================================

class DialogueRequest(BaseModel):
    """Single dialogue entry"""
    speaker: str
    content: str
    timestamp: Optional[str] = None
    role: Optional[str] = None  # 'user' or 'assistant' for Zep compatibility


class DialogueBatchRequest(BaseModel):
    """Batch of dialogues"""
    dialogues: List[DialogueRequest]


class MessageRequest(BaseModel):
    """Zep-compatible message format"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MessagesRequest(BaseModel):
    """Batch messages request (Zep-compatible)"""
    messages: List[MessageRequest]


class AskRequest(BaseModel):
    """Question request"""
    question: str
    enable_planning: Optional[bool] = None
    enable_reflection: Optional[bool] = None


class AskResponse(BaseModel):
    """Question response"""
    answer: str
    contexts: Optional[List[str]] = None
    memory_count: int = 0


class MemoryInstanceConfig(BaseModel):
    """Configuration for creating a memory instance"""
    memory_id: Optional[str] = None
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    clear_db: bool = False
    metadata: Optional[Dict[str, Any]] = None


class MemoryInstanceResponse(BaseModel):
    """Response for memory instance creation"""
    memory_id: str
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    created_at: str
    memory_count: int = 0


class MemoryEntryResponse(BaseModel):
    """Single memory entry response"""
    entry_id: str
    lossless_restatement: str
    keywords: List[str]
    timestamp: Optional[str] = None
    location: Optional[str] = None
    persons: List[str] = []
    entities: List[str] = []
    topic: Optional[str] = None


class MemoryResponse(BaseModel):
    """Full memory response (Zep-compatible structure)"""
    memory_id: str
    messages: List[Dict[str, Any]] = []
    summary: Optional[str] = None
    context: Optional[str] = None
    facts: Optional[List[str]] = None
    memory_entries: List[MemoryEntryResponse] = []
    memory_count: int = 0


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    memory_instances: int
    timestamp: str


class UserProfileResponse(BaseModel):
    """User profile response"""
    profile_id: str
    agent_id: str
    universal_user_id: str
    platform_type: str
    username: Optional[str] = None
    summary: str
    interests: List[Dict[str, Any]] = []
    expertise_level: Optional[Dict[str, Any]] = None
    communication_style: Optional[Dict[str, Any]] = None
    total_messages_processed: int = 0
    wallet_address: Optional[str] = None
    basename: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class GroupProfileResponse(BaseModel):
    """Group profile response"""
    profile_id: str
    agent_id: str
    group_id: str
    group_name: Optional[str] = None
    platform: str
    summary: str
    main_topics: List[str] = []
    group_purpose: Optional[str] = None
    tone: str = "casual"
    expertise_level: str = "intermediate"
    total_messages_processed: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class UserInGroupProfileResponse(BaseModel):
    """User in group profile response"""
    profile_id: str
    agent_id: str
    group_id: str
    universal_user_id: str
    username: Optional[str] = None
    summary: str
    role_in_group: str = "regular"
    participation_level: str = "moderate"
    expertise_in_group: str = "intermediate"
    topics_engaged: List[str] = []
    total_messages_in_group: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class AgentResponseRequest(BaseModel):
    """Request to add agent response for immediate context"""
    response: str
    trigger_message: Optional[str] = None  # User message that triggered this response
    trigger_message_id: Optional[str] = None  # ID of the trigger message
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ContextResponse(BaseModel):
    """Full context response with everything"""
    memory_id: str
    recent_messages: List[Dict[str, Any]] = []
    memories: List[MemoryEntryResponse] = []
    agent_responses: List[Dict[str, Any]] = []
    user_profile: Optional[UserProfileResponse] = None
    group_profile: Optional[GroupProfileResponse] = None
    user_in_group_profile: Optional[UserInGroupProfileResponse] = None
    memory_count: int = 0


# ============================================================================
# Unified API Request/Response Models
# ============================================================================

class PlatformIdentity(BaseModel):
    """Platform-specific identity info"""
    platform: str  # telegram, twitter, xmtp, farcaster
    telegramId: Optional[int] = None
    chatId: Optional[str] = None
    groupId: Optional[str] = None
    username: Optional[str] = None
    walletAddress: Optional[str] = None
    basename: Optional[str] = None


class PassiveMemoryRequest(BaseModel):
    """Request for passive memory (background processing)"""
    agent_id: str
    message: str
    platform_identity: PlatformIdentity
    speaker: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ActiveMemoryRequest(BaseModel):
    """Request for active memory (immediate context return)"""
    agent_id: str
    message: str
    platform_identity: PlatformIdentity
    speaker: Optional[str] = None
    return_context: bool = True
    context_limit: int = 10
    metadata: Optional[Dict[str, Any]] = None


class ContextQueryRequest(BaseModel):
    """Request for context retrieval"""
    agent_id: str
    query: Optional[str] = None
    platform_identity: PlatformIdentity
    include_recent: bool = True
    recent_limit: int = 10
    memory_limit: int = 5


class UnifiedContextResponse(BaseModel):
    """Response with full context"""
    success: bool
    recent_messages: List[Dict[str, Any]] = []
    relevant_memories: List[Dict[str, Any]] = []
    user_profile: Optional[Dict[str, Any]] = None
    group_profile: Optional[Dict[str, Any]] = None
    formatted_context: str = ""


# ============================================================================
# Multi-tenant Storage (Single Shared DB)
# ============================================================================

# Shared components (initialized on startup)
shared_embedding_model: Optional[EmbeddingModel] = None
shared_llm_client: Optional[LLMClient] = None
shared_db_path = config.LANCEDB_PATH
shared_table_name = os.environ.get("MEMORY_TABLE_NAME", "memories")

# Tenant metadata (lightweight, no LRU needed)
tenant_metadata: Dict[str, Dict[str, Any]] = {}

# Cache for SimpleMemSystem instances (by agent_id to avoid re-opening GCS tables)
_system_cache: Dict[str, SimpleMemSystem] = {}

# Startup time for uptime tracking
_startup_time: float = 0

# Known agents to pre-warm on startup (eliminates 5s cold start)
KNOWN_AGENTS = [
    "71f6f657-6800-0892-875f-f26e8c213756",  # jessexbt
]

# ============================================================================
# Background Worker (dedicated thread for maintenance tasks)
# ============================================================================

_task_queue: queue.Queue = queue.Queue()
_worker_thread: Optional[threading.Thread] = None
_worker_running: bool = False


def _background_worker():
    """Background worker thread for maintenance tasks."""
    global _worker_running
    print("[Worker] Background worker started")
    while _worker_running:
        try:
            task = _task_queue.get(timeout=1.0)
            if task is None:
                break
            func, args, kwargs = task
            try:
                func(*args, **kwargs)
            except Exception as e:
                print(f"[Worker] Task error: {e}")
            finally:
                _task_queue.task_done()
        except queue.Empty:
            continue
    print("[Worker] Background worker stopped")


def schedule_task(func: Callable, *args, **kwargs):
    """Schedule a task to run in the background worker."""
    _task_queue.put((func, args, kwargs))


# ============================================================================
# Scheduled Maintenance (LanceDB compaction, etc.)
# ============================================================================

def compact_all_tables():
    """Compact LanceDB tables for all cached agents."""
    print("[Maintenance] Starting LanceDB compaction...")
    compacted = 0
    for agent_id, system in list(_system_cache.items()):
        try:
            if hasattr(system, 'unified_store') and hasattr(system.unified_store, 'optimize_tables'):
                system.unified_store.optimize_tables()
                compacted += 1
                print(f"[Maintenance] Compacted tables for {agent_id[:8]}...")
        except Exception as e:
            print(f"[Maintenance] Compaction failed for {agent_id}: {e}")
    print(f"[Maintenance] Compaction complete: {compacted} agents")


def _scheduler_thread():
    """Thread that runs scheduled maintenance tasks."""
    # Schedule compaction at 4am UTC daily (low traffic)
    schedule.every().day.at("04:00").do(lambda: schedule_task(compact_all_tables))
    # Also compact every 6 hours for high-volume scenarios
    schedule.every(6).hours.do(lambda: schedule_task(compact_all_tables))

    print("[Scheduler] Maintenance scheduler started (compaction at 04:00 UTC + every 6h)")
    while _worker_running:
        schedule.run_pending()
        time.sleep(60)
    print("[Scheduler] Maintenance scheduler stopped")


# ============================================================================
# Shared Singletons
# ============================================================================

def get_shared_embedding_model() -> EmbeddingModel:
    """Get or create shared embedding model."""
    global shared_embedding_model
    if shared_embedding_model is None:
        shared_embedding_model = EmbeddingModel()
    return shared_embedding_model


def get_shared_llm_client() -> LLMClient:
    """Get or create shared LLM client (reduces memory, reuses HTTP connections)."""
    global shared_llm_client
    if shared_llm_client is None:
        shared_llm_client = LLMClient()
    return shared_llm_client


def parse_memory_id(memory_id: str) -> tuple:
    """Parse memory_id into (agent_id, user_id).

    Formats supported:
    - "agent_001:user_123" -> ("agent_001", "user_123")
    - "user_123" -> (None, "user_123")
    - "uuid-format" -> (None, "uuid-format")
    """
    if ":" in memory_id:
        parts = memory_id.split(":", 1)
        return (parts[0], parts[1])
    return (None, memory_id)


def get_memory_system(memory_id: str, config: Optional[MemoryInstanceConfig] = None) -> SimpleMemSystem:
    """Get SimpleMem instance for tenant (cached by agent_id to avoid re-opening GCS tables)."""
    agent_id, user_id = parse_memory_id(memory_id)

    # Override with config if provided
    if config:
        if config.metadata and config.metadata.get("agent_id"):
            agent_id = config.metadata["agent_id"]
        if config.user_id:
            user_id = config.user_id

    clear_db = config.clear_db if config else False

    # Use agent_id as cache key (tables are per-agent, user_id only affects filtering)
    cache_key = agent_id or "default"

    # Return cached system if exists and not clearing DB
    if cache_key in _system_cache and not clear_db:
        system = _system_cache[cache_key]
        # Update user_id for this request (affects Firestore context filtering)
        system.user_id = user_id
        return system

    # Create new tenant-scoped system (use shared singletons)
    system = SimpleMemSystem(
        db_path=shared_db_path,
        table_name=shared_table_name,
        clear_db=clear_db,
        agent_id=agent_id,
        user_id=user_id,
        embedding_model=get_shared_embedding_model(),
        llm_client=get_shared_llm_client()
    )

    # Cache it
    _system_cache[cache_key] = system
    print(f"[SimpleMem API] Cached system for agent: {cache_key}")

    # Track metadata
    if memory_id not in tenant_metadata:
        tenant_metadata[memory_id] = {
            "created_at": datetime.utcnow().isoformat(),
            "agent_id": agent_id,
            "user_id": user_id,
            "thread_id": config.thread_id if config else None,
            "metadata": config.metadata if config else {},
            "dialogue_count": 0
        }
        print(f"[SimpleMem API] New tenant: {memory_id} (agent={agent_id}, user={user_id})")

    return system


# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global _worker_running, _worker_thread, _startup_time

    # Startup
    _startup_time = time.time()
    print("=" * 60)
    print("SimpleMem API Starting (Multi-tenant Mode)")
    print(f"Shared DB: {shared_db_path}")
    print(f"Table: {shared_table_name}")
    print("=" * 60)

    # Create data directory
    os.makedirs("./data", exist_ok=True)

    # 1. Pre-initialize shared singletons
    print("[Startup] Initializing shared embedding model...")
    get_shared_embedding_model()
    print("[Startup] Initializing shared LLM client...")
    get_shared_llm_client()

    # 2. Start background worker thread
    _worker_running = True
    _worker_thread = threading.Thread(target=_background_worker, daemon=True)
    _worker_thread.start()

    # 3. Start maintenance scheduler thread
    scheduler_thread = threading.Thread(target=_scheduler_thread, daemon=True)
    scheduler_thread.start()

    # 4. Pre-warm known agents (eliminates 5s cold start for main agents)
    print(f"[Startup] Pre-warming {len(KNOWN_AGENTS)} known agents...")
    for agent_id in KNOWN_AGENTS:
        try:
            get_memory_system(f"{agent_id}:prewarm")
            print(f"  ✓ {agent_id[:12]}...")
        except Exception as e:
            print(f"  ✗ {agent_id[:12]}: {e}")

    print("=" * 60)
    print(f"[Startup] Ready! Pre-warmed {len(_system_cache)} agents")
    print("=" * 60)

    yield

    # Shutdown
    print("SimpleMem API Shutting down...")
    _worker_running = False
    _task_queue.put(None)  # Signal worker to stop
    if _worker_thread:
        _worker_thread.join(timeout=5)
    tenant_metadata.clear()


app = FastAPI(
    title="SimpleMem API",
    description="Efficient Lifelong Memory for LLM Agents - HTTP API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health & Info Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="2.0.0",  # Multi-tenant version
        memory_instances=len(tenant_metadata),
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/metrics")
async def get_metrics():
    """
    Detailed metrics endpoint for monitoring and debugging.
    Returns system state, cache stats, and per-agent info.
    """
    # Process memory usage
    process = psutil.Process()
    memory_info = process.memory_info()

    # Embedding cache stats
    embedding_stats = {}
    if shared_embedding_model:
        try:
            embedding_stats = shared_embedding_model.get_cache_stats()
        except:
            embedding_stats = {"error": "unavailable"}

    # Per-agent stats
    agent_stats = {}
    for agent_id, system in list(_system_cache.items()):
        try:
            memory_count = 0
            if hasattr(system, 'unified_store'):
                # Try to get memory count without expensive queries
                memory_count = getattr(system.unified_store, '_cached_count', 0)
            agent_stats[agent_id[:12] + "..."] = {
                "cached": True,
                "memory_count": memory_count,
            }
        except Exception as e:
            agent_stats[agent_id[:12] + "..."] = {"error": str(e)}

    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - _startup_time, 1),
        "uptime_human": _format_uptime(time.time() - _startup_time),
        "memory": {
            "rss_mb": round(memory_info.rss / 1024 / 1024, 1),
            "vms_mb": round(memory_info.vms / 1024 / 1024, 1),
        },
        "cache": {
            "cached_agents": len(_system_cache),
            "tenant_metadata": len(tenant_metadata),
        },
        "worker": {
            "queue_size": _task_queue.qsize(),
            "running": _worker_running,
        },
        "embedding": embedding_stats,
        "agents": agent_stats,
        "config": {
            "embedding_provider": config.EMBEDDING_PROVIDER,
            "embedding_dimension": config.EMBEDDING_DIMENSION,
            "lancedb_path": shared_db_path[:50] + "..." if len(shared_db_path) > 50 else shared_db_path,
        }
    }


def _format_uptime(seconds: float) -> str:
    """Format uptime in human readable format."""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


@app.post("/maintenance/compact")
async def trigger_compaction():
    """Manually trigger LanceDB table compaction."""
    schedule_task(compact_all_tables)
    return {"status": "scheduled", "message": "Compaction task queued"}


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "SimpleMem API",
        "version": "1.0.0",
        "description": "Efficient Lifelong Memory for LLM Agents",
        "docs": "/docs",
        "health": "/health"
    }


# ============================================================================
# Memory Instance Management (Zep-compatible: Thread/User concept)
# ============================================================================

@app.post("/memories", response_model=MemoryInstanceResponse)
async def create_memory_instance(config: Optional[MemoryInstanceConfig] = None):
    """
    Create/register a tenant context (equivalent to Zep thread creation).
    Uses single shared DB with tenant filtering.
    """
    if config and config.memory_id:
        memory_id = config.memory_id
    elif config and config.thread_id:
        memory_id = config.thread_id
    else:
        memory_id = str(uuid.uuid4())

    # Register tenant (creates metadata entry)
    system = get_memory_system(memory_id, config)
    meta = tenant_metadata.get(memory_id, {})

    return MemoryInstanceResponse(
        memory_id=memory_id,
        user_id=meta.get("user_id"),
        thread_id=meta.get("thread_id") or memory_id,
        created_at=meta.get("created_at", datetime.utcnow().isoformat()),
        memory_count=system.vector_store.count_entries()
    )


@app.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(memory_id: str, last_n: Optional[int] = None):
    """
    Get memory contents for tenant (equivalent to Zep getMemory).
    Returns all memory entries filtered by tenant.
    """
    system = get_memory_system(memory_id)
    meta = tenant_metadata.get(memory_id, {})

    # Get all memory entries (filtered by tenant)
    entries = system.get_all_memories()

    # Build response
    memory_entries = [
        MemoryEntryResponse(
            entry_id=entry.entry_id,
            lossless_restatement=entry.lossless_restatement,
            keywords=entry.keywords,
            timestamp=entry.timestamp,
            location=entry.location,
            persons=entry.persons,
            entities=entry.entities,
            topic=entry.topic
        )
        for entry in entries
    ]

    # Build facts from memory entries
    facts = [entry.lossless_restatement for entry in entries]

    # Build context summary
    context = None
    if entries:
        recent = entries[-5:] if last_n is None else entries[-last_n:]
        context = "\n".join([e.lossless_restatement for e in recent])

    return MemoryResponse(
        memory_id=memory_id,
        messages=[],
        summary=context,
        context=context,
        facts=facts,
        memory_entries=memory_entries,
        memory_count=len(entries)
    )


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete tenant's memory entries (not the whole DB)."""
    system = get_memory_system(memory_id)

    # Clear only this tenant's entries
    try:
        system.vector_store.clear()
        tenant_metadata.pop(memory_id, None)
    except Exception as e:
        print(f"[SimpleMem API] Warning: Could not clear tenant data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"success": True, "memory_id": memory_id}


# ============================================================================
# Dialogue/Message Operations
# ============================================================================

@app.post("/memories/{memory_id}/dialogues")
async def add_dialogues(
    memory_id: str,
    request: DialogueBatchRequest,
    background_tasks: BackgroundTasks
):
    """
    Add dialogues to tenant's memory (triggers memory building in background).
    """
    system = get_memory_system(memory_id)

    # Add each dialogue
    for dialogue in request.dialogues:
        speaker = dialogue.speaker or dialogue.role or "user"
        system.add_dialogue(
            speaker=speaker,
            content=dialogue.content,
            timestamp=dialogue.timestamp
        )

    # Update dialogue count
    if memory_id in tenant_metadata:
        tenant_metadata[memory_id]["dialogue_count"] = (
            tenant_metadata[memory_id].get("dialogue_count", 0) + len(request.dialogues)
        )

    # Finalize in background (process buffer)
    background_tasks.add_task(system.finalize)

    return {
        "success": True,
        "memory_id": memory_id,
        "dialogues_added": len(request.dialogues)
    }


@app.post("/memories/{memory_id}/messages")
async def add_messages(
    memory_id: str,
    request: MessagesRequest,
    background_tasks: BackgroundTasks
):
    """
    Add messages to tenant's memory (Zep-compatible format).
    """
    system = get_memory_system(memory_id)

    # Convert messages to dialogues
    for msg in request.messages:
        speaker = msg.name or msg.role
        system.add_dialogue(
            speaker=speaker,
            content=msg.content,
            timestamp=datetime.utcnow().isoformat()
        )

    # Update count
    if memory_id in tenant_metadata:
        tenant_metadata[memory_id]["dialogue_count"] = (
            tenant_metadata[memory_id].get("dialogue_count", 0) + len(request.messages)
        )

    # Finalize in background
    background_tasks.add_task(system.finalize)

    return {
        "success": True,
        "memory_id": memory_id,
        "messages_added": len(request.messages)
    }


# ============================================================================
# Query Operations
# ============================================================================

@app.post("/memories/{memory_id}/ask", response_model=AskResponse)
async def ask_memory(memory_id: str, request: AskRequest):
    """
    Ask a question against tenant's memory.
    Uses hybrid retrieval and answer generation.
    """
    system = get_memory_system(memory_id)
    memory_count = system.vector_store.count_entries()

    if memory_count == 0:
        return AskResponse(
            answer="No memories available to answer this question.",
            contexts=[],
            memory_count=0
        )

    # Use hybrid retriever (filtered by tenant)
    contexts = await asyncio.to_thread(
        system.hybrid_retriever.retrieve,
        request.question,
        enable_reflection=request.enable_reflection
    )

    # Generate answer
    answer = await asyncio.to_thread(
        system.answer_generator.generate_answer,
        request.question,
        contexts
    )

    # Convert contexts to strings for response model
    context_strings = [str(c) for c in contexts] if contexts else []

    return AskResponse(
        answer=answer,
        contexts=context_strings,
        memory_count=memory_count
    )


@app.post("/memories/{memory_id}/search")
async def search_memory(
    memory_id: str,
    query: str,
    limit: Optional[int] = 10,
    enable_reflection: bool = True
):
    """
    Search tenant's memory with hybrid retrieval.
    """
    system = get_memory_system(memory_id)

    # Use hybrid retriever (filtered by tenant)
    contexts = await asyncio.to_thread(
        system.hybrid_retriever.retrieve,
        query,
        enable_reflection=enable_reflection
    )

    # Limit results
    if limit and contexts:
        contexts = contexts[:limit]

    return {
        "query": query,
        "results": contexts or [],
        "count": len(contexts) if contexts else 0
    }


# ============================================================================
# Debug Endpoints
# ============================================================================

@app.get("/memories/{memory_id}/entries", response_model=List[MemoryEntryResponse])
async def list_memory_entries(memory_id: str):
    """List all memory entries for tenant (for debugging)"""
    system = get_memory_system(memory_id)
    entries = system.get_all_memories()

    return [
        MemoryEntryResponse(
            entry_id=entry.entry_id,
            lossless_restatement=entry.lossless_restatement,
            keywords=entry.keywords,
            timestamp=entry.timestamp,
            location=entry.location,
            persons=entry.persons,
            entities=entry.entities,
            topic=entry.topic
        )
        for entry in entries
    ]


@app.get("/tenants")
async def list_tenants():
    """List all registered tenants (for debugging)"""
    return {
        "tenants": list(tenant_metadata.keys()),
        "count": len(tenant_metadata),
        "metadata": {
            k: {
                "created_at": v.get("created_at"),
                "agent_id": v.get("agent_id"),
                "user_id": v.get("user_id"),
                "dialogue_count": v.get("dialogue_count", 0)
            }
            for k, v in tenant_metadata.items()
        }
    }


# Backwards compatibility alias
@app.get("/instances")
async def list_instances():
    """Alias for /tenants (backwards compatibility)"""
    return await list_tenants()


# ============================================================================
# Profile Endpoints
# ============================================================================

def get_profile_stores(memory_id: str):
    """Get profile stores for tenant."""
    agent_id, user_id = parse_memory_id(memory_id)

    user_store = UserProfileStore(
        db_path=shared_db_path,
        embedding_model=get_shared_embedding_model(),
        agent_id=agent_id
    )

    group_store = GroupProfileStore(
        db_path=shared_db_path,
        embedding_model=get_shared_embedding_model(),
        agent_id=agent_id
    )

    return user_store, group_store, agent_id


@app.get("/v1/profiles/user/{universal_user_id}", response_model=UserProfileResponse)
async def get_user_profile(universal_user_id: str, agent_id: Optional[str] = None):
    """Get user profile by universal_user_id (e.g., telegram:123456789)."""
    memory_id = agent_id or "default"
    user_store, _, _ = get_profile_stores(memory_id)

    profile = user_store.get_profile_by_universal_id(universal_user_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"User profile not found: {universal_user_id}")

    return UserProfileResponse(
        profile_id=profile.profile_id,
        agent_id=profile.agent_id,
        universal_user_id=profile.universal_user_id,
        platform_type=profile.platform_type,
        username=profile.username,
        summary=profile.summary,
        interests=[{"keyword": i.keyword, "score": i.score} for i in profile.interests],
        expertise_level={"value": profile.expertise_level.value, "confidence": profile.expertise_level.confidence} if profile.expertise_level else None,
        communication_style={"value": profile.communication_style.value, "confidence": profile.communication_style.confidence} if profile.communication_style else None,
        total_messages_processed=profile.total_messages_processed,
        wallet_address=profile.wallet_address,
        basename=profile.basename,
        created_at=profile.created_at,
        updated_at=profile.updated_at
    )


@app.get("/v1/profiles/group/{group_id}", response_model=GroupProfileResponse)
async def get_group_profile(group_id: str, agent_id: Optional[str] = None):
    """Get group profile by group_id."""
    memory_id = agent_id or "default"
    _, group_store, _ = get_profile_stores(memory_id)

    profile = group_store.get_group_profile(group_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Group profile not found: {group_id}")

    return GroupProfileResponse(
        profile_id=profile.profile_id,
        agent_id=profile.agent_id,
        group_id=profile.group_id,
        group_name=profile.group_name,
        platform=profile.platform,
        summary=profile.summary,
        main_topics=profile.main_topics,
        group_purpose=profile.group_purpose,
        tone=profile.tone.value if hasattr(profile.tone, 'value') else profile.tone,
        expertise_level=profile.expertise_level,
        total_messages_processed=profile.total_messages_processed,
        created_at=profile.created_at,
        updated_at=profile.updated_at
    )


@app.get("/v1/profiles/user/{universal_user_id}/group/{group_id}", response_model=UserInGroupProfileResponse)
async def get_user_in_group_profile(universal_user_id: str, group_id: str, agent_id: Optional[str] = None):
    """Get user-in-group profile."""
    memory_id = agent_id or "default"
    _, group_store, _ = get_profile_stores(memory_id)

    profile = group_store.get_user_in_group_profile(group_id, universal_user_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"User-in-group profile not found: {universal_user_id} in {group_id}")

    return UserInGroupProfileResponse(
        profile_id=profile.profile_id,
        agent_id=profile.agent_id,
        group_id=profile.group_id,
        universal_user_id=profile.universal_user_id,
        username=profile.username,
        summary=profile.summary,
        role_in_group=profile.role_in_group,
        participation_level=profile.participation_level,
        expertise_in_group=profile.expertise_in_group,
        topics_engaged=profile.topics_engaged,
        total_messages_in_group=profile.total_messages_in_group,
        created_at=profile.created_at,
        updated_at=profile.updated_at
    )


@app.get("/v1/profiles/group/{group_id}/members", response_model=List[UserInGroupProfileResponse])
async def get_group_members(group_id: str, agent_id: Optional[str] = None):
    """Get all user profiles for a group."""
    memory_id = agent_id or "default"
    _, group_store, _ = get_profile_stores(memory_id)

    profiles = group_store.get_users_in_group(group_id)

    return [
        UserInGroupProfileResponse(
            profile_id=p.profile_id,
            agent_id=p.agent_id,
            group_id=p.group_id,
            universal_user_id=p.universal_user_id,
            username=p.username,
            summary=p.summary,
            role_in_group=p.role_in_group,
            participation_level=p.participation_level,
            expertise_in_group=p.expertise_in_group,
            topics_engaged=p.topics_engaged,
            total_messages_in_group=p.total_messages_in_group,
            created_at=p.created_at,
            updated_at=p.updated_at
        )
        for p in profiles
    ]


# ============================================================================
# Context Endpoint (Everything in One Call)
# ============================================================================

@app.get("/v1/memory/context/{memory_id}", response_model=ContextResponse)
async def get_full_context(
    memory_id: str,
    limit: int = 10,
    include_profiles: bool = True
):
    """
    Get full context for a memory_id including:
    - Recent messages from Firestore sliding window
    - Consolidated memories from LanceDB
    - Agent responses
    - User, group, and user-in-group profiles (if available)
    """
    system = get_memory_system(memory_id)
    agent_id, user_id = parse_memory_id(memory_id)

    # Get memories
    entries = system.get_all_memories()
    memory_entries = [
        MemoryEntryResponse(
            entry_id=entry.entry_id,
            lossless_restatement=entry.lossless_restatement,
            keywords=entry.keywords,
            timestamp=entry.timestamp,
            location=entry.location,
            persons=entry.persons,
            entities=entry.entities,
            topic=entry.topic
        )
        for entry in entries[-limit:]
    ]

    # Get recent messages and agent responses from Firestore
    recent_messages = []
    agent_responses = []
    try:
        firestore_data = system.get_firestore_context()
        recent_messages = firestore_data.get("raw_messages", [])
        agent_responses = firestore_data.get("agent_responses", [])
    except Exception as e:
        print(f"[API] Warning: Could not get Firestore context: {e}")

    # Get profiles if requested
    user_profile = None
    group_profile = None
    user_in_group_profile = None

    if include_profiles and agent_id:
        try:
            user_store, group_store, _ = get_profile_stores(memory_id)

            # Try to get user profile
            if user_id:
                # Detect platform from user_id format
                if user_id.startswith("telegram:"):
                    universal_user_id = user_id
                elif ":" in user_id:
                    universal_user_id = user_id
                else:
                    # Default to telegram for backwards compatibility
                    universal_user_id = f"telegram:{user_id}"

                profile = user_store.get_profile_by_universal_id(universal_user_id)
                if profile:
                    user_profile = UserProfileResponse(
                        profile_id=profile.profile_id,
                        agent_id=profile.agent_id,
                        universal_user_id=profile.universal_user_id,
                        platform_type=profile.platform_type,
                        username=profile.username,
                        summary=profile.summary,
                        interests=[{"keyword": i.keyword, "score": i.score} for i in profile.interests],
                        expertise_level={"value": profile.expertise_level.value, "confidence": profile.expertise_level.confidence} if profile.expertise_level else None,
                        communication_style={"value": profile.communication_style.value, "confidence": profile.communication_style.confidence} if profile.communication_style else None,
                        total_messages_processed=profile.total_messages_processed,
                        wallet_address=profile.wallet_address,
                        basename=profile.basename,
                        created_at=profile.created_at,
                        updated_at=profile.updated_at
                    )

            # Try to get group profile if memory_id contains group info
            group_id = None
            meta = tenant_metadata.get(memory_id, {})
            if meta.get("metadata", {}).get("group_id"):
                group_id = meta["metadata"]["group_id"]
            elif user_id and user_id.startswith("telegram_-"):
                # Group ID format
                group_id = user_id

            if group_id:
                gp = group_store.get_group_profile(group_id)
                if gp:
                    group_profile = GroupProfileResponse(
                        profile_id=gp.profile_id,
                        agent_id=gp.agent_id,
                        group_id=gp.group_id,
                        group_name=gp.group_name,
                        platform=gp.platform,
                        summary=gp.summary,
                        main_topics=gp.main_topics,
                        group_purpose=gp.group_purpose,
                        tone=gp.tone.value if hasattr(gp.tone, 'value') else gp.tone,
                        expertise_level=gp.expertise_level,
                        total_messages_processed=gp.total_messages_processed,
                        created_at=gp.created_at,
                        updated_at=gp.updated_at
                    )

                # Try user-in-group profile
                if user_id and universal_user_id:
                    uigp = group_store.get_user_in_group_profile(group_id, universal_user_id)
                    if uigp:
                        user_in_group_profile = UserInGroupProfileResponse(
                            profile_id=uigp.profile_id,
                            agent_id=uigp.agent_id,
                            group_id=uigp.group_id,
                            universal_user_id=uigp.universal_user_id,
                            username=uigp.username,
                            summary=uigp.summary,
                            role_in_group=uigp.role_in_group,
                            participation_level=uigp.participation_level,
                            expertise_in_group=uigp.expertise_in_group,
                            topics_engaged=uigp.topics_engaged,
                            total_messages_in_group=uigp.total_messages_in_group,
                            created_at=uigp.created_at,
                            updated_at=uigp.updated_at
                        )
        except Exception as e:
            print(f"[API] Warning: Could not get profiles: {e}")

    return ContextResponse(
        memory_id=memory_id,
        recent_messages=recent_messages,
        memories=memory_entries,
        agent_responses=agent_responses,
        user_profile=user_profile,
        group_profile=group_profile,
        user_in_group_profile=user_in_group_profile,
        memory_count=len(entries)
    )


@app.post("/v1/memory/{memory_id}/add-response")
async def add_agent_response(memory_id: str, request: AgentResponseRequest):
    """
    Add agent response to Firestore sliding window for immediate context.
    """
    system = get_memory_system(memory_id)

    try:
        system.add_agent_response_to_window(
            response=request.response,
            trigger_message=request.trigger_message,
            trigger_message_id=request.trigger_message_id,
            timestamp=request.timestamp,
            metadata=request.metadata
        )
        return {"success": True, "memory_id": memory_id}
    except Exception as e:
        print(f"[API] Error adding agent response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Unified Memory Endpoints (Main API)
# ============================================================================

def _extract_identity_info(platform_identity: PlatformIdentity) -> Dict[str, Any]:
    """Extract group_id, user_id, etc from platform_identity."""
    platform = platform_identity.platform

    # Determine if group or DM
    chat_id = platform_identity.chatId or platform_identity.groupId
    is_group = False
    group_id = None

    if chat_id:
        # Negative chatId = group in Telegram
        if str(chat_id).startswith('-') or str(chat_id).startswith('telegram_-'):
            is_group = True
            group_id = chat_id if str(chat_id).startswith('telegram_') else f"telegram_{chat_id}"

    # Build user_id
    user_id = None
    if platform_identity.telegramId:
        user_id = f"{platform}:{platform_identity.telegramId}"
    elif platform_identity.walletAddress:
        user_id = f"{platform}:{platform_identity.walletAddress}"
    elif platform_identity.username:
        user_id = f"{platform}:{platform_identity.username}"

    return {
        "platform": platform,
        "is_group": is_group,
        "group_id": group_id,
        "user_id": user_id,
        "username": platform_identity.username,
        "wallet_address": platform_identity.walletAddress,
        "basename": platform_identity.basename
    }


@app.post("/v1/memory/passive")
async def add_passive_memory(request: PassiveMemoryRequest, background_tasks: BackgroundTasks):
    """
    Add message for background processing (fire and forget).

    Used for messages that don't need immediate context response.
    - Adds to Firestore window (fast, <100ms)
    - Checks batch threshold
    - If threshold met, schedules processing in background (non-blocking)
    - Returns immediately
    """
    identity = _extract_identity_info(request.platform_identity)

    # Build memory_id - always include colon so parse_memory_id extracts agent_id correctly
    memory_id = f"{request.agent_id}:{identity['user_id'] or ''}"
    system = get_memory_system(memory_id)

    # Step 1: Add to Firestore only (fast, no LLM processing)
    result = system.add_dialogue(
        speaker=request.speaker or identity['username'] or "user",
        content=request.message,
        platform=identity['platform'],
        group_id=identity['group_id'],
        user_id=identity['user_id'],
        username=identity['username'],
        add_to_firestore=True,
        use_stateless_processing=False  # Don't process synchronously
    )

    # Step 2: Check if we should trigger background processing
    effective_group_id = identity['group_id'] or f"dm_{identity['user_id']}"
    should_process = False

    if result.get("added") and system.firestore:
        try:
            # Check threshold without processing
            from core.adaptive_threshold import AdaptiveThresholdManager
            threshold_manager = AdaptiveThresholdManager(
                firestore_client=system.firestore.db
            )
            adaptive_threshold = threshold_manager.get_threshold(request.agent_id, effective_group_id)

            # Count unprocessed
            unprocessed = system.firestore.get_unprocessed(
                agent_id=request.agent_id,
                group_id=effective_group_id,
                min_count=adaptive_threshold
            )

            if unprocessed:
                should_process_now, _ = threshold_manager.should_process(
                    request.agent_id,
                    effective_group_id,
                    len(unprocessed)
                )
                should_process = should_process_now
        except Exception as e:
            print(f"[Passive] Warning checking threshold: {e}")

    # Step 3: Schedule background processing if needed (non-blocking)
    if should_process:
        def process_batch():
            try:
                processing_result = system._process_unprocessed_messages(
                    effective_group_id=effective_group_id,
                    original_group_id=identity['group_id'],
                    platform=identity['platform']
                )
                print(f"[Background] Processed batch for {effective_group_id}: {processing_result.get('memories_created', 0)} memories")
            except Exception as e:
                print(f"[Background] Error processing batch: {e}")

        schedule_task(process_batch)
        print(f"[Passive] Scheduled background processing for {effective_group_id}")

    return {
        "success": result.get("added", False),
        "is_group": identity['is_group'],
        "group_id": identity['group_id'],
        "user_id": identity['user_id'],
        "processing_scheduled": should_process,
        "processed": False,  # Always False now - processing is async
        "memories_created": 0,  # Will be created in background
        "is_spam": result.get("is_spam", False),
        "spam_score": result.get("spam_score", 0.0)
    }


@app.post("/v1/memory/active")
async def add_active_memory(request: ActiveMemoryRequest):
    """
    Add message AND return context immediately.

    Used when agent needs to respond - adds message and returns relevant context.
    - Adds to Firestore window (fast)
    - Returns context for response generation
    - Schedules batch processing in background if threshold reached
    """
    identity = _extract_identity_info(request.platform_identity)

    # Build memory_id - always include colon so parse_memory_id extracts agent_id correctly
    memory_id = f"{request.agent_id}:{identity['user_id'] or ''}"
    system = get_memory_system(memory_id)

    # Step 1: Add to Firestore only (fast, no LLM processing)
    result = system.add_dialogue(
        speaker=request.speaker or identity['username'] or "user",
        content=request.message,
        platform=identity['platform'],
        group_id=identity['group_id'],
        user_id=identity['user_id'],
        username=identity['username'],
        add_to_firestore=True,
        use_stateless_processing=False  # Don't process synchronously
    )

    effective_group_id = identity['group_id'] or f"dm_{identity['user_id']}"

    response = {
        "success": result.get("added", False),
        "added": result.get("added", False),
        "is_group": identity['is_group'],
        "group_id": identity['group_id'],
        "user_id": identity['user_id'],
        "processing_scheduled": False,
        "context": None
    }

    # Get context if requested (fast - just Firestore read)
    if request.return_context:
        try:
            context_data = system.get_firestore_context()
            response["context"] = {
                "recent_messages": context_data.get("raw_messages", [])[-request.context_limit:],
                "agent_responses": context_data.get("agent_responses", [])
            }
        except Exception as e:
            print(f"[API] Warning: Could not get context: {e}")

    # Step 2: Check if we should trigger background processing
    if result.get("added") and system.firestore:
        try:
            from core.adaptive_threshold import AdaptiveThresholdManager
            threshold_manager = AdaptiveThresholdManager(
                firestore_client=system.firestore.db
            )
            adaptive_threshold = threshold_manager.get_threshold(request.agent_id, effective_group_id)

            unprocessed = system.firestore.get_unprocessed(
                agent_id=request.agent_id,
                group_id=effective_group_id,
                min_count=adaptive_threshold
            )

            if unprocessed:
                should_process_now, _ = threshold_manager.should_process(
                    request.agent_id,
                    effective_group_id,
                    len(unprocessed)
                )
                if should_process_now:
                    def process_batch():
                        try:
                            processing_result = system._process_unprocessed_messages(
                                effective_group_id=effective_group_id,
                                original_group_id=identity['group_id'],
                                platform=identity['platform']
                            )
                            print(f"[Background] Processed batch for {effective_group_id}: {processing_result.get('memories_created', 0)} memories")
                        except Exception as e:
                            print(f"[Background] Error processing batch: {e}")

                    schedule_task(process_batch)
                    response["processing_scheduled"] = True
                    print(f"[Active] Scheduled background processing for {effective_group_id}")
        except Exception as e:
            print(f"[Active] Warning checking threshold: {e}")

    return response


@app.post("/v1/memory/context")
async def get_memory_context(request: ContextQueryRequest):
    """
    Get context for query without adding a message.

    Used for context retrieval only (e.g., for RAG).
    """
    identity = _extract_identity_info(request.platform_identity)

    # Build memory_id - always include colon so parse_memory_id extracts agent_id correctly
    memory_id = f"{request.agent_id}:{identity['user_id'] or ''}"
    system = get_memory_system(memory_id)

    response = {
        "success": True,
        "recent_messages": [],
        "relevant_memories": [],
        "user_profile": None,
        "group_profile": None,
        "formatted_context": ""
    }

    # Get recent messages from Firestore
    if request.include_recent:
        try:
            context_data = system.get_firestore_context()
            response["recent_messages"] = context_data.get("raw_messages", [])[-request.recent_limit:]
        except Exception as e:
            print(f"[API] Warning: Could not get recent messages: {e}")

    # Search memories if query provided
    if request.query:
        try:
            # Use retrieve_for_context for groups (includes group_memories, user_memories tables)
            if identity['is_group'] and identity['group_id']:
                context = {
                    'group_id': identity['group_id'],
                    'user_id': identity['user_id'],
                    'platform': identity['platform']
                }
                results = await asyncio.to_thread(
                    system.hybrid_retriever.retrieve_for_context,
                    request.query,
                    context,
                    limit_per_table=request.memory_limit
                )
                # Combine all memory types from the results
                all_memories = []
                all_memories.extend(results.get("individual_memories", []))
                all_memories.extend(results.get("group_memories", []))
                all_memories.extend(results.get("user_memories", []))
                all_memories.extend(results.get("interaction_memories", []))
                all_memories.extend(results.get("cross_group_memories", []))
                response["relevant_memories"] = all_memories[:request.memory_limit]
                response["group_profile"] = results.get("group_context")
                response["user_profile"] = results.get("relevant_profiles")
            else:
                # DM context - use basic retrieve
                memories = await asyncio.to_thread(
                    system.hybrid_retriever.retrieve,
                    request.query,
                    enable_reflection=True
                )
                response["relevant_memories"] = memories[:request.memory_limit] if memories else []
        except Exception as e:
            print(f"[API] Warning: Could not search memories: {e}")

    # Build formatted context
    context_parts = []
    if response["recent_messages"]:
        context_parts.append("Recent messages:")
        for msg in response["recent_messages"]:
            username = msg.get('username', 'unknown')
            content = msg.get('content', '')
            context_parts.append(f"  {username}: {content}")

    if response["relevant_memories"]:
        context_parts.append("\nRelevant memories:")
        for mem in response["relevant_memories"]:
            context_parts.append(f"  - {mem}")

    response["formatted_context"] = "\n".join(context_parts)

    return response


@app.post("/v1/memory/process-pending")
async def process_pending_messages(
    agent_id: str,
    group_id: Optional[str] = None
):
    """
    Manually trigger processing of pending messages.

    Useful for testing or forcing batch processing.
    """
    # Include colon so parse_memory_id extracts agent_id correctly
    memory_id = f"{agent_id}:"
    system = get_memory_system(memory_id)

    effective_group_id = group_id or f"dm_{agent_id}"

    # Force processing by calling internal method
    try:
        result = system._process_unprocessed_messages(
            effective_group_id=effective_group_id,
            original_group_id=group_id,
            platform="telegram"
        )
        return {
            "success": True,
            "processed": result.get("processed", False),
            "memories_created": result.get("memories_created", 0)
        }
    except Exception as e:
        import traceback
        print(f"[API] Error processing pending: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/v1/memory/stats/{agent_id}")
async def get_memory_stats(agent_id: str):
    """
    Get memory statistics for an agent.
    """
    memory_id = f"{agent_id}:"  # Include colon for proper agent_id parsing
    system = get_memory_system(memory_id)

    stats = {
        "agent_id": agent_id,
        "memory_count": 0,
        "user_profile_count": 0,
        "group_profile_count": 0,
        "memory_breakdown": {}
    }

    try:
        # Get stats from all tables
        if hasattr(system.vector_store, 'get_stats'):
            memory_stats = system.vector_store.get_stats()
            stats["memory_breakdown"] = memory_stats
            # Sum all memory tables for total count (including agent_responses)
            stats["memory_count"] = sum([
                memory_stats.get("dm_memories", 0),
                memory_stats.get("group_memories", 0),
                memory_stats.get("user_memories", 0),
                memory_stats.get("interaction_memories", 0),
                memory_stats.get("cross_group_memories", 0),
                memory_stats.get("agent_responses", 0)
            ])
        else:
            stats["memory_count"] = system.vector_store.count_entries()
    except:
        pass

    try:
        user_store, group_store, _ = get_profile_stores(memory_id)
        stats["user_profile_count"] = user_store.count_profiles()
    except:
        pass

    try:
        _, group_store, _ = get_profile_stores(memory_id)
        # Count group profiles for this agent
        results = group_store.group_profiles_table.search().where(
            f"agent_id = '{agent_id}'", prefilter=True
        ).limit(1000).to_list()
        stats["group_profile_count"] = len(results)
    except Exception as e:
        print(f"[Stats] Error counting group profiles: {e}")

    return stats


@app.delete("/v1/memory/reset/{agent_id}")
async def reset_agent_data(agent_id: str, confirm: bool = False):
    """
    Delete ALL data for an agent (Firestore + LanceDB).

    USE WITH CAUTION - this is irreversible!

    Args:
        agent_id: Agent ID to reset
        confirm: Must be True to actually delete

    Returns:
        Summary of deleted data
    """
    if not confirm:
        return {
            "error": "Must pass confirm=true to delete data",
            "warning": "This will delete ALL data for this agent from Firestore and LanceDB",
            "usage": f"DELETE /v1/memory/reset/{agent_id}?confirm=true"
        }

    result = {
        "agent_id": agent_id,
        "firestore_deleted": 0,
        "lancedb_deleted": {}
    }

    memory_id = f"{agent_id}:"

    # 1. Delete Firestore data
    try:
        from services.firestore_window import get_firestore_store
        firestore = get_firestore_store()
        if firestore and firestore._enabled:
            # Get all groups for this agent
            db = firestore.db
            agent_ref = db.collection(f"{config.FIRESTORE_COLLECTION_PREFIX}").document(agent_id)
            groups_ref = agent_ref.collection("groups")

            # Delete all documents in all group subcollections
            groups = groups_ref.stream()
            for group_doc in groups:
                group_id = group_doc.id
                messages_ref = groups_ref.document(group_id).collection("recent_messages")

                # Delete all messages in batch
                batch = db.batch()
                count = 0
                for msg_doc in messages_ref.stream():
                    batch.delete(msg_doc.reference)
                    count += 1
                    if count % 500 == 0:  # Firestore batch limit
                        batch.commit()
                        batch = db.batch()

                if count % 500 != 0:
                    batch.commit()

                result["firestore_deleted"] += count

                # Delete the group document itself
                groups_ref.document(group_id).delete()

            print(f"[Reset] Deleted {result['firestore_deleted']} Firestore messages for {agent_id}")
    except Exception as e:
        print(f"[Reset] Firestore error: {e}")
        result["firestore_error"] = str(e)

    # 2. Delete LanceDB data
    try:
        system = get_memory_system(memory_id)

        # Get current counts before deletion
        if hasattr(system.vector_store, 'get_stats'):
            before_stats = system.vector_store.get_stats()
        else:
            before_stats = {}

        # Delete from each table by dropping and recreating
        tables_to_clear = [
            ("dm_memories", system.vector_store.dm_memories if hasattr(system.vector_store, 'dm_memories') else None),
            ("group_memories", system.vector_store.group_memories if hasattr(system.vector_store, 'group_memories') else None),
            ("user_memories", system.vector_store.user_memories if hasattr(system.vector_store, 'user_memories') else None),
            ("interaction_memories", system.vector_store.interaction_memories if hasattr(system.vector_store, 'interaction_memories') else None),
            ("cross_group_memories", system.vector_store.cross_group_memories if hasattr(system.vector_store, 'cross_group_memories') else None),
            ("agent_responses", system.vector_store.agent_responses if hasattr(system.vector_store, 'agent_responses') else None),
            ("conversation_summaries", system.vector_store.conversation_summaries if hasattr(system.vector_store, 'conversation_summaries') else None),
        ]

        for table_name, table_obj in tables_to_clear:
            if table_obj and hasattr(table_obj, 'table'):
                try:
                    count_before = table_obj.table.count_rows()
                    if count_before > 0:
                        # Delete all rows for this agent
                        table_obj.table.delete(f"agent_id = '{agent_id}'")
                        result["lancedb_deleted"][table_name] = count_before
                except Exception as e:
                    print(f"[Reset] Error clearing {table_name}: {e}")

        # Clear from cache so next request creates fresh system
        cache_key = agent_id or "default"
        if cache_key in _system_cache:
            del _system_cache[cache_key]

        print(f"[Reset] Deleted LanceDB data for {agent_id}: {result['lancedb_deleted']}")
    except Exception as e:
        print(f"[Reset] LanceDB error: {e}")
        result["lancedb_error"] = str(e)

    result["success"] = True
    return result


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"Starting SimpleMem API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
