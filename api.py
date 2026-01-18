"""
SimpleMem HTTP API - Multi-tenant memory service

Single shared LanceDB, tenant isolation via agent_id + user_id filtering.
Compatible with Zep-like memory operations.
"""
import os
import uuid
import asyncio
import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from main import SimpleMemSystem
from models.memory_entry import MemoryEntry
from database.vector_store import VectorStore
from utils.embedding import EmbeddingModel


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


# ============================================================================
# Multi-tenant Storage (Single Shared DB)
# ============================================================================

# Shared components (initialized on startup)
shared_embedding_model: Optional[EmbeddingModel] = None
shared_db_path = os.environ.get("LANCEDB_PATH", "./data/lancedb_shared")
shared_table_name = os.environ.get("MEMORY_TABLE_NAME", "memories")

# Tenant metadata (lightweight, no LRU needed)
tenant_metadata: Dict[str, Dict[str, Any]] = {}


def get_shared_embedding_model() -> EmbeddingModel:
    """Get or create shared embedding model."""
    global shared_embedding_model
    if shared_embedding_model is None:
        shared_embedding_model = EmbeddingModel()
    return shared_embedding_model


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
    """Get SimpleMem instance for tenant (creates lightweight tenant-scoped system)."""
    agent_id, user_id = parse_memory_id(memory_id)

    # Override with config if provided
    if config:
        if config.metadata and config.metadata.get("agent_id"):
            agent_id = config.metadata["agent_id"]
        if config.user_id:
            user_id = config.user_id

    clear_db = config.clear_db if config else False

    # Create tenant-scoped system (shares DB, filters by tenant)
    system = SimpleMemSystem(
        db_path=shared_db_path,
        table_name=shared_table_name,
        clear_db=clear_db,
        agent_id=agent_id,
        user_id=user_id
    )

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
    # Startup
    print("=" * 60)
    print("SimpleMem API Starting (Multi-tenant Mode)")
    print(f"Shared DB: {shared_db_path}")
    print(f"Table: {shared_table_name}")
    print("=" * 60)

    # Create data directory
    os.makedirs("./data", exist_ok=True)

    # Pre-initialize shared embedding model
    get_shared_embedding_model()

    yield

    # Shutdown
    print("SimpleMem API Shutting down...")
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

    return AskResponse(
        answer=answer,
        contexts=contexts if contexts else [],
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
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"Starting SimpleMem API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
