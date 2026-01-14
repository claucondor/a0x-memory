"""
SimpleMem HTTP API
FastAPI server exposing SimpleMem as a REST API for integration with TypeScript services.

Endpoints designed to be compatible with Zep-like memory operations.
"""
import os
import uuid
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from main import SimpleMemSystem
from models.memory_entry import MemoryEntry


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
# In-Memory Storage for Memory Instances
# ============================================================================

# Global storage for memory instances (in production, use Redis or persistent storage)
memory_instances: Dict[str, SimpleMemSystem] = {}
memory_metadata: Dict[str, Dict[str, Any]] = {}


def get_or_create_memory(memory_id: str, config: Optional[MemoryInstanceConfig] = None) -> SimpleMemSystem:
    """Get existing memory instance or create new one"""
    if memory_id not in memory_instances:
        # Create new instance with isolated database
        db_path = f"./data/lancedb_{memory_id}"
        table_name = f"memory_{memory_id.replace('-', '_')}"

        clear_db = config.clear_db if config else False

        memory_instances[memory_id] = SimpleMemSystem(
            db_path=db_path,
            table_name=table_name,
            clear_db=clear_db
        )

        memory_metadata[memory_id] = {
            "created_at": datetime.utcnow().isoformat(),
            "user_id": config.user_id if config else None,
            "thread_id": config.thread_id if config else None,
            "metadata": config.metadata if config else {},
            "dialogue_count": 0
        }

        print(f"[SimpleMem API] Created new memory instance: {memory_id}")

    return memory_instances[memory_id]


# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("=" * 60)
    print("SimpleMem API Starting...")
    print("=" * 60)

    # Create data directory
    os.makedirs("./data", exist_ok=True)

    yield

    # Shutdown
    print("SimpleMem API Shutting down...")
    memory_instances.clear()
    memory_metadata.clear()


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
        version="1.0.0",
        memory_instances=len(memory_instances),
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
    Create a new memory instance (equivalent to Zep thread creation).
    Each memory instance has isolated storage.
    """
    if config and config.memory_id:
        memory_id = config.memory_id
    elif config and config.thread_id:
        memory_id = config.thread_id
    else:
        memory_id = str(uuid.uuid4())

    # Get or create the memory instance
    get_or_create_memory(memory_id, config)

    meta = memory_metadata.get(memory_id, {})

    return MemoryInstanceResponse(
        memory_id=memory_id,
        user_id=meta.get("user_id"),
        thread_id=meta.get("thread_id") or memory_id,
        created_at=meta.get("created_at", datetime.utcnow().isoformat()),
        memory_count=0
    )


@app.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(memory_id: str, last_n: Optional[int] = None):
    """
    Get memory contents (equivalent to Zep getMemory).
    Returns all memory entries and summary.
    """
    if memory_id not in memory_instances:
        raise HTTPException(status_code=404, detail=f"Memory instance {memory_id} not found")

    system = memory_instances[memory_id]
    meta = memory_metadata.get(memory_id, {})

    # Get all memory entries
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
        # Simple context: last few memories
        recent = entries[-5:] if last_n is None else entries[-last_n:]
        context = "\n".join([e.lossless_restatement for e in recent])

    return MemoryResponse(
        memory_id=memory_id,
        messages=[],  # We don't store raw messages, only processed memories
        summary=context,
        context=context,
        facts=facts,
        memory_entries=memory_entries,
        memory_count=len(entries)
    )


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a memory instance"""
    if memory_id not in memory_instances:
        raise HTTPException(status_code=404, detail=f"Memory instance {memory_id} not found")

    # Clean up
    system = memory_instances.pop(memory_id)
    memory_metadata.pop(memory_id, None)

    # Clear the vector store
    try:
        system.vector_store.clear()
    except Exception as e:
        print(f"[SimpleMem API] Warning: Could not clear vector store: {e}")

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
    Add dialogues to memory (triggers memory building in background).
    Equivalent to Zep addMessages.
    """
    system = get_or_create_memory(memory_id)

    # Add each dialogue
    for dialogue in request.dialogues:
        speaker = dialogue.speaker or dialogue.role or "user"
        system.add_dialogue(
            speaker=speaker,
            content=dialogue.content,
            timestamp=dialogue.timestamp
        )

    # Update dialogue count
    if memory_id in memory_metadata:
        memory_metadata[memory_id]["dialogue_count"] = (
            memory_metadata[memory_id].get("dialogue_count", 0) + len(request.dialogues)
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
    Add messages to memory (Zep-compatible format).
    Converts role-based messages to speaker-based dialogues.
    """
    system = get_or_create_memory(memory_id)

    # Convert messages to dialogues
    for msg in request.messages:
        speaker = msg.name or msg.role
        system.add_dialogue(
            speaker=speaker,
            content=msg.content,
            timestamp=datetime.utcnow().isoformat()
        )

    # Update count
    if memory_id in memory_metadata:
        memory_metadata[memory_id]["dialogue_count"] = (
            memory_metadata[memory_id].get("dialogue_count", 0) + len(request.messages)
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
    Ask a question against the memory (equivalent to Zep graph search).
    Uses SimpleMem's hybrid retrieval and answer generation.
    """
    if memory_id not in memory_instances:
        raise HTTPException(status_code=404, detail=f"Memory instance {memory_id} not found")

    system = memory_instances[memory_id]

    # Get memory count
    entries = system.get_all_memories()

    if not entries:
        return AskResponse(
            answer="No memories available to answer this question.",
            contexts=[],
            memory_count=0
        )

    # Use hybrid retriever to get contexts
    contexts = system.hybrid_retriever.retrieve(
        request.question,
        enable_planning=request.enable_planning,
        enable_reflection=request.enable_reflection
    )

    # Generate answer
    answer = system.answer_generator.generate_answer(request.question, contexts)

    return AskResponse(
        answer=answer,
        contexts=contexts if contexts else [],
        memory_count=len(entries)
    )


@app.post("/memories/{memory_id}/search")
async def search_memory(
    memory_id: str,
    query: str,
    limit: Optional[int] = 10
):
    """
    Search memory entries (simpler than ask, just retrieval without generation).
    Equivalent to Zep searchUserGraph.
    """
    if memory_id not in memory_instances:
        raise HTTPException(status_code=404, detail=f"Memory instance {memory_id} not found")

    system = memory_instances[memory_id]

    # Use hybrid retriever for search only
    contexts = system.hybrid_retriever.retrieve(
        query,
        enable_planning=False,
        enable_reflection=False
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
    """List all memory entries (for debugging)"""
    if memory_id not in memory_instances:
        raise HTTPException(status_code=404, detail=f"Memory instance {memory_id} not found")

    system = memory_instances[memory_id]
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


@app.get("/instances")
async def list_instances():
    """List all memory instances (for debugging)"""
    return {
        "instances": list(memory_instances.keys()),
        "count": len(memory_instances),
        "metadata": {
            k: {
                "created_at": v.get("created_at"),
                "user_id": v.get("user_id"),
                "dialogue_count": v.get("dialogue_count", 0)
            }
            for k, v in memory_metadata.items()
        }
    }


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"Starting SimpleMem API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
