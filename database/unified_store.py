"""
Unified Memory Store - Fusion of VectorStore + GroupMemoryStore

This is the single store that handles:
1. DM memories (memories.lance) - compatible with SimpleMem MemoryEntry
2. Group memories (group_memories.lance) - GroupMemory
3. User memories (user_memories.lance) - UserMemory
4. Interaction memories (interaction_memories.lance) - InteractionMemory
5. Cross-group memories (cross_group_memories.lance) - CrossGroupMemory
6. Cross-agent links (global/cross_agent_links.lance) - CrossAgentLink

Architecture:
- /data/agents/{agent_id}/ - One DB per agent
  - memories.lance (DMs - backward compatible with SimpleMem)
  - group_memories.lance
  - user_memories.lance
  - interaction_memories.lance
  - cross_group_memories.lance
- /data/global/ - Shared DB for cross-agent linking
  - cross_agent_links.lance
"""
import os
import json
from typing import List, Optional, Dict, Any, Tuple, Union
from datetime import datetime, timezone
from pathlib import Path

import lancedb
import pyarrow as pa
import numpy as np

from models.memory_entry import MemoryEntry, MemoryType as DM_MemoryType, PrivacyScope as DM_PrivacyScope
from models.group_memory import (
    GroupMemory, UserMemory, InteractionMemory,
    CrossGroupMemory, CrossAgentLink, AgentResponse,
    MemoryLevel, MemoryType, PrivacyScope
)
from utils.embedding import EmbeddingModel
from utils.structured_schemas import AGENT_RESPONSE_METADATA_SCHEMA
import config


class UnifiedMemoryStore:
    """
    Unified Memory Store - Single interface for all memory types.

    Supports:
    - DMs: MemoryEntry (SimpleMem compatible)
    - Groups: GroupMemory, UserMemory, InteractionMemory
    - Cross-group: CrossGroupMemory
    - Cross-agent: CrossAgentLink
    """

    def __init__(
        self,
        agent_id: str,
        db_base_path: str = None,
        embedding_model: EmbeddingModel = None,
        storage_options: Optional[Dict[str, Any]] = None,
        llm_client = None
    ):
        self.agent_id = agent_id
        self.embedding_model = embedding_model or EmbeddingModel()
        self.storage_options = storage_options
        self.llm_client = llm_client

        # Paths
        self.db_base_path = db_base_path or config.LANCEDB_PATH
        self.agent_db_path = f"{self.db_base_path}/agents/{agent_id}"
        self.global_db_path = f"{self.db_base_path}/global"

        # Detect cloud storage
        self._is_cloud_storage = self.db_base_path.startswith(("gs://", "s3://", "az://"))

        # FTS index status
        self._fts_initialized = {
            'memories': False,
            'group_memories': False,
            'user_memories': False,
            'interaction_memories': False,
            'cross_group_memories': False,
            'agent_responses': False,
            'conversation_summaries': False
        }

        # Initialize databases
        self._init_agent_db()
        self._init_global_db()

        print(f"[UnifiedMemoryStore] Initialized for agent {agent_id}")
        print(f"  - Agent DB: {self.agent_db_path}")
        print(f"  - Global DB: {self.global_db_path}")

    def _init_agent_db(self):
        """Initialize agent-specific database with all tables."""
        if self._is_cloud_storage:
            self.agent_db = lancedb.connect(self.agent_db_path, storage_options=self.storage_options)
        else:
            os.makedirs(self.agent_db_path, exist_ok=True)
            self.agent_db = lancedb.connect(self.agent_db_path)

        # Initialize all tables
        self._init_dm_memories_table()      # memories.lance
        self._init_group_memories_table()   # group_memories.lance
        self._init_user_memories_table()    # user_memories.lance
        self._init_interaction_memories_table()  # interaction_memories.lance
        self._init_cross_group_memories_table()  # cross_group_memories.lance
        self._init_agent_responses_table()  # agent_responses.lance
        self._init_conversation_summaries_table()  # conversation_summaries.lance

    def _init_global_db(self):
        """Initialize global shared database for cross-agent linking."""
        if self._is_cloud_storage:
            self.global_db = lancedb.connect(self.global_db_path, storage_options=self.storage_options)
        else:
            os.makedirs(self.global_db_path, exist_ok=True)
            self.global_db = lancedb.connect(self.global_db_path)

        # Initialize cross-agent links table
        self._init_cross_agent_links_table()

    # ============================================================
    # Table Initialization
    # ============================================================

    def _init_dm_memories_table(self):
        """Initialize DM memories table (compatible with SimpleMem VectorStore)."""
        schema = pa.schema([
            # Multi-tenant
            pa.field("agent_id", pa.string()),
            pa.field("user_id", pa.string()),

            # Core memory (SimpleMem compatible)
            pa.field("entry_id", pa.string()),
            pa.field("lossless_restatement", pa.string()),
            pa.field("keywords", pa.list_(pa.string())),
            pa.field("timestamp", pa.string()),
            pa.field("location", pa.string()),
            pa.field("persons", pa.list_(pa.string())),
            pa.field("entities", pa.list_(pa.string())),
            pa.field("topic", pa.string()),

            # Group context (for DMs we use group_id = "dm_{user_id}")
            pa.field("group_id", pa.string()),
            pa.field("username", pa.string()),
            pa.field("platform", pa.string()),

            # Classification
            pa.field("memory_type", pa.string()),
            pa.field("privacy_scope", pa.string()),
            pa.field("importance_score", pa.float32()),

            # Vector
            pa.field("vector", pa.list_(pa.float32(), self.embedding_model.dimension))
        ])

        table_name = "memories"
        if table_name not in self.agent_db.table_names():
            self.memories_table = self.agent_db.create_table(table_name, schema=schema)
            print(f"[{self.agent_id}] Created {table_name} table")
        else:
            self.memories_table = self.agent_db.open_table(table_name)
            print(f"[{self.agent_id}] Opened {table_name} ({self.memories_table.count_rows()} rows)")

    def _init_group_memories_table(self):
        """Initialize group_memories table."""
        schema = pa.schema([
            pa.field("memory_id", pa.string()),
            pa.field("agent_id", pa.string()),
            pa.field("group_id", pa.string()),
            pa.field("memory_level", pa.string()),
            pa.field("memory_type", pa.string()),
            pa.field("privacy_scope", pa.string()),
            pa.field("content", pa.string()),
            pa.field("speaker", pa.string()),
            pa.field("keywords", pa.list_(pa.string())),
            pa.field("topics", pa.list_(pa.string())),
            pa.field("importance_score", pa.float32()),
            pa.field("evidence_count", pa.int32()),
            pa.field("first_seen", pa.string()),
            pa.field("last_seen", pa.string()),
            pa.field("last_updated", pa.string()),
            pa.field("source_message_id", pa.string()),
            pa.field("source_timestamp", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self.embedding_model.dimension))
        ])

        table_name = "group_memories"
        if table_name not in self.agent_db.table_names():
            self.group_memories_table = self.agent_db.create_table(table_name, schema=schema)
            print(f"[{self.agent_id}] Created {table_name} table")
        else:
            self.group_memories_table = self.agent_db.open_table(table_name)
            print(f"[{self.agent_id}] Opened {table_name} ({self.group_memories_table.count_rows()} rows)")

        # Create indices
        self._create_scalar_index(self.group_memories_table, "group_id")

    def _init_user_memories_table(self):
        """Initialize user_memories table."""
        schema = pa.schema([
            pa.field("memory_id", pa.string()),
            pa.field("agent_id", pa.string()),
            pa.field("group_id", pa.string()),
            pa.field("user_id", pa.string()),
            pa.field("memory_level", pa.string()),
            pa.field("memory_type", pa.string()),
            pa.field("privacy_scope", pa.string()),
            pa.field("content", pa.string()),
            pa.field("keywords", pa.list_(pa.string())),
            pa.field("topics", pa.list_(pa.string())),
            pa.field("importance_score", pa.float32()),
            pa.field("evidence_count", pa.int32()),
            pa.field("first_seen", pa.string()),
            pa.field("last_seen", pa.string()),
            pa.field("last_updated", pa.string()),
            pa.field("source_message_id", pa.string()),
            pa.field("source_timestamp", pa.string()),
            pa.field("username", pa.string()),
            pa.field("platform", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self.embedding_model.dimension))
        ])

        table_name = "user_memories"
        if table_name not in self.agent_db.table_names():
            self.user_memories_table = self.agent_db.create_table(table_name, schema=schema)
            print(f"[{self.agent_id}] Created {table_name} table")
        else:
            self.user_memories_table = self.agent_db.open_table(table_name)
            print(f"[{self.agent_id}] Opened {table_name} ({self.user_memories_table.count_rows()} rows)")

        # Create indices
        self._create_scalar_index(self.user_memories_table, "group_id")
        self._create_scalar_index(self.user_memories_table, "user_id")

    def _init_interaction_memories_table(self):
        """Initialize interaction_memories table."""
        schema = pa.schema([
            pa.field("memory_id", pa.string()),
            pa.field("agent_id", pa.string()),
            pa.field("group_id", pa.string()),
            pa.field("speaker_id", pa.string()),
            pa.field("listener_id", pa.string()),
            pa.field("mentioned_users", pa.list_(pa.string())),
            pa.field("memory_level", pa.string()),
            pa.field("memory_type", pa.string()),
            pa.field("privacy_scope", pa.string()),
            pa.field("content", pa.string()),
            pa.field("keywords", pa.list_(pa.string())),
            pa.field("topics", pa.list_(pa.string())),
            pa.field("importance_score", pa.float32()),
            pa.field("evidence_count", pa.int32()),
            pa.field("first_seen", pa.string()),
            pa.field("last_seen", pa.string()),
            pa.field("last_updated", pa.string()),
            pa.field("source_message_id", pa.string()),
            pa.field("source_timestamp", pa.string()),
            pa.field("interaction_type", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self.embedding_model.dimension))
        ])

        table_name = "interaction_memories"
        if table_name not in self.agent_db.table_names():
            self.interaction_memories_table = self.agent_db.create_table(table_name, schema=schema)
            print(f"[{self.agent_id}] Created {table_name} table")
        else:
            self.interaction_memories_table = self.agent_db.open_table(table_name)
            print(f"[{self.agent_id}] Opened {table_name} ({self.interaction_memories_table.count_rows()} rows)")

        # Create indices
        self._create_scalar_index(self.interaction_memories_table, "group_id")
        self._create_scalar_index(self.interaction_memories_table, "speaker_id")
        self._create_scalar_index(self.interaction_memories_table, "listener_id")

    def _init_cross_group_memories_table(self):
        """Initialize cross_group_memories table."""
        schema = pa.schema([
            pa.field("memory_id", pa.string()),
            pa.field("agent_id", pa.string()),
            pa.field("universal_user_id", pa.string()),
            pa.field("user_identities", pa.list_(pa.string())),
            pa.field("groups_involved", pa.list_(pa.string())),
            pa.field("group_count", pa.int32()),
            pa.field("memory_level", pa.string()),
            pa.field("memory_type", pa.string()),
            pa.field("privacy_scope", pa.string()),
            pa.field("content", pa.string()),
            pa.field("keywords", pa.list_(pa.string())),
            pa.field("topics", pa.list_(pa.string())),
            pa.field("confidence_score", pa.float32()),
            pa.field("pattern_type", pa.string()),
            pa.field("evidence_count", pa.int32()),
            pa.field("first_seen", pa.string()),
            pa.field("last_seen", pa.string()),
            pa.field("last_updated", pa.string()),
            pa.field("consolidated_at", pa.string()),
            pa.field("source_memory_ids", pa.list_(pa.string())),
            pa.field("vector", pa.list_(pa.float32(), self.embedding_model.dimension))
        ])

        table_name = "cross_group_memories"
        if table_name not in self.agent_db.table_names():
            self.cross_group_memories_table = self.agent_db.create_table(table_name, schema=schema)
            print(f"[{self.agent_id}] Created {table_name} table")
        else:
            self.cross_group_memories_table = self.agent_db.open_table(table_name)
            print(f"[{self.agent_id}] Opened {table_name} ({self.cross_group_memories_table.count_rows()} rows)")

        # Create indices
        self._create_scalar_index(self.cross_group_memories_table, "universal_user_id")

    def _init_cross_agent_links_table(self):
        """Initialize cross_agent_links table in global DB."""
        schema = pa.schema([
            pa.field("link_id", pa.string()),
            pa.field("universal_user_id", pa.string()),
            pa.field("agent_mappings", pa.string()),  # JSON string
            pa.field("linking_confidence", pa.float32()),
            pa.field("evidence_count", pa.int32()),
            pa.field("first_linked", pa.string()),
            pa.field("last_updated", pa.string()),
            pa.field("last_verified", pa.string()),
            pa.field("linking_method", pa.string()),
            pa.field("linking_evidence", pa.list_(pa.string()))
        ])

        table_name = "cross_agent_links"
        if table_name not in self.global_db.table_names():
            self.cross_agent_links_table = self.global_db.create_table(table_name, schema=schema)
            print(f"[Global] Created {table_name} table")
        else:
            self.cross_agent_links_table = self.global_db.open_table(table_name)
            print(f"[Global] Opened {table_name} ({self.cross_agent_links_table.count_rows()} rows)")

        # Create indices
        self._create_scalar_index(self.cross_agent_links_table, "universal_user_id")

    def _init_agent_responses_table(self):
        """Initialize agent_responses table for storing agent-generated responses."""
        schema = pa.schema([
            pa.field("response_id", pa.string()),
            pa.field("agent_id", pa.string()),
            pa.field("group_id", pa.string()),
            pa.field("user_id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("content_hash", pa.string()),
            pa.field("summary", pa.string()),
            pa.field("response_type", pa.string()),
            pa.field("topics", pa.list_(pa.string())),
            pa.field("keywords", pa.list_(pa.string())),
            pa.field("trigger_message", pa.string()),
            pa.field("trigger_message_id", pa.string()),
            pa.field("timestamp", pa.string()),
            pa.field("token_count", pa.int32()),
            pa.field("was_repeated", pa.bool_()),
            pa.field("importance_score", pa.float32()),
            pa.field("vector", pa.list_(pa.float32(), self.embedding_model.dimension))
        ])

        table_name = "agent_responses"
        if table_name not in self.agent_db.table_names():
            self.agent_responses_table = self.agent_db.create_table(table_name, schema=schema)
            print(f"[{self.agent_id}] Created {table_name} table")
        else:
            self.agent_responses_table = self.agent_db.open_table(table_name)
            print(f"[{self.agent_id}] Opened {table_name} ({self.agent_responses_table.count_rows()} rows)")

        # Create indices for fast lookups
        self._create_scalar_index(self.agent_responses_table, "agent_id")
        self._create_scalar_index(self.agent_responses_table, "user_id")
        self._create_scalar_index(self.agent_responses_table, "group_id")
        self._create_scalar_index(self.agent_responses_table, "content_hash")

    def _init_conversation_summaries_table(self):
        """Initialize conversation_summaries table for running conversation summaries per thread."""
        schema = pa.schema([
            pa.field("thread_id", pa.string()),
            pa.field("agent_id", pa.string()),
            pa.field("summary", pa.string()),
            pa.field("message_count", pa.int32()),
            pa.field("last_updated", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self.embedding_model.dimension))
        ])

        table_name = "conversation_summaries"
        if table_name not in self.agent_db.table_names():
            self.conversation_summaries_table = self.agent_db.create_table(table_name, schema=schema)
            print(f"[{self.agent_id}] Created {table_name} table")
        else:
            self.conversation_summaries_table = self.agent_db.open_table(table_name)
            print(f"[{self.agent_id}] Opened {table_name} ({self.conversation_summaries_table.count_rows()} rows)")

        # Create indices for fast lookups
        self._create_scalar_index(self.conversation_summaries_table, "thread_id")
        self._create_scalar_index(self.conversation_summaries_table, "agent_id")

    def _create_scalar_index(self, table, column: str):
        """Create scalar index for fast lookups."""
        try:
            table.create_scalar_index(column, replace=True)
        except Exception as e:
            pass  # Index may already exist

    def _init_fts_index(self, table, column: str, table_name: str):
        """Initialize Full-Text Search index."""
        if self._fts_initialized.get(table_name, False):
            return

        try:
            if self._is_cloud_storage:
                table.create_fts_index(column, use_tantivy=False, replace=True)
            else:
                table.create_fts_index(column, use_tantivy=True, tokenizer_name="en_stem", replace=True)
            self._fts_initialized[table_name] = True
            print(f"[{self.agent_id}] FTS index created for {table_name}.{column}")
        except Exception as e:
            pass  # Index may already exist or not supported

    # ============================================================
    # DM Memory Operations (SimpleMem Compatible)
    # ============================================================

    def add_memory_entries(self, entries: List[MemoryEntry], user_id: str = None):
        """
        Add DM memory entries (SimpleMem compatible).

        Args:
            entries: List of MemoryEntry objects
            user_id: Optional user_id override
        """
        if not entries:
            return

        # Batch encode
        restatements = [entry.lossless_restatement for entry in entries]
        vectors = self.embedding_model.encode_documents(restatements)

        data = []
        for entry, vector in zip(entries, vectors):
            # Determine group_id for DMs
            dm_group_id = entry.group_id or f"dm_{entry.user_id or user_id or 'unknown'}"

            data.append({
                "agent_id": self.agent_id,
                "user_id": entry.user_id or user_id or "",
                "entry_id": entry.entry_id,
                "lossless_restatement": entry.lossless_restatement,
                "keywords": entry.keywords,
                "timestamp": entry.timestamp or "",
                "location": entry.location or "",
                "persons": entry.persons,
                "entities": entry.entities,
                "topic": entry.topic or "",
                "group_id": dm_group_id,
                "username": entry.username or "",
                "platform": entry.platform or "direct",
                "memory_type": entry.memory_type.value if isinstance(entry.memory_type, DM_MemoryType) else str(entry.memory_type),
                "privacy_scope": entry.privacy_scope.value if isinstance(entry.privacy_scope, DM_PrivacyScope) else str(entry.privacy_scope),
                "importance_score": entry.importance_score,
                "vector": vector.tolist()
            })

        self.memories_table.add(data)
        print(f"[{self.agent_id}] Added {len(entries)} DM memory entries")

        # Initialize FTS index after first data insertion
        if not self._fts_initialized.get('memories'):
            self._init_fts_index(self.memories_table, "lossless_restatement", "memories")

    def search_memories(
        self,
        query: str,
        user_id: str = None,
        top_k: int = 10
    ) -> List[MemoryEntry]:
        """
        Search DM memories (SimpleMem compatible).

        Args:
            query: Search query
            user_id: Optional user filter
            top_k: Max results
        """
        if self.memories_table.count_rows() == 0:
            return []

        query_vector = self.embedding_model.encode_single(query, is_query=True)
        search = self.memories_table.search(query_vector.tolist())

        # Apply filters
        conditions = [f"agent_id = '{self.agent_id}'"]
        if user_id:
            conditions.append(f"user_id = '{user_id}'")

        where_clause = " AND ".join(conditions)
        search = search.where(where_clause, prefilter=True)

        results = search.limit(top_k).to_list()
        memories = [self._row_to_memory_entry(r) for r in results]

        # Track access for returned memories (non-blocking)
        self.track_search_results_access(memories, "memories")

        return memories

    def _row_to_memory_entry(self, row: Dict) -> MemoryEntry:
        """Convert LanceDB row to MemoryEntry."""
        # Parse memory_type enum
        memory_type_str = row.get("memory_type", "conversation")
        try:
            memory_type = DM_MemoryType(memory_type_str) if memory_type_str else DM_MemoryType.CONVERSATION
        except ValueError:
            memory_type = DM_MemoryType.CONVERSATION

        # Parse privacy_scope enum
        privacy_scope_str = row.get("privacy_scope", "private")
        try:
            privacy_scope = DM_PrivacyScope(privacy_scope_str) if privacy_scope_str else DM_PrivacyScope.PRIVATE
        except ValueError:
            privacy_scope = DM_PrivacyScope.PRIVATE

        return MemoryEntry(
            entry_id=row["entry_id"],
            lossless_restatement=row["lossless_restatement"],
            keywords=list(row.get("keywords") or []),
            timestamp=row.get("timestamp") or None,
            location=row.get("location") or None,
            persons=list(row.get("persons") or []),
            entities=list(row.get("entities") or []),
            topic=row.get("topic") or None,
            group_id=row.get("group_id") or None,
            user_id=row.get("user_id") or None,
            username=row.get("username") or None,
            platform=row.get("platform") or "direct",
            memory_type=memory_type,
            privacy_scope=privacy_scope,
            importance_score=row.get("importance_score", 0.5)
        )

    # ============================================================
    # Group Memory Operations
    # ============================================================

    def add_group_memory(self, memory: GroupMemory) -> GroupMemory:
        """Add a group-level memory."""
        vector = self.embedding_model.encode_single(memory.content, is_query=False)

        data = {
            "memory_id": memory.memory_id,
            "agent_id": memory.agent_id,
            "group_id": memory.group_id,
            "memory_level": memory.memory_level.value,
            "memory_type": memory.memory_type.value,
            "privacy_scope": memory.privacy_scope.value,
            "content": memory.content,
            "speaker": memory.speaker or "",
            "keywords": memory.keywords,
            "topics": memory.topics,
            "importance_score": memory.importance_score,
            "evidence_count": memory.evidence_count,
            "first_seen": memory.first_seen,
            "last_seen": memory.last_seen,
            "last_updated": memory.last_updated,
            "source_message_id": memory.source_message_id or "",
            "source_timestamp": memory.source_timestamp or "",
            "vector": vector.tolist()
        }

        self.group_memories_table.add([data])

        if not self._fts_initialized.get('group_memories'):
            self._init_fts_index(self.group_memories_table, "content", "group_memories")

        return memory

    def add_group_memories_batch(self, memories: List[GroupMemory]) -> List[GroupMemory]:
        """Add multiple group-level memories with batch embeddings and deduplication."""
        if not memories:
            return []

        contents = [m.content for m in memories]
        vectors = self.embedding_model.encode_documents(contents)

        # Track memories to be added (for in-batch deduplication)
        pending_memories = {}  # group_id -> list of (memory, vector) tuples
        merged_count = 0
        new_count = 0

        # First, check for duplicates within the batch
        for memory, vector in zip(memories, vectors):
            group_id = memory.group_id

            if group_id not in pending_memories:
                pending_memories[group_id] = []

            # Check against other memories in the same group from this batch
            found_duplicate_in_batch = False
            for existing_memory, existing_vector in pending_memories[group_id]:
                similarity = self._compute_cosine_similarity(vector, existing_vector)
                if similarity >= 0.85:
                    # Duplicate found in batch - merge into the existing one
                    found_duplicate_in_batch = True
                    merged_count += 1
                    # Update the existing memory's evidence count
                    existing_memory.evidence_count += 1
                    existing_memory.last_seen = memory.last_seen
                    existing_memory.last_updated = datetime.now(timezone.utc).isoformat()
                    break

            if not found_duplicate_in_batch:
                # No duplicate in batch, check against database
                existing = self._find_similar_existing(
                    self.group_memories_table,
                    vector.tolist(),
                    group_id,
                    threshold=0.85
                )

                if existing:
                    # Merge with existing memory in database (delete + re-add pattern)
                    merged_count += 1
                    updated_memory = GroupMemory(
                        memory_id=existing["memory_id"],
                        agent_id=existing["agent_id"],
                        group_id=existing["group_id"],
                        memory_level=MemoryLevel(existing["memory_level"]),
                        memory_type=MemoryType(existing["memory_type"]),
                        privacy_scope=PrivacyScope(existing["privacy_scope"]),
                        content=existing["content"],
                        speaker=existing.get("speaker"),
                        keywords=list(existing.get("keywords") or []),
                        topics=list(existing.get("topics") or []),
                        importance_score=existing["importance_score"],
                        evidence_count=existing["evidence_count"] + 1,
                        first_seen=existing["first_seen"],
                        last_seen=memory.last_seen,
                        last_updated=datetime.now(timezone.utc).isoformat(),
                        source_message_id=existing.get("source_message_id"),
                        source_timestamp=existing.get("source_timestamp")
                    )

                    # Delete old from database
                    self.group_memories_table.delete(f"memory_id = '{existing['memory_id']}'")

                    # Add to pending as updated
                    pending_memories[group_id].append((updated_memory, vector))
                else:
                    # No duplicate found - add as new
                    new_count += 1
                    pending_memories[group_id].append((memory, vector))

        # Convert all pending memories to data format
        all_data = []
        for group_id, memory_list in pending_memories.items():
            for memory, vector in memory_list:
                all_data.append({
                    "memory_id": memory.memory_id,
                    "agent_id": memory.agent_id,
                    "group_id": memory.group_id,
                    "memory_level": memory.memory_level.value,
                    "memory_type": memory.memory_type.value,
                    "privacy_scope": memory.privacy_scope.value,
                    "content": memory.content,
                    "speaker": memory.speaker or "",
                    "keywords": memory.keywords,
                    "topics": memory.topics,
                    "importance_score": memory.importance_score,
                    "evidence_count": memory.evidence_count,
                    "first_seen": memory.first_seen,
                    "last_seen": memory.last_seen,
                    "last_updated": memory.last_updated,
                    "source_message_id": memory.source_message_id or "",
                    "source_timestamp": memory.source_timestamp or "",
                    "vector": vector.tolist() if hasattr(vector, 'tolist') else vector
                })

        if all_data:
            self.group_memories_table.add(all_data)

        print(f"[{self.agent_id}] Added {len(memories)} group memories (merged: {merged_count}, new: {new_count})")

        if not self._fts_initialized.get('group_memories'):
            self._init_fts_index(self.group_memories_table, "content", "group_memories")

        return memories

    def add_user_memory(self, memory: UserMemory) -> UserMemory:
        """Add a user-level memory."""
        vector = self.embedding_model.encode_single(memory.content, is_query=False)

        data = {
            "memory_id": memory.memory_id,
            "agent_id": memory.agent_id,
            "group_id": memory.group_id,
            "user_id": memory.user_id,
            "memory_level": memory.memory_level.value,
            "memory_type": memory.memory_type.value,
            "privacy_scope": memory.privacy_scope.value,
            "content": memory.content,
            "keywords": memory.keywords,
            "topics": memory.topics,
            "importance_score": memory.importance_score,
            "evidence_count": memory.evidence_count,
            "first_seen": memory.first_seen,
            "last_seen": memory.last_seen,
            "last_updated": memory.last_updated,
            "source_message_id": memory.source_message_id or "",
            "source_timestamp": memory.source_timestamp or "",
            "username": memory.username or "",
            "platform": memory.platform or "",
            "vector": vector.tolist()
        }

        self.user_memories_table.add([data])

        if not self._fts_initialized.get('user_memories'):
            self._init_fts_index(self.user_memories_table, "content", "user_memories")

        return memory

    def add_user_memories_batch(self, memories: List[UserMemory]) -> List[UserMemory]:
        """Add multiple user-level memories with batch embeddings and deduplication."""
        if not memories:
            return []

        contents = [m.content for m in memories]
        vectors = self.embedding_model.encode_documents(contents)

        # Track memories to be added (for in-batch deduplication)
        pending_memories = {}  # user_id -> list of (memory, vector) tuples
        merged_count = 0
        new_count = 0

        # First, check for duplicates within the batch
        for memory, vector in zip(memories, vectors):
            user_id = memory.user_id

            if user_id not in pending_memories:
                pending_memories[user_id] = []

            # Check against other memories for the same user from this batch
            found_duplicate_in_batch = False
            for existing_memory, existing_vector in pending_memories[user_id]:
                similarity = self._compute_cosine_similarity(vector, existing_vector)
                if similarity >= 0.85:
                    # Duplicate found in batch - merge into the existing one
                    found_duplicate_in_batch = True
                    merged_count += 1
                    # Update the existing memory's evidence count
                    existing_memory.evidence_count += 1
                    existing_memory.last_seen = memory.last_seen
                    existing_memory.last_updated = datetime.now(timezone.utc).isoformat()
                    break

            if not found_duplicate_in_batch:
                # No duplicate in batch, check against database
                existing = self._find_similar_existing(
                    self.user_memories_table,
                    vector.tolist(),
                    user_id,
                    filter_field="user_id",
                    threshold=0.85
                )

                if existing:
                    # Merge with existing memory in database (delete + re-add pattern)
                    merged_count += 1
                    updated_memory = UserMemory(
                        memory_id=existing["memory_id"],
                        agent_id=existing["agent_id"],
                        group_id=existing["group_id"],
                        user_id=existing["user_id"],
                        memory_level=MemoryLevel(existing["memory_level"]),
                        memory_type=MemoryType(existing["memory_type"]),
                        privacy_scope=PrivacyScope(existing["privacy_scope"]),
                        content=existing["content"],
                        keywords=list(existing.get("keywords") or []),
                        topics=list(existing.get("topics") or []),
                        importance_score=existing["importance_score"],
                        evidence_count=existing["evidence_count"] + 1,
                        first_seen=existing["first_seen"],
                        last_seen=memory.last_seen,
                        last_updated=datetime.now(timezone.utc).isoformat(),
                        source_message_id=existing.get("source_message_id"),
                        source_timestamp=existing.get("source_timestamp"),
                        username=existing.get("username"),
                        platform=existing.get("platform")
                    )

                    # Delete old from database
                    self.user_memories_table.delete(f"memory_id = '{existing['memory_id']}'")

                    # Add to pending as updated
                    pending_memories[user_id].append((updated_memory, vector))
                else:
                    # No duplicate found - add as new
                    new_count += 1
                    pending_memories[user_id].append((memory, vector))

        # Convert all pending memories to data format
        all_data = []
        for user_id, memory_list in pending_memories.items():
            for memory, vector in memory_list:
                all_data.append({
                    "memory_id": memory.memory_id,
                    "agent_id": memory.agent_id,
                    "group_id": memory.group_id,
                    "user_id": memory.user_id,
                    "memory_level": memory.memory_level.value,
                    "memory_type": memory.memory_type.value,
                    "privacy_scope": memory.privacy_scope.value,
                    "content": memory.content,
                    "keywords": memory.keywords,
                    "topics": memory.topics,
                    "importance_score": memory.importance_score,
                    "evidence_count": memory.evidence_count,
                    "first_seen": memory.first_seen,
                    "last_seen": memory.last_seen,
                    "last_updated": memory.last_updated,
                    "source_message_id": memory.source_message_id or "",
                    "source_timestamp": memory.source_timestamp or "",
                    "username": memory.username or "",
                    "platform": memory.platform or "",
                    "vector": vector.tolist() if hasattr(vector, 'tolist') else vector
                })

        if all_data:
            self.user_memories_table.add(all_data)

        print(f"[ingestion-dedup] Merged {merged_count} memories, inserted {new_count} new")
        print(f"[{self.agent_id}] Added {len(memories)} user memories (merged: {merged_count}, new: {new_count})")

        if not self._fts_initialized.get('user_memories'):
            self._init_fts_index(self.user_memories_table, "content", "user_memories")

        return memories

    def add_interaction_memory(self, memory: InteractionMemory) -> InteractionMemory:
        """Add an interaction memory."""
        vector = self.embedding_model.encode_single(memory.content, is_query=False)

        data = {
            "memory_id": memory.memory_id,
            "agent_id": memory.agent_id,
            "group_id": memory.group_id,
            "speaker_id": memory.speaker_id,
            "listener_id": memory.listener_id,
            "mentioned_users": memory.mentioned_users,
            "memory_level": memory.memory_level.value,
            "memory_type": memory.memory_type.value,
            "privacy_scope": memory.privacy_scope.value,
            "content": memory.content,
            "keywords": memory.keywords,
            "topics": memory.topics,
            "importance_score": memory.importance_score,
            "evidence_count": memory.evidence_count,
            "first_seen": memory.first_seen,
            "last_seen": memory.last_seen,
            "last_updated": memory.last_updated,
            "source_message_id": memory.source_message_id or "",
            "source_timestamp": memory.source_timestamp or "",
            "interaction_type": memory.interaction_type or "",
            "vector": vector.tolist()
        }

        self.interaction_memories_table.add([data])

        if not self._fts_initialized.get('interaction_memories'):
            self._init_fts_index(self.interaction_memories_table, "content", "interaction_memories")

        return memory

    def add_interaction_memories_batch(self, memories: List[InteractionMemory]) -> List[InteractionMemory]:
        """Add multiple interaction memories with batch embeddings and deduplication."""
        if not memories:
            return []

        contents = [m.content for m in memories]
        vectors = self.embedding_model.encode_documents(contents)

        # Track memories to be added (for in-batch deduplication)
        pending_memories = {}  # speaker_id -> list of (memory, vector) tuples
        merged_count = 0
        new_count = 0

        # First, check for duplicates within the batch
        for memory, vector in zip(memories, vectors):
            speaker_id = memory.speaker_id

            if speaker_id not in pending_memories:
                pending_memories[speaker_id] = []

            # Check against other memories for the same speaker from this batch
            found_duplicate_in_batch = False
            for existing_memory, existing_vector in pending_memories[speaker_id]:
                similarity = self._compute_cosine_similarity(vector, existing_vector)
                if similarity >= 0.85:
                    # Duplicate found in batch - merge into the existing one
                    found_duplicate_in_batch = True
                    merged_count += 1
                    # Update the existing memory's evidence count
                    existing_memory.evidence_count += 1
                    existing_memory.last_seen = memory.last_seen
                    existing_memory.last_updated = datetime.now(timezone.utc).isoformat()
                    break

            if not found_duplicate_in_batch:
                # No duplicate in batch, check against database
                existing = self._find_similar_existing(
                    self.interaction_memories_table,
                    vector.tolist(),
                    speaker_id,
                    filter_field="speaker_id",
                    threshold=0.85
                )

                if existing:
                    # Merge with existing memory in database (delete + re-add pattern)
                    merged_count += 1
                    updated_memory = InteractionMemory(
                        memory_id=existing["memory_id"],
                        agent_id=existing["agent_id"],
                        group_id=existing["group_id"],
                        speaker_id=existing["speaker_id"],
                        listener_id=existing["listener_id"],
                        mentioned_users=list(existing.get("mentioned_users") or []),
                        memory_level=MemoryLevel(existing["memory_level"]),
                        memory_type=MemoryType(existing["memory_type"]),
                        privacy_scope=PrivacyScope(existing["privacy_scope"]),
                        content=existing["content"],
                        keywords=list(existing.get("keywords") or []),
                        topics=list(existing.get("topics") or []),
                        importance_score=existing["importance_score"],
                        evidence_count=existing["evidence_count"] + 1,
                        first_seen=existing["first_seen"],
                        last_seen=memory.last_seen,
                        last_updated=datetime.now(timezone.utc).isoformat(),
                        source_message_id=existing.get("source_message_id"),
                        source_timestamp=existing.get("source_timestamp"),
                        interaction_type=existing.get("interaction_type")
                    )

                    # Delete old from database
                    self.interaction_memories_table.delete(f"memory_id = '{existing['memory_id']}'")

                    # Add to pending as updated
                    pending_memories[speaker_id].append((updated_memory, vector))
                else:
                    # No duplicate found - add as new
                    new_count += 1
                    pending_memories[speaker_id].append((memory, vector))

        # Convert all pending memories to data format
        all_data = []
        for speaker_id, memory_list in pending_memories.items():
            for memory, vector in memory_list:
                all_data.append({
                    "memory_id": memory.memory_id,
                    "agent_id": memory.agent_id,
                    "group_id": memory.group_id,
                    "speaker_id": memory.speaker_id,
                    "listener_id": memory.listener_id,
                    "mentioned_users": memory.mentioned_users,
                    "memory_level": memory.memory_level.value,
                    "memory_type": memory.memory_type.value,
                    "privacy_scope": memory.privacy_scope.value,
                    "content": memory.content,
                    "keywords": memory.keywords,
                    "topics": memory.topics,
                    "importance_score": memory.importance_score,
                    "evidence_count": memory.evidence_count,
                    "first_seen": memory.first_seen,
                    "last_seen": memory.last_seen,
                    "last_updated": memory.last_updated,
                    "source_message_id": memory.source_message_id or "",
                    "source_timestamp": memory.source_timestamp or "",
                    "interaction_type": memory.interaction_type or "",
                    "vector": vector.tolist() if hasattr(vector, 'tolist') else vector
                })

        if all_data:
            self.interaction_memories_table.add(all_data)

        print(f"[ingestion-dedup] Merged {merged_count} memories, inserted {new_count} new")
        print(f"[{self.agent_id}] Added {len(memories)} interaction memories (merged: {merged_count}, new: {new_count})")

        if not self._fts_initialized.get('interaction_memories'):
            self._init_fts_index(self.interaction_memories_table, "content", "interaction_memories")

        return memories

    def add_cross_group_memory(self, memory: CrossGroupMemory) -> CrossGroupMemory:
        """Add a cross-group consolidated memory."""
        vector = self.embedding_model.encode_single(memory.content, is_query=False)

        data = {
            "memory_id": memory.memory_id,
            "agent_id": memory.agent_id,
            "universal_user_id": memory.universal_user_id,
            "user_identities": memory.user_identities,
            "groups_involved": memory.groups_involved,
            "group_count": memory.group_count,
            "memory_level": memory.memory_level.value,
            "memory_type": memory.memory_type.value,
            "privacy_scope": memory.privacy_scope.value,
            "content": memory.content,
            "keywords": memory.keywords,
            "topics": memory.topics,
            "confidence_score": memory.confidence_score,
            "pattern_type": memory.pattern_type,
            "evidence_count": memory.evidence_count,
            "first_seen": memory.first_seen,
            "last_seen": memory.last_seen,
            "last_updated": memory.last_updated,
            "consolidated_at": memory.consolidated_at,
            "source_memory_ids": memory.source_memory_ids,
            "vector": vector.tolist()
        }

        self.cross_group_memories_table.add([data])

        if not self._fts_initialized.get('cross_group_memories'):
            self._init_fts_index(self.cross_group_memories_table, "content", "cross_group_memories")

        return memory

    # ============================================================
    # Agent Response Operations (Anti-Repetition)
    # ============================================================

    def add_agent_response(self, response: AgentResponse) -> str:
        """Add an agent-generated response to memory."""
        # Encode summary for semantic search (fallback to content if no summary)
        search_text = response.summary or response.content
        vector = self.embedding_model.encode_single(search_text, is_query=False)

        data = {
            "response_id": response.response_id,
            "agent_id": response.agent_id,
            "group_id": response.group_id or "",
            "user_id": response.user_id,
            "content": response.content,
            "content_hash": response.content_hash,
            "summary": response.summary or "",
            "response_type": response.response_type.value,
            "topics": response.topics,
            "keywords": response.keywords,
            "trigger_message": response.trigger_message,
            "trigger_message_id": response.trigger_message_id or "",
            "timestamp": response.timestamp,
            "token_count": response.token_count,
            "was_repeated": response.was_repeated,
            "importance_score": response.importance_score,
            "vector": vector.tolist()
        }

        self.agent_responses_table.add([data])

        if not self._fts_initialized.get('agent_responses'):
            self._init_fts_index(self.agent_responses_table, "summary", "agent_responses")

        print(f"[{self.agent_id}] Added agent response: {response.response_id}")
        return response.response_id

    def search_agent_responses(
        self,
        query: str,
        group_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 5,
        max_age_hours: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search agent responses by semantic similarity with multi-level filtering.

        Priority levels (when retrieving context):
        1. group_id + user_id (most specific - direct conversation)
        2. group_id only (what was said to others in this group)
        3. user_id only (what was said to this user in other groups)
        4. None (global - fallback)

        Args:
            query: Search query
            group_id: Filter by group (None = all groups/DMs)
            user_id: Filter by user (None = all users)
            limit: Max results
            max_age_hours: Only return responses within this time window

        Returns:
            List of agent responses with metadata
        """
        if self.agent_responses_table.count_rows() == 0:
            return []

        query_vector = self.embedding_model.encode_single(query, is_query=True)
        search = self.agent_responses_table.search(query_vector.tolist())

        # Build filter conditions
        conditions = [f"agent_id = '{self.agent_id}'"]

        if group_id:
            conditions.append(f"group_id = '{group_id}'")
        else:
            # If not filtering by group, exclude empty group_id (DMs handled separately)
            pass

        if user_id:
            conditions.append(f"user_id = '{user_id}'")

        where_clause = " AND ".join(conditions)
        search = search.where(where_clause, prefilter=True)

        results = search.limit(limit * 2).to_list()  # Get more to filter by age

        # Filter by age and format results
        filtered_results = []
        cutoff_time = None
        if max_age_hours:
            from datetime import datetime, timezone, timedelta
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

        for r in results:
            # Check age filter
            if cutoff_time:
                try:
                    resp_time = datetime.fromisoformat(r.get("timestamp", "").replace('Z', '+00:00'))
                    if resp_time < cutoff_time:
                        continue
                except:
                    pass  # Skip entries with invalid timestamps

            # Format time ago
            time_ago = self._format_time_ago(r.get("timestamp", ""))

            filtered_results.append({
                "response_id": r["response_id"],
                "content": r["content"],
                "summary": r.get("summary", "") or r["content"][:100] + "..." if len(r["content"]) > 100 else r["content"],
                "timestamp": time_ago,
                "raw_timestamp": r.get("timestamp", ""),
                "response_type": r["response_type"],
                "topics": list(r.get("topics") or []),
                "group_id": r.get("group_id"),
                "user_id": r["user_id"],
                "trigger_message": r.get("trigger_message", "")
            })

            if len(filtered_results) >= limit:
                break

        return filtered_results

    def check_duplicate_response(
        self,
        content_hash: str,
        user_id: str,
        group_id: Optional[str] = None,
        within_hours: int = 24
    ) -> Optional[Dict[str, Any]]:
        """
        Check if a similar response was recently given to avoid repetition.

        Args:
            content_hash: Hash of the response content
            user_id: User ID
            group_id: Group ID (optional)
            within_hours: Time window to check

        Returns:
            Previous response if found, None otherwise
        """
        conditions = [
            f"agent_id = '{self.agent_id}'",
            f"user_id = '{user_id}'",
            f"content_hash = '{content_hash}'"
        ]

        if group_id:
            conditions.append(f"group_id = '{group_id}'")

        where_clause = " AND ".join(conditions)

        try:
            results = self.agent_responses_table.search().where(where_clause, prefilter=True).limit(1).to_list()
            if results:
                r = results[0]
                return {
                    "response_id": r["response_id"],
                    "content": r["content"],
                    "timestamp": self._format_time_ago(r.get("timestamp", "")),
                    "raw_timestamp": r.get("timestamp", "")
                }
        except Exception as e:
            print(f"[{self.agent_id}] Error checking duplicate: {e}")

        return None

    def _format_time_ago(self, timestamp: str) -> str:
        """Format timestamp as 'X time ago' string."""
        try:
            from datetime import datetime, timezone

            # Handle various timestamp formats
            ts = timestamp.replace('Z', '+00:00')
            if '+' not in ts:
                ts += '+00:00'

            resp_time = datetime.fromisoformat(ts)
            now = datetime.now(timezone.utc)
            delta = now - resp_time

            seconds = int(delta.total_seconds())

            if seconds < 60:
                return f"{seconds}s ago"
            elif seconds < 3600:
                minutes = seconds // 60
                return f"{minutes}m ago"
            elif seconds < 86400:
                hours = seconds // 3600
                return f"{hours}h ago"
            elif seconds < 604800:
                days = seconds // 86400
                return f"{days}d ago"
            else:
                weeks = seconds // 604800
                return f"{weeks}w ago"
        except:
            return timestamp

    def add_agent_response_with_llm(
        self,
        llm_client,
        agent_id: str,
        group_id: Optional[str],
        user_id: str,
        content: str,
        trigger_message: str,
        trigger_message_id: Optional[str] = None,
        platform: str = "direct"
    ) -> str:
        """
        Add agent response with LLM-extracted metadata (SAME PATTERN as group memories).

        Uses llama-3.1-8b-instruct (via OpenRouter) to extract:
        - summary: One-sentence summary
        - response_type: greeting, answer, clarification, etc.
        - topics: Main topics covered
        - keywords: Important terms for retrieval
        - importance_score: 0.0-1.0 (how important is this response)

        Args:
            llm_client: LLMClient instance (shared with MemoryBuilder)
            agent_id: Agent ID
            group_id: Group ID (None for DMs)
            user_id: User ID
            content: Agent's response content
            trigger_message: What the user asked
            trigger_message_id: Original message ID
            platform: Platform (telegram, xmtp, etc.)

        Returns:
            response_id: The created response ID
        """
        from datetime import datetime, timezone
        import hashlib

        # Step 1: Extract metadata with LLM (SAME PATTERN as _generate_group_memories)
        metadata = self._extract_agent_response_metadata(
            llm_client=llm_client,
            content=content,
            trigger_message=trigger_message,
            group_id=group_id,
            user_id=user_id,
            platform=platform
        )

        # Step 2: Create AgentResponse with LLM-extracted metadata
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        from models.group_memory import AgentResponse, ResponseType

        agent_response = AgentResponse(
            agent_id=agent_id,
            group_id=group_id,
            user_id=user_id,
            content=content,
            content_hash=content_hash,
            summary=metadata["summary"],
            response_type=ResponseType(metadata["response_type"]),
            topics=metadata["topics"],
            keywords=metadata["keywords"],
            trigger_message=trigger_message,
            trigger_message_id=trigger_message_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            token_count=len(content.split()),
            was_repeated=False,
            importance_score=metadata["importance_score"]
        )

        # Step 3: Store with embedding
        return self.add_agent_response(agent_response)

    def _extract_agent_response_metadata(
        self,
        llm_client,
        content: str,
        trigger_message: str,
        group_id: Optional[str],
        user_id: str,
        platform: str
    ) -> Dict[str, Any]:
        """
        Extract metadata from agent response using LLM.

        SAME PATTERN as _generate_group_memories() - uses structured prompt
        with llama-3.1-8b-instruct for extraction.

        Args:
            llm_client: LLMClient instance
            content: Agent's response
            trigger_message: What the user asked
            group_id: Group ID (None for DMs)
            user_id: User ID
            platform: Platform

        Returns:
            Dict with: summary, response_type, topics, keywords, importance_score
        """
        prompt = f"""
Analyze this agent response and extract structured metadata for memory storage.

[Context]
Platform: {platform}
Group ID: {group_id or 'DM'}
User ID: {user_id}

[Trigger Message - What the user asked]
{trigger_message}

[Agent Response - What the agent said]
{content}

[Your Task]
Extract key metadata from this response to enable:
1. Semantic search (finding similar responses later)
2. Anti-repetition (knowing what was already said)
3. Quality assessment (knowing which responses are important)

[Output Format]
Return JSON:

```json
{{
  "summary": "One-sentence summary of what the agent explained",
  "response_type": "greeting|answer|clarification|recommendation|question|acknowledgment|other",
  "topics": ["topic1", "topic2"],
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "importance_score": 0.0 to 1.0
}}
```

[Response Type Guidelines]
- greeting: Agent welcomes user or says hello
- answer: Direct answer to a question
- clarification: Agent clarifies something confusing
- recommendation: Agent suggests something (tool, resource, approach)
- question: Agent asks user for information
- acknowledgment: Agent confirms understanding (ok, got it, thanks)
- other: Anything else

[Importance Score Guidelines]
- 0.8-1.0: Critical info (instructions, explanations of complex topics, important recommendations)
- 0.5-0.7: Useful info (standard answers, moderate explanations)
- 0.3-0.4: Low value (simple confirmations, brief responses)
- 0.0-0.2: Trivial (hi, ok, thanks, greetings)

[Rules]
1. summary MUST be self-contained - understandable without context
2. topics should be specific (e.g., "gas fees", "yield farming" not "crypto")
3. keywords should include technical terms and entities mentioned
4. Focus on WHAT the agent communicated, not HOW
5. If response is very short (< 20 chars), importance should be low

Return ONLY the JSON object.
"""

        # LLM call with retry (MISMO PATRÓN que group memories)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                messages = [
                    {
                        "role": "system",
                        "content": "You are a metadata extraction specialist for AI agent responses. You must output valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

                response = llm_client.chat_completion(
                    messages,
                    temperature=0.1,
                    response_format=AGENT_RESPONSE_METADATA_SCHEMA
                )

                # Parse response - structured outputs return valid JSON
                try:
                    data = json.loads(response)
                except (json.JSONDecodeError, TypeError):
                    data = llm_client.extract_json(response)

                # Validate required fields
                if not isinstance(data, dict):
                    raise ValueError(f"Expected JSON object but got: {type(data)}")

                # Validate response_type
                valid_types = ["greeting", "answer", "clarification", "recommendation",
                               "question", "acknowledgment", "other"]
                response_type = data.get("response_type", "other")
                if response_type not in valid_types:
                    response_type = "other"

                return {
                    "summary": data.get("summary", content[:100] + "..." if len(content) > 100 else content),
                    "response_type": response_type,
                    "topics": data.get("topics", []),
                    "keywords": data.get("keywords", []),
                    "importance_score": data.get("importance_score", 0.5)
                }

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[AgentResponse] Attempt {attempt + 1}/{max_retries} failed: {e}")
                else:
                    print(f"[AgentResponse] All attempts failed, using fallback")
                    # Fallback simple
                    return {
                        "summary": content[:100] + "..." if len(content) > 100 else content,
                        "response_type": "other",
                        "topics": [],
                        "keywords": [],
                        "importance_score": 0.5
                    }

    # ============================================================
    # Search Operations
    # ============================================================

    def search_group_memories(
        self,
        group_id: str,
        query: str,
        limit: int = 10,
        memory_type: Optional[MemoryType] = None
    ) -> List[GroupMemory]:
        """Search group memories by semantic similarity."""
        if self.group_memories_table.count_rows() == 0:
            return []

        query_vector = self.embedding_model.encode_single(query, is_query=True)
        search = self.group_memories_table.search(query_vector.tolist())

        conditions = [f"group_id = '{group_id}'"]
        if memory_type:
            conditions.append(f"memory_type = '{memory_type.value}'")

        where_clause = " AND ".join(conditions)
        search = search.where(where_clause, prefilter=True)

        results = search.limit(limit).to_list()
        group_memories = [self._row_to_group_memory(r) for r in results]

        # Track access for returned memories (non-blocking)
        # This is fire-and-forget - we don't wait for it to complete
        self.track_search_results_access(group_memories, "group_memories")

        return group_memories

    def search_user_memories(
        self,
        group_id: str,
        user_id: str,
        query: str,
        limit: int = 10
    ) -> List[UserMemory]:
        """Search user memories by semantic similarity for a specific user."""
        if self.user_memories_table.count_rows() == 0:
            return []

        query_vector = self.embedding_model.encode_single(query, is_query=True)
        search = self.user_memories_table.search(query_vector.tolist())

        where_clause = f"group_id = '{group_id}' AND user_id = '{user_id}'"
        search = search.where(where_clause, prefilter=True)

        results = search.limit(limit).to_list()
        user_memories = [self._row_to_user_memory(r) for r in results]

        # Track access for returned memories (non-blocking)
        self.track_search_results_access(user_memories, "user_memories")

        return user_memories

    def search_user_memories_in_group(
        self,
        group_id: str,
        query: str,
        limit: int = 10,
        exclude_user_id: Optional[str] = None
    ) -> List[UserMemory]:
        """Search user memories across ALL users in a group."""
        if self.user_memories_table.count_rows() == 0:
            return []

        query_vector = self.embedding_model.encode_single(query, is_query=True)
        search = self.user_memories_table.search(query_vector.tolist())

        where_clause = f"group_id = '{group_id}'"
        if exclude_user_id:
            where_clause += f" AND user_id != '{exclude_user_id}'"

        search = search.where(where_clause, prefilter=True)

        results = search.limit(limit).to_list()
        user_memories = [self._row_to_user_memory(r) for r in results]

        # Track access for returned memories (non-blocking)
        self.track_search_results_access(user_memories, "user_memories")

        return user_memories

    def search_interactions(
        self,
        group_id: str,
        speaker_id: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 10
    ) -> List[InteractionMemory]:
        """Search interaction memories."""
        if self.interaction_memories_table.count_rows() == 0:
            return []

        conditions = [f"group_id = '{group_id}'"]
        if speaker_id:
            conditions.append(f"(speaker_id = '{speaker_id}' OR listener_id = '{speaker_id}')")

        where_clause = " AND ".join(conditions)

        if query:
            query_vector = self.embedding_model.encode_single(query, is_query=True)
            search = self.interaction_memories_table.search(query_vector.tolist())
            search = search.where(where_clause, prefilter=True)
            results = search.limit(limit).to_list()
        else:
            results = self.interaction_memories_table.search().where(where_clause, prefilter=True).limit(limit).to_list()

        interaction_memories = [self._row_to_interaction_memory(r) for r in results]

        # Track access for returned memories (non-blocking)
        self.track_search_results_access(interaction_memories, "interaction_memories")

        return interaction_memories

    def search_cross_group(
        self,
        universal_user_id: str,
        query: str,
        limit: int = 10
    ) -> List[CrossGroupMemory]:
        """Search cross-group memories for a user."""
        if self.cross_group_memories_table.count_rows() == 0:
            return []

        query_vector = self.embedding_model.encode_single(query, is_query=True)
        search = self.cross_group_memories_table.search(query_vector.tolist())

        where_clause = f"universal_user_id = '{universal_user_id}'"
        search = search.where(where_clause, prefilter=True)

        results = search.limit(limit).to_list()
        cross_group_memories = [self._row_to_cross_group_memory(r) for r in results]

        # Track access for returned memories (non-blocking)
        self.track_search_results_access(cross_group_memories, "cross_group_memories")

        return cross_group_memories

    # ============================================================
    # Unified Search (Multi-table)
    # ============================================================

    def search_all(
        self,
        query: str,
        context: Dict[str, Any],
        limit_per_table: int = 5
    ) -> Dict[str, List[Any]]:
        """
        Search across all relevant tables based on context.

        Args:
            query: Search query
            context: Dict with group_id, user_id, etc.
            limit_per_table: Max results per table

        Returns:
            Dict with results from each table type
        """
        results = {
            "dm_memories": [],
            "group_memories": [],
            "user_memories": [],
            "interaction_memories": [],
            "cross_group_memories": []
        }

        group_id = context.get('group_id')
        user_id = context.get('user_id')
        is_group = group_id is not None and not group_id.startswith('dm_')

        if is_group:
            # Group context - search group tables
            results["group_memories"] = self.search_group_memories(
                group_id, query, limit=limit_per_table
            )
            results["user_memories"] = self.search_user_memories_in_group(
                group_id, query, limit=limit_per_table, exclude_user_id=user_id
            )
            results["interaction_memories"] = self.search_interactions(
                group_id, speaker_id=user_id, query=query, limit=limit_per_table
            )

            # Cross-group if user_id available
            if user_id:
                universal_id = user_id  # Assuming user_id is already platform:id format
                results["cross_group_memories"] = self.search_cross_group(
                    universal_id, query, limit=limit_per_table
                )
        else:
            # DM context - search memories table
            results["dm_memories"] = self.search_memories(
                query, user_id=user_id, top_k=limit_per_table
            )

        return results

    # ============================================================
    # Row Converters
    # ============================================================

    def _row_to_group_memory(self, row: Dict) -> GroupMemory:
        """Convert LanceDB row to GroupMemory."""
        return GroupMemory(
            memory_id=row["memory_id"],
            agent_id=row["agent_id"],
            group_id=row["group_id"],
            memory_level=MemoryLevel(row["memory_level"]),
            memory_type=MemoryType(row["memory_type"]),
            privacy_scope=PrivacyScope(row["privacy_scope"]),
            content=row["content"],
            speaker=row["speaker"] or None,
            keywords=list(row.get("keywords") or []),
            topics=list(row.get("topics") or []),
            importance_score=row["importance_score"],
            evidence_count=row["evidence_count"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            last_updated=row["last_updated"],
            source_message_id=row["source_message_id"] or None,
            source_timestamp=row["source_timestamp"] or None
        )

    def _row_to_user_memory(self, row: Dict) -> UserMemory:
        """Convert LanceDB row to UserMemory."""
        return UserMemory(
            memory_id=row["memory_id"],
            agent_id=row["agent_id"],
            group_id=row["group_id"],
            user_id=row["user_id"],
            memory_level=MemoryLevel(row["memory_level"]),
            memory_type=MemoryType(row["memory_type"]),
            privacy_scope=PrivacyScope(row["privacy_scope"]),
            content=row["content"],
            keywords=list(row.get("keywords") or []),
            topics=list(row.get("topics") or []),
            importance_score=row["importance_score"],
            evidence_count=row["evidence_count"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            last_updated=row["last_updated"],
            source_message_id=row["source_message_id"] or None,
            source_timestamp=row["source_timestamp"] or None,
            username=row["username"] or None,
            platform=row["platform"] or None
        )

    def _row_to_interaction_memory(self, row: Dict) -> InteractionMemory:
        """Convert LanceDB row to InteractionMemory."""
        return InteractionMemory(
            memory_id=row["memory_id"],
            agent_id=row["agent_id"],
            group_id=row["group_id"],
            speaker_id=row["speaker_id"],
            listener_id=row["listener_id"],
            mentioned_users=list(row.get("mentioned_users") or []),
            memory_level=MemoryLevel(row["memory_level"]),
            memory_type=MemoryType(row["memory_type"]),
            privacy_scope=PrivacyScope(row["privacy_scope"]),
            content=row["content"],
            keywords=list(row.get("keywords") or []),
            topics=list(row.get("topics") or []),
            importance_score=row["importance_score"],
            evidence_count=row["evidence_count"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            last_updated=row["last_updated"],
            source_message_id=row["source_message_id"] or None,
            source_timestamp=row["source_timestamp"] or None,
            interaction_type=row["interaction_type"] or None
        )

    def _row_to_cross_group_memory(self, row: Dict) -> CrossGroupMemory:
        """Convert LanceDB row to CrossGroupMemory."""
        return CrossGroupMemory(
            memory_id=row["memory_id"],
            agent_id=row["agent_id"],
            universal_user_id=row["universal_user_id"],
            user_identities=list(row.get("user_identities") or []),
            groups_involved=list(row.get("groups_involved") or []),
            group_count=row["group_count"],
            memory_level=MemoryLevel(row["memory_level"]),
            memory_type=MemoryType(row["memory_type"]),
            privacy_scope=PrivacyScope(row["privacy_scope"]),
            content=row["content"],
            keywords=list(row.get("keywords") or []),
            topics=list(row.get("topics") or []),
            confidence_score=row["confidence_score"],
            pattern_type=row["pattern_type"],
            evidence_count=row["evidence_count"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            last_updated=row["last_updated"],
            consolidated_at=row["consolidated_at"],
            source_memory_ids=list(row.get("source_memory_ids") or [])
        )

    # ============================================================
    # SimpleMem VectorStore Compatibility Methods
    # ============================================================

    def semantic_search(self, query: str, top_k: int = 5, user_id: str = None) -> List[MemoryEntry]:
        """Semantic search (SimpleMem VectorStore compatible)."""
        return self.search_memories(query, user_id=user_id, top_k=top_k)

    def keyword_search(self, keywords: List[str], top_k: int = 3, user_id: str = None) -> List[MemoryEntry]:
        """Keyword search (SimpleMem VectorStore compatible)."""
        if not keywords or self.memories_table.count_rows() == 0:
            return []

        query = " ".join(keywords)

        try:
            search = self.memories_table.search(query)

            conditions = [f"agent_id = '{self.agent_id}'"]
            if user_id:
                conditions.append(f"user_id = '{user_id}'")

            where_clause = " AND ".join(conditions)
            search = search.where(where_clause, prefilter=True)

            results = search.limit(top_k).to_list()
            return [self._row_to_memory_entry(r) for r in results]
        except Exception as e:
            print(f"Error during keyword search: {e}")
            return []

    def keyword_search_with_scores(self, keywords: List[str], top_k: int = 3) -> List[Tuple[MemoryEntry, float]]:
        """Keyword search with scores (SimpleMem VectorStore compatible)."""
        if not keywords or self.memories_table.count_rows() == 0:
            return []

        query = " ".join(keywords)

        try:
            search = self.memories_table.search(query)

            conditions = [f"agent_id = '{self.agent_id}'"]
            where_clause = " AND ".join(conditions)
            search = search.where(where_clause, prefilter=True)

            results = search.limit(top_k).to_list()

            if not results:
                return []

            scored_entries = []
            max_score = 0

            for r in results:
                score = r.get("_score", 0.0)
                max_score = max(max_score, score)
                try:
                    entry = self._row_to_memory_entry(r)
                    scored_entries.append((entry, score))
                except Exception as e:
                    print(f"Warning: Failed to parse FTS result: {e}")
                    continue

            # Normalize scores to [0, 1]
            if max_score > 0:
                scored_entries = [(entry, score / max_score) for entry, score in scored_entries]

            return scored_entries
        except Exception as e:
            print(f"Error during keyword search with scores: {e}")
            return []

    def keyword_search_group_memories(self, group_id: str, keywords: List[str], top_k: int = 5) -> List[Tuple[GroupMemory, float]]:
        """FTS keyword search on group_memories table with scores."""
        if not keywords or self.group_memories_table.count_rows() == 0:
            return []

        query = " ".join(keywords)

        try:
            search = self.group_memories_table.search(query)
            where_clause = f"group_id = '{group_id}'"
            search = search.where(where_clause, prefilter=True)

            results = search.limit(top_k).to_list()

            if not results:
                return []

            scored_entries = []
            max_score = 0

            for r in results:
                score = r.get("_score", 0.0)
                max_score = max(max_score, score)
                try:
                    entry = self._row_to_group_memory(r)
                    scored_entries.append((entry, score))
                except Exception as e:
                    print(f"Warning: Failed to parse group FTS result: {e}")
                    continue

            # Normalize scores to [0, 1]
            if max_score > 0:
                scored_entries = [(entry, score / max_score) for entry, score in scored_entries]

            return scored_entries
        except Exception as e:
            print(f"Error during group keyword search with scores: {e}")
            return []

    def keyword_search_user_memories(self, group_id: str, keywords: List[str], top_k: int = 5) -> List[Tuple[UserMemory, float]]:
        """FTS keyword search on user_memories table with scores."""
        if not keywords or self.user_memories_table.count_rows() == 0:
            return []

        query = " ".join(keywords)

        try:
            search = self.user_memories_table.search(query)
            where_clause = f"group_id = '{group_id}'"
            search = search.where(where_clause, prefilter=True)

            results = search.limit(top_k).to_list()

            if not results:
                return []

            scored_entries = []
            max_score = 0

            for r in results:
                score = r.get("_score", 0.0)
                max_score = max(max_score, score)
                try:
                    entry = self._row_to_user_memory(r)
                    scored_entries.append((entry, score))
                except Exception as e:
                    print(f"Warning: Failed to parse user FTS result: {e}")
                    continue

            # Normalize scores to [0, 1]
            if max_score > 0:
                scored_entries = [(entry, score / max_score) for entry, score in scored_entries]

            return scored_entries
        except Exception as e:
            print(f"Error during user keyword search with scores: {e}")
            return []

    def keyword_search_interactions(self, group_id: str, keywords: List[str], top_k: int = 5) -> List[Tuple[InteractionMemory, float]]:
        """FTS keyword search on interaction_memories table with scores."""
        if not keywords or self.interaction_memories_table.count_rows() == 0:
            return []

        query = " ".join(keywords)

        try:
            search = self.interaction_memories_table.search(query)
            where_clause = f"group_id = '{group_id}'"
            search = search.where(where_clause, prefilter=True)

            results = search.limit(top_k).to_list()

            if not results:
                return []

            scored_entries = []
            max_score = 0

            for r in results:
                score = r.get("_score", 0.0)
                max_score = max(max_score, score)
                try:
                    entry = self._row_to_interaction_memory(r)
                    scored_entries.append((entry, score))
                except Exception as e:
                    print(f"Warning: Failed to parse interaction FTS result: {e}")
                    continue

            # Normalize scores to [0, 1]
            if max_score > 0:
                scored_entries = [(entry, score / max_score) for entry, score in scored_entries]

            return scored_entries
        except Exception as e:
            print(f"Error during interaction keyword search with scores: {e}")
            return []

    def add_entries(self, entries: List[MemoryEntry], agent_id: str = None, user_id: str = None):
        """Add entries (SimpleMem VectorStore compatible)."""
        self.add_memory_entries(entries, user_id=user_id)

    def get_all_entries(self) -> List[MemoryEntry]:
        """Get all entries (SimpleMem VectorStore compatible)."""
        conditions = [f"agent_id = '{self.agent_id}'"]
        where_clause = " AND ".join(conditions)
        results = self.memories_table.search().where(where_clause, prefilter=True).to_list()
        return [self._row_to_memory_entry(r) for r in results]

    def count_entries(self) -> int:
        """Count entries for this agent."""
        return self.memories_table.count_rows()

    # ============================================================
    # Deduplication Methods
    # ============================================================

    def _compute_cosine_similarity(self, vec1, vec2) -> float:
        """Compute cosine similarity between two vectors."""
        try:
            import numpy as np
            v1 = np.array(vec1) if not isinstance(vec1, np.ndarray) else vec1
            v2 = np.array(vec2) if not isinstance(vec2, np.ndarray) else vec2
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0

    def _find_similar_existing(
        self,
        table,
        vector: List[float],
        filter_value: str,
        filter_field: str = "group_id",
        threshold: float = 0.85
    ) -> Optional[Dict]:
        """
        Find a similar existing memory using cosine similarity.

        Args:
            table: LanceDB table to search
            vector: Embedding vector to search for
            filter_value: Value to filter by (e.g., group_id, user_id)
            filter_field: Field name to filter on (default: group_id)
            threshold: Similarity threshold (default: 0.85)

        Returns:
            Dictionary with existing memory data if found, None otherwise
        """
        try:
            # Search with vector similarity and filter
            results = (
                table.search(vector)
                .where(f"{filter_field} = '{filter_value}'", prefilter=True)
                .limit(1)
                .to_list()
            )

            if results:
                result = results[0]
                # LanceDB returns cosine distance, need to convert to similarity
                # distance = 1 - similarity, so similarity = 1 - distance
                distance = result.get("_distance", 1.0)
                similarity = 1.0 - distance

                if similarity >= threshold:
                    return result

        except Exception as e:
            print(f"[{self.agent_id}] Error finding similar memory: {e}")

        return None

    # ============================================================
    # Conversation Summary Methods
    # ============================================================

    def get_or_create_summary(self, thread_id: str) -> str:
        """
        Get or create a conversation summary for a thread.

        Args:
            thread_id: Thread identifier (format: "{agent_id}_{context_type}_{group_or_user_id}")

        Returns:
            Current summary text (empty string if not exists)
        """
        try:
            results = self.conversation_summaries_table.search().where(
                f"thread_id = '{thread_id}'",
                prefilter=True
            ).limit(1).to_list()

            if results:
                return results[0].get("summary", "")
            return ""
        except Exception as e:
            print(f"[get_or_create_summary] Error: {e}")
            return ""

    def update_summary(
        self,
        thread_id: str,
        new_messages: List[str],
        current_summary: str
    ) -> str:
        """
        Update conversation summary with new messages using LLM.

        Args:
            thread_id: Thread identifier
            new_messages: List of new message contents to incorporate
            current_summary: Current summary text

        Returns:
            Updated summary text
        """
        if not self.llm_client:
            print("[update_summary] No LLM client available, returning current summary")
            return current_summary

        if not new_messages:
            return current_summary

        messages_joined = "\n".join([f"- {msg}" for msg in new_messages])

        prompt = f"""Given this existing summary and these new messages, write an updated summary in 2-3 sentences. Keep key facts, update what changed, drop irrelevant details.

Existing summary: {current_summary}

New messages:
{messages_joined}

Write a concise updated summary:"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a conversation summarizer. Write concise 2-3 sentence summaries preserving key information."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            updated_summary = self.llm_client.chat_completion(
                messages,
                temperature=0.3
            )

            # Clean up the response
            updated_summary = updated_summary.strip()

            # Get current message count
            current_count = 0
            try:
                results = self.conversation_summaries_table.search().where(
                    f"thread_id = '{thread_id}'",
                    prefilter=True
                ).limit(1).to_list()
                if results:
                    current_count = results[0].get("message_count", 0)
            except:
                pass

            new_count = current_count + len(new_messages)
            last_updated = datetime.now(timezone.utc).isoformat()

            # Create or update summary record
            vector = self.embedding_model.encode_single(updated_summary, is_query=False)

            # Check if exists
            if current_count > 0 or results:
                # Delete old and insert new (upsert pattern)
                self.conversation_summaries_table.delete(f"thread_id = '{thread_id}'")

            data = {
                "thread_id": thread_id,
                "agent_id": self.agent_id,
                "summary": updated_summary,
                "message_count": new_count,
                "last_updated": last_updated,
                "vector": vector.tolist()
            }

            self.conversation_summaries_table.add([data])
            print(f"[update_summary] Updated summary for {thread_id} ({new_count} messages)")

            return updated_summary

        except Exception as e:
            print(f"[update_summary] Error updating summary: {e}")
            return current_summary

    def get_conversation_summary_by_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full conversation summary record for a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            Dict with summary, message_count, last_updated or None
        """
        try:
            results = self.conversation_summaries_table.search().where(
                f"thread_id = '{thread_id}'",
                prefilter=True
            ).limit(1).to_list()

            if results:
                r = results[0]
                return {
                    "thread_id": r["thread_id"],
                    "summary": r.get("summary", ""),
                    "message_count": r.get("message_count", 0),
                    "last_updated": r.get("last_updated", "")
                }
            return None
        except Exception as e:
            print(f"[get_conversation_summary_by_thread] Error: {e}")
            return None

    # ============================================================
    # Utility Methods
    # ============================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about all tables."""
        return {
            "agent_id": self.agent_id,
            "memories_count": self.memories_table.count_rows(),
            "group_memories_count": self.group_memories_table.count_rows(),
            "user_memories_count": self.user_memories_table.count_rows(),
            "interaction_memories_count": self.interaction_memories_table.count_rows(),
            "cross_group_memories_count": self.cross_group_memories_table.count_rows(),
            "agent_responses_count": self.agent_responses_table.count_rows(),
            "cross_agent_links_count": self.cross_agent_links_table.count_rows(),
            "conversation_summaries_count": self.conversation_summaries_table.count_rows()
        }

    def cleanup_decayed_memories(self, days_threshold: int = 180) -> Dict[str, int]:
        """
        Remove decayed (old) conversation memories from all tables.

        Only removes memories where:
        - last_seen > days_threshold AND
        - memory_type is "conversation" (not "expertise" or "preference" which never expire)

        This is a maintenance job that should be run periodically, not during normal operation.

        Args:
            days_threshold: Age in days after which conversation memories are removed (default 180)

        Returns:
            Dict with counts of removed memories per table
        """
        from datetime import datetime, timezone, timedelta

        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days_threshold)
        cutoff_iso = cutoff_time.isoformat()

        counts = {
            "group_memories": 0,
            "user_memories": 0,
            "interaction_memories": 0,
            "total": 0
        }

        # Clean group_memories
        try:
            if self.group_memories_table.count_rows() > 0:
                # Find old conversation memories
                old_memories = self.group_memories_table.search().where(
                    f"last_seen < '{cutoff_iso}' AND memory_type = 'conversation'",
                    prefilter=True
                ).to_list()

                if old_memories:
                    memory_ids = [m["memory_id"] for m in old_memories]
                    # Delete in batches
                    for memory_id in memory_ids:
                        self.group_memories_table.delete(f"memory_id = '{memory_id}'")
                    counts["group_memories"] = len(memory_ids)
                    print(f"[cleanup] Removed {len(memory_ids)} decayed conversation memories from group_memories")
        except Exception as e:
            print(f"[cleanup] Error cleaning group_memories: {e}")

        # Clean user_memories
        try:
            if self.user_memories_table.count_rows() > 0:
                old_memories = self.user_memories_table.search().where(
                    f"last_seen < '{cutoff_iso}' AND memory_type = 'conversation'",
                    prefilter=True
                ).to_list()

                if old_memories:
                    memory_ids = [m["memory_id"] for m in old_memories]
                    for memory_id in memory_ids:
                        self.user_memories_table.delete(f"memory_id = '{memory_id}'")
                    counts["user_memories"] = len(memory_ids)
                    print(f"[cleanup] Removed {len(memory_ids)} decayed conversation memories from user_memories")
        except Exception as e:
            print(f"[cleanup] Error cleaning user_memories: {e}")

        # Clean interaction_memories
        try:
            if self.interaction_memories_table.count_rows() > 0:
                old_memories = self.interaction_memories_table.search().where(
                    f"last_seen < '{cutoff_iso}' AND memory_type = 'conversation'",
                    prefilter=True
                ).to_list()

                if old_memories:
                    memory_ids = [m["memory_id"] for m in old_memories]
                    for memory_id in memory_ids:
                        self.interaction_memories_table.delete(f"memory_id = '{memory_id}'")
                    counts["interaction_memories"] = len(memory_ids)
                    print(f"[cleanup] Removed {len(memory_ids)} decayed conversation memories from interaction_memories")
        except Exception as e:
            print(f"[cleanup] Error cleaning interaction_memories: {e}")

        counts["total"] = sum([counts["group_memories"], counts["user_memories"], counts["interaction_memories"]])
        print(f"[cleanup] Total removed: {counts['total']} decayed conversation memories")

        return counts

    def _track_memory_access(
        self,
        memory_id: str,
        table_name: str = "group_memories"
    ):
        """
        Track access to a memory by updating access_count.

        This is meant to be called asynchronously/non-blocking after search results.
        The access_count is stored but not currently used in scoring - it's available
        for future features like "frequently accessed memories".

        Args:
            memory_id: ID of the memory that was accessed
            table_name: Table name (default "group_memories")
        """
        try:
            # Map table_name to actual table
            table_map = {
                "group_memories": self.group_memories_table,
                "user_memories": self.user_memories_table,
                "interaction_memories": self.interaction_memories_table,
                "cross_group_memories": self.cross_group_memories_table,
                "memories": self.memories_table
            }

            table = table_map.get(table_name)
            if not table:
                return

            # Check if access_count field exists in schema
            # For now, we'll skip this since adding new fields requires schema migration
            # This is a placeholder for future implementation
            pass

            # TODO: Implement actual access_count tracking when schema is updated
            # This would require:
            # 1. Adding access_count field to table schemas
            # 2. Updating the field here (requires delete + re-add pattern for LanceDB)

        except Exception as e:
            # Silently fail to avoid blocking search operations
            pass

    def track_search_results_access(
        self,
        results: List[Any],
        table_name: str = "group_memories"
    ):
        """
        Track access for multiple search results.

        This should be called after search results are returned.
        It's a fire-and-forget operation that doesn't block.

        Args:
            results: List of memory objects returned from search
            table_name: Table name for the results
        """
        if not results:
            return

        # Extract memory IDs from results
        memory_ids = []
        for item in results:
            if hasattr(item, 'memory_id'):
                memory_ids.append(item.memory_id)
            elif hasattr(item, 'entry_id'):
                memory_ids.append(item.entry_id)

        # Track access for each memory (non-blocking)
        for memory_id in memory_ids:
            self._track_memory_access(memory_id, table_name)

    def clear_agent_data(self):
        """Clear all data for this agent."""
        tables = ["memories", "group_memories", "user_memories",
                  "interaction_memories", "cross_group_memories", "agent_responses",
                  "conversation_summaries"]
        for table_name in tables:
            if table_name in self.agent_db.table_names():
                self.agent_db.drop_table(table_name)
        self._init_agent_db()
        print(f"[{self.agent_id}] Cleared all agent data")

    def clear(self):
        """Clear DM memories only (SimpleMem VectorStore compatible)."""
        if "memories" in self.agent_db.table_names():
            self.agent_db.drop_table("memories")
        self._init_dm_memories_table()
        self._fts_initialized['memories'] = False
        print(f"[{self.agent_id}] Cleared DM memories")

    # ============================================================
    # Memory Consolidation (SimpleMem Stage 2)
    # ============================================================

    def consolidate_similar_memories(
        self,
        group_id: str,
        similarity_threshold: float = 0.80,
        min_cluster_size: int = 2
    ) -> Dict[str, Any]:
        """
        Consolidate similar memories within a group (SimpleMem Stage 2 - Atom to Molecule Evolution).

        Merges related atomic memories into consolidated molecules using:
        1. Embedding similarity to find clusters (cosine > threshold)
        2. LLM to synthesize consolidated content
        3. Transactional delete + insert for each cluster

        Args:
            group_id: Group ID to consolidate
            similarity_threshold: Cosine similarity threshold (0.80 for consolidation, lower than 0.85 dedup)
            min_cluster_size: Minimum memories in a cluster to consolidate

        Returns:
            Dict with consolidation statistics per table
        """
        from utils.llm_client import LLMClient
        import uuid

        results = {
            "group_memories": {"originals": 0, "consolidated": 0},
            "user_memories": {"originals": 0, "consolidated": 0},
            "interaction_memories": {"originals": 0, "consolidated": 0},
            "total_originals": 0,
            "total_consolidated": 0
        }

        # Process each table
        tables_to_process = [
            ("group_memories", self.group_memories_table, "group_id"),
            ("user_memories", self.user_memories_table, "group_id"),
            ("interaction_memories", self.interaction_memories_table, "group_id")
        ]

        for table_name, table, filter_field in tables_to_process:
            try:
                # Get all memories for this group
                where_clause = f"agent_id = '{self.agent_id}' AND {filter_field} = '{group_id}'"
                memories = table.search().where(where_clause, prefilter=True).to_list()

                if len(memories) < min_cluster_size:
                    continue

                # Extract memory IDs and vectors for clustering
                memory_data = []
                for mem in memories:
                    memory_data.append({
                        "id": mem.get("memory_id"),
                        "vector": mem.get("vector", []),
                        "data": mem
                    })

                # Find clusters using greedy approach
                processed_ids = set()
                clusters = []

                for i, mem_i in enumerate(memory_data):
                    if mem_i["id"] in processed_ids:
                        continue

                    # Find similar memories
                    cluster = [mem_i]
                    for j, mem_j in enumerate(memory_data):
                        if i == j or mem_j["id"] in processed_ids:
                            continue

                        similarity = self._compute_cosine_similarity(
                            mem_i["vector"],
                            mem_j["vector"]
                        )

                        if similarity >= similarity_threshold:
                            cluster.append(mem_j)
                            processed_ids.add(mem_j["id"])

                    if len(cluster) >= min_cluster_size:
                        clusters.append(cluster)
                        for mem in cluster:
                            processed_ids.add(mem["id"])

                # Consolidate each cluster
                llm_client = LLMClient()

                for cluster in clusters:
                    # Merge memories using LLM
                    consolidated_memory = self._merge_memory_cluster(
                        cluster,
                        table_name,
                        llm_client
                    )

                    if consolidated_memory:
                        # Delete originals
                        for mem in cluster:
                            table.delete(f"memory_id = '{mem['id']}'")

                        # Insert consolidated
                        if table_name == "group_memories":
                            self.add_group_memory(consolidated_memory)
                        elif table_name == "user_memories":
                            self.add_user_memory(consolidated_memory)
                        elif table_name == "interaction_memories":
                            self.add_interaction_memory(consolidated_memory)

                        results[table_name]["originals"] += len(cluster)
                        results[table_name]["consolidated"] += 1

            except Exception as e:
                print(f"[consolidation] Error processing {table_name} for group {group_id}: {e}")
                continue

        results["total_originals"] = sum(r["originals"] for r in results.values() if isinstance(r, dict))
        results["total_consolidated"] = sum(r["consolidated"] for r in results.values() if isinstance(r, dict))

        print(f"[consolidation] Consolidated {results['total_originals']} memories into {results['total_consolidated']} for group {group_id}")

        return results

    def _merge_memory_cluster(
        self,
        cluster: List[Dict[str, Any]],
        table_name: str,
        llm_client: Any
    ) -> Optional[Any]:
        """
        Merge a cluster of similar memories using LLM.

        Args:
            cluster: List of memory dicts with 'data' key containing full memory
            table_name: Name of table (group_memories, user_memories, interaction_memories)
            llm_client: LLMClient instance

        Returns:
            Consolidated memory object or None
        """
        from models.group_memory import GroupMemory, UserMemory, InteractionMemory
        from datetime import datetime, timezone

        # Build memories description for LLM
        memories_desc = []
        for i, mem in enumerate(cluster, 1):
            data = mem["data"]
            desc = f"\n[Memory {i}]\n"
            desc += f"Content: {data.get('content', 'N/A')}\n"
            desc += f"Keywords: {', '.join(data.get('keywords', []))}\n"
            desc += f"Topics: {', '.join(data.get('topics', []))}\n"

            if table_name == "group_memories":
                if data.get("speaker"):
                    desc += f"Speaker: {data['speaker']}\n"
            elif table_name == "user_memories":
                if data.get("username"):
                    desc += f"Username: {data['username']}\n"
            elif table_name == "interaction_memories":
                desc += f"Speaker: {data.get('speaker_id', 'N/A')}\n"
                desc += f"Listener: {data.get('listener_id', 'N/A')}\n"

            desc += f"First seen: {data.get('first_seen', 'N/A')}\n"
            desc += f"Last seen: {data.get('last_seen', 'N/A')}\n"
            memories_desc.append(desc)

        prompt = f"""Given these related memories, write ONE consolidated memory that captures all the information.

Be specific and factual. Preserve key details like names, dates, locations, and technical terms.
Create a unified narrative that flows naturally.

Memories to consolidate:
{''.join(memories_desc)}

Output ONLY the consolidated memory content (no prefixes, labels, or explanations).
"""

        try:
            consolidated_content = llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation specialist. Merge related memories into clear, factual summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )

            # Extract metadata from cluster
            first_mem = cluster[0]["data"]

            # Aggregate fields
            all_keywords = set()
            all_topics = set()
            total_evidence = 0
            earliest_first = None
            latest_last = None
            max_importance = 0.0

            for mem in cluster:
                data = mem["data"]
                all_keywords.update(data.get("keywords", []))
                all_topics.update(data.get("topics", []))
                total_evidence += data.get("evidence_count", 1)

                if earliest_first is None or data.get("first_seen", "") < earliest_first:
                    earliest_first = data.get("first_seen", "")

                if latest_last is None or data.get("last_seen", "") > latest_last:
                    latest_last = data.get("last_seen", "")

                if data.get("importance_score", 0.0) > max_importance:
                    max_importance = data.get("importance_score", 0.0)

            # Create consolidated memory based on table type
            now = datetime.now(timezone.utc).isoformat()

            if table_name == "group_memories":
                return GroupMemory(
                    memory_id=str(uuid.uuid4()),
                    agent_id=self.agent_id,
                    group_id=first_mem.get("group_id"),
                    memory_level=first_mem.get("memory_level", "group"),
                    memory_type=first_mem.get("memory_type", "conversation"),
                    privacy_scope=first_mem.get("privacy_scope", "public"),
                    content=consolidated_content.strip(),
                    speaker=first_mem.get("speaker"),
                    keywords=list(all_keywords),
                    topics=list(all_topics),
                    importance_score=max_importance,
                    evidence_count=total_evidence,
                    first_seen=earliest_first,
                    last_seen=latest_last,
                    last_updated=now,
                    source_message_id=first_mem.get("source_message_id"),
                    source_timestamp=first_mem.get("source_timestamp")
                )

            elif table_name == "user_memories":
                return UserMemory(
                    memory_id=str(uuid.uuid4()),
                    agent_id=self.agent_id,
                    group_id=first_mem.get("group_id"),
                    user_id=first_mem.get("user_id"),
                    memory_level=first_mem.get("memory_level", "individual"),
                    memory_type=first_mem.get("memory_type", "preference"),
                    privacy_scope=first_mem.get("privacy_scope", "protected"),
                    content=consolidated_content.strip(),
                    keywords=list(all_keywords),
                    topics=list(all_topics),
                    importance_score=max_importance,
                    evidence_count=total_evidence,
                    first_seen=earliest_first,
                    last_seen=latest_last,
                    last_updated=now,
                    source_message_id=first_mem.get("source_message_id"),
                    source_timestamp=first_mem.get("source_timestamp"),
                    username=first_mem.get("username"),
                    platform=first_mem.get("platform")
                )

            elif table_name == "interaction_memories":
                return InteractionMemory(
                    memory_id=str(uuid.uuid4()),
                    agent_id=self.agent_id,
                    group_id=first_mem.get("group_id"),
                    speaker_id=first_mem.get("speaker_id"),
                    listener_id=first_mem.get("listener_id"),
                    mentioned_users=first_mem.get("mentioned_users", []),
                    memory_level=first_mem.get("memory_level", "individual"),
                    memory_type=first_mem.get("memory_type", "interaction"),
                    privacy_scope=first_mem.get("privacy_scope", "protected"),
                    content=consolidated_content.strip(),
                    keywords=list(all_keywords),
                    topics=list(all_topics),
                    importance_score=max_importance,
                    evidence_count=total_evidence,
                    first_seen=earliest_first,
                    last_seen=latest_last,
                    last_updated=now,
                    source_message_id=first_mem.get("source_message_id"),
                    source_timestamp=first_mem.get("source_timestamp"),
                    interaction_type=first_mem.get("interaction_type")
                )

        except Exception as e:
            print(f"[consolidation] Error merging cluster: {e}")
            return None

    def consolidate_all_groups(self) -> Dict[str, Any]:
        """
        Consolidate memories across all groups.

        Returns:
            Dict with overall consolidation statistics
        """
        results = {
            "groups_processed": 0,
            "total_originals": 0,
            "total_consolidated": 0,
            "group_details": {}
        }

        # Get all unique group_ids from each table
        all_group_ids = set()

        try:
            group_memories = self.group_memories_table.search().where(
                f"agent_id = '{self.agent_id}'", prefilter=True
            ).to_list()
            all_group_ids.update(m.get("group_id") for m in group_memories)
        except:
            pass

        try:
            user_memories = self.user_memories_table.search().where(
                f"agent_id = '{self.agent_id}'", prefilter=True
            ).to_list()
            all_group_ids.update(m.get("group_id") for m in user_memories)
        except:
            pass

        try:
            interaction_memories = self.interaction_memories_table.search().where(
                f"agent_id = '{self.agent_id}'", prefilter=True
            ).to_list()
            all_group_ids.update(m.get("group_id") for m in interaction_memories)
        except:
            pass

        # Consolidate each group
        for group_id in all_group_ids:
            if not group_id:
                continue

            group_result = self.consolidate_similar_memories(group_id)
            if group_result["total_originals"] > 0:
                results["groups_processed"] += 1
                results["total_originals"] += group_result["total_originals"]
                results["total_consolidated"] += group_result["total_consolidated"]
                results["group_details"][group_id] = group_result

        print(f"[consolidation] Processed {results['groups_processed']} groups")
        print(f"[consolidation] Total: {results['total_originals']} memories -> {results['total_consolidated']} consolidated")

        return results
