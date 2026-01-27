"""
Group Memory Store - Hybrid Arch2 + Arch5 Implementation

Arch2 (Partitioning): Separate tables for each memory type
Arch5 (Hybrid Multi-Level): Cross-group consolidation and adaptive retrieval

Architecture:
- /data/agents/{agent_id}/ - One DB per agent (isolation + scalability)
  - group_memories.lance
  - user_memories.lance
  - interaction_memories.lance
  - cross_group_memories.lance
- /data/global/ - Shared DB for cross-agent linking
  - user_profiles.lance (existing)
  - cross_agent_links.lance (new)
"""
import os
import json
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone
from pathlib import Path

import lancedb
import pyarrow as pa
import numpy as np

from models.group_memory import (
    GroupMemory, UserMemory, InteractionMemory,
    CrossGroupMemory, CrossAgentLink,
    MemoryLevel, MemoryType, PrivacyScope
)
from utils.embedding import EmbeddingModel
import config


class GroupMemoryStore:
    """
    Hybrid multi-level group memory store.

    Implements:
    - Arch2: Partitioned storage by memory type
    - Arch5: Cross-group consolidation and adaptive retrieval
    - Multi-tenant: One DB per agent
    - Cross-agent: Shared global DB for identity linking
    """

    def __init__(
        self,
        agent_id: str,
        db_base_path: str = "/data/a0x-memory",
        embedding_model: EmbeddingModel = None,
        storage_options: Optional[Dict[str, Any]] = None
    ):
        self.agent_id = agent_id
        self.embedding_model = embedding_model or EmbeddingModel()
        self.storage_options = storage_options

        # Paths
        self.db_base_path = db_base_path
        self.agent_db_path = f"{db_base_path}/agents/{agent_id}"
        self.global_db_path = f"{db_base_path}/global"

        # Detect cloud storage
        self._is_cloud_storage = db_base_path.startswith(("gs://", "s3://", "az://"))

        # Initialize databases
        self._init_agent_db()
        self._init_global_db()

        print(f"[GroupMemoryStore] Initialized for agent {agent_id}")
        print(f"  - Agent DB: {self.agent_db_path}")
        print(f"  - Global DB: {self.global_db_path}")

    def _init_agent_db(self):
        """Initialize agent-specific database with partitioned tables."""
        if self._is_cloud_storage:
            self.agent_db = lancedb.connect(self.agent_db_path, storage_options=self.storage_options)
        else:
            os.makedirs(self.agent_db_path, exist_ok=True)
            self.agent_db = lancedb.connect(self.agent_db_path)

        # Initialize partitioned tables
        self._init_group_memories_table()
        self._init_user_memories_table()
        self._init_interaction_memories_table()
        self._init_cross_group_memories_table()

    def _init_global_db(self):
        """Initialize global shared database for cross-agent linking."""
        if self._is_cloud_storage:
            self.global_db = lancedb.connect(self.global_db_path, storage_options=self.storage_options)
        else:
            os.makedirs(self.global_db_path, exist_ok=True)
            self.global_db = lancedb.connect(self.global_db_path)

        # Initialize cross-agent links table
        self._init_cross_agent_links_table()

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
        self._create_fts_index(self.group_memories_table, "content")

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
        self._create_fts_index(self.user_memories_table, "content")

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
        self._create_fts_index(self.interaction_memories_table, "content")

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
        self._create_fts_index(self.cross_group_memories_table, "content")

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

    def _create_scalar_index(self, table, column: str):
        """Create scalar index for fast lookups."""
        try:
            table.create_scalar_index(column, replace=True)
        except Exception as e:
            pass  # Index may already exist

    def _create_fts_index(self, table, column: str):
        """Create full-text search index."""
        try:
            if self._is_cloud_storage:
                table.create_fts_index(column, use_tantivy=False, replace=True)
            else:
                table.create_fts_index(column, use_tantivy=True, replace=True)
        except Exception as e:
            pass  # Index may already exist or not supported

    # ============================================================
    # Arch2: Memory Type Operations
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
        return memory

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
        return memory

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
        return memory

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
        return memory

    # ============================================================
    # Batch Operations - Optimized for Parallel Processing
    # ============================================================

    def add_group_memories_batch(self, memories: List[GroupMemory]) -> List[GroupMemory]:
        """Add multiple group-level memories with batch embeddings.

        Performance: ~100ms for N memories vs ~100ms*N for individual calls.
        """
        if not memories:
            return []

        # Batch encode all contents at once
        contents = [m.content for m in memories]
        vectors = self.embedding_model.encode_documents(contents)

        # Prepare all data
        all_data = []
        for memory, vector in zip(memories, vectors):
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
                "vector": vector.tolist()
            })

        self.group_memories_table.add(all_data)
        return memories

    def add_user_memories_batch(self, memories: List[UserMemory]) -> List[UserMemory]:
        """Add multiple user-level memories with batch embeddings.

        Performance: ~100ms for N memories vs ~100ms*N for individual calls.
        """
        if not memories:
            return []

        # Batch encode all contents at once
        contents = [m.content for m in memories]
        vectors = self.embedding_model.encode_documents(contents)

        # Prepare all data
        all_data = []
        for memory, vector in zip(memories, vectors):
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
                "vector": vector.tolist()
            })

        self.user_memories_table.add(all_data)
        return memories

    def add_interaction_memories_batch(self, memories: List[InteractionMemory]) -> List[InteractionMemory]:
        """Add multiple interaction memories with batch embeddings.

        Performance: ~100ms for N memories vs ~100ms*N for individual calls.
        """
        if not memories:
            return []

        # Batch encode all contents at once
        contents = [m.content for m in memories]
        vectors = self.embedding_model.encode_documents(contents)

        # Prepare all data
        all_data = []
        for memory, vector in zip(memories, vectors):
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
                "vector": vector.tolist()
            })

        self.interaction_memories_table.add(all_data)
        return memories

    # ============================================================
    # Arch2: Search Operations
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

        # Apply filters
        conditions = [f"group_id = '{group_id}'"]
        if memory_type:
            conditions.append(f"memory_type = '{memory_type.value}'")

        where_clause = " AND ".join(conditions)
        search = search.where(where_clause, prefilter=True)

        results = search.limit(limit).to_list()
        return [self._row_to_group_memory(r) for r in results]

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
        return [self._row_to_user_memory(r) for r in results]

    def search_user_memories_in_group(
        self,
        group_id: str,
        query: str,
        limit: int = 10,
        exclude_user_id: Optional[str] = None
    ) -> List[UserMemory]:
        """
        Search user memories by semantic similarity across ALL users in a group.

        Use this when someone asks about another user - search the entire group's memories.

        Args:
            group_id: The group ID
            query: The search query
            limit: Max results
            exclude_user_id: Optional user ID to exclude (e.g., the asker)
        """
        if self.user_memories_table.count_rows() == 0:
            return []

        query_vector = self.embedding_model.encode_single(query, is_query=True)
        search = self.user_memories_table.search(query_vector.tolist())

        where_clause = f"group_id = '{group_id}'"
        if exclude_user_id:
            where_clause += f" AND user_id != '{exclude_user_id}'"

        search = search.where(where_clause, prefilter=True)

        results = search.limit(limit).to_list()
        return [self._row_to_user_memory(r) for r in results]

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

        return [self._row_to_interaction_memory(r) for r in results]

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
        return [self._row_to_cross_group_memory(r) for r in results]

    # ============================================================
    # Arch5: Context Retrieval (Multi-level)
    # ============================================================

    def get_group_context(
        self,
        group_id: str,
        user_id: Optional[str] = None,
        query: Optional[str] = None,
        limit_per_level: int = 5
    ) -> Dict[str, List[Any]]:
        """
        Get comprehensive group context with multi-level retrieval.

        Returns:
            {
                "group_context": List[GroupMemory],
                "user_context": List[UserMemory] (if user_id provided),
                "interaction_context": List[InteractionMemory] (if user_id provided),
                "cross_group_context": List[CrossGroupMemory] (if user_id provided)
            }
        """
        context = {
            "group_context": [],
            "user_context": [],
            "interaction_context": [],
            "cross_group_context": []
        }

        # Group-level memories
        if query:
            context["group_context"] = self.search_group_memories(group_id, query, limit_per_level)
        else:
            context["group_context"] = self._get_recent_group_memories(group_id, limit_per_level)

        if user_id:
            # User-level memories
            # When there's a query, search ALL users in the group (not just the asker)
            # This allows finding info about other users when someone asks about them
            if query:
                context["user_context"] = self.search_user_memories_in_group(group_id, query, limit_per_level, exclude_user_id=user_id)
            else:
                context["user_context"] = self._get_recent_user_memories(group_id, user_id, limit_per_level)

            # Interaction memories
            context["interaction_context"] = self.search_interactions(group_id, user_id, limit=limit_per_level)

            # Cross-group context
            universal_id = f"telegram:{user_id}"  # Assuming telegram for now
            if query:
                context["cross_group_context"] = self.search_cross_group(universal_id, query, limit_per_level)
            else:
                context["cross_group_context"] = self._get_recent_cross_group_memories(universal_id, limit_per_level)

        return context

    def _get_recent_group_memories(self, group_id: str, limit: int) -> List[GroupMemory]:
        """Get recent group memories ordered by last_seen."""
        results = self.group_memories_table.search().where(
            f"group_id = '{group_id}'", prefilter=True
        ).limit(limit).to_list()
        return [self._row_to_group_memory(r) for r in results]

    def _get_recent_user_memories(self, group_id: str, user_id: str, limit: int) -> List[UserMemory]:
        """Get recent user memories."""
        results = self.user_memories_table.search().where(
            f"group_id = '{group_id}' AND user_id = '{user_id}'", prefilter=True
        ).limit(limit).to_list()
        return [self._row_to_user_memory(r) for r in results]

    def _get_recent_cross_group_memories(self, universal_user_id: str, limit: int) -> List[CrossGroupMemory]:
        """Get recent cross-group memories."""
        results = self.cross_group_memories_table.search().where(
            f"universal_user_id = '{universal_user_id}'", prefilter=True
        ).limit(limit).to_list()
        return [self._row_to_cross_group_memory(r) for r in results]

    # ============================================================
    # Arch5: Cross-Group Consolidation
    # ============================================================

    def detect_cross_group_patterns(
        self,
        user_id: str,
        min_groups: int = 2,
        min_evidence: int = 2
    ) -> List[CrossGroupMemory]:
        """
        Detect patterns across multiple groups for consolidation.

        Args:
            user_id: User to analyze
            min_groups: Minimum number of groups to qualify as cross-group
            min_evidence: Minimum evidence count per group

        Returns:
            List of consolidated cross-group memories
        """
        # Get all user memories across groups
        universal_id = f"telegram:{user_id}"
        user_memories = self.user_memories_table.search().where(
            f"user_id = '{user_id}'", prefilter=True
        ).to_list()

        if not user_memories:
            return []

        # Group memories by topic and memory type
        patterns = self._analyze_patterns(user_memories)

        # Filter by min_groups
        consolidated = []
        for pattern_key, pattern_data in patterns.items():
            if len(pattern_data["groups"]) >= min_groups and pattern_data["total_evidence"] >= min_evidence:
                cross_group_memory = self.consolidate_cross_group_memory(pattern_data)
                if cross_group_memory:
                    consolidated.append(cross_group_memory)

        return consolidated

    def _analyze_patterns(self, user_memories: List[Dict]) -> Dict[str, Dict]:
        """Analyze memories to find cross-group patterns."""
        patterns = {}

        for memory_row in user_memories:
            memory = self._row_to_user_memory(memory_row)

            # Create pattern key from memory type and topics
            if not memory.topics:
                continue

            primary_topic = memory.topics[0] if memory.topics else "general"
            pattern_key = f"{memory.memory_type.value}:{primary_topic}"

            if pattern_key not in patterns:
                patterns[pattern_key] = {
                    "memory_type": memory.memory_type,
                    "topics": memory.topics,
                    "keywords": set(),
                    "groups": set(),
                    "total_evidence": 0,
                    "first_seen": memory.first_seen,
                    "last_seen": memory.last_seen,
                    "source_memory_ids": [],
                    "contents": []
                }

            pattern = patterns[pattern_key]
            pattern["keywords"].update(memory.keywords)
            pattern["groups"].add(memory.group_id)
            pattern["total_evidence"] += memory.evidence_count
            pattern["source_memory_ids"].append(memory.memory_id)
            pattern["contents"].append(memory.content)

            # Update timestamps
            if memory.first_seen < pattern["first_seen"]:
                pattern["first_seen"] = memory.first_seen
            if memory.last_seen > pattern["last_seen"]:
                pattern["last_seen"] = memory.last_seen

        return patterns

    def consolidate_cross_group_memory(self, pattern: Dict) -> Optional[CrossGroupMemory]:
        """Create a cross-group memory from pattern data."""
        if len(pattern["groups"]) < 2:
            return None

        # Generate consolidated content
        consolidated_content = self._generate_consolidated_content(pattern)

        # Calculate confidence
        confidence = min(1.0, pattern["total_evidence"] / 10.0)

        # Determine pattern type
        pattern_type = self._determine_pattern_type(pattern)

        return CrossGroupMemory(
            agent_id=self.agent_id,
            universal_user_id=f"telegram:{pattern['source_memory_ids'][0]}",  # Simplified
            user_identities=list(pattern["groups"]),  # Simplified
            groups_involved=list(pattern["groups"]),
            group_count=len(pattern["groups"]),
            memory_type=pattern["memory_type"],
            content=consolidated_content,
            keywords=list(pattern["keywords"]),
            topics=pattern["topics"],
            confidence_score=confidence,
            pattern_type=pattern_type,
            evidence_count=pattern["total_evidence"],
            first_seen=pattern["first_seen"],
            last_seen=pattern["last_seen"],
            source_memory_ids=pattern["source_memory_ids"]
        )

    def _generate_consolidated_content(self, pattern: Dict) -> str:
        """Generate consolidated content from pattern data."""
        memory_type = pattern["memory_type"].value
        topics = ", ".join(pattern["topics"][:2])
        group_count = len(pattern["groups"])

        if memory_type == "expertise":
            return f"User demonstrated expertise in {topics} across {group_count} groups"
        elif memory_type == "preference":
            return f"User showed consistent preferences for {topics} across {group_count} groups"
        elif memory_type == "fact":
            return f"User mentioned facts about {topics} in multiple groups"
        else:
            return f"User engaged with {topics} across {group_count} groups"

    def _determine_pattern_type(self, pattern: Dict) -> str:
        """Determine pattern type from memory type and content."""
        memory_type = pattern["memory_type"].value
        if memory_type == "expertise":
            return "expertise"
        elif memory_type == "preference":
            return "preference"
        elif memory_type == "conversation":
            return "behavior"
        else:
            return "general"

    def update_cross_group_memory(self, memory_id: str, new_evidence: Dict) -> Optional[CrossGroupMemory]:
        """Update an existing cross-group memory with new evidence."""
        results = self.cross_group_memories_table.search().where(
            f"memory_id = '{memory_id}'", prefilter=True
        ).to_list()

        if not results:
            return None

        existing = self._row_to_cross_group_memory(results[0])

        # Update with new evidence
        existing.last_seen = new_evidence.get("last_seen", existing.last_seen)
        existing.last_updated = datetime.now(timezone.utc).isoformat()
        existing.evidence_count += new_evidence.get("evidence_count", 1)
        existing.confidence_score = min(1.0, existing.confidence_score + 0.05)

        # Delete old and insert updated
        self.cross_group_memories_table.delete(f"memory_id = '{memory_id}'")
        self.add_cross_group_memory(existing)

        return existing

    # ============================================================
    # Cross-Agent Linking
    # ============================================================

    def link_user_across_agents(
        self,
        agent1_id: str,
        agent2_id: str,
        universal_user_id: str,
        linking_method: str = "manual",
        linking_evidence: Optional[List[str]] = None
    ) -> CrossAgentLink:
        """Create or update a cross-agent identity link."""
        # Check if link exists
        existing = self._get_cross_agent_link(universal_user_id)

        if existing:
            # Update existing link
            existing.add_agent_mapping(agent2_id, universal_user_id)
            existing.last_verified = datetime.now(timezone.utc).isoformat()
            existing.evidence_count += 1

            # Delete and re-insert
            self.cross_agent_links_table.delete(f"universal_user_id = '{universal_user_id}'")
            self._add_cross_agent_link(existing)
            return existing
        else:
            # Create new link
            link = CrossAgentLink(
                universal_user_id=universal_user_id,
                agent_mappings={
                    agent1_id: universal_user_id,
                    agent2_id: universal_user_id
                },
                linking_method=linking_method,
                linking_evidence=linking_evidence or []
            )
            self._add_cross_agent_link(link)
            return link

    def get_agent_mappings(self, universal_user_id: str) -> Dict[str, str]:
        """Get all agent mappings for a universal user ID."""
        link = self._get_cross_agent_link(universal_user_id)
        if link:
            return link.agent_mappings
        return {}

    def _get_cross_agent_link(self, universal_user_id: str) -> Optional[CrossAgentLink]:
        """Get cross-agent link by universal user ID."""
        results = self.cross_agent_links_table.search().where(
            f"universal_user_id = '{universal_user_id}'", prefilter=True
        ).to_list()

        if not results:
            return None

        return self._row_to_cross_agent_link(results[0])

    def _add_cross_agent_link(self, link: CrossAgentLink):
        """Add cross-agent link to database."""
        data = {
            "link_id": link.link_id,
            "universal_user_id": link.universal_user_id,
            "agent_mappings": json.dumps(link.agent_mappings),
            "linking_confidence": link.linking_confidence,
            "evidence_count": link.evidence_count,
            "first_linked": link.first_linked,
            "last_updated": link.last_updated,
            "last_verified": link.last_verified,
            "linking_method": link.linking_method,
            "linking_evidence": link.linking_evidence
        }

        self.cross_agent_links_table.add([data])

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

    def _row_to_cross_agent_link(self, row: Dict) -> CrossAgentLink:
        """Convert LanceDB row to CrossAgentLink."""
        return CrossAgentLink(
            link_id=row["link_id"],
            universal_user_id=row["universal_user_id"],
            agent_mappings=json.loads(row["agent_mappings"]),
            linking_confidence=row["linking_confidence"],
            evidence_count=row["evidence_count"],
            first_linked=row["first_linked"],
            last_updated=row["last_updated"],
            last_verified=row["last_verified"],
            linking_method=row["linking_method"],
            linking_evidence=list(row.get("linking_evidence") or [])
        )

    # ============================================================
    # Utility Methods
    # ============================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the store."""
        return {
            "agent_id": self.agent_id,
            "group_memories_count": self.group_memories_table.count_rows(),
            "user_memories_count": self.user_memories_table.count_rows(),
            "interaction_memories_count": self.interaction_memories_table.count_rows(),
            "cross_group_memories_count": self.cross_group_memories_table.count_rows(),
            "cross_agent_links_count": self.cross_agent_links_table.count_rows()
        }

    def clear_agent_data(self):
        """Clear all data for this agent."""
        self.agent_db.drop_table("group_memories")
        self.agent_db.drop_table("user_memories")
        self.agent_db.drop_table("interaction_memories")
        self.agent_db.drop_table("cross_group_memories")
        self._init_agent_db()
        print(f"[{self.agent_id}] Cleared all agent data")
