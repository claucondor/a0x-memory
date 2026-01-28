"""
CrossGroupMemories table operations.
Extracted from unified_store.py.
"""
from typing import List

import pyarrow as pa

from models.group_memory import CrossGroupMemory, MemoryLevel, MemoryType, PrivacyScope
from database.base import LanceDBConnection
from utils.embedding import EmbeddingModel
import config


class CrossGroupMemoriesTable:
    """CRUD for cross_group_memories table."""

    def __init__(self, agent_id: str, embedding_model: EmbeddingModel = None, storage_options: dict = None):
        self.agent_id = agent_id
        self.embedding_model = embedding_model or EmbeddingModel()
        self.storage_options = storage_options
        self.db = LanceDBConnection.get_agent_db(agent_id, storage_options=storage_options)
        self.table = self._init_table()

    def _init_table(self):
        """Initialize table with schema."""
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
        if table_name not in self.db.table_names():
            table = self.db.create_table(table_name, schema=schema)
            print(f"[{self.agent_id}] Created {table_name} table")
        else:
            table = self.db.open_table(table_name)
            print(f"[{self.agent_id}] Opened {table_name} ({table.count_rows()} rows)")

        self._create_scalar_index(table, "universal_user_id")
        self._create_vector_index(table, "cross_group_memories")

        return table

    def _create_scalar_index(self, table, column: str):
        """Create scalar index."""
        try:
            table.create_scalar_index(column, replace=True)
        except Exception:
            pass

    def _create_vector_index(self, table, table_name: str, min_rows: int = 256):
        """Create vector index."""
        try:
            row_count = table.count_rows()
            if row_count < min_rows:
                return
            table.create_index(
                metric="cosine",
                num_partitions=min(row_count // 20, 64),
                num_sub_vectors=min(self.embedding_model.dimension // 16, 24),
                replace=True
            )
        except Exception:
            pass

    def add(self, memory: CrossGroupMemory) -> CrossGroupMemory:
        """Add a cross-group memory."""
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

        self.table.add([data])
        return memory

    def count(self) -> int:
        """Count all rows."""
        return self.table.count_rows()

    def optimize(self):
        """Compact the table."""
        try:
            self.table.optimize()
        except Exception:
            pass

    def search_semantic(self, universal_user_id: str, query_vector, limit: int = 10) -> List[CrossGroupMemory]:
        """Search cross-group memories by semantic similarity."""
        try:
            results = self.table.search(query_vector).where(
                f"universal_user_id = '{universal_user_id}' AND agent_id = '{self.agent_id}'",
                prefilter=True
            ).limit(limit).to_list()
            return [self._row_to_cross_group_memory(r) for r in results]
        except Exception as e:
            print(f"[CrossGroupMemories] Search error: {e}")
            return []

    def _row_to_cross_group_memory(self, row: dict) -> CrossGroupMemory:
        """Convert LanceDB row to CrossGroupMemory."""
        return CrossGroupMemory(
            memory_id=row["memory_id"],
            agent_id=row["agent_id"],
            universal_user_id=row["universal_user_id"],
            user_identities=list(row.get("user_identities") or []),
            groups_involved=list(row.get("groups_involved") or []),
            group_count=row.get("group_count", 1),
            memory_level=MemoryLevel(row["memory_level"]),
            memory_type=MemoryType(row["memory_type"]),
            privacy_scope=PrivacyScope(row["privacy_scope"]),
            content=row["content"],
            keywords=list(row.get("keywords") or []),
            topics=list(row.get("topics") or []),
            confidence_score=row.get("confidence_score", 0.5),
            pattern_type=row.get("pattern_type", "unknown"),
            evidence_count=row.get("evidence_count", 1),
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            last_updated=row.get("last_updated"),
            consolidated_at=row.get("consolidated_at"),
            source_memory_ids=list(row.get("source_memory_ids") or [])
        )
