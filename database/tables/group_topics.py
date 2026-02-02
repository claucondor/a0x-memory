"""
Group Topics table - derived index for topic-based search.

This table is DERIVED from group_memories.topics - it's automatically
maintained as a topic index for efficient semantic topic search.

Topics are created/updated when group memories are added, and provide:
1. Semantic topic search (by name/summary)
2. Memory counts per topic
3. Topic evolution tracking
"""
from typing import List, Optional
from datetime import datetime, timezone

import pyarrow as pa

from models.group_memory import GroupMemory
from database.base import LanceDBConnection
from utils.embedding import EmbeddingModel


class GroupTopicsTable:
    """CRUD for group_topics table (derived topic index)."""

    def __init__(self, agent_id: str, embedding_model: EmbeddingModel = None, storage_options: dict = None):
        self.agent_id = agent_id
        self.embedding_model = embedding_model or EmbeddingModel()
        self.storage_options = storage_options
        self.db = LanceDBConnection.get_agent_db(agent_id, storage_options=storage_options)
        self.table = self._init_table()

    def _init_table(self):
        """Initialize table with schema."""
        schema = pa.schema([
            pa.field("topic_id", pa.string()),
            pa.field("agent_id", pa.string()),
            pa.field("group_id", pa.string()),
            pa.field("name", pa.string()),  # "defi", "farming", "uniswap"
            pa.field("summary", pa.string()),  # Auto-generated summary
            pa.field("memory_count", pa.int32()),
            pa.field("last_seen", pa.string()),
            pa.field("last_updated", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self.embedding_model.dimension))
        ])

        table_name = "group_topics"
        if table_name not in self.db.table_names():
            table = self.db.create_table(table_name, schema=schema)
            print(f"[{self.agent_id}] Created {table_name} table")
        else:
            table = self.db.open_table(table_name)
            print(f"[{self.agent_id}] Opened {table_name} ({table.count_rows()} rows)")

        # Create vector index for semantic search
        self._create_vector_index(table, "group_topics")

        return table

    def _create_vector_index(self, table, table_name: str, min_rows: int = 256):
        """Create IVF_PQ vector index for fast ANN search."""
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
            print(f"[{self.agent_id}] Vector index created for {table_name} ({row_count} rows)")
        except Exception:
            pass  # Index may already exist or not enough data

    def upsert_topic(
        self,
        group_id: str,
        name: str,
        summary: Optional[str] = None,
        memory_count_delta: int = 1
    ) -> str:
        """
        Create or update a topic entry.

        Args:
            group_id: Group identifier
            name: Topic name (e.g., "defi", "farming")
            summary: Optional summary (auto-generated if None)
            memory_count_delta: Increment to memory_count

        Returns:
            topic_id
        """
        topic_id = f"{group_id}:{name}"

        # Check if topic exists
        existing = self.table.search().where(
            f"group_id = '{group_id}' AND name = '{name}'"
        ).to_list()

        if existing:
            # Update existing topic
            existing_row = existing[0]
            current_count = existing_row.get("memory_count", 0)
            new_count = current_count + memory_count_delta

            self.table.delete(f"topic_id = '{topic_id}'")

            # Generate summary if not provided
            if not summary:
                summary = f"Discussions about {name} in {group_id}"

            # Updated topic
            new_vector = self.embedding_model.encode_single(summary, is_query=False)

            data = {
                "topic_id": topic_id,
                "agent_id": self.agent_id,
                "group_id": group_id,
                "name": name,
                "summary": summary,
                "memory_count": new_count,
                "last_seen": datetime.now(timezone.utc).isoformat(),
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "vector": new_vector.tolist()
            }
        else:
            # Create new topic
            if not summary:
                summary = f"Discussions about {name} in {group_id}"

            topic_vector = self.embedding_model.encode_single(f"{name}: {summary}", is_query=False)

            data = {
                "topic_id": topic_id,
                "agent_id": self.agent_id,
                "group_id": group_id,
                "name": name,
                "summary": summary,
                "memory_count": memory_count_delta,
                "last_seen": datetime.now(timezone.utc).isoformat(),
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "vector": topic_vector.tolist()
            }

        self.table.add([data])
        return topic_id

    def search_semantic(self, group_id: str, query_vector, limit: int = 5) -> List[dict]:
        """
        Search topics by semantic similarity.

        Args:
            group_id: Group to search within
            query_vector: Query embedding
            limit: Max results

        Returns:
            List of topic dicts with name, summary, memory_count
        """
        if self.table.count_rows() == 0:
            return []

        vec = query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector
        search = self.table.search(vec).distance_type("cosine")

        where_clause = f"group_id = '{group_id}'"
        search = search.where(where_clause, prefilter=True)

        results = search.limit(limit).to_list()

        return [
            {
                "topic_id": r["topic_id"],
                "name": r["name"],
                "summary": r["summary"],
                "memory_count": r.get("memory_count", 0),
                "_distance": 1.0 - r.get("_distance", 1.0)  # Convert distance to similarity
            }
            for r in results
        ]

    def get_topics_for_group(self, group_id: str) -> List[dict]:
        """Get all topics for a group, sorted by memory count."""
        if self.table.count_rows() == 0:
            return []

        results = self.table.search().where(
            f"group_id = '{group_id}'"
        ).to_list()

        # Sort by memory_count descending
        results.sort(key=lambda x: x.get("memory_count", 0), reverse=True)

        return [
            {
                "topic_id": r["topic_id"],
                "name": r["name"],
                "summary": r["summary"],
                "memory_count": r.get("memory_count", 0),
                "last_seen": r.get("last_seen"),
                "last_updated": r.get("last_updated"),
            }
            for r in results
        ]

    def get_topic_memory_count(self, group_id: str, topic_name: str) -> int:
        """Get memory count for a specific topic."""
        result = self.table.search().where(
            f"group_id = '{group_id}' AND name = '{topic_name}'"
        ).to_list()

        if result:
            return result[0].get("memory_count", 0)
        return 0

    def batch_update_memory_counts(self, group_id: str, topic_updates: dict[str, int]):
        """
        Batch update memory counts for multiple topics.

        Args:
            group_id: Group identifier
            topic_updates: {topic_name: count_delta}
        """
        if not topic_updates:
            return

        for topic_name, count_delta in topic_updates.items():
            if count_delta != 0:
                topic_id = f"{group_id}:{topic_name}"
                existing = self.table.search().where(
                    f"topic_id = '{topic_id}'"
                ).to_list()

                if existing:
                    current_count = existing[0].get("memory_count", 0)
                    new_count = current_count + count_delta

                    self.table.delete(f"topic_id = '{topic_id}'")

                    # Re-insert with updated count
                    # (simplified - in production would preserve other fields)
                    if new_count > 0:
                        self.upsert_topic(group_id, topic_name, memory_count_delta=new_count)

    def count(self) -> int:
        """Count all topics."""
        return self.table.count_rows()

    def optimize(self):
        """Compact the table."""
        try:
            self.table.optimize()
        except Exception:
            pass
