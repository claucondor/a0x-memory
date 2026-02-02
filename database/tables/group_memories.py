"""
GroupMemories table operations.
Extracted from unified_store.py - will be refactored to standalone.
"""
from typing import List, Optional
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow as pa

from models.group_memory import GroupMemory, MemoryLevel, MemoryType, PrivacyScope
from database.base import LanceDBConnection
from database.tables.group_topics import GroupTopicsTable
from utils.embedding import EmbeddingModel
import config


class GroupMemoriesTable:
    """CRUD for group_memories table."""

    def __init__(self, agent_id: str, embedding_model: EmbeddingModel = None, storage_options: dict = None):
        self.agent_id = agent_id
        self.embedding_model = embedding_model or EmbeddingModel()
        self.storage_options = storage_options
        self.db = LanceDBConnection.get_agent_db(agent_id, storage_options=storage_options)
        self.table = self._init_table()
        self._fts_initialized = False
        # Initialize derived topics table for efficient topic search
        self.topics_table = GroupTopicsTable(agent_id, embedding_model, storage_options)

    def _init_table(self):
        """Initialize table with schema."""
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
        if table_name not in self.db.table_names():
            table = self.db.create_table(table_name, schema=schema)
            print(f"[{self.agent_id}] Created {table_name} table")
        else:
            table = self.db.open_table(table_name)
            print(f"[{self.agent_id}] Opened {table_name} ({table.count_rows()} rows)")

        # Create indices
        self._create_scalar_index(table, "group_id")
        self._create_vector_index(table, "group_memories")

        return table

    def _create_scalar_index(self, table, column: str):
        """Create scalar index for fast lookups."""
        try:
            table.create_scalar_index(column, replace=True)
        except Exception:
            pass  # Index may already exist

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

    def _init_fts_index(self):
        """Initialize Full-Text Search index."""
        if self._fts_initialized:
            return

        try:
            is_cloud_storage = config.LANCEDB_PATH.startswith(("gs://", "s3://", "az://"))
            if is_cloud_storage:
                self.table.create_fts_index("content", use_tantivy=False, replace=True)
            else:
                self.table.create_fts_index("content", use_tantivy=True, tokenizer_name="en_stem", replace=True)
            self._fts_initialized = True
            print(f"[{self.agent_id}] FTS index created for group_memories.content")
        except Exception:
            pass  # Index may already exist or not supported

    def _update_topics_for_memory(self, memory: GroupMemory):
        """Update topics table when a memory is added/updated."""
        if not memory.topics:
            return

        group_id = memory.group_id
        for topic_name in memory.topics:
            try:
                # Generate topic summary from memory content
                summary = f"Discussions about {topic_name} in {group_id}. Recent: {memory.content[:100]}..."
                self.topics_table.upsert_topic(
                    group_id=group_id,
                    name=topic_name,
                    summary=summary,
                    memory_count_delta=1
                )
            except Exception as e:
                print(f"[{self.agent_id}] Error updating topic {topic_name}: {e}")

    def add(self, memory: GroupMemory) -> GroupMemory:
        """Add a single group memory."""
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

        self.table.add([data])
        self._init_fts_index()

        # Update topics index (derived index maintenance)
        self._update_topics_for_memory(memory)

        return memory

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

    def _find_similar_existing(self, vector: List[float], group_id: str, threshold: float = 0.85) -> Optional[dict]:
        """Find a similar existing memory using cosine similarity."""
        try:
            results = (
                self.table.search(vector)
                .distance_type("cosine")
                .where(f"group_id = '{group_id}'", prefilter=True)
                .limit(1)
                .to_list()
            )

            if results:
                result = results[0]
                distance = result.get("_distance", 1.0)
                similarity = 1.0 - distance

                if similarity >= threshold:
                    return result

        except Exception as e:
            print(f"[{self.agent_id}] Error finding similar memory: {e}")

        return None

    def add_batch(self, memories: List[GroupMemory]) -> List[GroupMemory]:
        """Add multiple group memories with deduplication."""
        if not memories:
            return []

        contents = [m.content for m in memories]
        vectors = self.embedding_model.encode_documents(contents)

        pending_memories = {}  # group_id -> list of (memory, vector) tuples
        merged_count = 0
        new_count = 0

        for memory, vector in zip(memories, vectors):
            group_id = memory.group_id

            if group_id not in pending_memories:
                pending_memories[group_id] = []

            # Check against other memories in the same group from this batch
            found_duplicate_in_batch = False
            for existing_memory, existing_vector in pending_memories[group_id]:
                similarity = self._compute_cosine_similarity(vector, existing_vector)
                if similarity >= 0.85:
                    found_duplicate_in_batch = True
                    merged_count += 1
                    existing_memory.evidence_count += 1
                    existing_memory.last_seen = memory.last_seen
                    existing_memory.last_updated = datetime.now(timezone.utc).isoformat()
                    break

            if not found_duplicate_in_batch:
                existing = self._find_similar_existing(vector.tolist(), group_id, threshold=0.85)

                if existing:
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

                    self.table.delete(f"memory_id = '{existing['memory_id']}'")
                    pending_memories[group_id].append((updated_memory, vector))
                else:
                    new_count += 1
                    pending_memories[group_id].append((memory, vector))

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
                    "vector": vector.tolist()
                })

        if all_data:
            self.table.add(all_data)
            self._init_fts_index()

            # Update topics index for all new/updated memories
            for group_id, memory_list in pending_memories.items():
                for memory, _ in memory_list:
                    self._update_topics_for_memory(memory)

        print(f"[{self.agent_id}] Added {new_count} new group memories, merged {merged_count}")
        return memories

    def search_semantic(self, group_id: str, query_vector, limit: int = 10, memory_type: Optional[MemoryType] = None) -> List[GroupMemory]:
        """Search by semantic similarity."""
        if self.table.count_rows() == 0:
            return []

        vec = query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector
        search = self.table.search(vec).distance_type("cosine")

        conditions = [f"group_id = '{group_id}'"]
        if memory_type:
            conditions.append(f"memory_type = '{memory_type.value}'")

        where_clause = " AND ".join(conditions)
        search = search.where(where_clause, prefilter=True)

        search = search.select([
            "_distance", "memory_id", "agent_id", "group_id", "memory_level",
            "memory_type", "privacy_scope", "content", "speaker",
            "keywords", "topics", "importance_score", "evidence_count",
            "first_seen", "last_seen", "last_updated",
            "source_message_id", "source_timestamp"
        ])

        results = search.limit(limit).to_list()
        return [self._row_to_group_memory(r) for r in results]

    def search_keyword(self, group_id: str, keywords: List[str], top_k: int = 5) -> List[tuple]:
        """FTS keyword search with scores."""
        if not keywords or self.table.count_rows() == 0:
            return []

        query = " ".join(keywords)

        try:
            search = self.table.search(query)
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
                except Exception:
                    continue

            if max_score > 0:
                scored_entries = [(entry, score / max_score) for entry, score in scored_entries]

            return scored_entries
        except Exception as e:
            print(f"Error during group keyword search: {e}")
            return []

    def search_by_topic(
        self,
        group_id: str,
        query: str,
        topic_names: Optional[List[str]] = None,
        limit: int = 10
    ) -> dict:
        """
        Search by topic with parallel semantic + topic-based retrieval.

        Args:
            group_id: Group to search within
            query: Search query for semantic search
            topic_names: Optional list of specific topics to filter by
            limit: Max results per search type

        Returns:
            Dict with:
                - topic_results: List of matching topics with similarity
                - memory_results: List of relevant memories
                - combined: Fused results ranked by relevance
        """
        query_vector = self.embedding_model.encode_single(query, is_query=True)

        # Parallel execution of topic and memory searches
        results = {"topic_results": [], "memory_results": [], "combined": []}

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both searches in parallel
            topic_future = executor.submit(
                self.topics_table.search_semantic,
                group_id, query_vector, limit
            )
            memory_future = executor.submit(
                self.search_semantic,
                group_id, query_vector, limit
            )

            # Collect results
            try:
                results["topic_results"] = topic_future.result(timeout=10)
            except Exception as e:
                print(f"[{self.agent_id}] Topic search error: {e}")

            try:
                results["memory_results"] = memory_future.result(timeout=10)
            except Exception as e:
                print(f"[{self.agent_id}] Memory search error: {e}")

        # Filter by specific topics if provided
        if topic_names:
            filtered_topics = [
                t for t in results["topic_results"]
                if t["name"] in topic_names
            ]
            filtered_memories = [
                m for m in results["memory_results"]
                if any(t in topic_names for t in m.topics)
            ]
            results["topic_results"] = filtered_topics
            results["memory_results"] = filtered_memories

        # Combine and rank results
        topic_set = {t["name"] for t in results["topic_results"]}
        combined = []

        # Add memories from matching topics first
        for memory in results["memory_results"]:
            relevance_boost = 0
            matching_topics = [t for t in memory.topics if t in topic_set]
            if matching_topics:
                # Boost relevance if memory has matching topics
                relevance_boost = len(matching_topics) * 0.1
            combined.append({
                "type": "memory",
                "data": memory,
                "relevance_boost": relevance_boost
            })

        # Add topic headers
        for topic in results["topic_results"]:
            combined.append({
                "type": "topic",
                "data": topic,
                "relevance_boost": 0
            })

        # Sort by topic name for consistent results
        results["combined"] = combined

        return results

    def _row_to_group_memory(self, row: dict) -> GroupMemory:
        """Convert LanceDB row to GroupMemory."""
        return GroupMemory(
            memory_id=row["memory_id"],
            agent_id=row["agent_id"],
            group_id=row["group_id"],
            memory_level=MemoryLevel(row["memory_level"]),
            memory_type=MemoryType(row["memory_type"]),
            privacy_scope=PrivacyScope(row["privacy_scope"]),
            content=row["content"],
            speaker=row.get("speaker"),
            keywords=list(row.get("keywords") or []),
            topics=list(row.get("topics") or []),
            importance_score=row.get("importance_score", 0.5),
            evidence_count=row.get("evidence_count", 1),
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            last_updated=row["last_updated"],
            source_message_id=row.get("source_message_id"),
            source_timestamp=row.get("source_timestamp")
        )

    def count(self) -> int:
        """Count all rows."""
        return self.table.count_rows()

    def optimize(self):
        """Compact the table."""
        try:
            self.table.optimize()
        except Exception:
            pass
