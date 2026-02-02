"""
UserMemories table operations.
Extracted from unified_store.py.
"""
from typing import List, Optional
from datetime import datetime, timezone

import pyarrow as pa

from models.group_memory import UserMemory, MemoryLevel, MemoryType, PrivacyScope
from database.base import LanceDBConnection
from utils.embedding import EmbeddingModel
import config


class UserMemoriesTable:
    """CRUD for user_memories table."""

    def __init__(self, agent_id: str, embedding_model: EmbeddingModel = None, storage_options: dict = None):
        self.agent_id = agent_id
        self.embedding_model = embedding_model or EmbeddingModel()
        self.storage_options = storage_options
        self.db = LanceDBConnection.get_agent_db(agent_id, storage_options=storage_options)
        self.table = self._init_table()
        self._fts_initialized = False

    def _init_table(self):
        """Initialize table with schema."""
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
        if table_name not in self.db.table_names():
            table = self.db.create_table(table_name, schema=schema)
            print(f"[{self.agent_id}] Created {table_name} table")
        else:
            table = self.db.open_table(table_name)
            print(f"[{self.agent_id}] Opened {table_name} ({table.count_rows()} rows)")

        self._create_scalar_index(table, "group_id")
        self._create_scalar_index(table, "user_id")
        self._create_vector_index(table, "user_memories")

        return table

    def _create_scalar_index(self, table, column: str):
        """Create scalar index for fast lookups."""
        try:
            table.create_scalar_index(column, replace=True)
        except Exception:
            pass

    def _create_vector_index(self, table, table_name: str, min_rows: int = 256):
        """Create IVF_PQ vector index."""
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
        except Exception:
            pass

    def add(self, memory: UserMemory, dedup_threshold: float = 0.85) -> UserMemory:
        """Add a single user memory with deduplication."""
        vector = self.embedding_model.encode_single(memory.content, is_query=False)

        # === DEDUPLICACIÓN INSERT-TIME ===
        if self.table.count_rows() > 0:
            try:
                where_clause = f"group_id = '{memory.group_id}' AND user_id = '{memory.user_id}'"
                similar = (
                    self.table.search(vector.tolist())
                    .distance_type("cosine")
                    .where(where_clause, prefilter=True)
                    .limit(1)
                    .to_list()
                )

                if similar:
                    distance = similar[0].get('_distance', 2.0)
                    similarity = 1 - (distance / 2)

                    if similarity >= dedup_threshold:
                        print(f"[dedup] Skip user memory (sim={similarity:.2f}): {memory.content[:40]}...")
                        return memory
            except Exception as e:
                print(f"[dedup] Search failed: {e}")

        # === Insertar si no es duplicado ===
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

        self.table.add([data])
        self._init_fts_index()
        return memory

    def _compute_cosine_similarity(self, vec1, vec2) -> float:
        """Compute cosine similarity."""
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

    def _find_similar_existing(self, vector: List[float], group_id: str, user_id: str = None, threshold: float = 0.85) -> Optional[dict]:
        """Find similar existing memory."""
        try:
            where_clause = f"group_id = '{group_id}'"
            if user_id:
                where_clause += f" AND user_id = '{user_id}'"

            results = (
                self.table.search(vector)
                .distance_type("cosine")
                .where(where_clause, prefilter=True)
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
            print(f"[{self.agent_id}] Error finding similar: {e}")
        return None

    def add_batch(self, memories: List[UserMemory]) -> List[UserMemory]:
        """Add multiple user memories with deduplication."""
        if not memories:
            return []

        contents = [m.content for m in memories]
        vectors = self.embedding_model.encode_documents(contents)

        pending_memories = {}
        merged_count = 0
        new_count = 0

        for memory, vector in zip(memories, vectors):
            key = (memory.group_id, memory.user_id)
            if key not in pending_memories:
                pending_memories[key] = []

            found_duplicate = False
            for existing_memory, existing_vector in pending_memories[key]:
                similarity = self._compute_cosine_similarity(vector, existing_vector)
                if similarity >= 0.85:
                    found_duplicate = True
                    merged_count += 1
                    existing_memory.evidence_count += 1
                    existing_memory.last_seen = memory.last_seen
                    existing_memory.last_updated = datetime.now(timezone.utc).isoformat()
                    break

            if not found_duplicate:
                existing = self._find_similar_existing(vector.tolist(), memory.group_id, memory.user_id, threshold=0.85)
                if existing:
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
                    self.table.delete(f"memory_id = '{existing['memory_id']}'")
                    pending_memories[key].append((updated_memory, vector))
                else:
                    new_count += 1
                    pending_memories[key].append((memory, vector))

        all_data = []
        for memory_list in pending_memories.values():
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
                    "vector": vector.tolist()
                })

        if all_data:
            self.table.add(all_data)
            self._init_fts_index()

        print(f"[{self.agent_id}] Added {new_count} new user memories, merged {merged_count}")
        return memories

    def search_semantic(self, group_id: str, user_id: str, query_vector, limit: int = 10) -> List[UserMemory]:
        """Search by semantic similarity."""
        if self.table.count_rows() == 0:
            return []

        vec = query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector
        search = self.table.search(vec).distance_type("cosine")

        where_clause = f"group_id = '{group_id}' AND user_id = '{user_id}'"
        search = search.where(where_clause, prefilter=True)

        results = search.limit(limit).to_list()
        return [self._row_to_user_memory(r) for r in results]

    def search_semantic_in_group(self, group_id: str, query_vector, limit: int = 10, exclude_user_id: str = None) -> List[UserMemory]:
        """Search all user memories in a group."""
        if self.table.count_rows() == 0:
            return []

        vec = query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector
        search = self.table.search(vec).distance_type("cosine")

        where_clause = f"group_id = '{group_id}'"
        if exclude_user_id:
            where_clause += f" AND user_id != '{exclude_user_id}'"

        search = search.where(where_clause, prefilter=True)

        search = search.select([
            "_distance", "memory_id", "agent_id", "group_id", "user_id",
            "memory_level", "memory_type", "privacy_scope", "content",
            "keywords", "topics", "importance_score", "evidence_count",
            "first_seen", "last_seen", "last_updated",
            "source_message_id", "source_timestamp", "username", "platform"
        ])

        results = search.limit(limit).to_list()
        return [self._row_to_user_memory(r) for r in results]

    def search_by_user(self, user_id: str, query_vector, limit: int = 10) -> List[UserMemory]:
        """Search all user memories for a specific user across ALL groups.

        This is used for DM context to retrieve what the agent knows about a user
        from their interactions in various groups.
        """
        if self.table.count_rows() == 0:
            return []

        vec = query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector
        search = self.table.search(vec).distance_type("cosine")

        # Search by user_id only, across all groups
        where_clause = f"user_id = '{user_id}'"
        search = search.where(where_clause, prefilter=True)

        search = search.select([
            "_distance", "memory_id", "agent_id", "group_id", "user_id",
            "memory_level", "memory_type", "privacy_scope", "content",
            "keywords", "topics", "importance_score", "evidence_count",
            "first_seen", "last_seen", "last_updated",
            "source_message_id", "source_timestamp", "username", "platform"
        ])

        results = search.limit(limit).to_list()
        return [self._row_to_user_memory(r) for r in results]

    def search_keyword(self, group_id: str, keywords: List[str], top_k: int = 5) -> List[tuple]:
        """FTS keyword search with scores."""
        if not keywords or self.table.count_rows() == 0:
            return []

        query = " ".join(keywords)
        try:
            search = self.table.search(query)
            search = search.where(f"group_id = '{group_id}'", prefilter=True)
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
                except Exception:
                    continue

            if max_score > 0:
                scored_entries = [(entry, score / max_score) for entry, score in scored_entries]
            return scored_entries
        except Exception as e:
            print(f"Error during user keyword search: {e}")
            return []

    def _row_to_user_memory(self, row: dict) -> UserMemory:
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
            importance_score=row.get("importance_score", 0.5),
            evidence_count=row.get("evidence_count", 1),
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            last_updated=row["last_updated"],
            source_message_id=row.get("source_message_id"),
            source_timestamp=row.get("source_timestamp"),
            username=row.get("username"),
            platform=row.get("platform")
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
