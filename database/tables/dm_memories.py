"""
DMMemories table operations (SimpleMem compatible).
Extracted from unified_store.py.
"""
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa

from models.memory_entry import MemoryEntry, MemoryType as DM_MemoryType, PrivacyScope as DM_PrivacyScope
from database.base import LanceDBConnection
from database.tables.dm_topics import DMTopicsTable
from utils.embedding import EmbeddingModel
import config


class DMMemoriesTable:
    """CRUD for dm_memories table (memories.lance - SimpleMem compatible)."""

    def __init__(self, agent_id: str, embedding_model: EmbeddingModel = None, storage_options: dict = None):
        self.agent_id = agent_id
        self.embedding_model = embedding_model or EmbeddingModel()
        self.storage_options = storage_options
        self.db = LanceDBConnection.get_agent_db(agent_id, storage_options=storage_options)
        self.table = self._init_table()
        self._fts_initialized = False
        # Initialize derived topics table for efficient topic search
        self.topics_table = DMTopicsTable(agent_id, embedding_model, storage_options)

    def _init_table(self):
        """Initialize table with schema."""
        schema = pa.schema([
            pa.field("agent_id", pa.string()),
            pa.field("user_id", pa.string()),
            pa.field("entry_id", pa.string()),
            pa.field("lossless_restatement", pa.string()),
            pa.field("keywords", pa.list_(pa.string())),
            pa.field("timestamp", pa.string()),
            pa.field("location", pa.string()),
            pa.field("persons", pa.list_(pa.string())),
            pa.field("entities", pa.list_(pa.string())),
            pa.field("topic", pa.string()),
            pa.field("group_id", pa.string()),
            pa.field("username", pa.string()),
            pa.field("platform", pa.string()),
            pa.field("memory_type", pa.string()),
            pa.field("privacy_scope", pa.string()),
            pa.field("importance_score", pa.float32()),
            pa.field("is_shareable", pa.bool_()),
            pa.field("vector", pa.list_(pa.float32(), self.embedding_model.dimension))
        ])

        table_name = "memories"
        if table_name not in self.db.table_names():
            table = self.db.create_table(table_name, schema=schema)
            print(f"[{self.agent_id}] Created {table_name} table")
        else:
            table = self.db.open_table(table_name)
            print(f"[{self.agent_id}] Opened {table_name} ({table.count_rows()} rows)")

        self._create_vector_index(table, "memories")
        return table

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

    def _init_fts_index(self):
        """Initialize FTS index."""
        if self._fts_initialized:
            return
        try:
            is_cloud_storage = config.LANCEDB_PATH.startswith(("gs://", "s3://", "az://"))
            if is_cloud_storage:
                self.table.create_fts_index("lossless_restatement", use_tantivy=False, replace=True)
            else:
                self.table.create_fts_index("lossless_restatement", use_tantivy=True, tokenizer_name="en_stem", replace=True)
            self._fts_initialized = True
        except Exception:
            pass

    def _update_topics_for_entry(self, entry: MemoryEntry, user_id: str = None):
        """Update topics table when a DM memory is added/updated."""
        if not entry.topic:
            return

        uid = entry.user_id or user_id or "unknown"

        try:
            # Generate topic summary from memory content
            summary = f"Discussions about {entry.topic} with {uid}. Recent: {entry.lossless_restatement[:100]}..."
            self.topics_table.upsert_topic(
                user_id=uid,
                name=entry.topic,
                summary=summary,
                memory_count_delta=1
            )
        except Exception as e:
            print(f"[{self.agent_id}] Error updating topic {entry.topic}: {e}")

    def add_batch(self, entries: List[MemoryEntry], user_id: str = None, dedup_threshold: float = 0.85):
        """Add DM memory entries with deduplication."""
        if not entries:
            return

        restatements = [entry.lossless_restatement for entry in entries]
        vectors = self.embedding_model.encode_documents(restatements)

        # === DEDUPLICACIÓN INSERT-TIME ===
        entries_to_insert = []
        vectors_to_insert = []
        duplicates_skipped = 0

        for entry, vector in zip(entries, vectors):
            is_duplicate = False

            if self.table.count_rows() > 0:
                dm_group_id = entry.group_id or f"dm_{entry.user_id or user_id or 'unknown'}"

                # Buscar memorias similares existentes usando LanceDB nativo
                try:
                    similar = (
                        self.table.search(vector.tolist())
                        .distance_type("cosine")
                        .where(f"agent_id = '{self.agent_id}' AND group_id = '{dm_group_id}'", prefilter=True)
                        .limit(1)
                        .to_list()
                    )

                    if similar:
                        # cosine distance en LanceDB: 0 = idéntico, 2 = opuesto
                        # similarity = 1 - (distance / 2) para normalizar a [0,1]
                        distance = similar[0].get('_distance', 2.0)
                        similarity = 1 - (distance / 2)

                        if similarity >= dedup_threshold:
                            is_duplicate = True
                            duplicates_skipped += 1
                            print(f"[dedup] Skip (sim={similarity:.2f}): {entry.lossless_restatement[:40]}...")
                except Exception as e:
                    print(f"[dedup] Search failed: {e}")

            if not is_duplicate:
                entries_to_insert.append(entry)
                vectors_to_insert.append(vector)

        if not entries_to_insert:
            print(f"[{self.agent_id}] All {len(entries)} entries were duplicates")
            return

        # === Insertar solo las no-duplicadas ===
        data = []
        for entry, vector in zip(entries_to_insert, vectors_to_insert):
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
                "is_shareable": entry.is_shareable,
                "vector": vector.tolist()
            })

        self.table.add(data)
        print(f"[{self.agent_id}] Added {len(entries_to_insert)}/{len(entries)} DM entries (skipped {duplicates_skipped} duplicates)")
        self._init_fts_index()

        # Update topics index for all entries
        for entry in entries:
            self._update_topics_for_entry(entry, user_id)

    def search_semantic(self, query: str, user_id: str = None, top_k: int = 10, query_vector=None) -> List[MemoryEntry]:
        """Search DM memories."""
        if self.table.count_rows() == 0:
            return []

        if query_vector is None:
            query_vector = self.embedding_model.encode_single(query, is_query=True)
        vec = query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector
        search = self.table.search(vec).distance_type("cosine")

        conditions = [f"agent_id = '{self.agent_id}'"]
        if user_id:
            conditions.append(f"user_id = '{user_id}'")

        where_clause = " AND ".join(conditions)
        search = search.where(where_clause, prefilter=True)

        search = search.select([
            "_distance", "entry_id", "lossless_restatement", "keywords", "timestamp",
            "location", "persons", "entities", "topic", "group_id",
            "user_id", "username", "platform", "memory_type",
            "privacy_scope", "importance_score", "is_shareable"
        ])

        results = search.limit(top_k).to_list()
        return [self._row_to_memory_entry(r) for r in results]

    def search_keyword(self, keywords: List[str], top_k: int = 3, user_id: str = None) -> List[MemoryEntry]:
        """Keyword search."""
        if not keywords or self.table.count_rows() == 0:
            return []

        query = " ".join(keywords)
        try:
            search = self.table.search(query)
            conditions = [f"agent_id = '{self.agent_id}'"]
            if user_id:
                conditions.append(f"user_id = '{user_id}'")

            where_clause = " AND ".join(conditions)
            search = search.where(where_clause, prefilter=True)

            results = search.limit(top_k).to_list()
            return [self._row_to_memory_entry(r) for r in results]
        except Exception:
            return []

    def search_by_topic(
        self,
        user_id: str,
        query: str,
        topic_name: Optional[str] = None,
        limit: int = 10
    ) -> dict:
        """
        Search by topic with parallel semantic + topic-based retrieval.

        Args:
            user_id: User to search within
            query: Search query for semantic search
            topic_name: Optional specific topic to filter by
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
                user_id, query_vector, limit
            )
            memory_future = executor.submit(
                self.search_semantic,
                query, user_id, limit, query_vector
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

        # Filter by specific topic if provided
        if topic_name:
            filtered_topics = [
                t for t in results["topic_results"]
                if t["name"] == topic_name
            ]
            filtered_memories = [
                m for m in results["memory_results"]
                if m.topic == topic_name
            ]
            results["topic_results"] = filtered_topics
            results["memory_results"] = filtered_memories

        # Combine and rank results
        topic_set = {t["name"] for t in results["topic_results"]}
        combined = []

        # Add memories from matching topics first
        for memory in results["memory_results"]:
            relevance_boost = 0
            if memory.topic and memory.topic in topic_set:
                relevance_boost = 0.1
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

        results["combined"] = combined

        return results

    def get_all(self) -> List[MemoryEntry]:
        """Get all entries."""
        where_clause = f"agent_id = '{self.agent_id}'"
        results = self.table.search().where(where_clause, prefilter=True).to_list()
        return [self._row_to_memory_entry(r) for r in results]

    def _row_to_memory_entry(self, row: dict) -> MemoryEntry:
        """Convert LanceDB row to MemoryEntry."""
        memory_type_str = row.get("memory_type", "conversation")
        try:
            memory_type = DM_MemoryType(memory_type_str) if memory_type_str else DM_MemoryType.CONVERSATION
        except ValueError:
            memory_type = DM_MemoryType.CONVERSATION

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
            importance_score=row.get("importance_score", 0.5),
            is_shareable=row.get("is_shareable", False)
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
