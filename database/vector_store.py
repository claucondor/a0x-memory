"""
Vector Store - Multi-tenant storage with Structured Multi-View Indexing

Supports multi-tenant isolation via agent_id and user_id filtering.
Single shared database, queries filtered by tenant context.

Indexing dimensions:
- Semantic Layer: Dense vectors (embedding similarity)
- Lexical Layer: Full-text search with Tantivy FTS
- Symbolic Layer: SQL-based metadata filtering
"""
from typing import List, Optional, Dict, Any, Tuple
import lancedb
import pyarrow as pa
from models.memory_entry import MemoryEntry
from utils.embedding import EmbeddingModel
import config
import os


class VectorStore:
    """
    Structured Multi-View Indexing - Storage and retrieval for Atomic Entries

    Paper Reference: Section 3.2 - Structured Indexing
    Implements M(m_k) with three structured layers:
    1. Semantic Layer: Dense embedding vectors for conceptual similarity
    2. Lexical Layer: Full-text search via Tantivy FTS index
    3. Symbolic Layer: SQL-based metadata filtering with DataFusion
    """

    def __init__(
        self,
        db_path: str = None,
        embedding_model: EmbeddingModel = None,
        table_name: str = None,
        storage_options: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        self.db_path = db_path or config.LANCEDB_PATH
        self.embedding_model = embedding_model or EmbeddingModel()
        self.table_name = table_name or config.MEMORY_TABLE_NAME
        self.table = None
        self._fts_initialized = False

        # Multi-tenant context (None = no filtering, backwards compatible)
        self.agent_id = agent_id
        self.user_id = user_id

        # Detect if using cloud storage (GCS, S3, Azure)
        self._is_cloud_storage = self.db_path.startswith(("gs://", "s3://", "az://"))

        # Connect to database
        if self._is_cloud_storage:
            self.db = lancedb.connect(self.db_path, storage_options=storage_options)
        else:
            os.makedirs(self.db_path, exist_ok=True)
            self.db = lancedb.connect(self.db_path)

        self._init_table()

    def _init_table(self):
        """Initialize table schema with multi-tenant support."""
        schema = pa.schema([
            pa.field("agent_id", pa.string()),  # Multi-tenant: agent identifier
            pa.field("user_id", pa.string()),   # Multi-tenant: user identifier
            pa.field("entry_id", pa.string()),
            pa.field("lossless_restatement", pa.string()),
            pa.field("keywords", pa.list_(pa.string())),
            pa.field("timestamp", pa.string()),
            pa.field("location", pa.string()),
            pa.field("persons", pa.list_(pa.string())),
            pa.field("entities", pa.list_(pa.string())),
            pa.field("topic", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self.embedding_model.dimension))
        ])

        if self.table_name not in self.db.table_names():
            self.table = self.db.create_table(self.table_name, schema=schema)
            print(f"Created new table: {self.table_name}")
        else:
            self.table = self.db.open_table(self.table_name)
            print(f"Opened existing table: {self.table_name}")

    def _get_tenant_filter(self) -> Optional[str]:
        """Build SQL filter clause for multi-tenant isolation."""
        conditions = []
        if self.agent_id:
            conditions.append(f"agent_id = '{self.agent_id}'")
        if self.user_id:
            conditions.append(f"user_id = '{self.user_id}'")
        return " AND ".join(conditions) if conditions else None

    def _init_fts_index(self):
        """Initialize Full-Text Search index on lossless_restatement column."""
        if self._fts_initialized:
            return

        try:
            if self._is_cloud_storage:
                # Use native FTS for cloud storage (Tantivy only works with local filesystem)
                self.table.create_fts_index(
                    "lossless_restatement",
                    use_tantivy=False,
                    replace=True
                )
                print("FTS index created (native mode for cloud storage)")
            else:
                # Use Tantivy FTS for local storage (better performance)
                self.table.create_fts_index(
                    "lossless_restatement",
                    use_tantivy=True,
                    tokenizer_name="en_stem",
                    replace=True
                )
                print("FTS index created (Tantivy mode)")
            self._fts_initialized = True
        except Exception as e:
            print(f"FTS index creation skipped: {e}")

    def _results_to_entries(self, results: List[dict]) -> List[MemoryEntry]:
        """Convert LanceDB results to MemoryEntry objects."""
        entries = []
        for r in results:
            try:
                entries.append(MemoryEntry(
                    entry_id=r["entry_id"],
                    lossless_restatement=r["lossless_restatement"],
                    keywords=list(r.get("keywords") or []),
                    timestamp=r.get("timestamp") or None,
                    location=r.get("location") or None,
                    persons=list(r.get("persons") or []),
                    entities=list(r.get("entities") or []),
                    topic=r.get("topic") or None
                ))
            except Exception as e:
                print(f"Warning: Failed to parse result: {e}")
                continue
        return entries

    def add_entries(self, entries: List[MemoryEntry], agent_id: str = None, user_id: str = None):
        """Batch add memory entries with optional tenant override."""
        if not entries:
            return

        # Use instance tenant or override
        _agent_id = agent_id or self.agent_id or ""
        _user_id = user_id or self.user_id or ""

        restatements = [entry.lossless_restatement for entry in entries]
        vectors = self.embedding_model.encode_documents(restatements)

        data = []
        for entry, vector in zip(entries, vectors):
            data.append({
                "agent_id": _agent_id,
                "user_id": _user_id,
                "entry_id": entry.entry_id,
                "lossless_restatement": entry.lossless_restatement,
                "keywords": entry.keywords,
                "timestamp": entry.timestamp or "",
                "location": entry.location or "",
                "persons": entry.persons,
                "entities": entry.entities,
                "topic": entry.topic or "",
                "vector": vector.tolist()
            })

        self.table.add(data)
        print(f"Added {len(entries)} entries (agent={_agent_id or 'none'}, user={_user_id or 'none'})")

        # Initialize FTS index after first data insertion
        if not self._fts_initialized:
            self._init_fts_index()

    def semantic_search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Semantic search with multi-tenant filtering."""
        try:
            if self.table.count_rows() == 0:
                return []

            query_vector = self.embedding_model.encode_single(query, is_query=True)
            search = self.table.search(query_vector.tolist())

            # Apply tenant filter
            tenant_filter = self._get_tenant_filter()
            if tenant_filter:
                search = search.where(tenant_filter, prefilter=True)

            results = search.limit(top_k).to_list()
            return self._results_to_entries(results)

        except Exception as e:
            print(f"Error during semantic search: {e}")
            return []

    def keyword_search(self, keywords: List[str], top_k: int = 3) -> List[MemoryEntry]:
        """Keyword search (BM25 FTS) with multi-tenant filtering."""
        try:
            if not keywords or self.table.count_rows() == 0:
                return []

            query = " ".join(keywords)
            search = self.table.search(query)

            # Apply tenant filter
            tenant_filter = self._get_tenant_filter()
            if tenant_filter:
                search = search.where(tenant_filter, prefilter=True)

            results = search.limit(top_k).to_list()
            return self._results_to_entries(results)

        except Exception as e:
            print(f"Error during keyword search: {e}")
            return []

    def structured_search(
        self,
        persons: Optional[List[str]] = None,
        timestamp_range: Optional[tuple] = None,
        location: Optional[str] = None,
        entities: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ) -> List[MemoryEntry]:
        """Structured metadata search with multi-tenant filtering."""
        try:
            if self.table.count_rows() == 0:
                return []

            if not any([persons, timestamp_range, location, entities]):
                return []

            conditions = []

            # Add tenant filter first
            tenant_filter = self._get_tenant_filter()
            if tenant_filter:
                conditions.append(tenant_filter)

            if persons:
                values = ", ".join([f"'{p}'" for p in persons])
                conditions.append(f"array_has_any(persons, make_array({values}))")

            if location:
                safe_location = location.replace("'", "''")
                conditions.append(f"location LIKE '%{safe_location}%'")

            if entities:
                values = ", ".join([f"'{e}'" for e in entities])
                conditions.append(f"array_has_any(entities, make_array({values}))")

            if timestamp_range:
                start_time, end_time = timestamp_range
                conditions.append(f"timestamp >= '{start_time}' AND timestamp <= '{end_time}'")

            where_clause = " AND ".join(conditions)
            query = self.table.search().where(where_clause, prefilter=True)

            if top_k:
                query = query.limit(top_k)

            results = query.to_list()
            return self._results_to_entries(results)

        except Exception as e:
            print(f"Error during structured search: {e}")
            return []

    def get_all_entries(self) -> List[MemoryEntry]:
        """Get all memory entries for current tenant."""
        tenant_filter = self._get_tenant_filter()
        if tenant_filter:
            results = self.table.search().where(tenant_filter, prefilter=True).to_list()
        else:
            results = self.table.to_arrow().to_pylist()
        return self._results_to_entries(results)

    def count_entries(self) -> int:
        """Count entries for current tenant."""
        tenant_filter = self._get_tenant_filter()
        if tenant_filter:
            results = self.table.search().where(tenant_filter, prefilter=True).to_list()
            return len(results)
        return self.table.count_rows()

    def optimize(self):
        """Optimize table after bulk insertions for better query performance."""
        self.table.optimize()
        print("Table optimized")

    def clear(self):
        """Clear entries for current tenant (or all if no tenant set)."""
        tenant_filter = self._get_tenant_filter()
        if tenant_filter:
            # Delete only tenant's entries
            self.table.delete(tenant_filter)
            print(f"Cleared entries for agent={self.agent_id}, user={self.user_id}")
        else:
            # No tenant = drop and recreate table
            self.db.drop_table(self.table_name)
            self._fts_initialized = False
            self._init_table()
            print("Database cleared")

    def keyword_search_with_scores(self, keywords: List[str], top_k: int = 3) -> List[Tuple[MemoryEntry, float]]:
        """Keyword search with scores and multi-tenant filtering."""
        try:
            if not keywords or self.table.count_rows() == 0:
                return []

            query = " ".join(keywords)
            search = self.table.search(query)

            # Apply tenant filter
            tenant_filter = self._get_tenant_filter()
            if tenant_filter:
                search = search.where(tenant_filter, prefilter=True)

            results = search.limit(top_k).to_list()

            if not results:
                return []

            scored_entries = []
            max_score = 0

            for r in results:
                score = r.get("_score", 0.0)
                max_score = max(max_score, score)
                try:
                    entry = MemoryEntry(
                        entry_id=r["entry_id"],
                        lossless_restatement=r["lossless_restatement"],
                        keywords=list(r.get("keywords") or []),
                        timestamp=r.get("timestamp") or None,
                        location=r.get("location") or None,
                        persons=list(r.get("persons") or []),
                        entities=list(r.get("entities") or []),
                        topic=r.get("topic") or None
                    )
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
