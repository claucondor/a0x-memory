"""
AgentResponses table operations.
Extracted from unified_store.py.

Dual-vector schema:
- trigger_vector: for searching similar questions
- response_vector: for searching similar responses
"""
from typing import List, Optional

import pyarrow as pa

from models.group_memory import AgentResponse
from database.base import LanceDBConnection
from utils.embedding import EmbeddingModel
import config


class AgentResponsesTable:
    """CRUD for agent_responses table with dual-vector search."""

    def __init__(self, agent_id: str, embedding_model: EmbeddingModel = None, storage_options: dict = None):
        self.agent_id = agent_id
        self.embedding_model = embedding_model or EmbeddingModel()
        self.storage_options = storage_options
        self.db = LanceDBConnection.get_agent_db(agent_id, storage_options=storage_options)
        self.table = self._init_table()

    def _init_table(self):
        """Initialize table with dual-vector schema."""
        schema = pa.schema([
            pa.field("response_id", pa.string()),
            pa.field("agent_id", pa.string()),
            # Scope
            pa.field("scope", pa.string()),
            pa.field("user_id", pa.string()),
            pa.field("group_id", pa.string()),
            # Trigger (user's message)
            pa.field("trigger_message", pa.string()),
            pa.field("trigger_vector", pa.list_(pa.float32(), self.embedding_model.dimension)),
            # Response
            pa.field("response_content", pa.string()),
            pa.field("response_summary", pa.string()),
            pa.field("response_vector", pa.list_(pa.float32(), self.embedding_model.dimension)),
            # Classification
            pa.field("response_type", pa.string()),
            pa.field("topics", pa.list_(pa.string())),
            pa.field("keywords", pa.list_(pa.string())),
            # Metadata
            pa.field("timestamp", pa.string()),
            pa.field("importance_score", pa.float32()),
        ])

        table_name = "agent_responses"
        if table_name not in self.db.table_names():
            table = self.db.create_table(table_name, schema=schema)
            print(f"[{self.agent_id}] Created {table_name} table")
        else:
            table = self.db.open_table(table_name)
            print(f"[{self.agent_id}] Opened {table_name} ({table.count_rows()} rows)")

        # Scalar indices for filtering
        self._create_scalar_index(table, "scope")
        self._create_scalar_index(table, "user_id")
        self._create_scalar_index(table, "group_id")

        return table

    def _create_scalar_index(self, table, column: str):
        """Create scalar index."""
        try:
            table.create_scalar_index(column, replace=True)
        except Exception:
            pass

    def add(self, response: AgentResponse) -> str:
        """Add an agent response (legacy method for backward compatibility)."""
        trigger_vector = self.embedding_model.encode_single(response.trigger_message, is_query=False)
        response_vector = self.embedding_model.encode_single(
            response.summary or response.content,
            is_query=False
        )

        data = {
            "response_id": response.response_id,
            "agent_id": response.agent_id,
            "scope": getattr(response, "scope", "user"),
            "user_id": response.user_id,
            "group_id": response.group_id,
            "trigger_message": response.trigger_message,
            "trigger_vector": trigger_vector.tolist(),
            "response_content": response.content,
            "response_summary": response.summary,
            "response_vector": response_vector.tolist(),
            "response_type": response.response_type.value,
            "topics": response.topics,
            "keywords": response.keywords,
            "timestamp": response.timestamp,
            "importance_score": response.importance_score,
        }

        self.table.add([data])
        return response.response_id

    def count(self) -> int:
        """Count all rows."""
        return self.table.count_rows()

    def search_semantic(
        self,
        query_vector,
        group_id: str = None,
        user_id: str = None,
        limit: int = 5
    ) -> list:
        """
        Search agent responses by semantic similarity (uses response_vector).

        Args:
            query_vector: Pre-computed query vector
            group_id: Filter by group (None = all)
            user_id: Filter by user (None = all)
            limit: Max results

        Returns:
            List of dicts with response data
        """
        if self.table.count_rows() == 0:
            return []

        search = self.table.search(query_vector, vector_column_name="response_vector")

        # Build filter
        conditions = [f"agent_id = '{self.agent_id}'"]
        if group_id:
            conditions.append(f"group_id = '{group_id}'")
        if user_id:
            conditions.append(f"user_id = '{user_id}'")

        where_clause = " AND ".join(conditions)
        search = search.where(where_clause, prefilter=True)

        try:
            results = search.limit(limit).to_list()
            return results
        except Exception as e:
            print(f"[AgentResponses] Search error: {e}")
            return []

    def optimize(self):
        """Compact the table."""
        try:
            self.table.optimize()
        except Exception:
            pass
