"""
AgentResponses table operations.
Extracted from unified_store.py.
"""
from typing import List

import pyarrow as pa

from models.group_memory import AgentResponse
from database.base import LanceDBConnection
from utils.embedding import EmbeddingModel
import config


class AgentResponsesTable:
    """CRUD for agent_responses table."""

    def __init__(self, agent_id: str, embedding_model: EmbeddingModel = None, storage_options: dict = None):
        self.agent_id = agent_id
        self.embedding_model = embedding_model or EmbeddingModel()
        self.storage_options = storage_options
        self.db = LanceDBConnection.get_agent_db(agent_id, storage_options=storage_options)
        self.table = self._init_table()

    def _init_table(self):
        """Initialize table with schema."""
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
        if table_name not in self.db.table_names():
            table = self.db.create_table(table_name, schema=schema)
            print(f"[{self.agent_id}] Created {table_name} table")
        else:
            table = self.db.open_table(table_name)
            print(f"[{self.agent_id}] Opened {table_name} ({table.count_rows()} rows)")

        self._create_scalar_index(table, "agent_id")
        self._create_scalar_index(table, "user_id")
        self._create_scalar_index(table, "group_id")
        self._create_scalar_index(table, "content_hash")

        return table

    def _create_scalar_index(self, table, column: str):
        """Create scalar index."""
        try:
            table.create_scalar_index(column, replace=True)
        except Exception:
            pass

    def add(self, response: AgentResponse) -> str:
        """Add an agent response."""
        vector = self.embedding_model.encode_single(response.summary or response.content, is_query=False)

        data = {
            "response_id": response.response_id,
            "agent_id": response.agent_id,
            "group_id": response.group_id,
            "user_id": response.user_id,
            "content": response.content,
            "content_hash": response.content_hash,
            "summary": response.summary,
            "response_type": response.response_type.value,
            "topics": response.topics,
            "keywords": response.keywords,
            "trigger_message": response.trigger_message,
            "trigger_message_id": response.trigger_message_id,
            "timestamp": response.timestamp,
            "token_count": response.token_count,
            "was_repeated": response.was_repeated,
            "importance_score": response.importance_score,
            "vector": vector.tolist()
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
        Search agent responses by semantic similarity.

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

        search = self.table.search(query_vector)

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
