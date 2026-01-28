"""
ConversationSummaries table operations.
Extracted from unified_store.py.
"""
from typing import Optional, Dict, Any

import pyarrow as pa

from database.base import LanceDBConnection
from utils.embedding import EmbeddingModel
import config


class ConversationSummariesTable:
    """CRUD for conversation_summaries table."""

    def __init__(self, agent_id: str, embedding_model: EmbeddingModel = None, storage_options: dict = None):
        self.agent_id = agent_id
        self.embedding_model = embedding_model or EmbeddingModel()
        self.storage_options = storage_options
        self.db = LanceDBConnection.get_agent_db(agent_id, storage_options=storage_options)
        self.table = self._init_table()

    def _init_table(self):
        """Initialize table with schema."""
        schema = pa.schema([
            pa.field("thread_id", pa.string()),
            pa.field("agent_id", pa.string()),
            pa.field("summary", pa.string()),
            pa.field("message_count", pa.int32()),
            pa.field("last_updated", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self.embedding_model.dimension))
        ])

        table_name = "conversation_summaries"
        if table_name not in self.db.table_names():
            table = self.db.create_table(table_name, schema=schema)
            print(f"[{self.agent_id}] Created {table_name} table")
        else:
            table = self.db.open_table(table_name)
            print(f"[{self.agent_id}] Opened {table_name} ({table.count_rows()} rows)")

        self._create_scalar_index(table, "thread_id")
        self._create_scalar_index(table, "agent_id")

        return table

    def _create_scalar_index(self, table, column: str):
        """Create scalar index."""
        try:
            table.create_scalar_index(column, replace=True)
        except Exception:
            pass

    def get(self, thread_id: str) -> str:
        """Get summary by thread_id."""
        try:
            results = self.table.search().where(
                f"thread_id = '{thread_id}'",
                prefilter=True
            ).limit(1).to_list()

            if results:
                return results[0].get("summary", "")
            return ""
        except Exception:
            return ""

    def update(self, thread_id: str, summary: str, message_count: int, last_updated: str):
        """Update conversation summary."""
        vector = self.embedding_model.encode_single(summary, is_query=False)

        # Delete existing
        try:
            self.table.delete(f"thread_id = '{thread_id}'")
        except Exception:
            pass

        # Add new
        data = {
            "thread_id": thread_id,
            "agent_id": self.agent_id,
            "summary": summary,
            "message_count": message_count,
            "last_updated": last_updated,
            "vector": vector.tolist()
        }
        self.table.add([data])

    def get_by_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get full record by thread_id."""
        try:
            results = self.table.search().where(
                f"thread_id = '{thread_id}'",
                prefilter=True
            ).limit(1).to_list()

            if results:
                return {
                    "thread_id": results[0].get("thread_id"),
                    "agent_id": results[0].get("agent_id"),
                    "summary": results[0].get("summary"),
                    "message_count": results[0].get("message_count"),
                    "last_updated": results[0].get("last_updated")
                }
            return None
        except Exception:
            return None

    def count(self) -> int:
        """Count all rows."""
        return self.table.count_rows()

    def optimize(self):
        """Compact the table."""
        try:
            self.table.optimize()
        except Exception:
            pass
