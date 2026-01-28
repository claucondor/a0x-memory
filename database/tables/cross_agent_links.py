"""
CrossAgentLinks table operations (global DB).
Extracted from unified_store.py.
"""
from typing import List

import pyarrow as pa

from models.group_memory import CrossAgentLink
from database.base import LanceDBConnection
import config


class CrossAgentLinksTable:
    """CRUD for cross_agent_links table in global DB."""

    def __init__(self, storage_options: dict = None):
        self.storage_options = storage_options
        self.db = LanceDBConnection.get_global_db(storage_options=storage_options)
        self.table = self._init_table()

    def _init_table(self):
        """Initialize table with schema."""
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
        if table_name not in self.db.table_names():
            table = self.db.create_table(table_name, schema=schema)
            print(f"[Global] Created {table_name} table")
        else:
            table = self.db.open_table(table_name)
            print(f"[Global] Opened {table_name} ({table.count_rows()} rows)")

        self._create_scalar_index(table, "universal_user_id")

        return table

    def _create_scalar_index(self, table, column: str):
        """Create scalar index."""
        try:
            table.create_scalar_index(column, replace=True)
        except Exception:
            pass

    def add(self, link: CrossAgentLink) -> CrossAgentLink:
        """Add a cross-agent link."""
        import json

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

        self.table.add([data])
        return link

    def count(self) -> int:
        """Count all rows."""
        return self.table.count_rows()

    def optimize(self):
        """Compact the table."""
        try:
            self.table.optimize()
        except Exception:
            pass
