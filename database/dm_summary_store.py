"""
DMSummaryStore - LanceDB storage for DM conversation summaries.

Lower thresholds than group summaries because DMs have less volume.
Uses same GroupSummary model but with DM_SUMMARY_CONFIG.
"""
import lancedb
import pyarrow as pa

from database.group_summary_store import GroupSummaryStore, DM_SUMMARY_CONFIG
from models.group_memory import GroupSummary
from utils.embedding import EmbeddingModel
import config


class DMSummaryStore(GroupSummaryStore):
    """Storage for DM summaries with lower volume thresholds."""

    def __init__(
        self,
        db_path: str = None,
        embedding_model: EmbeddingModel = None,
        agent_id: str = None
    ):
        super().__init__(db_path, embedding_model, agent_id)
        # Override config with DM thresholds
        self.config = DM_SUMMARY_CONFIG

    def _init_table(self):
        """Initialize dm_summaries table."""
        schema = pa.schema([
            pa.field("summary_id", pa.string()),
            pa.field("agent_id", pa.string()),
            pa.field("group_id", pa.string()),  # Will be dm_{user_id}
            pa.field("level", pa.string()),
            pa.field("message_start", pa.int64()),
            pa.field("message_end", pa.int64()),
            pa.field("message_count", pa.int64()),
            pa.field("time_start", pa.string()),
            pa.field("time_end", pa.string()),
            pa.field("duration_hours", pa.float32()),
            pa.field("activity_rate", pa.float32()),
            pa.field("summary", pa.string()),
            pa.field("topics", pa.list_(pa.string())),
            pa.field("highlights", pa.list_(pa.string())),
            pa.field("active_users", pa.list_(pa.string())),
            pa.field("decay_score", pa.float32()),
            pa.field("messages_since_created", pa.int64()),
            pa.field("aggregated_from", pa.list_(pa.string())),
            pa.field("is_aggregated", pa.bool_()),
            pa.field("created_at", pa.string()),
            pa.field("summary_vector", pa.list_(pa.float32(), self.embedding_model.dimension)),
        ])

        table_name = "dm_summaries"
        if table_name not in self.db.table_names():
            self.table = self.db.create_table(table_name, schema=schema)
            print(f"[DMSummaryStore] Created {table_name} table")
        else:
            self.table = self.db.open_table(table_name)
            print(f"[DMSummaryStore] Opened {table_name} ({self.table.count_rows()} rows)")

        # Create scalar indexes for faster filtering
        self._create_indexes()

    def _create_indexes(self):
        """Create scalar indexes for common query patterns."""
        try:
            self.table.create_scalar_index("group_id", replace=True)
            self.table.create_scalar_index("level", replace=True)
            self.table.create_scalar_index("is_aggregated", replace=True)
        except Exception as e:
            print(f"[DMSummaryStore] Index creation note: {e}")
