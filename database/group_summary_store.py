"""
GroupSummaryStore - LanceDB storage for volume-based hierarchical group summaries.

Handles:
- CRUD for GroupSummary objects (micro, chunk, block levels)
- Decay score updates based on message volume (not time)
- Aggregation queries (get summaries ready for aggregation)
- Pruning old summaries

Volume-based summarization:
- Micro: Every ~50 messages
- Chunk: Every 5 micros (~250 messages)
- Block: Every 5 chunks (~1250 messages)
"""
import math
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import lancedb
import pyarrow as pa

from models.group_memory import GroupSummary, SummaryLevel
from utils.embedding import EmbeddingModel
import config


# Volume-based configuration
SUMMARY_CONFIG = {
    "micro": {
        "message_threshold": 50,      # Trigger after 50 messages
        "aggregate_count": 5,          # 5 micros → 1 chunk
        "decay_start_messages": 250,   # Start decaying after 250 new messages
        "max_messages": 500,           # Fully decayed after 500 new messages
    },
    "chunk": {
        "message_threshold": 250,      # 5 micros
        "aggregate_count": 5,          # 5 chunks → 1 block
        "decay_start_messages": 1250,  # Start decaying after 1250 new messages
        "max_messages": 2500,          # Fully decayed after 2500 new messages
    },
    "block": {
        "message_threshold": 1250,     # 5 chunks
        "aggregate_count": 5,          # 5 blocks → 1 archive (future)
        "decay_start_messages": 6250,  # Start decaying after 6250 new messages
        "max_messages": 12500,         # Fully decayed after 12500 new messages
    }
}

# DM-specific configuration (lower thresholds for 1-on-1 conversations)
DM_SUMMARY_CONFIG = {
    "micro": {
        "message_threshold": 20,       # Lower than group (50)
        "aggregate_count": 5,          # 5 micros → 1 chunk
        "decay_start_messages": 100,
        "max_messages": 200,
    },
    "chunk": {
        "message_threshold": 100,      # 5 micros
        "aggregate_count": 5,          # 5 chunks → 1 block
        "decay_start_messages": 500,
        "max_messages": 1000,
    },
    "block": {
        "message_threshold": 500,      # 5 chunks
        "aggregate_count": 5,
        "decay_start_messages": 2500,
        "max_messages": 5000,
    }
}


class GroupSummaryStore:
    """LanceDB storage for volume-based hierarchical group summaries."""

    def __init__(
        self,
        db_path: str = None,
        embedding_model: EmbeddingModel = None,
        agent_id: Optional[str] = None
    ):
        self.db_path = db_path or config.LANCEDB_PATH
        self.embedding_model = embedding_model or EmbeddingModel()
        self.agent_id = agent_id

        # Connect to database
        self.db = lancedb.connect(self.db_path)
        self._init_table()

    def _init_table(self):
        """Initialize group_summaries table with volume-based schema."""
        schema = pa.schema([
            pa.field("summary_id", pa.string()),
            pa.field("agent_id", pa.string()),
            pa.field("group_id", pa.string()),
            # Level (micro, chunk, block)
            pa.field("level", pa.string()),
            # Message range
            pa.field("message_start", pa.int64()),
            pa.field("message_end", pa.int64()),
            pa.field("message_count", pa.int64()),
            # Temporal context
            pa.field("time_start", pa.string()),
            pa.field("time_end", pa.string()),
            pa.field("duration_hours", pa.float32()),
            pa.field("activity_rate", pa.float32()),
            # Content
            pa.field("summary", pa.string()),
            pa.field("topics", pa.list_(pa.string())),
            pa.field("highlights", pa.list_(pa.string())),
            pa.field("active_users", pa.list_(pa.string())),
            # Decay (volume-based)
            pa.field("decay_score", pa.float32()),
            pa.field("messages_since_created", pa.int64()),
            # Hierarchy
            pa.field("aggregated_from", pa.list_(pa.string())),
            pa.field("is_aggregated", pa.bool_()),
            # Metadata
            pa.field("created_at", pa.string()),
            pa.field("summary_vector", pa.list_(pa.float32(), self.embedding_model.dimension)),
        ])

        table_name = "group_summaries"
        if table_name not in self.db.table_names():
            self.table = self.db.create_table(table_name, schema=schema)
            print(f"[GroupSummaryStore] Created {table_name} table")
        else:
            self.table = self.db.open_table(table_name)
            print(f"[GroupSummaryStore] Opened {table_name} ({self.table.count_rows()} rows)")

        # Create scalar indexes for faster filtering
        self._create_indexes()

    def _create_indexes(self):
        """Create scalar indexes for common query patterns."""
        try:
            self.table.create_scalar_index("group_id", replace=True)
            self.table.create_scalar_index("level", replace=True)
            self.table.create_scalar_index("is_aggregated", replace=True)
        except Exception as e:
            print(f"[GroupSummaryStore] Index creation note: {e}")

    def add_summary(self, summary: GroupSummary) -> str:
        """Add a new group summary."""
        vector = self.embedding_model.encode_single(summary.summary, is_query=False)

        level = summary.level.value if hasattr(summary.level, 'value') else summary.level

        data = {
            "summary_id": summary.summary_id,
            "agent_id": summary.agent_id,
            "group_id": summary.group_id,
            "level": level,
            "message_start": summary.message_start,
            "message_end": summary.message_end,
            "message_count": summary.message_count,
            "time_start": summary.time_start,
            "time_end": summary.time_end,
            "duration_hours": summary.duration_hours,
            "activity_rate": summary.activity_rate,
            "summary": summary.summary,
            "topics": summary.topics,
            "highlights": summary.highlights,
            "active_users": summary.active_users,
            "decay_score": summary.decay_score,
            "messages_since_created": summary.messages_since_created,
            "aggregated_from": summary.aggregated_from,
            "is_aggregated": summary.is_aggregated,
            "created_at": summary.created_at,
            "summary_vector": vector.tolist()
        }

        self.table.add([data])
        return summary.summary_id

    def get_summaries_by_level(
        self,
        group_id: str,
        level: str,
        min_decay_score: float = 0.1
    ) -> List[GroupSummary]:
        """Get all summaries of a level for a group, filtered by decay."""
        results = self.table.search().where(
            f"group_id = '{group_id}' AND level = '{level}' AND decay_score >= {min_decay_score}",
            prefilter=True
        ).limit(100).to_list()

        return [self._row_to_summary(r) for r in results]

    def get_summaries_to_aggregate(
        self,
        group_id: str,
        level: str
    ) -> List[GroupSummary]:
        """
        Get summaries ready for aggregation to next level.
        Returns non-aggregated summaries of the given level.
        """
        results = self.table.search().where(
            f"group_id = '{group_id}' AND level = '{level}' AND is_aggregated = false",
            prefilter=True
        ).limit(100).to_list()

        return [self._row_to_summary(r) for r in results]

    def get_pending_micro_count(self, group_id: str) -> int:
        """Get count of non-aggregated micros for a group."""
        results = self.table.search().where(
            f"group_id = '{group_id}' AND level = 'micro' AND is_aggregated = false",
            prefilter=True
        ).limit(100).to_list()
        return len(results)

    def get_pending_chunk_count(self, group_id: str) -> int:
        """Get count of non-aggregated chunks for a group."""
        results = self.table.search().where(
            f"group_id = '{group_id}' AND level = 'chunk' AND is_aggregated = false",
            prefilter=True
        ).limit(100).to_list()
        return len(results)

    def get_last_message_index(self, group_id: str) -> int:
        """Get the last message index that has been summarized."""
        results = self.table.search().where(
            f"group_id = '{group_id}' AND level = 'micro'",
            prefilter=True
        ).limit(1000).to_list()

        if not results:
            return -1

        return max(r["message_end"] for r in results)

    def mark_as_aggregated(self, summary_ids: List[str]):
        """Mark summaries as aggregated after they've been used in a higher-level summary."""
        if not summary_ids:
            return

        rows_to_update = []
        for summary_id in summary_ids:
            results = self.table.search().where(
                f"summary_id = '{summary_id}'",
                prefilter=True
            ).limit(1).to_list()
            if results:
                rows_to_update.append(results[0])

        if not rows_to_update:
            return

        id_list = "', '".join(summary_ids)
        self.table.delete(f"summary_id IN ('{id_list}')")

        for row in rows_to_update:
            row["is_aggregated"] = True

        self.table.add(rows_to_update)
        print(f"[GroupSummaryStore] Marked {len(rows_to_update)} summaries as aggregated")

    def update_decay_scores(self, group_id: str, current_message_count: int):
        """
        Update decay scores based on how many new messages have arrived.

        Args:
            group_id: Group to update
            current_message_count: Total messages in the group now
        """
        results = self.table.search().where(
            f"group_id = '{group_id}'",
            prefilter=True
        ).limit(1000).to_list()

        if not results:
            return

        rows_to_update = []

        for row in results:
            level = row["level"]
            message_end = row["message_end"]

            # Calculate how many messages have arrived since this summary was created
            messages_since = current_message_count - message_end - 1
            new_score = self._calculate_decay_score(messages_since, level)

            if abs(new_score - row["decay_score"]) > 0.01 or row["messages_since_created"] != messages_since:
                row["decay_score"] = new_score
                row["messages_since_created"] = messages_since
                rows_to_update.append(row)

        if not rows_to_update:
            return

        ids_to_update = [r["summary_id"] for r in rows_to_update]
        id_list = "', '".join(ids_to_update)
        self.table.delete(f"summary_id IN ('{id_list}')")
        self.table.add(rows_to_update)
        print(f"[GroupSummaryStore] Updated decay scores for {len(rows_to_update)} summaries")

    def prune_expired(self, group_id: str, min_decay_score: float = 0.1):
        """Delete summaries with decay_score below threshold."""
        self.table.delete(
            f"group_id = '{group_id}' AND decay_score < {min_decay_score}"
        )

    def _calculate_decay_score(self, messages_since: int, level: str) -> float:
        """Calculate decay score based on messages since creation."""
        level_config = SUMMARY_CONFIG.get(level, SUMMARY_CONFIG["micro"])

        if messages_since < level_config["decay_start_messages"]:
            return 1.0

        decay_range = level_config["max_messages"] - level_config["decay_start_messages"]
        if decay_range <= 0:
            return 0.0

        # Exponential decay with half-life at 1/3 of decay range
        half_life = decay_range / 3
        effective_messages = messages_since - level_config["decay_start_messages"]
        lambda_ = math.log(2) / half_life

        return max(0.0, math.exp(-lambda_ * effective_messages))

    def _row_to_summary(self, row: dict) -> GroupSummary:
        """Convert LanceDB row to GroupSummary."""
        level = row["level"]
        if isinstance(level, str):
            level = SummaryLevel(level)

        return GroupSummary(
            summary_id=row["summary_id"],
            agent_id=row["agent_id"],
            group_id=row["group_id"],
            level=level,
            message_start=row["message_start"],
            message_end=row["message_end"],
            message_count=row.get("message_count", 0),
            time_start=row["time_start"],
            time_end=row["time_end"],
            duration_hours=row.get("duration_hours", 0.0),
            activity_rate=row.get("activity_rate", 0.0),
            summary=row["summary"],
            topics=row.get("topics", []),
            highlights=row.get("highlights", []),
            active_users=row.get("active_users", []),
            decay_score=row.get("decay_score", 1.0),
            messages_since_created=row.get("messages_since_created", 0),
            aggregated_from=row.get("aggregated_from", []),
            is_aggregated=row.get("is_aggregated", False),
            created_at=row["created_at"]
        )

    def get_context_summaries(
        self,
        group_id: str,
        limit_micro: int = 5,
        limit_chunk: int = 3,
        limit_block: int = 2
    ) -> Dict[str, List[GroupSummary]]:
        """
        Get summaries for context building.
        Returns most recent summaries at each level.
        """
        result = {}

        for level, limit in [
            ("micro", limit_micro),
            ("chunk", limit_chunk),
            ("block", limit_block)
        ]:
            summaries = self.get_summaries_by_level(group_id, level)
            # Sort by message_end desc (most recent first)
            summaries.sort(key=lambda s: s.message_end, reverse=True)
            result[level] = summaries[:limit]

        return result

    def search_summaries(
        self,
        group_id: str,
        query: str,
        levels: List[str] = None,
        limit: int = 10
    ) -> List[GroupSummary]:
        """Semantic search across summaries."""
        query_vector = self.embedding_model.encode_single(query, is_query=True)

        where_clause = f"group_id = '{group_id}'"
        if levels:
            levels_str = "', '".join(levels)
            where_clause += f" AND level IN ('{levels_str}')"

        results = (
            self.table.search(query_vector.tolist())
            .where(where_clause, prefilter=True)
            .limit(limit)
            .to_list()
        )

        return [self._row_to_summary(r) for r in results]

    def get_summary_by_id(self, summary_id: str) -> Optional[GroupSummary]:
        """Get a specific summary by ID."""
        results = self.table.search().where(
            f"summary_id = '{summary_id}'",
            prefilter=True
        ).limit(1).to_list()

        if results:
            return self._row_to_summary(results[0])
        return None

    def count_summaries(self, group_id: str = None, level: str = None) -> int:
        """Count summaries with optional filters."""
        if not group_id and not level:
            return self.table.count_rows()

        conditions = []
        if group_id:
            conditions.append(f"group_id = '{group_id}'")
        if level:
            conditions.append(f"level = '{level}'")

        where_clause = " AND ".join(conditions)
        results = self.table.search().where(where_clause, prefilter=True).limit(10000).to_list()
        return len(results)
