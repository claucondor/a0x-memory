"""
GroupSummaryStore - LanceDB storage for hierarchical group summaries.

Handles:
- CRUD for GroupSummary objects
- Decay score updates
- Aggregation queries (get summaries for aggregation)
- Pruning old summaries
"""
import math
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any

import lancedb
import pyarrow as pa

from models.group_memory import GroupSummary, PeriodType
from utils.embedding import EmbeddingModel
import config


DECAY_CONFIG = {
    "daily": {
        "max_age_days": 14,
        "decay_start_days": 7,
        "aggregate_after_days": 7
    },
    "weekly": {
        "max_age_days": 90,
        "decay_start_days": 30,
        "aggregate_after_days": 30
    },
    "monthly": {
        "max_age_days": 365,
        "decay_start_days": 90,
        "archive_after_days": 180
    },
    "era": {
        "max_count": 10,
        "min_duration_days": 14
    }
}


class GroupSummaryStore:
    """LanceDB storage for hierarchical group summaries."""

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
        """Initialize group_summaries table."""
        schema = pa.schema([
            pa.field("summary_id", pa.string()),
            pa.field("agent_id", pa.string()),
            pa.field("group_id", pa.string()),
            pa.field("period_type", pa.string()),
            pa.field("period_start", pa.string()),
            pa.field("period_end", pa.string()),
            pa.field("summary", pa.string()),
            pa.field("topics", pa.list_(pa.string())),
            pa.field("highlights", pa.list_(pa.string())),
            pa.field("active_users", pa.list_(pa.string())),
            pa.field("message_count", pa.int64()),
            pa.field("activity_score", pa.float32()),
            pa.field("decay_score", pa.float32()),
            pa.field("aggregated_from", pa.list_(pa.string())),
            pa.field("created_at", pa.string()),
            pa.field("summary_vector", pa.list_(pa.float32(), self.embedding_model.dimension)),
        ])

        table_name = "group_summaries"
        if table_name not in self.db.table_names():
            self.table = self.db.create_table(table_name, schema=schema)
            print(f"[GroupSummaryStore] Created {table_name} table")
        else:
            self.table = self.db.open_table(table_name)
            print(f"[GroupSummaryStore] Opened {table_name}")

    def add_summary(self, summary: GroupSummary) -> str:
        """Add a new group summary."""
        vector = self.embedding_model.encode_single(summary.summary, is_query=False)

        data = {
            "summary_id": summary.summary_id,
            "agent_id": summary.agent_id,
            "group_id": summary.group_id,
            "period_type": summary.period_type.value,
            "period_start": summary.period_start,
            "period_end": summary.period_end,
            "summary": summary.summary,
            "topics": summary.topics,
            "highlights": summary.highlights,
            "active_users": summary.active_users,
            "message_count": summary.message_count,
            "activity_score": summary.activity_score,
            "decay_score": summary.decay_score,
            "aggregated_from": summary.aggregated_from,
            "created_at": summary.created_at,
            "summary_vector": vector.tolist()
        }

        self.table.add([data])
        return summary.summary_id

    def get_summaries_for_period(
        self,
        group_id: str,
        period_type: str,
        min_decay_score: float = 0.1
    ) -> List[GroupSummary]:
        """Get all summaries of a type for a group, filtered by decay."""
        results = self.table.search().where(
            f"group_id = '{group_id}' AND period_type = '{period_type}' AND decay_score >= {min_decay_score}",
            prefilter=True
        ).limit(100).to_list()

        return [self._row_to_summary(r) for r in results]

    def get_summaries_to_aggregate(
        self,
        group_id: str,
        period_type: str
    ) -> List[GroupSummary]:
        """Get summaries ready for aggregation to next level."""
        config = DECAY_CONFIG.get(period_type, {})
        aggregate_after = config.get("aggregate_after_days", 7)

        cutoff = (datetime.now(timezone.utc) - timedelta(days=aggregate_after)).strftime("%Y-%m-%d")

        results = self.table.search().where(
            f"group_id = '{group_id}' AND period_type = '{period_type}' AND period_end < '{cutoff}'",
            prefilter=True
        ).limit(100).to_list()

        # Filter out already aggregated
        return [
            self._row_to_summary(r) for r in results
            if not self._is_already_aggregated(r["summary_id"])
        ]

    def _is_already_aggregated(self, summary_id: str) -> bool:
        """Check if summary was already used in aggregation."""
        results = self.table.search().where(
            f"aggregated_from IS NOT NULL",
            prefilter=True
        ).limit(1000).to_list()

        for r in results:
            if summary_id in (r.get("aggregated_from") or []):
                return True
        return False

    def update_decay_scores(self, group_id: str):
        """Update decay scores for all summaries in a group."""
        results = self.table.search().where(
            f"group_id = '{group_id}'",
            prefilter=True
        ).limit(1000).to_list()

        now = datetime.now(timezone.utc)

        for row in results:
            period_type = row["period_type"]
            created = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
            new_score = self._calculate_decay_score(created, period_type, now)

            if abs(new_score - row["decay_score"]) > 0.01:
                # Update in place (delete + add)
                self.table.delete(f"summary_id = '{row['summary_id']}'")
                row["decay_score"] = new_score
                self.table.add([row])

    def prune_expired(self, group_id: str, min_decay_score: float = 0.1):
        """Delete summaries with decay_score below threshold."""
        self.table.delete(
            f"group_id = '{group_id}' AND decay_score < {min_decay_score}"
        )

    def _calculate_decay_score(
        self,
        created_at: datetime,
        period_type: str,
        now: datetime
    ) -> float:
        """Calculate decay score using exponential decay."""
        config = DECAY_CONFIG.get(period_type, {"decay_start_days": 7, "max_age_days": 30})

        age_days = (now - created_at).days

        if age_days < config["decay_start_days"]:
            return 1.0

        decay_period = config["max_age_days"] - config["decay_start_days"]
        if decay_period <= 0:
            return 0.0

        half_life = decay_period / 3
        effective_age = age_days - config["decay_start_days"]
        lambda_ = math.log(2) / half_life

        return max(0.0, math.exp(-lambda_ * effective_age))

    def _row_to_summary(self, row: dict) -> GroupSummary:
        """Convert LanceDB row to GroupSummary."""
        return GroupSummary(
            summary_id=row["summary_id"],
            agent_id=row["agent_id"],
            group_id=row["group_id"],
            period_type=PeriodType(row["period_type"]),
            period_start=row["period_start"],
            period_end=row["period_end"],
            summary=row["summary"],
            topics=row.get("topics", []),
            highlights=row.get("highlights", []),
            active_users=row.get("active_users", []),
            message_count=row.get("message_count", 0),
            activity_score=row.get("activity_score", 0.5),
            decay_score=row.get("decay_score", 1.0),
            aggregated_from=row.get("aggregated_from", []),
            created_at=row["created_at"]
        )

    def get_context_summaries(
        self,
        group_id: str,
        limit_daily: int = 7,
        limit_weekly: int = 4,
        limit_monthly: int = 3
    ) -> Dict[str, List[GroupSummary]]:
        """
        Get summaries for context building.
        Returns most recent summaries at each level.
        """
        result = {}

        for period_type, limit in [
            ("daily", limit_daily),
            ("weekly", limit_weekly),
            ("monthly", limit_monthly)
        ]:
            summaries = self.get_summaries_for_period(group_id, period_type)
            # Sort by period_end desc, take limit
            summaries.sort(key=lambda s: s.period_end, reverse=True)
            result[period_type] = summaries[:limit]

        return result
