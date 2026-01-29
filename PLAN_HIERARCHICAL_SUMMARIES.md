# Plan: Hierarchical Group Summaries with Temporal Decay

## Problem

Current `GroupProfile.summary` is a single 1-2 sentence string regenerated from only the last 4000 chars of messages. For groups with 10k+ messages:
- Loses all historical context
- No temporal awareness (what was discussed 3 months ago)
- No topic evolution tracking
- Profile only reflects recent activity

## Solution

Implement hierarchical summarization with decay:
1. **Daily summaries** → aggregate to **weekly** → aggregate to **monthly** → detect **eras**
2. **Decay mechanism**: Older summaries lose detail, very old ones get archived/deleted
3. **Topic timeline**: Track topic evolution over time
4. **Fast retrieval**: Only load recent + aggregated historical, not full history

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 GroupProfile (updated)              │
├─────────────────────────────────────────────────────┤
│ summary: str              # Current snapshot        │
│ summary_history_ref: str  # Reference to summaries  │
│ topic_evolution: [{       # Last 12 weeks of topics │
│   week: "2024-W05",                                 │
│   topics: ["grants", "audits"],                     │
│   activity_score: 0.8                               │
│ }]                                                  │
│ current_era: {            # Active era              │
│   started: "2024-01-15",                            │
│   theme: "Base ecosystem grants",                   │
│   key_topics: [...]                                 │
│ }                                                   │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│            GroupSummary (new model)                 │
├─────────────────────────────────────────────────────┤
│ summary_id: str                                     │
│ agent_id: str                                       │
│ group_id: str                                       │
│ period_type: "daily" | "weekly" | "monthly" | "era" │
│ period_start: str         # ISO date                │
│ period_end: str           # ISO date                │
│ summary: str              # 2-5 sentences           │
│ topics: List[str]         # 3-7 topics              │
│ highlights: List[str]     # 2-4 notable events      │
│ active_users: List[str]   # Top contributors        │
│ message_count: int        # Messages in period      │
│ activity_score: float     # 0-1, relative activity  │
│ decay_score: float        # 1.0 → 0.0 over time     │
│ created_at: str                                     │
│ aggregated_from: List[str] # Child summary IDs      │
└─────────────────────────────────────────────────────┘
```

## Decay Rules

```python
DECAY_CONFIG = {
    "daily": {
        "max_age_days": 14,       # Keep 2 weeks of daily
        "decay_start_days": 7,    # Start decay after 1 week
        "aggregate_after_days": 7 # Aggregate to weekly after 7 days
    },
    "weekly": {
        "max_age_days": 90,       # Keep 3 months of weekly
        "decay_start_days": 30,   # Start decay after 1 month
        "aggregate_after_days": 30 # Aggregate to monthly after 30 days
    },
    "monthly": {
        "max_age_days": 365,      # Keep 1 year of monthly
        "decay_start_days": 90,   # Start decay after 3 months
        "archive_after_days": 180 # Archive (compress) after 6 months
    },
    "era": {
        "max_count": 10,          # Keep last 10 eras
        "min_duration_days": 14   # Era must be at least 2 weeks
    }
}
```

## Decay Score Calculation

```python
def calculate_decay_score(created_at: str, period_type: str) -> float:
    """
    Returns 1.0 for fresh, decays to 0.0 at max_age.

    Uses exponential decay: score = e^(-λt)
    where λ = ln(2) / half_life_days
    """
    age_days = (now - created_at).days
    config = DECAY_CONFIG[period_type]

    if age_days < config["decay_start_days"]:
        return 1.0

    decay_period = config["max_age_days"] - config["decay_start_days"]
    half_life = decay_period / 3  # Reach ~0.125 at max_age

    effective_age = age_days - config["decay_start_days"]
    lambda_ = math.log(2) / half_life

    return math.exp(-lambda_ * effective_age)
```

## Implementation Plan

### Commit 1: Add GroupSummary model

File: `models/group_memory.py`

Add after `GroupProfile` class:

```python
class PeriodType(str, Enum):
    """Summary period granularity"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ERA = "era"


class GroupSummary(BaseModel):
    """
    Hierarchical group summary for a specific time period.

    Summaries aggregate upward:
    - Daily summaries aggregate to weekly
    - Weekly summaries aggregate to monthly
    - Monthly summaries detect and form eras

    Decay mechanism:
    - decay_score starts at 1.0
    - Decays exponentially based on age and period_type
    - Old summaries get pruned when decay_score < 0.1
    """
    summary_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = Field(..., description="Agent this summary belongs to")
    group_id: str = Field(..., description="Group identifier")

    # Period definition
    period_type: PeriodType = Field(..., description="Granularity: daily, weekly, monthly, era")
    period_start: str = Field(..., description="Period start ISO date (YYYY-MM-DD)")
    period_end: str = Field(..., description="Period end ISO date (YYYY-MM-DD)")

    # Content
    summary: str = Field(..., description="2-5 sentence summary of the period")
    topics: List[str] = Field(default_factory=list, description="3-7 main topics")
    highlights: List[str] = Field(default_factory=list, description="2-4 notable events/discussions")
    active_users: List[str] = Field(default_factory=list, description="Top 5 contributors")

    # Metrics
    message_count: int = Field(default=0, description="Messages processed in period")
    activity_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Relative activity level")

    # Decay
    decay_score: float = Field(default=1.0, ge=0.0, le=1.0, description="1.0=fresh, 0.0=expired")

    # Hierarchy
    aggregated_from: List[str] = Field(default_factory=list, description="Child summary IDs used")

    # Metadata
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
```

Also update `GroupProfile` to add:

```python
class GroupProfile(BaseModel):
    # ... existing fields ...

    # NEW: Topic evolution (last 12 data points)
    topic_evolution: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Topic trends: [{period, topics, activity_score}]"
    )

    # NEW: Current era tracking
    current_era: Optional[Dict[str, Any]] = Field(
        None,
        description="Active era: {started, theme, key_topics}"
    )
```

### Commit 2: Add GroupSummaryStore

File: `database/group_summary_store.py` (new file)

```python
"""
GroupSummaryStore - LanceDB storage for hierarchical group summaries.

Handles:
- CRUD for GroupSummary objects
- Decay score updates
- Aggregation queries (get summaries for aggregation)
- Pruning old summaries
"""
import json
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
        # Search for any summary that has this ID in aggregated_from
        # For now, simple check - could be optimized with index
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
                # Update in place (LanceDB doesn't support update, so delete + add)
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
```

### Commit 3: Add SummaryAggregator service

File: `services/summary_aggregator.py` (new file)

```python
"""
SummaryAggregator - Generates and aggregates hierarchical summaries.

Responsibilities:
- Generate daily summaries from messages
- Aggregate daily → weekly → monthly
- Detect era changes
- Update GroupProfile with topic evolution
"""
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from collections import Counter

from models.group_memory import GroupSummary, PeriodType, GroupProfile
from database.group_summary_store import GroupSummaryStore, DECAY_CONFIG
from utils.llm_client import LLMClient


DAILY_SUMMARY_PROMPT = """Summarize this day's group conversation.

Group: {group_id}
Date: {date}
Message count: {message_count}

Messages:
{messages}

Return JSON:
{{
  "summary": "2-3 sentences summarizing the day's discussions",
  "topics": ["topic1", "topic2", ...],  // 3-5 main topics
  "highlights": ["highlight1", ...],     // 1-3 notable events
  "active_users": ["user1", ...]         // Top 3-5 contributors
}}"""


WEEKLY_SUMMARY_PROMPT = """Aggregate these daily summaries into a weekly summary.

Group: {group_id}
Week: {week_start} to {week_end}

Daily summaries:
{daily_summaries}

Return JSON:
{{
  "summary": "3-4 sentences summarizing the week",
  "topics": ["topic1", ...],        // 5-7 main topics (merged from dailies)
  "highlights": ["highlight1", ...], // 2-4 key moments of the week
  "trend": "increasing|stable|decreasing"  // Activity trend
}}"""


MONTHLY_SUMMARY_PROMPT = """Aggregate these weekly summaries into a monthly summary.

Group: {group_id}
Month: {month}

Weekly summaries:
{weekly_summaries}

Return JSON:
{{
  "summary": "4-5 sentences summarizing the month",
  "topics": ["topic1", ...],        // Top 7 topics
  "highlights": ["highlight1", ...], // 3-4 significant events
  "theme": "Brief theme of the month",
  "is_era_shift": true/false        // Did major focus change?
}}"""


class SummaryAggregator:
    """Generates and aggregates hierarchical group summaries."""

    def __init__(
        self,
        summary_store: GroupSummaryStore,
        llm_client: LLMClient = None
    ):
        self.summary_store = summary_store
        self.llm_client = llm_client or LLMClient(use_streaming=False)

    def generate_daily_summary(
        self,
        agent_id: str,
        group_id: str,
        date: str,  # YYYY-MM-DD
        messages: List[Dict[str, Any]]
    ) -> Optional[GroupSummary]:
        """
        Generate a daily summary from messages.

        Args:
            agent_id: Agent ID
            group_id: Group ID
            date: Date string YYYY-MM-DD
            messages: List of message dicts with 'content', 'speaker', 'timestamp'

        Returns:
            GroupSummary or None if too few messages
        """
        if len(messages) < 3:  # Skip days with very low activity
            return None

        # Format messages for prompt
        messages_text = "\n".join([
            f"[{m.get('speaker', 'unknown')}]: {m.get('content', '')[:200]}"
            for m in messages[:50]  # Limit to 50 messages
        ])

        prompt = DAILY_SUMMARY_PROMPT.format(
            group_id=group_id,
            date=date,
            message_count=len(messages),
            messages=messages_text
        )

        try:
            response = self.llm_client.chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=0.3
            )
            data = self.llm_client.extract_json(response)

            # Calculate activity score (relative to expected)
            expected_daily = 20  # Baseline expectation
            activity_score = min(1.0, len(messages) / expected_daily)

            summary = GroupSummary(
                agent_id=agent_id,
                group_id=group_id,
                period_type=PeriodType.DAILY,
                period_start=date,
                period_end=date,
                summary=data.get("summary", f"Group discussion on {date}"),
                topics=data.get("topics", [])[:5],
                highlights=data.get("highlights", [])[:3],
                active_users=data.get("active_users", [])[:5],
                message_count=len(messages),
                activity_score=activity_score
            )

            self.summary_store.add_summary(summary)
            return summary

        except Exception as e:
            print(f"[SummaryAggregator] Daily summary generation failed: {e}")
            return None

    def aggregate_to_weekly(
        self,
        agent_id: str,
        group_id: str,
        week_start: str,  # YYYY-MM-DD (Monday)
        daily_summaries: List[GroupSummary]
    ) -> Optional[GroupSummary]:
        """Aggregate daily summaries into a weekly summary."""
        if len(daily_summaries) < 2:
            return None

        # Format daily summaries for prompt
        daily_text = "\n\n".join([
            f"**{s.period_start}** ({s.message_count} msgs, activity: {s.activity_score:.1f})\n"
            f"Summary: {s.summary}\n"
            f"Topics: {', '.join(s.topics)}\n"
            f"Highlights: {', '.join(s.highlights)}"
            for s in sorted(daily_summaries, key=lambda x: x.period_start)
        ])

        week_end = (datetime.strptime(week_start, "%Y-%m-%d") + timedelta(days=6)).strftime("%Y-%m-%d")

        prompt = WEEKLY_SUMMARY_PROMPT.format(
            group_id=group_id,
            week_start=week_start,
            week_end=week_end,
            daily_summaries=daily_text
        )

        try:
            response = self.llm_client.chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=0.3
            )
            data = self.llm_client.extract_json(response)

            # Aggregate metrics
            total_messages = sum(s.message_count for s in daily_summaries)
            avg_activity = sum(s.activity_score for s in daily_summaries) / len(daily_summaries)

            summary = GroupSummary(
                agent_id=agent_id,
                group_id=group_id,
                period_type=PeriodType.WEEKLY,
                period_start=week_start,
                period_end=week_end,
                summary=data.get("summary", f"Week of {week_start}"),
                topics=data.get("topics", [])[:7],
                highlights=data.get("highlights", [])[:4],
                active_users=self._merge_active_users(daily_summaries),
                message_count=total_messages,
                activity_score=avg_activity,
                aggregated_from=[s.summary_id for s in daily_summaries]
            )

            self.summary_store.add_summary(summary)
            return summary

        except Exception as e:
            print(f"[SummaryAggregator] Weekly aggregation failed: {e}")
            return None

    def aggregate_to_monthly(
        self,
        agent_id: str,
        group_id: str,
        month: str,  # YYYY-MM
        weekly_summaries: List[GroupSummary]
    ) -> Optional[GroupSummary]:
        """Aggregate weekly summaries into a monthly summary."""
        if len(weekly_summaries) < 2:
            return None

        # Format weekly summaries
        weekly_text = "\n\n".join([
            f"**Week {s.period_start}**\n"
            f"Summary: {s.summary}\n"
            f"Topics: {', '.join(s.topics)}\n"
            f"Activity: {s.activity_score:.1f}, Messages: {s.message_count}"
            for s in sorted(weekly_summaries, key=lambda x: x.period_start)
        ])

        # Calculate month boundaries
        year, mon = int(month[:4]), int(month[5:7])
        month_start = f"{month}-01"
        if mon == 12:
            month_end = f"{year + 1}-01-01"
        else:
            month_end = f"{year}-{mon + 1:02d}-01"
        month_end = (datetime.strptime(month_end, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

        prompt = MONTHLY_SUMMARY_PROMPT.format(
            group_id=group_id,
            month=month,
            weekly_summaries=weekly_text
        )

        try:
            response = self.llm_client.chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=0.3
            )
            data = self.llm_client.extract_json(response)

            total_messages = sum(s.message_count for s in weekly_summaries)
            avg_activity = sum(s.activity_score for s in weekly_summaries) / len(weekly_summaries)

            summary = GroupSummary(
                agent_id=agent_id,
                group_id=group_id,
                period_type=PeriodType.MONTHLY,
                period_start=month_start,
                period_end=month_end,
                summary=data.get("summary", f"Month of {month}"),
                topics=data.get("topics", [])[:7],
                highlights=data.get("highlights", [])[:4],
                active_users=self._merge_active_users(weekly_summaries),
                message_count=total_messages,
                activity_score=avg_activity,
                aggregated_from=[s.summary_id for s in weekly_summaries]
            )

            self.summary_store.add_summary(summary)

            # Check for era shift
            if data.get("is_era_shift"):
                self._handle_era_shift(agent_id, group_id, summary, data.get("theme", ""))

            return summary

        except Exception as e:
            print(f"[SummaryAggregator] Monthly aggregation failed: {e}")
            return None

    def _merge_active_users(self, summaries: List[GroupSummary]) -> List[str]:
        """Merge active users from multiple summaries, ranked by frequency."""
        counter = Counter()
        for s in summaries:
            for user in s.active_users:
                counter[user] += 1
        return [user for user, _ in counter.most_common(5)]

    def _handle_era_shift(
        self,
        agent_id: str,
        group_id: str,
        monthly_summary: GroupSummary,
        theme: str
    ):
        """Handle era shift detection - to be implemented with GroupProfile update."""
        print(f"[SummaryAggregator] Era shift detected for {group_id}: {theme}")
        # TODO: Create era summary and update GroupProfile.current_era

    def run_aggregation_cycle(self, agent_id: str, group_id: str):
        """
        Run full aggregation cycle:
        1. Update decay scores
        2. Aggregate pending dailies to weekly
        3. Aggregate pending weeklies to monthly
        4. Prune expired summaries
        """
        # Update decay scores
        self.summary_store.update_decay_scores(group_id)

        # Get dailies ready for weekly aggregation
        pending_dailies = self.summary_store.get_summaries_to_aggregate(group_id, "daily")
        if pending_dailies:
            # Group by week
            by_week = self._group_by_week(pending_dailies)
            for week_start, dailies in by_week.items():
                self.aggregate_to_weekly(agent_id, group_id, week_start, dailies)

        # Get weeklies ready for monthly aggregation
        pending_weeklies = self.summary_store.get_summaries_to_aggregate(group_id, "weekly")
        if pending_weeklies:
            by_month = self._group_by_month(pending_weeklies)
            for month, weeklies in by_month.items():
                self.aggregate_to_monthly(agent_id, group_id, month, weeklies)

        # Prune expired
        self.summary_store.prune_expired(group_id)

    def _group_by_week(self, summaries: List[GroupSummary]) -> Dict[str, List[GroupSummary]]:
        """Group summaries by ISO week (Monday start)."""
        result = {}
        for s in summaries:
            date = datetime.strptime(s.period_start, "%Y-%m-%d")
            # Get Monday of that week
            monday = date - timedelta(days=date.weekday())
            week_key = monday.strftime("%Y-%m-%d")
            if week_key not in result:
                result[week_key] = []
            result[week_key].append(s)
        return result

    def _group_by_month(self, summaries: List[GroupSummary]) -> Dict[str, List[GroupSummary]]:
        """Group summaries by month (YYYY-MM)."""
        result = {}
        for s in summaries:
            month_key = s.period_start[:7]  # YYYY-MM
            if month_key not in result:
                result[month_key] = []
            result[month_key].append(s)
        return result
```

### Commit 4: Integrate with main.py

File: `main.py`

Add import at top:
```python
from services.summary_aggregator import SummaryAggregator
from database.group_summary_store import GroupSummaryStore
```

In `SimpleMemSystem.__init__`, after other store initializations:
```python
# Group summary store for hierarchical summaries
self.group_summary_store = GroupSummaryStore(
    db_path=self.db_path,
    embedding_model=self.embedding_model,
    agent_id=agent_id
)
self.summary_aggregator = SummaryAggregator(
    summary_store=self.group_summary_store,
    llm_client=self.llm_client
)
```

In `_post_batch_processing`, after profile generation:
```python
# Generate daily summary if enough messages processed today
today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
today_messages = [m for m in unprocessed if m.get('timestamp', '').startswith(today)]
if len(today_messages) >= 5:
    self.summary_aggregator.generate_daily_summary(
        agent_id=self.agent_id,
        group_id=original_group_id,
        date=today,
        messages=today_messages
    )

# Run aggregation cycle periodically (every 10 batches)
if hasattr(self, '_batch_count'):
    self._batch_count += 1
else:
    self._batch_count = 1

if self._batch_count % 10 == 0:
    self.summary_aggregator.run_aggregation_cycle(self.agent_id, original_group_id)
```

### Commit 5: Update hybrid_retriever to use summaries

File: `core/hybrid_retriever.py`

In `retrieve_for_context`, after getting group_profile:
```python
# Get hierarchical summaries for context
if self.group_summary_store:
    summaries = self.group_summary_store.get_context_summaries(
        group_id=group_id,
        limit_daily=3,   # Last 3 days
        limit_weekly=2,  # Last 2 weeks
        limit_monthly=1  # Last month
    )

    # Build summary context string
    summary_context = self._build_summary_context(summaries)
    context["historical_context"] = summary_context
```

Add method:
```python
def _build_summary_context(self, summaries: Dict[str, List]) -> str:
    """Build context string from hierarchical summaries."""
    parts = []

    if summaries.get("monthly"):
        s = summaries["monthly"][0]
        parts.append(f"Last month ({s.period_start[:7]}): {s.summary}")

    if summaries.get("weekly"):
        for s in summaries["weekly"][:2]:
            parts.append(f"Week of {s.period_start}: {s.summary}")

    if summaries.get("daily"):
        recent = summaries["daily"][:3]
        if recent:
            parts.append("Recent days: " + " | ".join([
                f"{s.period_start[-5:]}: {s.topics[0] if s.topics else 'general'}"
                for s in recent
            ]))

    return "\n".join(parts) if parts else ""
```

### Commit 6: Add API endpoint for summaries

File: `api.py`

Add response model:
```python
class GroupSummaryResponse(BaseModel):
    summary_id: str
    period_type: str
    period_start: str
    period_end: str
    summary: str
    topics: List[str]
    message_count: int
    activity_score: float
    decay_score: float
```

Add endpoint:
```python
@app.get("/v1/groups/{group_id}/summaries")
async def get_group_summaries(
    group_id: str,
    agent_id: str,
    period_type: Optional[str] = None,
    limit: int = 10
):
    """Get hierarchical summaries for a group."""
    memory_id = f"{agent_id}:"
    system = get_memory_system(memory_id)

    if period_type:
        summaries = system.group_summary_store.get_summaries_for_period(
            group_id, period_type
        )
    else:
        summaries_dict = system.group_summary_store.get_context_summaries(
            group_id,
            limit_daily=limit,
            limit_weekly=limit,
            limit_monthly=limit
        )
        summaries = []
        for period_summaries in summaries_dict.values():
            summaries.extend(period_summaries)

    return {
        "group_id": group_id,
        "summaries": [
            GroupSummaryResponse(
                summary_id=s.summary_id,
                period_type=s.period_type.value,
                period_start=s.period_start,
                period_end=s.period_end,
                summary=s.summary,
                topics=s.topics,
                message_count=s.message_count,
                activity_score=s.activity_score,
                decay_score=s.decay_score
            )
            for s in sorted(summaries, key=lambda x: x.period_end, reverse=True)[:limit]
        ]
    }
```

### Commit 7: Add manual aggregation trigger endpoint

File: `api.py`

```python
@app.post("/v1/groups/{group_id}/aggregate")
async def trigger_aggregation(group_id: str, agent_id: str):
    """Manually trigger summary aggregation cycle."""
    memory_id = f"{agent_id}:"
    system = get_memory_system(memory_id)

    try:
        system.summary_aggregator.run_aggregation_cycle(agent_id, group_id)

        # Get updated counts
        summaries = system.group_summary_store.get_context_summaries(group_id)

        return {
            "success": True,
            "counts": {
                "daily": len(summaries.get("daily", [])),
                "weekly": len(summaries.get("weekly", [])),
                "monthly": len(summaries.get("monthly", []))
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
```

## Testing

After implementation, test with:

```bash
# 1. Send 50+ messages over multiple "days" (simulate with timestamps)
# 2. Trigger aggregation
curl -X POST "http://VM:8080/v1/groups/telegram_-200001/aggregate?agent_id=test"

# 3. Check summaries
curl "http://VM:8080/v1/groups/telegram_-200001/summaries?agent_id=test"

# 4. Verify decay by checking decay_score values
# 5. Check context endpoint includes historical_context
```

## Summary

| Commit | Files | Description |
|--------|-------|-------------|
| 1 | models/group_memory.py | Add GroupSummary model, update GroupProfile |
| 2 | database/group_summary_store.py | New store with decay logic |
| 3 | services/summary_aggregator.py | Aggregation service with LLM |
| 4 | main.py | Integration with batch processing |
| 5 | core/hybrid_retriever.py | Use summaries in context |
| 6 | api.py | GET endpoint for summaries |
| 7 | api.py | POST endpoint for manual aggregation |

## Notes for Implementation

1. Start with commits 1-3 (models and stores) before integrating
2. Test each component independently before integration
3. The decay calculation uses exponential decay with configurable half-life
4. Era detection is basic - can be enhanced later with more sophisticated topic drift detection
5. All LLM calls use structured output for reliable JSON parsing
