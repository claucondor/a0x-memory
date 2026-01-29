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
