"""
SummaryAggregator - Generates and aggregates volume-based hierarchical summaries.

Volume-based summarization:
- Micro: Every ~50 messages (raw messages → summary)
- Chunk: Every 5 micros (~250 messages, aggregates micros)
- Block: Every 5 chunks (~1250 messages, aggregates chunks)

Responsibilities:
- Generate micro summaries from raw messages
- Aggregate micros → chunks → blocks
- Track temporal context (when messages occurred)
- Update decay based on message volume
"""
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from collections import Counter

from models.group_memory import GroupSummary, SummaryLevel
from database.group_summary_store import GroupSummaryStore, SUMMARY_CONFIG
from utils.llm_client import LLMClient


MICRO_SUMMARY_PROMPT = """Summarize this batch of group messages.

Group: {group_id}
Messages {msg_start} to {msg_end} ({msg_count} messages)
Time span: {time_start} to {time_end}

Messages:
{messages}

Return JSON:
{{
  "summary": "2-3 sentences summarizing the key discussions",
  "topics": ["topic1", "topic2", ...],  // 3-5 main topics
  "highlights": ["highlight1", ...],     // 1-2 notable events
  "active_users": ["user1", ...]         // Top 3-5 contributors
}}"""


CHUNK_SUMMARY_PROMPT = """Aggregate these micro summaries into a chunk summary.

Group: {group_id}
Covering messages {msg_start} to {msg_end} ({total_msgs} messages)
Time span: {time_start} to {time_end} ({duration_hours:.1f} hours)

Micro summaries:
{micro_summaries}

Return JSON:
{{
  "summary": "3-4 sentences summarizing this period",
  "topics": ["topic1", ...],        // 5-7 main topics (merged from micros)
  "highlights": ["highlight1", ...], // 2-3 key moments
  "trend": "increasing|stable|decreasing"  // Activity trend
}}"""


BLOCK_SUMMARY_PROMPT = """Aggregate these chunk summaries into a block summary.

Group: {group_id}
Covering messages {msg_start} to {msg_end} ({total_msgs} messages)
Time span: {time_start} to {time_end} ({duration_hours:.1f} hours)

Chunk summaries:
{chunk_summaries}

Return JSON:
{{
  "summary": "4-5 sentences summarizing this larger period",
  "topics": ["topic1", ...],        // Top 7 topics
  "highlights": ["highlight1", ...], // 3-4 significant events
  "theme": "Brief theme of this period",
  "activity_pattern": "Description of activity patterns observed"
}}"""


class SummaryAggregator:
    """Generates and aggregates volume-based hierarchical group summaries."""

    def __init__(
        self,
        summary_store: GroupSummaryStore,
        llm_client: LLMClient = None
    ):
        self.summary_store = summary_store
        self.llm_client = llm_client or LLMClient(use_streaming=False)
        self.config = SUMMARY_CONFIG

    def generate_micro_summary(
        self,
        agent_id: str,
        group_id: str,
        messages: List[Dict[str, Any]],
        message_start_index: int
    ) -> Optional[GroupSummary]:
        """
        Generate a micro summary from raw messages.

        Args:
            agent_id: Agent ID
            group_id: Group ID
            messages: List of message dicts with 'content', 'speaker', 'timestamp'
            message_start_index: Global index of first message

        Returns:
            GroupSummary or None if too few messages
        """
        if len(messages) < 3:
            return None

        # Extract temporal info
        timestamps = [m.get('timestamp', '') for m in messages if m.get('timestamp')]
        time_start = min(timestamps) if timestamps else datetime.now(timezone.utc).isoformat()
        time_end = max(timestamps) if timestamps else datetime.now(timezone.utc).isoformat()

        # Calculate duration
        try:
            t_start = datetime.fromisoformat(time_start.replace('Z', '+00:00'))
            t_end = datetime.fromisoformat(time_end.replace('Z', '+00:00'))
            duration_hours = (t_end - t_start).total_seconds() / 3600
        except:
            duration_hours = 0.0

        activity_rate = len(messages) / max(duration_hours, 0.1)

        # Format messages for prompt
        messages_text = "\n".join([
            f"[{m.get('speaker', 'unknown')}]: {m.get('content', '')[:200]}"
            for m in messages[:50]
        ])

        message_end_index = message_start_index + len(messages) - 1

        prompt = MICRO_SUMMARY_PROMPT.format(
            group_id=group_id,
            msg_start=message_start_index,
            msg_end=message_end_index,
            msg_count=len(messages),
            time_start=time_start[:19],
            time_end=time_end[:19],
            messages=messages_text
        )

        try:
            response = self.llm_client.chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=0.3
            )
            data = self.llm_client.extract_json(response)

            summary = GroupSummary(
                agent_id=agent_id,
                group_id=group_id,
                level=SummaryLevel.MICRO,
                message_start=message_start_index,
                message_end=message_end_index,
                message_count=len(messages),
                time_start=time_start,
                time_end=time_end,
                duration_hours=duration_hours,
                activity_rate=activity_rate,
                summary=data.get("summary", f"Messages {message_start_index}-{message_end_index}"),
                topics=data.get("topics", [])[:5],
                highlights=data.get("highlights", [])[:2],
                active_users=data.get("active_users", [])[:5]
            )

            self.summary_store.add_summary(summary)
            print(f"[SummaryAggregator] Created micro summary: msgs {message_start_index}-{message_end_index}")
            return summary

        except Exception as e:
            print(f"[SummaryAggregator] Micro summary generation failed: {e}")
            return None

    def aggregate_to_chunk(
        self,
        agent_id: str,
        group_id: str,
        micro_summaries: List[GroupSummary]
    ) -> Optional[GroupSummary]:
        """Aggregate micro summaries into a chunk summary."""
        if len(micro_summaries) < 2:
            return None

        # Sort by message_start
        micros = sorted(micro_summaries, key=lambda x: x.message_start)

        # Calculate ranges
        msg_start = micros[0].message_start
        msg_end = micros[-1].message_end
        total_msgs = sum(m.message_count for m in micros)
        time_start = micros[0].time_start
        time_end = micros[-1].time_end

        # Calculate duration
        try:
            t_start = datetime.fromisoformat(time_start.replace('Z', '+00:00'))
            t_end = datetime.fromisoformat(time_end.replace('Z', '+00:00'))
            duration_hours = (t_end - t_start).total_seconds() / 3600
        except:
            duration_hours = sum(m.duration_hours for m in micros)

        activity_rate = total_msgs / max(duration_hours, 0.1)

        # Format micro summaries for prompt
        micro_text = "\n\n".join([
            f"**Messages {m.message_start}-{m.message_end}** ({m.message_count} msgs, {m.activity_rate:.1f} msg/hr)\n"
            f"Summary: {m.summary}\n"
            f"Topics: {', '.join(m.topics)}\n"
            f"Highlights: {', '.join(m.highlights)}"
            for m in micros
        ])

        prompt = CHUNK_SUMMARY_PROMPT.format(
            group_id=group_id,
            msg_start=msg_start,
            msg_end=msg_end,
            total_msgs=total_msgs,
            time_start=time_start[:19],
            time_end=time_end[:19],
            duration_hours=duration_hours,
            micro_summaries=micro_text
        )

        try:
            response = self.llm_client.chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=0.3
            )
            data = self.llm_client.extract_json(response)

            summary = GroupSummary(
                agent_id=agent_id,
                group_id=group_id,
                level=SummaryLevel.CHUNK,
                message_start=msg_start,
                message_end=msg_end,
                message_count=total_msgs,
                time_start=time_start,
                time_end=time_end,
                duration_hours=duration_hours,
                activity_rate=activity_rate,
                summary=data.get("summary", f"Chunk: msgs {msg_start}-{msg_end}"),
                topics=data.get("topics", [])[:7],
                highlights=data.get("highlights", [])[:3],
                active_users=self._merge_active_users(micros),
                aggregated_from=[m.summary_id for m in micros]
            )

            self.summary_store.add_summary(summary)

            # Mark micros as aggregated
            self.summary_store.mark_as_aggregated([m.summary_id for m in micros])

            print(f"[SummaryAggregator] Created chunk summary: msgs {msg_start}-{msg_end}")
            return summary

        except Exception as e:
            print(f"[SummaryAggregator] Chunk aggregation failed: {e}")
            return None

    def aggregate_to_block(
        self,
        agent_id: str,
        group_id: str,
        chunk_summaries: List[GroupSummary]
    ) -> Optional[GroupSummary]:
        """Aggregate chunk summaries into a block summary."""
        if len(chunk_summaries) < 2:
            return None

        # Sort by message_start
        chunks = sorted(chunk_summaries, key=lambda x: x.message_start)

        # Calculate ranges
        msg_start = chunks[0].message_start
        msg_end = chunks[-1].message_end
        total_msgs = sum(c.message_count for c in chunks)
        time_start = chunks[0].time_start
        time_end = chunks[-1].time_end

        # Calculate duration
        try:
            t_start = datetime.fromisoformat(time_start.replace('Z', '+00:00'))
            t_end = datetime.fromisoformat(time_end.replace('Z', '+00:00'))
            duration_hours = (t_end - t_start).total_seconds() / 3600
        except:
            duration_hours = sum(c.duration_hours for c in chunks)

        activity_rate = total_msgs / max(duration_hours, 0.1)

        # Format chunk summaries for prompt
        chunk_text = "\n\n".join([
            f"**Messages {c.message_start}-{c.message_end}** ({c.message_count} msgs, {c.duration_hours:.1f}h)\n"
            f"Summary: {c.summary}\n"
            f"Topics: {', '.join(c.topics)}\n"
            f"Activity: {c.activity_rate:.1f} msgs/hr"
            for c in chunks
        ])

        prompt = BLOCK_SUMMARY_PROMPT.format(
            group_id=group_id,
            msg_start=msg_start,
            msg_end=msg_end,
            total_msgs=total_msgs,
            time_start=time_start[:19],
            time_end=time_end[:19],
            duration_hours=duration_hours,
            chunk_summaries=chunk_text
        )

        try:
            response = self.llm_client.chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=0.3
            )
            data = self.llm_client.extract_json(response)

            summary = GroupSummary(
                agent_id=agent_id,
                group_id=group_id,
                level=SummaryLevel.BLOCK,
                message_start=msg_start,
                message_end=msg_end,
                message_count=total_msgs,
                time_start=time_start,
                time_end=time_end,
                duration_hours=duration_hours,
                activity_rate=activity_rate,
                summary=data.get("summary", f"Block: msgs {msg_start}-{msg_end}"),
                topics=data.get("topics", [])[:7],
                highlights=data.get("highlights", [])[:4],
                active_users=self._merge_active_users(chunks),
                aggregated_from=[c.summary_id for c in chunks]
            )

            self.summary_store.add_summary(summary)

            # Mark chunks as aggregated
            self.summary_store.mark_as_aggregated([c.summary_id for c in chunks])

            print(f"[SummaryAggregator] Created block summary: msgs {msg_start}-{msg_end}")
            return summary

        except Exception as e:
            print(f"[SummaryAggregator] Block aggregation failed: {e}")
            return None

    def _merge_active_users(self, summaries: List[GroupSummary]) -> List[str]:
        """Merge active users from multiple summaries, ranked by frequency."""
        counter = Counter()
        for s in summaries:
            for user in s.active_users:
                counter[user] += 1
        return [user for user, _ in counter.most_common(5)]

    def check_and_aggregate(
        self,
        agent_id: str,
        group_id: str,
        current_message_count: int
    ):
        """
        Check if aggregation is needed and perform it.

        Call this after adding new messages to check if:
        1. We have enough micros for a chunk
        2. We have enough chunks for a block

        Args:
            agent_id: Agent ID
            group_id: Group ID
            current_message_count: Total messages in the group now
        """
        # Update decay scores first
        self.summary_store.update_decay_scores(group_id, current_message_count)

        # Check if we have enough micros for a chunk
        pending_micros = self.summary_store.get_summaries_to_aggregate(group_id, "micro")
        micro_threshold = self.config["micro"]["aggregate_count"]

        if len(pending_micros) >= micro_threshold:
            # Take the oldest N micros for aggregation
            micros_to_aggregate = sorted(pending_micros, key=lambda x: x.message_start)[:micro_threshold]
            self.aggregate_to_chunk(agent_id, group_id, micros_to_aggregate)

        # Check if we have enough chunks for a block
        pending_chunks = self.summary_store.get_summaries_to_aggregate(group_id, "chunk")
        chunk_threshold = self.config["chunk"]["aggregate_count"]

        if len(pending_chunks) >= chunk_threshold:
            chunks_to_aggregate = sorted(pending_chunks, key=lambda x: x.message_start)[:chunk_threshold]
            self.aggregate_to_block(agent_id, group_id, chunks_to_aggregate)

        # Prune expired summaries
        self.summary_store.prune_expired(group_id)

    def should_generate_micro(self, group_id: str, total_messages: int) -> bool:
        """
        Check if we should generate a new micro summary.

        Returns True if there are enough unsummarized messages.
        """
        last_summarized = self.summary_store.get_last_message_index(group_id)
        unsummarized_count = total_messages - last_summarized - 1

        return unsummarized_count >= self.config["micro"]["message_threshold"]

    def get_unsummarized_range(self, group_id: str, total_messages: int) -> tuple:
        """
        Get the range of messages that need to be summarized.

        Returns:
            (start_index, end_index) or (None, None) if nothing to summarize
        """
        last_summarized = self.summary_store.get_last_message_index(group_id)
        start_index = last_summarized + 1

        threshold = self.config["micro"]["message_threshold"]
        if total_messages - start_index < threshold:
            return (None, None)

        end_index = start_index + threshold - 1
        return (start_index, end_index)
