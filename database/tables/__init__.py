"""
Database tables - individual table classes.
"""
from .dm_memories import DMMemoriesTable
from .group_memories import GroupMemoriesTable
from .user_memories import UserMemoriesTable
from .interaction_memories import InteractionMemoriesTable
from .cross_group_memories import CrossGroupMemoriesTable
from .agent_responses import AgentResponsesTable
from .conversation_summaries import ConversationSummariesTable
from .cross_agent_links import CrossAgentLinksTable

__all__ = [
    'DMMemoriesTable',
    'GroupMemoriesTable',
    'UserMemoriesTable',
    'InteractionMemoriesTable',
    'CrossGroupMemoriesTable',
    'AgentResponsesTable',
    'ConversationSummariesTable',
    'CrossAgentLinksTable',
]
