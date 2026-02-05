"""
Unified Memory Store - facade over individual tables.

This provides the same interface as the old UnifiedMemoryStore,
but delegates to individual table classes for cleaner separation.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from models.memory_entry import MemoryEntry
from models.group_memory import (
    GroupMemory, UserMemory, InteractionMemory,
    CrossGroupMemory, CrossAgentLink, AgentResponse,
    MemoryType
)
from database.tables import (
    DMMemoriesTable,
    GroupMemoriesTable,
    UserMemoriesTable,
    InteractionMemoriesTable,
    CrossGroupMemoriesTable,
    AgentResponsesTable,
    ConversationSummariesTable,
    CrossAgentLinksTable
)
from database.base import LanceDBConnection
from utils.embedding import EmbeddingModel
import config


class MemoryStore:
    """
    Facade that provides same interface as UnifiedMemoryStore.
    Delegates to individual table classes.
    """

    def __init__(
        self,
        agent_id: str,
        db_base_path: str = None,
        embedding_model: EmbeddingModel = None,
        storage_options: Optional[Dict[str, Any]] = None,
        llm_client = None
    ):
        self.agent_id = agent_id
        self.embedding_model = embedding_model or EmbeddingModel()
        self.storage_options = storage_options
        self.llm_client = llm_client

        # Configure connection
        LanceDBConnection.configure(storage_options)

        # Initialize tables
        self.dm_memories = DMMemoriesTable(agent_id, embedding_model, storage_options)
        self.group_memories = GroupMemoriesTable(agent_id, embedding_model, storage_options)
        self.user_memories = UserMemoriesTable(agent_id, embedding_model, storage_options)
        self.interaction_memories = InteractionMemoriesTable(agent_id, embedding_model, storage_options)
        self.cross_group_memories = CrossGroupMemoriesTable(agent_id, embedding_model, storage_options)
        self.agent_responses = AgentResponsesTable(agent_id, embedding_model, storage_options)
        self.conversation_summaries = ConversationSummariesTable(agent_id, embedding_model, storage_options)

        # Cross-agent links (global DB)
        self.cross_agent_links = CrossAgentLinksTable(storage_options)

        print(f"[MemoryStore] Initialized for agent {agent_id}")

    # ============================================================
    # DM Memory Operations (SimpleMem Compatible)
    # ============================================================

    def add_memory_entries(self, entries: List[MemoryEntry], user_id: str = None):
        """Add DM memory entries (SimpleMem compatible)."""
        self.dm_memories.add_batch(entries, user_id)

    def search_memories(
        self,
        query: str,
        user_id: str = None,
        top_k: int = 10,
        query_vector=None
    ) -> List[MemoryEntry]:
        """Search DM memories (SimpleMem compatible)."""
        return self.dm_memories.search_semantic(query, user_id, top_k, query_vector)

    # ============================================================
    # Group Memory Operations
    # ============================================================

    def add_group_memory(self, memory: GroupMemory) -> GroupMemory:
        """Add a group-level memory."""
        return self.group_memories.add(memory)

    def add_group_memories_batch(self, memories: List[GroupMemory]) -> List[GroupMemory]:
        """Add multiple group-level memories."""
        return self.group_memories.add_batch(memories)

    def search_group_memories(
        self,
        group_id: str,
        query: str,
        limit: int = 10,
        memory_type: Optional[MemoryType] = None,
        query_vector=None
    ) -> List[GroupMemory]:
        """Search group memories by semantic similarity."""
        if query_vector is None:
            query_vector = self.embedding_model.encode_single(query, is_query=True)
        return self.group_memories.search_semantic(group_id, query_vector, limit, memory_type)

    def keyword_search_group_memories(self, group_id: str, keywords: List[str], top_k: int = 5):
        """FTS keyword search on group_memories."""
        return self.group_memories.search_keyword(group_id, keywords, top_k)

    # ============================================================
    # User Memory Operations
    # ============================================================

    def add_user_memory(self, memory: UserMemory) -> UserMemory:
        """Add a user memory."""
        return self.user_memories.add(memory)

    def add_user_memories_batch(self, memories: List[UserMemory]) -> List[UserMemory]:
        """Add multiple user memories."""
        return self.user_memories.add_batch(memories)

    def search_user_memories(self, group_id: str, user_id: str, query: str, limit: int = 10) -> List[UserMemory]:
        """Search user memories for a specific user."""
        query_vector = self.embedding_model.encode_single(query, is_query=True)
        return self.user_memories.search_semantic(group_id, user_id, query_vector, limit)

    def search_user_memories_in_group(
        self,
        group_id: str,
        query: str,
        limit: int = 10,
        exclude_user_id: Optional[str] = None,
        query_vector=None
    ) -> List[UserMemory]:
        """Search user memories across ALL users in a group."""
        if query_vector is None:
            query_vector = self.embedding_model.encode_single(query, is_query=True)
        return self.user_memories.search_semantic_in_group(group_id, query_vector, limit, exclude_user_id)

    def keyword_search_user_memories(self, group_id: str, keywords: List[str], top_k: int = 5):
        """FTS keyword search on user_memories."""
        return self.user_memories.search_keyword(group_id, keywords, top_k)

    # ============================================================
    # Interaction Memory Operations
    # ============================================================

    def add_interaction_memory(self, memory: InteractionMemory) -> InteractionMemory:
        """Add an interaction memory."""
        return self.interaction_memories.add(memory)

    def add_interaction_memories_batch(self, memories: List[InteractionMemory]) -> List[InteractionMemory]:
        """Add multiple interaction memories."""
        return self.interaction_memories.add_batch(memories)

    def search_interaction_memories(
        self,
        group_id: str,
        query: str,
        speaker_id: str = None,
        listener_id: str = None,
        limit: int = 10
    ):
        """Search interaction memories."""
        query_vector = self.embedding_model.encode_single(query, is_query=True)
        return self.interaction_memories.search_semantic(group_id, speaker_id, listener_id, query_vector, limit)

    def keyword_search_interactions(self, group_id: str, keywords: List[str], top_k: int = 5):
        """FTS keyword search on interaction_memories."""
        return self.interaction_memories.search_keyword(group_id, keywords, top_k)

    def search_interactions(
        self,
        group_id: str,
        speaker_id: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 10,
        query_vector=None
    ) -> List:
        """Search interaction memories (compatibility method)."""
        if query_vector is None and query:
            query_vector = self.embedding_model.encode_single(query, is_query=True)
        return self.interaction_memories.search_semantic(group_id, speaker_id, None, query_vector, limit)

    # ============================================================
    # Cross-Group Memory Operations
    # ============================================================

    def add_cross_group_memory(self, memory: CrossGroupMemory) -> CrossGroupMemory:
        """Add a cross-group memory."""
        return self.cross_group_memories.add(memory)

    def search_cross_group(
        self,
        universal_user_id: str,
        query: str,
        limit: int = 10,
        query_vector=None
    ) -> List:
        """Search cross-group memories for a user."""
        if query_vector is None:
            query_vector = self.embedding_model.encode_single(query, is_query=True)
        return self.cross_group_memories.search_semantic(universal_user_id, query_vector, limit)

    # ============================================================
    # Agent Response Operations
    # ============================================================

    def add_agent_response(self, response: AgentResponse) -> str:
        """Add an agent response."""
        return self.agent_responses.add(response)

    def search_agent_responses_by_trigger(
        self,
        query: str,
        scope: str = None,
        group_id: str = None,
        user_id: str = None,
        limit: int = 5
    ) -> List[dict]:
        """
        Search for similar questions asked before.

        Uses trigger_vector to find what users asked previously.

        Args:
            query: The search query
            scope: Filter by scope ("global" | "user" | "group")
            group_id: Filter by group (for group context)
            user_id: Filter by user (for DM context)
            limit: Max results
        """
        query_vector = self.embedding_model.encode_single(query, is_query=True)
        return self.agent_responses.search_by_trigger(query_vector, scope, group_id, user_id, limit)

    def search_agent_responses_by_response(
        self,
        query: str,
        user_id: str = None,
        group_id: str = None,
        limit: int = 5
    ) -> List[dict]:
        """
        Search for similar responses given before.

        Uses response_vector to find what the agent said previously.
        """
        query_vector = self.embedding_model.encode_single(query, is_query=True)
        return self.agent_responses.search_by_response(query_vector, user_id, group_id, limit)

    def add_agent_response_with_vectors(self, response: AgentResponse) -> str:
        """
        Add response with dual vectors.

        This is the new recommended method for storing agent responses.
        """
        return self.agent_responses.add_with_vectors(response)

    # ============================================================
    # Conversation Summary Operations
    # ============================================================

    def get_or_create_summary(self, thread_id: str) -> str:
        """Get or create a conversation summary for a thread."""
        return self.conversation_summaries.get(thread_id)

    def update_summary(
        self,
        thread_id: str,
        new_messages: List[str],
        current_summary: str
    ) -> str:
        """Update conversation summary with new messages using LLM."""
        if not self.llm_client:
            return current_summary

        if not new_messages:
            return current_summary

        # For now, just append - full LLM logic can be added later
        messages_joined = "\n".join([f"- {msg}" for msg in new_messages])
        updated_summary = f"{current_summary}\n{messages_joined}"

        # Update in table
        self.conversation_summaries.update(
            thread_id,
            updated_summary,
            len(new_messages),
            datetime.now(timezone.utc).isoformat()
        )

        return updated_summary

    def get_conversation_summary_by_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get full conversation summary record."""
        return self.conversation_summaries.get_by_thread(thread_id)

    # ============================================================
    # Cross-Agent Link Operations
    # ============================================================

    def add_cross_agent_link(self, link: CrossAgentLink) -> CrossAgentLink:
        """Add a cross-agent link."""
        return self.cross_agent_links.add(link)

    # ============================================================
    # Unified Search (Multi-table)
    # ============================================================

    def search_all(
        self,
        query: str,
        context: Dict[str, Any],
        limit_per_table: int = 5
    ) -> Dict[str, List[Any]]:
        """
        Search across all relevant tables based on context.

        Args:
            query: Search query
            context: Dict with group_id, user_id, etc.
            limit_per_table: Max results per table

        Returns:
            Dict with results from each table type
        """
        results = {
            "dm_memories": [],
            "group_memories": [],
            "user_memories": [],
            "interaction_memories": [],
            "cross_group_memories": []
        }

        group_id = context.get('group_id')
        user_id = context.get('user_id')
        is_group = group_id is not None and not str(group_id).startswith('dm_')

        if is_group:
            # Group context - search group tables
            results["group_memories"] = self.search_group_memories(
                group_id, query, limit=limit_per_table
            )
            results["user_memories"] = self.search_user_memories_in_group(
                group_id, query, limit=limit_per_table, exclude_user_id=user_id
            )
            results["interaction_memories"] = self.search_interactions(
                group_id, speaker_id=user_id, query=query, limit=limit_per_table
            )

            # Cross-group if user_id available
            if user_id:
                results["cross_group_memories"] = self.search_cross_group(
                    user_id, query, limit=limit_per_table
                )
        else:
            # DM context - search memories table
            results["dm_memories"] = self.search_memories(
                query, user_id=user_id, top_k=limit_per_table
            )

        return results

    # ============================================================
    # Compatibility Methods (VectorStore)
    # ============================================================

    def semantic_search(self, query: str, top_k: int = 5, user_id: str = None) -> List[MemoryEntry]:
        """Semantic search (SimpleMem VectorStore compatible)."""
        return self.search_memories(query, user_id=user_id, top_k=top_k)

    def keyword_search(self, keywords: List[str], top_k: int = 3, user_id: str = None) -> List[MemoryEntry]:
        """Keyword search (SimpleMem VectorStore compatible)."""
        return self.dm_memories.search_keyword(keywords, top_k, user_id)

    def keyword_search_with_scores(self, keywords: List[str], top_k: int = 3):
        """Keyword search with scores (SimpleMem VectorStore compatible)."""
        # For simplicity, return basic search
        results = self.dm_memories.search_keyword(keywords, top_k)
        return [(r, 1.0) for r in results]

    def add_entries(self, entries: List[MemoryEntry], agent_id: str = None, user_id: str = None):
        """Add entries (SimpleMem VectorStore compatible)."""
        self.add_memory_entries(entries, user_id)

    def get_all_entries(self) -> List[MemoryEntry]:
        """Get all entries (SimpleMem VectorStore compatible)."""
        return self.dm_memories.get_all()

    def count_entries(self) -> int:
        """Count entries for this agent."""
        return self.dm_memories.count()

    # ============================================================
    # Optimization
    # ============================================================

    def optimize_tables(self):
        """Compact all LanceDB tables."""
        self.dm_memories.optimize()
        self.group_memories.optimize()
        self.user_memories.optimize()
        self.interaction_memories.optimize()
        self.cross_group_memories.optimize()
        self.agent_responses.optimize()
        self.conversation_summaries.optimize()
        self.cross_agent_links.optimize()

    # ============================================================
    # Stats
    # ============================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about all tables."""
        return {
            "dm_memories": self.dm_memories.count(),
            "group_memories": self.group_memories.count(),
            "user_memories": self.user_memories.count(),
            "interaction_memories": self.interaction_memories.count(),
            "cross_group_memories": self.cross_group_memories.count(),
            "agent_responses": self.agent_responses.count(),
            "conversation_summaries": self.conversation_summaries.count(),
            "cross_agent_links": self.cross_agent_links.count()
        }

    # ============================================================
    # Internal Table Access (for compatibility with legacy code)
    # ============================================================

    @property
    def memories_table(self):
        """Access to DM memories table (for compatibility)."""
        return self.dm_memories.table

    @property
    def group_memories_table(self):
        """Access to group memories table (for compatibility)."""
        return self.group_memories.table

    @property
    def user_memories_table(self):
        """Access to user memories table (for compatibility)."""
        return self.user_memories.table

    @property
    def interaction_memories_table(self):
        """Access to interaction memories table (for compatibility)."""
        return self.interaction_memories.table

    @property
    def agent_responses_table(self):
        """Access to agent responses table (for compatibility)."""
        return self.agent_responses.table

    @property
    def conversation_summaries_table(self):
        """Access to conversation summaries table (for compatibility)."""
        return self.conversation_summaries.table

    @property
    def cross_group_memories_table(self):
        """Access to cross-group memories table (for compatibility)."""
        return self.cross_group_memories.table

    # ============================================================
    # Additional compatibility methods
    # ============================================================

    def clear_agent_data(self):
        """Clear all data for this agent."""
        # For now, just log - actual deletion can be added later
        print(f"[MemoryStore] clear_agent_data called for {self.agent_id}")

    def search_agent_responses(
        self,
        group_id: str,
        query: str,
        user_id: str = None,
        top_k: int = 5
    ):
        """Search agent responses by semantic similarity."""
        query_vector = self.embedding_model.encode_single(query, is_query=True)
        return self.agent_responses.search_semantic(
            query_vector=query_vector,
            group_id=group_id,
            user_id=user_id,
            limit=top_k
        )

    def consolidate_similar_memories(self, group_id: str):
        """Consolidate similar memories in a group."""
        # Placeholder for consolidation logic
        return {"consolidated": 0, "merged": 0}

    def consolidate_all_groups(self):
        """Consolidate similar memories across all groups."""
        # Placeholder for consolidation logic
        return {"groups_processed": 0, "consolidated": 0, "merged": 0}

    # ============================================================
    # Row Conversion Methods (for hybrid_retriever compatibility)
    # ============================================================

    def _row_to_user_memory(self, row: dict):
        """Convert LanceDB row to UserMemory."""
        return self.user_memories._row_to_user_memory(row)

    def _row_to_cross_group_memory(self, row: dict):
        """Convert LanceDB row to CrossGroupMemory."""
        return self.cross_group_memories._row_to_cross_group_memory(row)
