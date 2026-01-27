"""
Group Memory Manager - High-level service for integrating GroupMemoryStore with a0x agents

This service provides a simplified interface for the a0x agent execution pipeline to:
1. Process group messages and create memories
2. Get context for agent responses (multi-level retrieval)
3. Consolidate cross-group patterns
4. Maintain backward compatibility with VectorStore and UserProfileStore
5. Maintain Firestore recent messages window for immediate context

Integration Pattern:
- Incoming message → is_group_message() → process_group_message()
- Agent needs context → get_context_for_agent()
- Background job → consolidate_cross_group_patterns()
"""
import os
import json
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread

from database.group_memory_store import GroupMemoryStore
from database.vector_store import VectorStore
from database.user_profile_store import UserProfileStore
from models.group_memory import (
    GroupMemory, UserMemory, InteractionMemory,
    CrossGroupMemory, MemoryType, PrivacyScope
)
from models.user_profile import UserProfile
from models.memory_entry import MemoryEntry
from utils.embedding import EmbeddingModel
from services.memory_classifier import get_message_analyzer
from services.firestore_window import get_firestore_store
import config


class GroupMemoryManager:
    """
    High-level manager for group memory operations in a0x agents.

    Provides:
    - Message processing and memory creation
    - Multi-level context retrieval for agents
    - Cross-group pattern consolidation
    - Integration with existing VectorStore and UserProfileStore
    """

    def __init__(
        self,
        agent_id: str,
        db_base_path: str = None,
        embedding_model: EmbeddingModel = None
    ):
        """
        Initialize the GroupMemoryManager.

        Args:
            agent_id: The agent ID (e.g., "71f6f657-6800-0892-875f-f26e8c213756")
            db_base_path: Base path for LanceDB storage (default from config)
            embedding_model: Embedding model instance (default creates new one)
        """
        self.agent_id = agent_id
        self.db_base_path = db_base_path or getattr(config, 'DB_PATH', '/data/a0x-memory')
        self.embedding_model = embedding_model or EmbeddingModel()

        # Initialize stores
        self.group_store = GroupMemoryStore(
            agent_id=agent_id,
            db_base_path=self.db_base_path,
            embedding_model=self.embedding_model
        )

        # Optional: Initialize existing stores for backward compatibility
        self._vector_stores = {}  # Cache for VectorStore instances
        self._user_profile_store = None  # Lazy loaded UserProfileStore

        # Initialize Firestore window store for immediate context
        self.firestore = get_firestore_store()
        if self.firestore.is_enabled():
            print(f"[GroupMemoryManager] Firestore window enabled (last {config.RECENT_WINDOW_SIZE} messages)")
        else:
            print(f"[GroupMemoryManager] Firestore disabled - local-only mode")

        print(f"[GroupMemoryManager] Initialized for agent {agent_id}")

    # ==========================================================================
    # MESSAGE PROCESSING
    # ==========================================================================

    def is_group_message(self, platform_identity: Dict[str, Any]) -> bool:
        """
        Determine if a message is from a group context.

        Args:
            platform_identity: Platform identity dict with keys:
                - platform: 'telegram', 'xmtp', 'farcaster', 'twitter', 'direct'
                - chatId (for Telegram): present if group chat
                - conversationId (for XMTP): present if group conversation
                - For 'direct' platform: always False

        Returns:
            True if the message is from a group context
        """
        platform = platform_identity.get('platform', '').lower()

        # Direct messages are never groups
        if platform == 'direct':
            return False

        # Telegram: check if chatId exists and is not a DM
        if platform == 'telegram':
            # Telegram groups have chatId, DMs also have chatId but different format
            # Groups typically have negative chatId or specific patterns
            chat_id = platform_identity.get('chatId')
            if chat_id:
                # Negative chat IDs indicate groups/channels in Telegram
                return str(chat_id).startswith('-')
            return False

        # XMTP: check if conversationId indicates a group
        if platform == 'xmtp':
            # XMTP groups have conversationId with specific format
            conversation_id = platform_identity.get('conversationId')
            # Check if it's a group conversation (not 1:1)
            return conversation_id and '/groups/' in str(conversation_id)

        # Farcaster: casts can be considered group-like
        if platform == 'farcaster':
            # Casts are public, treat as group context
            return True

        # Twitter: tweets are public, treat as group context
        if platform == 'twitter':
            return True

        return False

    def get_group_id(self, platform_identity: Dict[str, Any]) -> Optional[str]:
        """
        Extract a consistent group ID from platform identity.

        Args:
            platform_identity: Platform identity dict

        Returns:
            Group ID string or None if not a group
        """
        if not self.is_group_message(platform_identity):
            return None

        platform = platform_identity.get('platform', '').lower()

        if platform == 'telegram':
            return f"telegram_group_{platform_identity.get('chatId')}"
        elif platform == 'xmtp':
            return f"xmtp_group_{platform_identity.get('conversationId')}"
        elif platform == 'farcaster':
            # For Farcaster, use channel or global
            channel = platform_identity.get('channel', 'global')
            return f"farcaster_{channel}"
        elif platform == 'twitter':
            # For Twitter, use timeline or search context
            return "twitter_public_timeline"

        return None

    def get_user_id(self, platform_identity: Dict[str, Any]) -> str:
        """
        Extract a consistent user ID from platform identity.

        Args:
            platform_identity: Platform identity dict

        Returns:
            User ID string
        """
        platform = platform_identity.get('platform', '').lower()

        if platform == 'telegram':
            # Use telegramId or username
            telegram_id = platform_identity.get('telegramId')
            username = platform_identity.get('username')
            return f"telegram_{telegram_id or username}"
        elif platform == 'xmtp':
            # Use wallet address
            return f"xmtp_{platform_identity.get('walletAddress', 'unknown')}"
        elif platform == 'farcaster':
            # Use fid
            return f"farcaster_{platform_identity.get('fid', 'unknown')}"
        elif platform == 'twitter':
            # Use username
            return f"twitter_{platform_identity.get('username', 'unknown')}"
        elif platform == 'direct':
            # Use clientId
            return f"direct_{platform_identity.get('clientId', 'unknown')}"

        return "unknown_user"

    def process_group_message(
        self,
        message: str,
        platform_identity: Dict[str, Any],
        speaker_info: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a group message and create appropriate memories.

        This analyzes the message and creates:
        - Group memory (if it contains group-level information)
        - User memory (if it reveals user preferences/expertise)
        - Interaction memory (if it's a reply or mention)

        Args:
            message: The message content
            platform_identity: Platform identity dict
            speaker_info: Additional info about the speaker (username, etc.)
            metadata: Additional metadata (message_id, timestamp, etc.)

        Returns:
            Dict with created memory IDs and counts
        """
        if not self.is_group_message(platform_identity):
            # Fall back to VectorStore for DMs
            return self._process_dm_message(message, platform_identity, metadata)

        group_id = self.get_group_id(platform_identity)
        user_id = self.get_user_id(platform_identity)

        # Get username from speaker_info or platform_identity
        username = (
            speaker_info.get('username') if speaker_info else None
        ) or platform_identity.get('username') or 'unknown'

        # ==========================================================================
        # FIRESTORE WINDOW - Immediate context storage
        # ==========================================================================
        # Add to Firestore immediately for immediate context access
        # This happens BEFORE LLM processing for low latency
        if self.firestore.is_enabled():
            self.firestore.add_message(
                agent_id=self.agent_id,
                group_id=group_id,
                message=message,
                username=username,
                platform_identity=platform_identity,
                metadata=metadata or {}
            )

            # Check if we have enough unprocessed messages for batch processing
            unprocessed = self.firestore.get_unprocessed(
                agent_id=self.agent_id,
                group_id=group_id,
                min_count=config.RECENT_BATCH_TRIGGER
            )

            if unprocessed:
                # Trigger batch processing in background
                self._process_batch_async(unprocessed, group_id)

        result = {
            "group_id": group_id,
            "user_id": user_id,
            "created_memories": {
                "group": [],
                "user": [],
                "interaction": []
            }
        }

        # Add recent context from Firestore window
        if self.firestore.is_enabled():
            result["recent_context"] = self.firestore.get_recent(
                agent_id=self.agent_id,
                group_id=group_id,
                limit=config.RECENT_WINDOW_SIZE
            )

        # ==========================================================================
        # LLM-BASED MESSAGE ANALYSIS
        # ==========================================================================
        # Get the LLM analyzer (singleton)
        analyzer = get_message_analyzer()

        # Analyze the message to decide what memories to create
        analysis = analyzer.analyze_message(
            message=message,
            username=username,
            group_context=f"Group ID: {group_id}",
            recent_messages=None  # Could be passed in for context
        )

        # If LLM says we shouldn't remember this message, skip it
        if not analysis.get("should_remember", True):
            return result

        # Accumulate memories for batch insertion
        group_memories = []
        user_memories = []

        # Create memories based on LLM analysis
        for mem_data in analysis.get("memories", []):
            mem_type = mem_data.get("type", "conversation")

            # Map LLM types to MemoryType enum
            type_mapping = {
                "expertise": MemoryType.EXPERTISE,
                "preference": MemoryType.PREFERENCE,
                "fact": MemoryType.FACT,
                "announcement": MemoryType.ANNOUNCEMENT,
                "need": MemoryType.PREFERENCE,  # NEED maps to PREFERENCE for now
                "conversation": MemoryType.CONVERSATION
            }

            memory_type = type_mapping.get(mem_type, MemoryType.CONVERSATION)

            # Determine if this should be a group memory or user memory
            if mem_type in ["announcement"]:
                # Group-level memory
                group_memory = GroupMemory(
                    agent_id=self.agent_id,
                    group_id=group_id,
                    memory_type=memory_type,
                    content=mem_data.get("content", message),
                    speaker=username,
                    keywords=mem_data.get("keywords", []),
                    topics=mem_data.get("topics", []),
                    importance_score=mem_data.get("importance", 0.5),
                    privacy_scope=PrivacyScope.PUBLIC,
                    source_message_id=metadata.get('message_id') if metadata else None,
                    source_timestamp=metadata.get('timestamp') if metadata else None
                )
                group_memories.append(group_memory)

            else:
                # User-level memory
                user_memory = UserMemory(
                    agent_id=self.agent_id,
                    group_id=group_id,
                    user_id=user_id,
                    username=username,
                    platform=platform_identity.get('platform', 'unknown'),
                    memory_type=memory_type,
                    content=mem_data.get("content", message),
                    topics=mem_data.get("topics", []),
                    keywords=mem_data.get("keywords", []),
                    importance_score=mem_data.get("importance", 0.5),
                    privacy_scope=PrivacyScope.PUBLIC,
                    source_message_id=metadata.get('message_id') if metadata else None
                )
                user_memories.append(user_memory)

        # ==========================================================================
        # INTERACTION DETECTION (mentions and replies)
        # ==========================================================================
        mentioned_users = metadata.get('mentioned_users', []) if metadata else []
        is_reply = metadata.get('is_reply', False) if metadata else False

        # Also check if LLM detected an interaction type
        llm_interaction_type = analysis.get("interaction_type")

        interaction_memories = []
        if mentioned_users or is_reply or llm_interaction_type:
            listener_id = mentioned_users[0] if mentioned_users else "unknown"

            # Use LLM interaction type if available, otherwise default
            interaction_type = llm_interaction_type or ("reply" if is_reply else "mention")

            interaction_memory = InteractionMemory(
                agent_id=self.agent_id,
                group_id=group_id,
                speaker_id=user_id,
                speaker_username=username,
                listener_id=listener_id,
                listener_username=mentioned_users[0] if mentioned_users else "unknown",
                content=message,
                mentioned_users=mentioned_users,
                interaction_type=interaction_type,
                source_message_id=metadata.get('message_id') if metadata else None
            )
            interaction_memories.append(interaction_memory)

        # ==========================================================================
        # BATCH INSERT - Optimized for parallel processing
        # ==========================================================================
        # Insert all memories in batch (much faster than individual inserts)
        if group_memories:
            self.group_store.add_group_memories_batch(group_memories)
            result["created_memories"]["group"] = group_memories

        if user_memories:
            self.group_store.add_user_memories_batch(user_memories)
            result["created_memories"]["user"] = user_memories

        if interaction_memories:
            self.group_store.add_interaction_memories_batch(interaction_memories)
            result["created_memories"]["interaction"] = interaction_memories

        return result

    def _process_batch_async(self, unprocessed_messages: List[Dict[str, Any]], group_id: str):
        """
        Process a batch of unprocessed messages in background.

        This is called when we have N unprocessed messages in Firestore.
        Runs in a separate thread to avoid blocking the main response.

        Args:
            unprocessed_messages: List of message dicts from Firestore
            group_id: Group ID
        """
        def process():
            try:
                print(f"[GroupMemoryManager] Processing batch of {len(unprocessed_messages)} messages...")

                doc_ids = []
                for msg_data in unprocessed_messages:
                    # Process each message
                    result = self._process_message_internal(
                        message=msg_data['content'],
                        username=msg_data['username'],
                        platform_identity=msg_data.get('platform_identity', {}),
                        metadata=msg_data.get('metadata', {}),
                        group_id=group_id
                    )

                    doc_ids.append(msg_data['doc_id'])

                # Mark all as processed
                if self.firestore.is_enabled():
                    self.firestore.mark_processed(self.agent_id, group_id, doc_ids)

                print(f"[GroupMemoryManager] Batch processing complete - {len(doc_ids)} messages processed")

            except Exception as e:
                print(f"[GroupMemoryManager] Error in batch processing: {e}")

        # Run in background thread
        thread = Thread(target=process, daemon=True)
        thread.start()

    def _process_message_internal(
        self,
        message: str,
        username: str,
        platform_identity: Dict[str, Any],
        metadata: Dict[str, Any],
        group_id: str
    ) -> Dict[str, Any]:
        """
        Internal message processing without Firestore (used by batch processing).

        This is the core logic extracted from process_group_message.
        """
        user_id = self.get_user_id(platform_identity)

        # Get the LLM analyzer
        analyzer = get_message_analyzer()

        # Analyze the message
        analysis = analyzer.analyze_message(
            message=message,
            username=username,
            group_context=f"Group ID: {group_id}",
            recent_messages=None
        )

        if not analysis.get("should_remember", True):
            return {"created_memories": {"group": [], "user": [], "interaction": []}}

        # Accumulate memories for batch insertion
        group_memories = []
        user_memories = []
        interaction_memories = []

        # Create memories based on LLM analysis
        for mem_data in analysis.get("memories", []):
            mem_type = mem_data.get("type", "conversation")

            type_mapping = {
                "expertise": MemoryType.EXPERTISE,
                "preference": MemoryType.PREFERENCE,
                "fact": MemoryType.FACT,
                "announcement": MemoryType.ANNOUNCEMENT,
                "need": MemoryType.PREFERENCE,
                "conversation": MemoryType.CONVERSATION
            }

            memory_type = type_mapping.get(mem_type, MemoryType.CONVERSATION)

            if mem_type in ["announcement"]:
                group_memory = GroupMemory(
                    agent_id=self.agent_id,
                    group_id=group_id,
                    memory_type=memory_type,
                    content=mem_data.get("content", message),
                    speaker=username,
                    keywords=mem_data.get("keywords", []),
                    topics=mem_data.get("topics", []),
                    importance_score=mem_data.get("importance", 0.5),
                    privacy_scope=PrivacyScope.PUBLIC,
                    source_message_id=metadata.get('message_id') if metadata else None,
                    source_timestamp=metadata.get('timestamp') if metadata else None
                )
                group_memories.append(group_memory)
            else:
                user_memory = UserMemory(
                    agent_id=self.agent_id,
                    group_id=group_id,
                    user_id=user_id,
                    username=username,
                    platform=platform_identity.get('platform', 'unknown'),
                    memory_type=memory_type,
                    content=mem_data.get("content", message),
                    topics=mem_data.get("topics", []),
                    keywords=mem_data.get("keywords", []),
                    importance_score=mem_data.get("importance", 0.5),
                    privacy_scope=PrivacyScope.PUBLIC,
                    source_message_id=metadata.get('message_id') if metadata else None
                )
                user_memories.append(user_memory)

        # Check for interactions
        mentioned_users = metadata.get('mentioned_users', []) if metadata else []
        is_reply = metadata.get('is_reply', False) if metadata else False
        llm_interaction_type = analysis.get("interaction_type")

        if mentioned_users or is_reply or llm_interaction_type:
            listener_id = mentioned_users[0] if mentioned_users else "unknown"
            interaction_type = llm_interaction_type or ("reply" if is_reply else "mention")

            interaction_memory = InteractionMemory(
                agent_id=self.agent_id,
                group_id=group_id,
                speaker_id=user_id,
                speaker_username=username,
                listener_id=listener_id,
                listener_username=mentioned_users[0] if mentioned_users else "unknown",
                content=message,
                mentioned_users=mentioned_users,
                interaction_type=interaction_type,
                source_message_id=metadata.get('message_id') if metadata else None
            )
            interaction_memories.append(interaction_memory)

        # Batch insert all memories
        if group_memories:
            self.group_store.add_group_memories_batch(group_memories)
        if user_memories:
            self.group_store.add_user_memories_batch(user_memories)
        if interaction_memories:
            self.group_store.add_interaction_memories_batch(interaction_memories)

        return {
            "created_memories": {
                "group": len(group_memories),
                "user": len(user_memories),
                "interaction": len(interaction_memories)
            }
        }

    def _process_dm_message(
        self,
        message: str,
        platform_identity: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a direct message using the existing VectorStore.

        This maintains backward compatibility with the existing system.
        """
        user_id = self.get_user_id(platform_identity)

        # Get or create VectorStore for this user
        if user_id not in self._vector_stores:
            self._vector_stores[user_id] = VectorStore(
                agent_id=self.agent_id,
                user_id=user_id,
                embedding_model=self.embedding_model
            )

        vector_store = self._vector_stores[user_id]

        # Create memory entry - use message as lossless_restatement for DMs
        # In production, this should use LLM for proper restatement
        entry = MemoryEntry(
            lossless_restatement=message,  # Required field
            keywords=self._extract_keywords(message),
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        # Add to vector store
        vector_store.add_entries([entry])

        return {
            "group_id": None,
            "user_id": user_id,
            "created_memories": {
                "vector_store": [1]  # Count of entries added
            }
        }

    # ==========================================================================
    # CONTEXT RETRIEVAL
    # ==========================================================================

    def get_context_for_agent(
        self,
        message: str,
        platform_identity: Dict[str, Any],
        limit_per_level: int = 5,
        include_user_profile: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive context for the agent to generate a response.

        This multi-level retrieval includes:
        - Group context (group decisions, culture, announcements)
        - User context (preferences, expertise, actions)
        - Interaction context (recent conversations)
        - Cross-group patterns (if user is in multiple groups)
        - User profile (if available)

        Args:
            message: The current user message (for semantic search)
            platform_identity: Platform identity dict
            limit_per_level: Max memories per level
            include_user_profile: Whether to include UserProfileStore data

        Returns:
            Dict with context from all levels
        """
        is_group = self.is_group_message(platform_identity)
        group_id = self.get_group_id(platform_identity) if is_group else None
        user_id = self.get_user_id(platform_identity)

        context = {
            "is_group": is_group,
            "group_id": group_id,
            "user_id": user_id,
            "message": message,
            "retrieved_at": datetime.now(timezone.utc).isoformat()
        }

        if is_group and group_id:
            # Multi-level context for groups
            group_context = self.group_store.get_group_context(
                group_id=group_id,
                user_id=user_id,
                query=message,
                limit_per_level=limit_per_level
            )
            context.update(group_context)

            # Add recent messages from Firestore window (immediate context)
            if self.firestore.is_enabled():
                context["recent_messages"] = self.firestore.get_recent(
                    agent_id=self.agent_id,
                    group_id=group_id,
                    limit=config.RECENT_WINDOW_SIZE
                )
        else:
            # Simple context for DMs
            context["dm_context"] = self._get_dm_context(user_id, message, limit_per_level)

        # Add user profile if requested
        if include_user_profile:
            context["user_profile"] = self._get_user_profile(user_id)

        return context

    def _get_dm_context(
        self,
        user_id: str,
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Get context for direct messages using VectorStore."""
        if user_id not in self._vector_stores:
            # Try to load existing vector store
            try:
                self._vector_stores[user_id] = VectorStore(
                    agent_id=self.agent_id,
                    user_id=user_id,
                    embedding_model=self.embedding_model
                )
            except Exception as e:
                print(f"[GroupMemoryManager] Could not load VectorStore for {user_id}: {e}")
                return []

        vector_store = self._vector_stores[user_id]

        try:
            results = vector_store.search(
                query_text=query,
                limit=limit
            )

            # Format results
            return [
                {
                    "content": r.get('content', ''),
                    "role": r.get('role', 'unknown'),
                    "timestamp": r.get('timestamp', ''),
                    "score": r.get('_score', 0.0)
                }
                for r in results
            ]
        except Exception as e:
            print(f"[GroupMemoryManager] Error searching VectorStore: {e}")
            return []

    def _get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile from UserProfileStore."""
        if self._user_profile_store is None:
            try:
                self._user_profile_store = UserProfileStore(
                    db_path=self.db_base_path,
                    embedding_model=self.embedding_model
                )
            except Exception as e:
                print(f"[GroupMemoryManager] Could not initialize UserProfileStore: {e}")
                return None

        try:
            # Try to get profile by universal_user_id
            # Format: "platform_user_identifier" (e.g., "telegram_123456789")
            profile = self._user_profile_store.get_profile_by_universal_id(user_id)
            if profile:
                return {
                    "profile_id": profile.profile_id,
                    "username": profile.username,
                    "summary": profile.summary,
                    "platform": profile.platform_type,
                    "expertise_areas": [i.name for i in profile.interests],
                    "interaction_count": profile.total_messages_processed
                }
        except Exception as e:
            print(f"[GroupMemoryManager] Error getting user profile: {e}")

        return None

    # ==========================================================================
    # CROSS-GROUP CONSOLIDATION
    # ==========================================================================

    def consolidate_cross_group_patterns(
        self,
        user_id: str,
        min_groups: int = 2,
        min_evidence: int = 2
    ) -> List[CrossGroupMemory]:
        """
        Detect and consolidate cross-group patterns for a user.

        This should be run periodically (e.g., daily) as a background job.

        Args:
            user_id: The user ID to consolidate
            min_groups: Minimum number of groups to form a pattern
            min_evidence: Minimum evidence count per group

        Returns:
            List of created/updated CrossGroupMemory objects
        """
        patterns = self.group_store.detect_cross_group_patterns(
            user_id=user_id,
            min_groups=min_groups,
            min_evidence=min_evidence
        )

        consolidated = []
        for pattern in patterns:
            # add_cross_group_memory returns the created/updated memory object
            created_memory = self.group_store.add_cross_group_memory(pattern)
            consolidated.append(created_memory)

        return consolidated

    def consolidate_cross_group_patterns_parallel(
        self,
        user_ids: List[str],
        min_groups: int = 2,
        min_evidence: int = 2,
        max_workers: int = 5
    ) -> Dict[str, List[CrossGroupMemory]]:
        """
        Detect and consolidate cross-group patterns for multiple users in parallel.

        This is significantly faster than calling consolidate_cross_group_patterns()
        for each user sequentially.

        Performance: ~1s for N users vs ~N*s for sequential processing.

        Args:
            user_ids: List of user IDs to consolidate
            min_groups: Minimum number of groups to form a pattern
            min_evidence: Minimum evidence count per group
            max_workers: Maximum parallel workers (default: 5)

        Returns:
            Dict mapping user_id to their list of CrossGroupMemory objects
        """
        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all consolidation tasks
            future_to_user = {
                executor.submit(
                    self.consolidate_cross_group_patterns,
                    user_id,
                    min_groups,
                    min_evidence
                ): user_id
                for user_id in user_ids
            }

            # Collect results as they complete
            for future in as_completed(future_to_user):
                user_id = future_to_user[future]
                try:
                    patterns = future.result(timeout=30)
                    results[user_id] = patterns
                except Exception as e:
                    print(f"[GroupMemoryManager] Error consolidating {user_id}: {e}")
                    results[user_id] = []

        return results

    def get_cross_group_patterns(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get consolidated cross-group patterns for a user.

        Args:
            user_id: The user ID
            limit: Maximum patterns to return

        Returns:
            List of cross-group patterns
        """
        patterns = self.group_store.search_cross_group(
            universal_user_id=user_id,
            query="",  # Empty query to get all
            limit=limit
        )

        return [
            {
                "pattern_type": p.pattern_type,
                "content": p.content,
                "groups_involved": p.groups_involved,
                "confidence_score": p.confidence_score,
                "evidence_count": p.evidence_count
            }
            for p in patterns
        ]

    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text for search optimization.

        This is a simple implementation - production should use NLP.
        """
        # Simple keyword extraction: remove common words, keep meaningful terms
        common_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
            'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are'
        }

        # Tokenize and filter
        words = text.lower().split()
        keywords = [w for w in words if len(w) > 3 and w not in common_words]

        # Remove duplicates and limit
        return list(set(keywords))[:10]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the group memory store.

        Returns:
            Dict with memory counts and stats
        """
        # This would require adding count methods to GroupMemoryStore
        # For now, return basic info
        return {
            "agent_id": self.agent_id,
            "db_path": self.group_store.agent_db_path,
            "global_db_path": self.group_store.global_db_path,
            "initialized_at": datetime.now(timezone.utc).isoformat()
        }
