"""
SimpleMem - Efficient Lifelong Memory for LLM Agents
Main system class integrating all components

Extended for Unified Memory:
- Support for both DMs and Groups
- Multi-table storage (group_memories, user_memories, interaction_memories)
- Firestore integration for immediate context
- UserProfile integration
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from models.memory_entry import Dialogue, MemoryEntry, MemoryType, PrivacyScope
from utils.llm_client import LLMClient
from utils.embedding import EmbeddingModel
from database.vector_store import VectorStore
from database.unified_store import UnifiedMemoryStore
from database.user_profile_store import UserProfileStore
from database.group_profile_store import GroupProfileStore
from core.memory_builder import MemoryBuilder
from core.hybrid_retriever import HybridRetriever
from core.answer_generator import AnswerGenerator
from services.firestore_window import get_firestore_store, FirestoreWindowStore
from core.adaptive_threshold import (
    AdaptiveThresholdManager,
    get_adaptive_threshold,
    record_batch_processed
)
import config


class SimpleMemSystem:
    """
    SimpleMem Main System

    Three-stage pipeline based on Semantic Lossless Compression:
    1. Semantic Structured Compression: add_dialogue() -> MemoryBuilder -> Store
    2. Structured Indexing and Recursive Consolidation: (background evolution - future work)
    3. Adaptive Query-Aware Retrieval: ask() -> HybridRetriever -> AnswerGenerator

    Extended for Groups:
    4. Firestore window for immediate context (recent messages)
    5. Multi-table storage for group memories
    6. Context-aware retrieval across tables
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        db_path: Optional[str] = None,
        table_name: Optional[str] = None,
        clear_db: bool = False,
        enable_thinking: Optional[bool] = None,
        use_streaming: Optional[bool] = None,
        enable_planning: Optional[bool] = None,
        enable_reflection: Optional[bool] = None,
        max_reflection_rounds: Optional[int] = None,
        enable_parallel_processing: Optional[bool] = None,
        max_parallel_workers: Optional[int] = None,
        enable_parallel_retrieval: Optional[bool] = None,
        max_retrieval_workers: Optional[int] = None,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        # New options for unified memory
        use_unified_store: bool = True,
        enable_firestore: bool = True
    ):
        """
        Initialize system

        Args:
        - api_key: OpenAI API key
        - model: LLM model name
        - base_url: Custom OpenAI base URL (for compatible APIs)
        - db_path: Database path
        - table_name: Memory table name (for parallel processing)
        - clear_db: Whether to clear existing database
        - enable_thinking: Enable deep thinking mode (for Qwen and compatible models)
        - use_streaming: Enable streaming responses
        - enable_planning: Enable multi-query planning for retrieval (None=use config default)
        - enable_reflection: Enable reflection-based additional retrieval (None=use config default)
        - max_reflection_rounds: Maximum number of reflection rounds (None=use config default)
        - enable_parallel_processing: Enable parallel processing for memory building (None=use config default)
        - max_parallel_workers: Maximum number of parallel workers for memory building (None=use config default)
        - enable_parallel_retrieval: Enable parallel processing for retrieval queries (None=use config default)
        - max_retrieval_workers: Maximum number of parallel workers for retrieval (None=use config default)
        - agent_id: Agent identifier for multi-tenant isolation (required for unified store)
        - user_id: User identifier for multi-tenant isolation (None=no filtering)
        - use_unified_store: Use UnifiedMemoryStore instead of VectorStore (default: True)
        - enable_firestore: Enable Firestore for immediate context (default: True)
        """
        # Store tenant context
        self.agent_id = agent_id or "default"
        self.user_id = user_id
        self.use_unified_store = use_unified_store

        print("=" * 60)
        print("Initializing SimpleMem System")
        if agent_id or user_id:
            print(f"Tenant: agent={self.agent_id}, user={user_id or 'any'}")
        print(f"Store: {'UnifiedMemoryStore' if use_unified_store else 'VectorStore'}")
        print("=" * 60)

        # Initialize core components
        self.llm_client = LLMClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            enable_thinking=enable_thinking,
            use_streaming=use_streaming
        )
        self.embedding_model = EmbeddingModel()

        # Initialize storage (unified or legacy)
        if use_unified_store:
            self.unified_store = UnifiedMemoryStore(
                agent_id=self.agent_id,
                db_base_path=db_path or config.LANCEDB_PATH,
                embedding_model=self.embedding_model,
                llm_client=self.llm_client
            )
            # Expose as vector_store for backward compatibility
            self.vector_store = self.unified_store
        else:
            self.vector_store = VectorStore(
                db_path=db_path,
                embedding_model=self.embedding_model,
                table_name=table_name,
                agent_id=agent_id,
                user_id=user_id
            )
            self.unified_store = self.vector_store

        if clear_db:
            print("\nClearing existing database...")
            if use_unified_store:
                self.unified_store.clear_agent_data()
            else:
                self.vector_store.clear()

        # Initialize Firestore (optional)
        self.firestore_enabled = enable_firestore
        if enable_firestore:
            self.firestore = get_firestore_store()
            if self.firestore.is_enabled():
                print("[Firestore] Connected for immediate context")
            else:
                print("[Firestore] Not available - using LanceDB only")
                self.firestore_enabled = False
        else:
            self.firestore = None
            print("[Firestore] Disabled")

        # Initialize three major modules
        self.memory_builder = MemoryBuilder(
            llm_client=self.llm_client,
            unified_store=self.unified_store,
            enable_parallel_processing=enable_parallel_processing,
            max_parallel_workers=max_parallel_workers,
            agent_id=self.agent_id
        )

        self.hybrid_retriever = HybridRetriever(
            llm_client=self.llm_client,
            unified_store=self.unified_store,
            enable_planning=enable_planning,
            enable_reflection=enable_reflection,
            max_reflection_rounds=max_reflection_rounds,
            enable_parallel_retrieval=enable_parallel_retrieval,
            max_retrieval_workers=max_retrieval_workers
        )

        self.answer_generator = AnswerGenerator(
            llm_client=self.llm_client
        )

        # Initialize UserProfileStore
        self.user_profile_store = UserProfileStore(
            db_path=db_path or config.LANCEDB_PATH,
            embedding_model=self.embedding_model,
            agent_id=self.agent_id
        )

        # Initialize GroupProfileStore
        self.group_profile_store = GroupProfileStore(
            db_path=db_path or config.LANCEDB_PATH,
            embedding_model=self.embedding_model,
            agent_id=self.agent_id
        )

        # Pass profile stores to hybrid_retriever for lightweight entity system
        self.hybrid_retriever.user_profile_store = self.user_profile_store
        self.hybrid_retriever.group_profile_store = self.group_profile_store

        print("\nSystem initialization complete!")
        print("=" * 60)

    def add_dialogue(
        self,
        speaker: str,
        content: str,
        timestamp: Optional[str] = None,
        # Group context
        platform: str = "direct",
        group_id: Optional[str] = None,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        # Message metadata
        message_id: Optional[str] = None,
        is_reply: bool = False,
        mentioned_users: Optional[List[str]] = None,
        reply_to_message_id: Optional[str] = None,
        # Processing options
        add_to_firestore: bool = True,
        use_stateless_processing: bool = True
    ) -> Dict[str, Any]:
        """
        Add a single dialogue with optional group context.

        STATELESS MODE (Cloud Run compatible):
        - Firestore is used as the buffer (not in-memory)
        - When 10+ unprocessed messages accumulate, they are processed
        - Messages are marked as processed after LLM extraction

        Args:
        - speaker: Speaker name
        - content: Dialogue content
        - timestamp: Timestamp (ISO 8601 format)
        - platform: Platform (telegram, xmtp, farcaster, twitter, direct)
        - group_id: Group identifier (None = DM, use "dm_{user_id}" internally)
        - user_id: User identifier
        - username: Username/handle
        - message_id: Original message ID
        - is_reply: Whether this is a reply
        - mentioned_users: List of mentioned usernames
        - reply_to_message_id: Message ID being replied to
        - add_to_firestore: Add to Firestore for immediate context (default: True)
        - use_stateless_processing: Use Firestore as buffer instead of in-memory (default: True)

        Returns:
        - Dict with processing status
        """
        from datetime import datetime, timezone

        # Generate timestamp if not provided
        if not timestamp:
            timestamp = datetime.now(timezone.utc).isoformat()

        # Determine effective group_id for DMs
        effective_group_id = group_id
        if group_id is None:
            effective_user_id = user_id or self.user_id
            effective_group_id = f"dm_{effective_user_id}" if effective_user_id else None

        result = {
            "added": False,
            "processed": False,
            "memories_created": 0,
            "effective_group_id": effective_group_id
        }

        # Add to Firestore (used as buffer in stateless mode)
        if self.firestore_enabled and self.firestore and effective_group_id:
            # Preserve original platform_identity fields and add computed user_id
            stored_platform_identity = {
                "platform": platform,
                "user_id": user_id,
                "username": username
            }
            doc_id = self.firestore.add_message(
                agent_id=self.agent_id,
                group_id=effective_group_id,
                message=content,
                username=username or speaker,
                platform_identity=stored_platform_identity,
                metadata={
                    "message_id": message_id,
                    "timestamp": timestamp,
                    "is_reply": is_reply,
                    "mentioned_users": mentioned_users,
                    "reply_to_message_id": reply_to_message_id,
                    "speaker": speaker
                }
            )
            result["added"] = doc_id is not None

            # Record message for adaptive threshold metrics
            # Use existing Firestore client to avoid creating new connections
            if result["added"] and self.firestore:
                try:
                    threshold_manager = AdaptiveThresholdManager(
                        firestore_client=self.firestore.db  # Reuse existing client
                    )
                    threshold_manager.record_message(
                        agent_id=self.agent_id,
                        group_id=effective_group_id,
                        importance_score=0.5
                    )
                except Exception as e:
                    print(f"[AdaptiveThreshold] Warning: {e}")

            # STATELESS PROCESSING: Check if we have enough unprocessed messages
            if use_stateless_processing and result["added"]:
                processing_result = self._process_unprocessed_messages(
                    effective_group_id=effective_group_id,
                    original_group_id=group_id,
                    platform=platform
                )
                result.update(processing_result)

        # FALLBACK: Use in-memory buffer if Firestore is not available
        elif not use_stateless_processing or not self.firestore_enabled:
            dialogue_id = self.memory_builder.processed_count + len(self.memory_builder.dialogue_buffer) + 1
            dialogue = Dialogue(
                dialogue_id=dialogue_id,
                speaker=speaker,
                content=content,
                timestamp=timestamp,
                platform=platform,
                group_id=group_id,
                user_id=user_id or self.user_id,
                username=username,
                message_id=message_id,
                is_reply=is_reply,
                mentioned_users=mentioned_users or [],
                reply_to_message_id=reply_to_message_id
            )
            self.memory_builder.add_dialogue(dialogue)
            result["added"] = True

        return result

    def _process_unprocessed_messages(
        self,
        effective_group_id: str,
        original_group_id: Optional[str],
        platform: str
    ) -> Dict[str, Any]:
        """
        Check for and process unprocessed messages from Firestore.

        This is the core of stateless processing - Firestore acts as the buffer.
        Uses adaptive thresholds based on group activity.

        Args:
            effective_group_id: The group ID used in Firestore (includes dm_ prefix for DMs)
            original_group_id: The original group ID (None for DMs)
            platform: The platform (telegram, xmtp, etc.)

        Returns:
            Dict with processing results
        """
        result = {"processed": False, "memories_created": 0}

        # Initialize adaptive threshold manager (reuse Firestore client)
        threshold_manager = AdaptiveThresholdManager(
            firestore_client=self.firestore.db if self.firestore else None
        )

        # Get adaptive threshold for this group
        adaptive_threshold = threshold_manager.get_threshold(self.agent_id, effective_group_id)
        print(f"[AdaptiveThreshold] Current threshold for {effective_group_id}: {adaptive_threshold}")

        # Get unprocessed messages with adaptive threshold
        unprocessed = self.firestore.get_unprocessed(
            agent_id=self.agent_id,
            group_id=effective_group_id,
            min_count=adaptive_threshold
        )

        if not unprocessed:
            return result

        # Check if we should actually process (considers time, urgency, activity)
        should_process, batch_size = threshold_manager.should_process(
            self.agent_id,
            effective_group_id,
            len(unprocessed)
        )

        if not should_process:
            print(f"[Stateless] {len(unprocessed)} messages pending, but threshold not met (adaptive: {adaptive_threshold})")
            return result

        print(f"\n[Stateless] Found {len(unprocessed)} unprocessed messages for {effective_group_id}")

        # Convert Firestore documents to Dialogue objects
        dialogues = []
        doc_ids = []
        for i, msg in enumerate(unprocessed):
            doc_ids.append(msg['doc_id'])
            metadata = msg.get('metadata', {})
            platform_identity = msg.get('platform_identity', {})

            dialogue = Dialogue(
                dialogue_id=i + 1,
                speaker=metadata.get('speaker', msg.get('username', 'unknown')),
                content=msg.get('content', ''),
                timestamp=metadata.get('timestamp', msg.get('timestamp')),
                platform=platform_identity.get('platform', platform),
                group_id=original_group_id,  # Use original (None for DMs)
                user_id=platform_identity.get('user_id'),
                username=msg.get('username'),
                message_id=metadata.get('message_id'),
                is_reply=metadata.get('is_reply', False),
                mentioned_users=metadata.get('mentioned_users') or [],
                reply_to_message_id=metadata.get('reply_to_message_id')
            )
            dialogues.append(dialogue)

        # Process dialogues with LLM
        processing_result = self.memory_builder.process_dialogues_direct(dialogues)

        # Mark messages as processed in Firestore
        self.firestore.mark_processed(
            agent_id=self.agent_id,
            group_id=effective_group_id,
            doc_ids=doc_ids
        )

        # Record batch processing for adaptive threshold metrics
        record_batch_processed(
            self.agent_id,
            effective_group_id,
            len(dialogues),
            firestore_client=self.firestore.db if self.firestore else None
        )

        result["processed"] = True
        result["memories_created"] = processing_result.get("total", 0)
        result["processing_details"] = processing_result

        # Update conversation summary after batch processing
        self._update_conversation_summary(
            effective_group_id=effective_group_id,
            original_group_id=original_group_id,
            unprocessed=unprocessed
        )

        # Auto-generate profiles after batch processing
        self._generate_all_profiles(
            unprocessed=unprocessed,
            platform=platform,
            original_group_id=original_group_id,
            effective_group_id=effective_group_id
        )

        return result

    def _generate_all_profiles(
        self,
        unprocessed: List[Dict],
        platform: str,
        original_group_id: Optional[str],
        effective_group_id: str
    ):
        """
        Generate all profile types after batch processing:
        1. UserProfile (global) - aggregates DM + group messages
        2. GroupProfile - group culture and context
        3. UserInGroupProfile - user behavior within this group

        Uses LLM (llama-3.1-8b-instruct) and a0x-models API.
        """
        from collections import defaultdict

        # Group messages by user
        user_data = defaultdict(lambda: {
            'messages': [],
            'universal_user_id': None,
            'username': None,
            'platform_type': platform
        })

        all_messages = []
        group_name = None

        for msg in unprocessed:
            content = msg.get('content', '')
            all_messages.append(content)

            platform_identity = msg.get('platform_identity', {})
            user_id = platform_identity.get('user_id')

            if user_id:
                user_data[user_id]['messages'].append(content)
                user_data[user_id]['universal_user_id'] = user_id
                user_data[user_id]['username'] = msg.get('username') or platform_identity.get('username')

            # Extract group name from first message
            if not group_name:
                metadata = msg.get('metadata', {})
                group_name = metadata.get('group_name') or platform_identity.get('groupName')

        # ============================================================
        # 1. GENERATE/UPDATE GROUP PROFILE (for groups only)
        # ============================================================
        if original_group_id is not None:  # It's a group, not DM
            total_group_messages = self._count_group_messages(original_group_id)
            print(f"\n[ProfileGen] Group {original_group_id}: {total_group_messages} memories (threshold: 10)")

            if total_group_messages >= 10:  # Threshold for group profile (10 memories = ~20 raw msgs)
                print(f"\n[GroupProfile] Generating profile for {original_group_id} ({total_group_messages} messages)")

                self.group_profile_store.generate_group_profile_from_messages(
                    agent_id=self.agent_id,
                    group_id=original_group_id,
                    platform=platform,
                    group_name=group_name,
                    messages=all_messages
                )

        # ============================================================
        # 2. GENERATE/UPDATE USER-IN-GROUP PROFILES
        # ============================================================
        if original_group_id is not None:  # Only for groups
            for user_id, data in user_data.items():
                if not data['messages']:
                    continue

                # Count user's messages in this group
                total_in_group = self._count_user_messages_in_group(original_group_id, data['universal_user_id'])
                print(f"[ProfileGen] User {data['universal_user_id']} in group: {total_in_group} memories (threshold: 10)")

                if total_in_group >= 10:  # Threshold for user-in-group profile
                    print(f"\n[UserInGroupProfile] Generating profile for {data['universal_user_id']} in {original_group_id}")

                    self.group_profile_store.generate_user_in_group_profile(
                        agent_id=self.agent_id,
                        group_id=original_group_id,
                        universal_user_id=data['universal_user_id'],
                        username=data['username'],
                        user_messages=data['messages']
                    )

        # ============================================================
        # 3. GENERATE/UPDATE GLOBAL USER PROFILES (aggregates all sources)
        # ============================================================
        for user_id, data in user_data.items():
            if not data['messages']:
                continue

            universal_user_id = data['universal_user_id']
            platform_type = data['platform_type']
            # Extract platform_specific_id from universal_user_id or use username as fallback
            if ':' in universal_user_id:
                platform_specific_id = universal_user_id.split(':', 1)[1]
            else:
                # Fallback to username if no colon
                platform_specific_id = data.get('username', universal_user_id)

            # Get existing profile to count total messages
            existing_profile = self.user_profile_store.get_profile_by_universal_id(universal_user_id)

            total_messages = len(data['messages'])
            if existing_profile:
                total_messages += existing_profile.total_messages_processed

            # Also count messages from memories (all sources)
            memory_messages = self._get_all_user_messages(universal_user_id, platform_type)
            if memory_messages:
                total_messages = max(total_messages, len(memory_messages))

            # Only generate if >= 10 messages
            if total_messages >= 10:
                print(f"\n[UserProfile] Generating global profile for {universal_user_id} ({total_messages} total messages)")

                # Use memory messages for better profile (includes all sources)
                messages_for_profile = memory_messages if memory_messages else data['messages']

                self.user_profile_store.generate_profile_from_messages(
                    agent_id=self.agent_id,
                    platform_type=platform_type,
                    platform_specific_id=platform_specific_id,
                    messages=messages_for_profile,
                    username=data['username']
                )

    def _get_all_user_messages(self, user_id: str, platform_type: str) -> Optional[List[str]]:
        """
        Collect all messages for a user from ALL sources (DM + groups).

        Args:
            user_id: User ID (e.g., "telegram:123456789")
            platform_type: Platform type

        Returns:
            List of message contents or None
        """
        all_messages = []

        try:
            # 1. DM memories (memories.lance)
            dm_memories = self.unified_store.memories_table.search().where(
                f"agent_id = '{self.agent_id}' AND user_id = '{user_id}'",
                prefilter=True
            ).to_list()

            all_messages.extend([m.get("content", "") for m in dm_memories if m.get("content")])

            # 2. User memories from groups (user_memories.lance)
            user_memories = self.unified_store.user_memories_table.search().where(
                f"agent_id = '{self.agent_id}' AND user_id = '{user_id}'",
                prefilter=True
            ).to_list()

            all_messages.extend([m.get("content", "") for m in user_memories if m.get("content")])

            # 3. Interaction memories where user is speaker
            interaction_memories = self.unified_store.interaction_memories_table.search().where(
                f"agent_id = '{self.agent_id}' AND speaker_id = '{user_id}'",
                prefilter=True
            ).to_list()

            all_messages.extend([m.get("content", "") for m in interaction_memories if m.get("content")])

            if all_messages:
                print(f"[UserProfile] Found {len(all_messages)} total messages for {user_id} (DM + groups)")
                return all_messages

        except Exception as e:
            print(f"[UserProfile] Error fetching historical messages: {e}")

        return None

    def _count_group_messages(self, group_id: str) -> int:
        """
        Count total messages in a group.

        Counts user_memories + interaction_memories since each represents
        a user message (group_memories are consolidated summaries).
        """
        try:
            user_count = self.unified_store.user_memories_table.search().where(
                f"agent_id = '{self.agent_id}' AND group_id = '{group_id}'",
                prefilter=True
            ).limit(500).to_list().__len__()

            interaction_count = self.unified_store.interaction_memories_table.search().where(
                f"agent_id = '{self.agent_id}' AND group_id = '{group_id}'",
                prefilter=True
            ).limit(500).to_list().__len__()

            return user_count + interaction_count
        except Exception as e:
            print(f"[_count_group_messages] Error: {e}")
            return 0

    def _count_user_messages_in_group(self, group_id: str, universal_user_id: str) -> int:
        """Count user's messages in a specific group."""
        try:
            return self.unified_store.user_memories_table.search().where(
                f"agent_id = '{self.agent_id}' AND group_id = '{group_id}' AND user_id = '{universal_user_id}'",
                prefilter=True
            ).limit(500).to_list().__len__()
        except:
            return 0

    def _update_conversation_summary(
        self,
        effective_group_id: str,
        original_group_id: Optional[str],
        unprocessed: List[Dict]
    ):
        """
        Update conversation summary after batch processing.

        This is called AFTER ingestion completes, never blocking the response path.

        Args:
            effective_group_id: Group ID (with dm_ prefix for DMs)
            original_group_id: Original group ID (None for DMs)
            unprocessed: List of processed message dicts
        """
        if not unprocessed:
            return

        # Determine thread_id format
        # For DMs: "{agent_id}_dm_{user_id}"
        # For groups: "{agent_id}_group_{group_id}"
        if original_group_id is None:
            # DM - extract user_id from effective_group_id (format: "dm_{user_id}")
            user_id = effective_group_id.replace("dm_", "", 1) if effective_group_id.startswith("dm_") else effective_group_id
            thread_id = f"{self.agent_id}_dm_{user_id}"
        else:
            # Group
            thread_id = f"{self.agent_id}_group_{original_group_id}"

        # Get current summary
        current_summary = self.unified_store.get_or_create_summary(thread_id)

        # Extract message contents
        new_messages = [msg.get('content', '') for msg in unprocessed if msg.get('content')]

        if new_messages:
            # Update summary (non-blocking, happens in background)
            try:
                self.unified_store.update_summary(thread_id, new_messages, current_summary)
            except Exception as e:
                print(f"[SimpleMemSystem] Error updating conversation summary: {e}")

    def get_conversation_summary(
        self,
        group_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        """
        Get the current conversation summary for a thread.

        Args:
            group_id: Group ID (None for DMs)
            user_id: User ID (for DMs)

        Returns:
            Current summary text (empty if not exists)
        """
        # Construct thread_id
        if group_id is None:
            # DM
            effective_user_id = user_id or self.user_id
            if not effective_user_id:
                return ""
            thread_id = f"{self.agent_id}_dm_{effective_user_id}"
        else:
            # Group
            thread_id = f"{self.agent_id}_group_{group_id}"

        return self.unified_store.get_or_create_summary(thread_id)

    def process_pending(self, group_id: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Manually trigger processing of pending messages.

        Useful for:
        - Processing remaining messages that haven't hit the threshold
        - Scheduled batch processing

        Args:
            group_id: Group ID to process (None for all groups)
            user_id: User ID for DM processing

        Returns:
            Dict with processing results
        """
        if not self.firestore_enabled or not self.firestore:
            return {"error": "Firestore not enabled"}

        effective_group_id = group_id
        if group_id is None and user_id:
            effective_group_id = f"dm_{user_id}"

        if not effective_group_id:
            return {"error": "Must specify group_id or user_id"}

        # Force processing by setting min_count to 1
        unprocessed = self.firestore.get_unprocessed(
            agent_id=self.agent_id,
            group_id=effective_group_id,
            min_count=1  # Process any pending messages
        )

        if not unprocessed:
            return {"processed": False, "message": "No pending messages"}

        return self._process_unprocessed_messages(
            effective_group_id=effective_group_id,
            original_group_id=group_id,
            platform="direct"
        )

    def add_group_message(
        self,
        message: str,
        platform_identity: Dict[str, Any],
        speaker_info: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Convenience method for adding group messages with platform_identity dict.

        This matches the interface used by a0x-agent-api for easier integration.

        Args:
        - message: The message content
        - platform_identity: Dict with platform, chatId/conversationId, telegramId/walletAddress, username
        - speaker_info: Optional dict with additional speaker info
        - metadata: Optional dict with message_id, timestamp, is_reply, mentioned_users, etc.

        Returns:
        - Dict with group_id, user_id, and processing status
        """
        # Extract platform info
        platform = platform_identity.get('platform', 'direct').lower()

        # Determine group_id
        group_id = None
        if platform == 'telegram':
            # Accept both chatId and groupId for flexibility
            chat_id = platform_identity.get('chatId') or platform_identity.get('groupId')
            if chat_id and (str(chat_id).startswith('-') or str(chat_id).startswith('telegram_')):
                # Normalize to telegram_{id} format
                group_id = chat_id if str(chat_id).startswith('telegram_') else f"telegram_{chat_id}"
        elif platform == 'xmtp':
            conv_id = platform_identity.get('conversationId')
            if conv_id and '/groups/' in str(conv_id):
                group_id = f"xmtp_{conv_id}"
        elif platform in ['farcaster', 'twitter']:
            # Public platforms are treated as groups
            group_id = f"{platform}_public"

        # Determine user_id
        if platform == 'telegram':
            # Priority: telegramId > user_id (from platform_identity) > username
            telegram_id = platform_identity.get('telegramId')
            platform_user_id = platform_identity.get('user_id')
            username_val = platform_identity.get('username')

            if telegram_id:
                user_id = f"telegram:{telegram_id}"
            elif platform_user_id and ':' in platform_user_id:
                user_id = platform_user_id  # Already has platform prefix
            elif username_val:
                user_id = f"telegram:{username_val}"
            else:
                user_id = f"telegram:unknown"
        elif platform == 'xmtp':
            user_id = f"xmtp:{platform_identity.get('walletAddress', 'unknown')}"
        elif platform == 'farcaster':
            user_id = f"farcaster:{platform_identity.get('fid', 'unknown')}"
        elif platform == 'twitter':
            user_id = f"twitter:{platform_identity.get('username', 'unknown')}"
        else:
            user_id = f"direct:{platform_identity.get('clientId', 'unknown')}"

        # Get username
        username = None
        if speaker_info:
            username = speaker_info.get('username')
        if not username:
            username = platform_identity.get('username')

        # Get speaker name
        speaker = username or user_id

        # Extract metadata
        meta = metadata or {}
        timestamp = meta.get('timestamp')
        message_id = meta.get('message_id')
        is_reply = meta.get('is_reply', False)
        mentioned_users = meta.get('mentioned_users', [])
        reply_to_message_id = meta.get('reply_to_message_id')

        # Add the dialogue
        self.add_dialogue(
            speaker=speaker,
            content=message,
            timestamp=timestamp,
            platform=platform,
            group_id=group_id,
            user_id=user_id,
            username=username,
            message_id=message_id,
            is_reply=is_reply,
            mentioned_users=mentioned_users,
            reply_to_message_id=reply_to_message_id
        )

        return {
            "group_id": group_id,
            "user_id": user_id,
            "platform": platform,
            "is_group": group_id is not None
        }

    def add_agent_response_to_window(
        self,
        response: str,
        user_id: str,
        group_id: Optional[str] = None,
        platform: str = "direct"
    ) -> bool:
        """
        Add agent response to Firestore recent window for immediate context.

        This ensures agent responses are immediately available in context
        without waiting for batch processing.

        Args:
            response: Agent's response content
            user_id: User who received this response
            group_id: Group ID (None for DMs)
            platform: Platform

        Returns:
            True if successful, False otherwise
        """
        if not self.firestore_enabled or not self.firestore:
            return False

        effective_group_id = group_id or f"dm_{user_id}"

        # Mark as processed=True so it won't be batch processed again
        return self.firestore.add_message(
            agent_id=self.agent_id,
            group_id=effective_group_id,
            message=response,
            username="agent",  # Special username for agent messages
            platform_identity={
                "platform": platform,
                "user_id": "agent",
                "is_agent": True
            },
            metadata={
                "is_agent_response": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "processed": True  # Already processed, don't re-process
            }
        ) is not None

    def get_firestore_context(self, group_id: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
        """
        Get recent context from Firestore sliding window.

        Args:
            group_id: Group ID to fetch (uses default if None)
            limit: Max messages to return

        Returns:
            Dict with raw_messages and agent_responses
        """
        result = {
            "raw_messages": [],
            "agent_responses": []
        }

        if not self.firestore_enabled or not self.firestore:
            return result

        # Use effective group_id
        effective_group_id = group_id or f"dm_{self.user_id}" if self.user_id else None
        if not effective_group_id:
            return result

        try:
            # Get recent messages from Firestore
            messages = self.firestore.get_recent(
                agent_id=self.agent_id,
                group_id=effective_group_id,
                limit=limit
            )

            for msg in messages:
                metadata = msg.get('metadata', {})
                platform_identity = msg.get('platform_identity', {})

                if metadata.get('is_agent_response') or platform_identity.get('is_agent'):
                    result["agent_responses"].append({
                        "content": msg.get('content', ''),
                        "timestamp": metadata.get('timestamp'),
                        "username": msg.get('username', 'agent')
                    })
                else:
                    result["raw_messages"].append({
                        "content": msg.get('content', ''),
                        "username": msg.get('username', 'unknown'),
                        "timestamp": metadata.get('timestamp'),
                        "platform_identity": platform_identity
                    })
        except Exception as e:
            print(f"[SimpleMemSystem] Error getting Firestore context: {e}")

        return result

    def add_dialogues(self, dialogues: List[Dialogue]):
        """
        Batch add dialogues

        Args:
        - dialogues: List of dialogues
        """
        self.memory_builder.add_dialogues(dialogues)

    def finalize(self):
        """
        Finalize dialogue input, process any remaining buffer (safety check)
        Note: In parallel mode, remaining dialogues are already processed
        """
        self.memory_builder.process_remaining()

    def search(
        self,
        query: str,
        group_id: Optional[str] = None,
        user_id: Optional[str] = None,
        include_window: bool = True,
        include_agent_responses: bool = True,
    ) -> Dict[str, Any]:
        """
        Search memory - returns raw structured context for the orchestrator.

        Combines:
        1. Firestore window (recent messages + agent responses)
        2. Intelligent retrieval (planning + BM25 + multi-table fan-out)
        3. Agent's previous responses from LanceDB

        Args:
            query: Search query
            group_id: Group context (None for DMs)
            user_id: User context
            include_window: Include Firestore recent messages
            include_agent_responses: Include agent's past responses

        Returns:
            Dict with:
            - recent_messages: List[Dict] from Firestore window
            - dm_memories: List[MemoryEntry]
            - group_memories: List[GroupMemory]
            - user_memories: List[UserMemory]
            - interaction_memories: List[InteractionMemory]
            - cross_group_memories: List[CrossGroupMemory]
            - agent_responses: List[Dict] from agent_responses table
            - relevant_profiles: List[UserProfile] from lightweight entity system
            - group_context: Dict with group profile and active users
            - formatted_context: str ready for LLM consumption
        """
        effective_user_id = user_id or self.user_id
        context = {
            "group_id": group_id,
            "user_id": effective_user_id,
        }

        result = {
            "recent_messages": [],
            "dm_memories": [],
            "group_memories": [],
            "user_memories": [],
            "interaction_memories": [],
            "cross_group_memories": [],
            "agent_responses": [],
            "relevant_profiles": [],
            "group_context": None,
            "formatted_context": "",
        }

        # 1. Firestore window (recent messages)
        if include_window and self.firestore_enabled and self.firestore:
            effective_group_id = group_id or (f"dm_{effective_user_id}" if effective_user_id else None)
            if effective_group_id:
                recent = self.firestore.get_recent(
                    agent_id=self.agent_id,
                    group_id=effective_group_id,
                    limit=config.RECENT_WINDOW_SIZE,
                )
                if recent:
                    result["recent_messages"] = recent
                    print(f"[search] Firestore window: {len(recent)} messages")

        # 2. Intelligent retrieval (planning + BM25 + all tables)
        if self.use_unified_store and hasattr(self.hybrid_retriever, 'retrieve_for_context'):
            retrieval = self.hybrid_retriever.retrieve_for_context(
                query=query, context=context
            )
            result["dm_memories"] = retrieval.get("dm_memories", [])
            result["group_memories"] = retrieval.get("group_memories", [])
            result["user_memories"] = retrieval.get("user_memories", [])
            result["interaction_memories"] = retrieval.get("interaction_memories", [])
            result["cross_group_memories"] = retrieval.get("cross_group_memories", [])
            # Include profiles and group context from lightweight entity system
            result["relevant_profiles"] = retrieval.get("relevant_profiles", [])
            result["group_context"] = retrieval.get("group_context")
        else:
            result["dm_memories"] = self.hybrid_retriever.retrieve(query)

        # 3. Agent's previous responses
        if include_agent_responses and self.use_unified_store:
            try:
                agent_responses = self.unified_store.search_agent_responses(
                    query=query,
                    group_id=group_id,
                    user_id=effective_user_id,
                    limit=5,
                )
                if agent_responses:
                    result["agent_responses"] = agent_responses
                    print(f"[search] Agent responses: {len(agent_responses)}")
            except Exception as e:
                print(f"[search] Agent responses search failed: {e}")

        # 4. Build formatted context
        sections = []

        if result["recent_messages"]:
            sections.append(f"## Recent Conversation\n{self._format_recent_messages(result['recent_messages'])}")

        # Memory context from retrieval
        if self.use_unified_store and hasattr(self.hybrid_retriever, 'retrieve_for_context'):
            memory_fmt = self.hybrid_retriever._format_multi_table_context(
                {k: result[k] for k in ["dm_memories", "group_memories", "user_memories",
                                         "interaction_memories", "cross_group_memories"]},
                context,
            )
            if memory_fmt and memory_fmt != "[No relevant memories found]":
                sections.append(memory_fmt)
        else:
            if result["dm_memories"]:
                sections.append(self._format_contexts(result["dm_memories"]))

        if result["agent_responses"]:
            resp_lines = ["## My Previous Responses"]
            for r in result["agent_responses"][:3]:
                trigger = r.get("trigger_message", "")
                summary = r.get("summary", r.get("content", "")[:100])
                resp_lines.append(f"- Q: {trigger[:80]}" if trigger else "")
                resp_lines.append(f"  A: {summary}")
            sections.append("\n".join(resp_lines))

        result["formatted_context"] = "\n\n".join(sections) if sections else ""

        return result

    def ask(
        self,
        question: str,
        group_id: Optional[str] = None,
        user_id: Optional[str] = None,
        include_firestore_context: bool = True
    ) -> str:
        """
        Ask question - Q&A interface. Uses search() + answer generator.

        Args:
        - question: User question
        - group_id: Group context for multi-table search
        - user_id: User context for personalized retrieval
        - include_firestore_context: Include recent messages from Firestore

        Returns:
        - Answer string
        """
        print("\n" + "=" * 60)
        print(f"Question: {question}")
        print("=" * 60)

        search_results = self.search(
            query=question,
            group_id=group_id,
            user_id=user_id,
            include_window=include_firestore_context,
        )

        full_context = search_results["formatted_context"]

        if full_context:
            dummy_entries = [MemoryEntry(lossless_restatement=full_context, keywords=[])]
            answer = self.answer_generator.generate_answer(question, dummy_entries)
        else:
            answer = self.answer_generator.generate_answer(question, [])

        print("\nAnswer:")
        print(answer)
        print("=" * 60 + "\n")

        return answer

    def _format_recent_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Format recent Firestore messages as context string."""
        lines = []
        for msg in messages:
            username = msg.get('username', 'unknown')
            content = msg.get('content', '')
            timestamp = msg.get('timestamp', '')
            if timestamp:
                lines.append(f"[{timestamp}] {username}: {content}")
            else:
                lines.append(f"{username}: {content}")
        return "\n".join(lines)

    def _format_contexts(self, contexts: List[MemoryEntry]) -> str:
        """Format memory entries as context string."""
        if not contexts:
            return ""

        formatted = []
        for i, entry in enumerate(contexts, 1):
            parts = [f"[{i}] {entry.lossless_restatement}"]
            if entry.timestamp:
                parts.append(f"({entry.timestamp})")
            formatted.append(" ".join(parts))

        return "\n".join(formatted)

    def get_all_memories(self) -> List[MemoryEntry]:
        """
        Get all memory entries (for debugging)
        """
        return self.vector_store.get_all_entries()

    def print_memories(self):
        """
        Print all memory entries (for debugging)
        """
        memories = self.get_all_memories()
        print("\n" + "=" * 60)
        print(f"All Memory Entries ({len(memories)} total)")
        print("=" * 60)

        for i, memory in enumerate(memories, 1):
            print(f"\n[Entry {i}]")
            print(f"ID: {memory.entry_id}")
            print(f"Restatement: {memory.lossless_restatement}")
            if memory.timestamp:
                print(f"Time: {memory.timestamp}")
            if memory.location:
                print(f"Location: {memory.location}")
            if memory.persons:
                print(f"Persons: {', '.join(memory.persons)}")
            if memory.entities:
                print(f"Entities: {', '.join(memory.entities)}")
            if memory.topic:
                print(f"Topic: {memory.topic}")
            print(f"Keywords: {', '.join(memory.keywords)}")

        print("\n" + "=" * 60)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory store."""
        if self.use_unified_store and hasattr(self.unified_store, 'get_stats'):
            return self.unified_store.get_stats()
        else:
            return {
                "memories_count": self.vector_store.count_entries() if hasattr(self.vector_store, 'count_entries') else len(self.get_all_memories())
            }

    def run_consolidation(self, group_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run memory consolidation (SimpleMem Stage 2 - Atom to Molecule Evolution).

        This is an offline/background process - NEVER run during user request handling.
        Intended for cron jobs or manual triggering.

        Consolidates similar atomic memories into higher-level molecules:
        - Finds clusters of similar memories using embedding similarity (cosine > 0.80)
        - Uses LLM to synthesize consolidated content
        - Replaces clusters with consolidated memories

        Args:
            group_id: Specific group ID to consolidate (None = all groups)

        Returns:
            Dict with consolidation results

        Example:
            # Consolidate all groups
            results = system.run_consolidation()

            # Consolidate specific group
            results = system.run_consolidation(group_id="telegram_group_123")
        """
        if not self.use_unified_store:
            return {"error": "Consolidation requires UnifiedMemoryStore"}

        print("\n" + "=" * 60)
        print("Starting Memory Consolidation (SimpleMem Stage 2)")
        print("=" * 60)

        if group_id:
            print(f"Group ID: {group_id}")
            results = self.unified_store.consolidate_similar_memories(group_id)
        else:
            print("Processing all groups...")
            results = self.unified_store.consolidate_all_groups()

        print("=" * 60)
        print("Consolidation Complete")
        print(f"  Groups processed: {results.get('groups_processed', 'N/A')}")
        print(f"  Original memories: {results.get('total_originals', 0)}")
        print(f"  Consolidated memories: {results.get('total_consolidated', 0)}")
        print(f"  Reduction: {results.get('total_originals', 0) - results.get('total_consolidated', 0)} memories")
        print("=" * 60 + "\n")

        return results


# Convenience function
def create_system(
    agent_id: Optional[str] = None,
    clear_db: bool = False,
    enable_planning: Optional[bool] = None,
    enable_reflection: Optional[bool] = None,
    max_reflection_rounds: Optional[int] = None,
    enable_parallel_processing: Optional[bool] = None,
    max_parallel_workers: Optional[int] = None,
    enable_parallel_retrieval: Optional[bool] = None,
    max_retrieval_workers: Optional[int] = None,
    use_unified_store: bool = True,
    enable_firestore: bool = True
) -> SimpleMemSystem:
    """
    Create SimpleMem system instance (uses config.py defaults when None)
    """
    return SimpleMemSystem(
        agent_id=agent_id,
        clear_db=clear_db,
        enable_planning=enable_planning,
        enable_reflection=enable_reflection,
        max_reflection_rounds=max_reflection_rounds,
        enable_parallel_processing=enable_parallel_processing,
        max_parallel_workers=max_parallel_workers,
        enable_parallel_retrieval=enable_parallel_retrieval,
        max_retrieval_workers=max_retrieval_workers,
        use_unified_store=use_unified_store,
        enable_firestore=enable_firestore
    )


if __name__ == "__main__":
    # Quick test with unified memory
    print("Running SimpleMem Quick Test with Unified Memory...")

    system = create_system(
        agent_id="test_agent",
        clear_db=True,
        use_unified_store=True,
        enable_firestore=False  # Disable for local testing
    )

    print(f"Using embedding model: {system.memory_builder.vector_store.embedding_model.model_name}")

    # Add some test dialogues (DM)
    system.add_dialogue(
        speaker="Alice",
        content="Bob, let's meet at Starbucks tomorrow at 2pm to discuss the new product",
        timestamp="2025-11-15T14:30:00",
        user_id="alice123"
    )
    system.add_dialogue(
        speaker="Bob",
        content="Okay, I'll prepare the materials",
        timestamp="2025-11-15T14:31:00",
        user_id="bob456"
    )
    system.add_dialogue(
        speaker="Alice",
        content="Remember to bring the market research report from last time",
        timestamp="2025-11-15T14:32:00",
        user_id="alice123"
    )

    # Finalize input
    system.finalize()

    # View memories
    system.print_memories()

    # Print stats
    print("\nMemory Store Stats:")
    stats = system.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Ask questions
    print("\nTesting retrieval with planning and reflection...")
    system.ask("When will Alice and Bob meet?")

    print("\nQuick test completed!")
