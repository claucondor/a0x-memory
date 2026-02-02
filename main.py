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
from database import MemoryStore, VectorStore, UserProfileStore
from database.group_profile_store import GroupProfileStore
from database.group_summary_store import GroupSummaryStore
from database.dm_summary_store import DMSummaryStore
from database.user_fact_store import UserFactStore
from services.summary_aggregator import SummaryAggregator
from services.dm_summary_aggregator import DMSummaryAggregator
from services.fact_extractor import FactExtractor
from core.memory_builder import MemoryBuilder
from core.hybrid_retriever import HybridRetriever
from core.answer_generator import AnswerGenerator
from core.response_extractor import ResponseExtractor
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
        enable_firestore: bool = True,
        # Shared components (singletons)
        embedding_model: Optional["EmbeddingModel"] = None,
        llm_client: Optional["LLMClient"] = None
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
        - use_unified_store: Use MemoryStore instead of VectorStore (default: True)
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
        print(f"Store: {'MemoryStore' if use_unified_store else 'VectorStore'}")
        print("=" * 60)

        # Initialize core components (use shared if provided)
        self.llm_client = llm_client or LLMClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            enable_thinking=enable_thinking,
            use_streaming=use_streaming
        )
        self.embedding_model = embedding_model or EmbeddingModel()

        # Initialize storage (unified or legacy)
        if use_unified_store:
            # Use new refactored MemoryStore
            self.unified_store = MemoryStore(
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

        # Group summary store for hierarchical summaries
        self.group_summary_store = GroupSummaryStore(
            db_path=db_path or config.LANCEDB_PATH,
            embedding_model=self.embedding_model,
            agent_id=self.agent_id
        )
        self.summary_aggregator = SummaryAggregator(
            summary_store=self.group_summary_store,
            llm_client=self.llm_client
        )

        # DM summary store for 1-on-1 conversations
        self.dm_summary_store = DMSummaryStore(
            db_path=db_path or config.LANCEDB_PATH,
            embedding_model=self.embedding_model,
            agent_id=self.agent_id
        )
        self.dm_summary_aggregator = DMSummaryAggregator(
            summary_store=self.dm_summary_store,
            llm_client=self.llm_client
        )

        # User fact store for evidence-based user profiles
        self.user_fact_store = UserFactStore(
            db_path=db_path or config.LANCEDB_PATH,
            embedding_model=self.embedding_model,
            agent_id=self.agent_id
        )
        self.fact_extractor = FactExtractor(
            fact_store=self.user_fact_store,
            llm_client=self.llm_client
        )

        # Adaptive threshold manager for spam detection
        self.threshold_manager = AdaptiveThresholdManager(
            firestore_client=self.firestore.db if self.firestore else None
        )

        # Pass profile stores to hybrid_retriever for lightweight entity system
        self.hybrid_retriever.user_profile_store = self.user_profile_store
        self.hybrid_retriever.group_profile_store = self.group_profile_store
        self.hybrid_retriever.fact_store = self.user_fact_store

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
        use_stateless_processing: bool = True,
        # Agent response support
        role: str = "user",
        metadata: Optional[Dict[str, Any]] = None
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
        - role: Role of the speaker ("user" or "assistant")
        - metadata: Additional metadata to merge with message metadata

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

            # Generate embedding for spam detection (reuse shared model)
            message_embedding = None
            try:
                message_embedding = self.embedding_model.encode_single(content, is_query=False)
                if hasattr(message_embedding, 'tolist'):
                    message_embedding = message_embedding.tolist()
            except Exception as e:
                print(f"[Spam] Warning: Could not generate embedding: {e}")

            # Build message metadata and merge with extra metadata
            msg_metadata = {
                "message_id": message_id,
                "timestamp": timestamp,
                "is_reply": is_reply,
                "mentioned_users": mentioned_users,
                "reply_to_message_id": reply_to_message_id,
                "speaker": speaker,
                "role": role
            }
            if metadata:
                msg_metadata.update(metadata)

            add_result = self.firestore.add_message(
                agent_id=self.agent_id,
                group_id=effective_group_id,
                message=content,
                username=username or speaker,
                platform_identity=stored_platform_identity,
                metadata=msg_metadata,
                embedding=message_embedding
            )

            # Handle new return format (dict with doc_id, is_spam, etc.)
            if isinstance(add_result, dict):
                doc_id = add_result.get('doc_id')
                result["added"] = doc_id is not None
                result["is_spam"] = add_result.get('is_spam', False)
                result["is_blocked"] = add_result.get('is_blocked', False)
                result["spam_score"] = add_result.get('spam_score', 0.0)
                if add_result.get('is_blocked'):
                    print(f"[Blocked] User {username} is blocked due to spam")
                elif add_result.get('is_spam'):
                    print(f"[Spam] Detected spam from {username}: {add_result.get('spam_reason')}")
            else:
                # Backwards compatibility
                doc_id = add_result
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

        # Update conversation summary + generate profiles (parallel LLM calls)
        self._post_batch_processing(
            unprocessed=unprocessed,
            platform=platform,
            original_group_id=original_group_id,
            effective_group_id=effective_group_id
        )

        # Process agent responses to LanceDB
        self._process_unprocessed_agent_responses(
            effective_group_id=effective_group_id,
            original_group_id=original_group_id,
            platform=platform
        )

        # Compact LanceDB fragments accumulated during this batch
        try:
            self.unified_store.optimize_tables()
        except Exception as e:
            print(f"[optimize] Non-fatal error: {e}")

        return result

    def _post_batch_processing(
        self,
        unprocessed: List[Dict],
        platform: str,
        original_group_id: Optional[str],
        effective_group_id: str
    ):
        """
        Post-batch processing: summary update + all profile generations in parallel.

        Fires all LLM calls concurrently (summary, group profile, user profiles)
        with sequential fallback on error.
        """
        import concurrent.futures
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
            metadata = msg.get('metadata', {})

            # Use speaker_user_id if available (for observed posts from other agents)
            # Otherwise fall back to the observer's user_id
            speaker_user_id = metadata.get('speaker_user_id')
            user_id = speaker_user_id or platform_identity.get('user_id')

            if user_id:
                user_data[user_id]['messages'].append(content)
                user_data[user_id]['universal_user_id'] = user_id
                # Use speaker name from metadata if available
                user_data[user_id]['username'] = metadata.get('speaker') or msg.get('username') or platform_identity.get('username')

            if not group_name:
                group_name = metadata.get('group_name') or platform_identity.get('groupName')

        # ============================================================
        # Collect all tasks to run in parallel
        # ============================================================
        tasks = []

        # Task: conversation summary
        def task_summary():
            self._update_conversation_summary(
                effective_group_id=effective_group_id,
                original_group_id=original_group_id,
                unprocessed=unprocessed
            )
            return "summary"

        tasks.append(("summary", task_summary))

        # Task: group profile
        if original_group_id is not None:
            total_group_messages = self._count_group_messages(original_group_id)
            print(f"\n[ProfileGen] Group {original_group_id}: {total_group_messages} memories (threshold: 10)")

            if total_group_messages >= 10:
                def task_group_profile():
                    print(f"\n[GroupProfile] Generating profile for {original_group_id} ({total_group_messages} messages)")
                    self.group_profile_store.generate_group_profile_from_messages(
                        agent_id=self.agent_id,
                        group_id=original_group_id,
                        platform=platform,
                        group_name=group_name,
                        messages=all_messages
                    )
                    return "group_profile"

                tasks.append(("group_profile", task_group_profile))

        # Task: user-in-group profiles
        if original_group_id is not None:
            for uid, data in user_data.items():
                if not data['messages']:
                    continue
                total_in_group = self._count_user_messages_in_group(original_group_id, data['universal_user_id'])
                print(f"[ProfileGen] User {data['universal_user_id']} in group: {total_in_group} memories (threshold: 10)")

                if total_in_group >= 10:
                    _uid = data['universal_user_id']
                    _uname = data['username']
                    _msgs = data['messages']
                    def task_user_in_group(uid=_uid, uname=_uname, msgs=_msgs):
                        print(f"\n[UserInGroupProfile] Generating profile for {uid} in {original_group_id}")
                        self.group_profile_store.generate_user_in_group_profile(
                            agent_id=self.agent_id,
                            group_id=original_group_id,
                            universal_user_id=uid,
                            username=uname,
                            user_messages=msgs
                        )
                        return f"user_in_group:{uid}"

                    tasks.append((f"user_in_group:{_uid}", task_user_in_group))

        # Tasks: global user profiles
        for uid, data in user_data.items():
            if not data['messages']:
                continue

            universal_user_id = data['universal_user_id']
            platform_type = data['platform_type']
            if ':' in universal_user_id:
                platform_specific_id = universal_user_id.split(':', 1)[1]
            else:
                platform_specific_id = data.get('username', universal_user_id)

            existing_profile = self.user_profile_store.get_profile_by_universal_id(universal_user_id)

            total_messages = len(data['messages'])
            if existing_profile:
                total_messages += existing_profile.total_messages_processed

            memory_messages = self._get_all_user_messages(universal_user_id, platform_type)
            if memory_messages:
                total_messages = max(total_messages, len(memory_messages))

            if total_messages >= 10:
                _uid = universal_user_id
                _pt = platform_type
                _psid = platform_specific_id
                _msgs = memory_messages if memory_messages else data['messages']
                _uname = data['username']
                _total = total_messages
                def task_user_profile(uid=_uid, pt=_pt, psid=_psid, msgs=_msgs, uname=_uname, total=_total):
                    print(f"\n[UserProfile] Generating global profile for {uid} ({total} total messages)")
                    self.user_profile_store.generate_profile_from_messages(
                        agent_id=self.agent_id,
                        platform_type=pt,
                        platform_specific_id=psid,
                        messages=msgs,
                        username=uname
                    )
                    return f"user_profile:{uid}"

                tasks.append((f"user_profile:{_uid}", task_user_profile))

        # Tasks: fact extraction for each user
        for uid, data in user_data.items():
            if not data['messages']:
                continue

            universal_user_id = data['universal_user_id']
            context_type = "dm" if original_group_id is None else "group"
            context_id = effective_group_id

            _uid = universal_user_id
            _msgs = [{"content": m, "speaker": data['username']} for m in data['messages']]
            _ctx_type = context_type
            _ctx_id = context_id
            def task_fact_extraction(uid=_uid, msgs=_msgs, ctx_type=_ctx_type, ctx_id=_ctx_id):
                try:
                    print(f"\n[FactExtractor] Extracting facts for {uid} from {ctx_type}")
                    self.fact_extractor.extract_from_messages(
                        agent_id=self.agent_id,
                        user_id=uid,
                        messages=msgs,
                        context_type=ctx_type,
                        context_id=ctx_id
                    )
                    return f"fact_extraction:{uid}"
                except Exception as e:
                    print(f"[FactExtractor] Error extracting facts for {uid}: {e}")
                    return f"fact_extraction:{uid}:error"

            tasks.append((f"fact_extraction:{_uid}", task_fact_extraction))

        # ============================================================
        # Execute all tasks in parallel
        # ============================================================
        if not tasks:
            return

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
                futures = {
                    executor.submit(fn): name
                    for name, fn in tasks
                }
                for future in concurrent.futures.as_completed(futures):
                    name = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"[PostBatch] Task '{name}' failed: {e}")
        except Exception as e:
            # Fallback: run everything sequentially
            print(f"[PostBatch] Parallel execution failed, running sequentially: {e}")
            for name, fn in tasks:
                try:
                    fn()
                except Exception as e2:
                    print(f"[PostBatch] Sequential task '{name}' failed: {e2}")

        # Generate micro summaries if threshold reached
        # Groups use summary_aggregator, DMs use dm_summary_aggregator
        try:
            self._generate_micro_if_needed(
                unprocessed=unprocessed,
                original_group_id=original_group_id,
                effective_group_id=effective_group_id
            )
        except Exception as e:
            print(f"[PostBatch] Micro summary generation failed: {e}")

    def _generate_micro_if_needed(
        self,
        unprocessed: List[Dict],
        original_group_id: Optional[str],
        effective_group_id: str
    ):
        """
        Generate micro summary if message threshold is reached.

        For groups: uses summary_aggregator (threshold: 50 messages)
        For DMs: uses dm_summary_aggregator (threshold: 20 messages)

        Micro summaries are later consolidated into chunks/blocks/eras
        by the a0x-memory-jobs service.
        """
        print(f"[MicroCheck] Checking micro generation for {effective_group_id} ({len(unprocessed)} messages)")
        is_dm = original_group_id is None

        if is_dm:
            # DM - use dm_summary_aggregator
            user_id = effective_group_id.replace("dm_", "") if effective_group_id.startswith("dm_") else effective_group_id

            # Get total message count from adaptive threshold metrics
            try:
                metrics = self.threshold_manager.get_metrics_debug(self.agent_id, effective_group_id)
                total_from_metrics = metrics.get("message_count", 0)
            except:
                total_from_metrics = 0

            # Also check last summarized index
            last_index = self.dm_summary_store.get_last_message_index(effective_group_id)
            # Use the higher of: metrics count or last_index+batch
            total_messages = max(total_from_metrics, last_index + 1 + len(unprocessed))
            print(f"[MicroCheck] DM metrics_count={total_from_metrics}, last_index={last_index}, total={total_messages}, threshold=20")

            should_gen = self.dm_summary_aggregator.should_generate_micro(user_id, total_messages)
            print(f"[MicroCheck] should_generate_micro={should_gen}")
            if should_gen:
                # Convert messages to expected format
                messages = [
                    {
                        "speaker": m.get('username', 'unknown'),
                        "content": m.get('content', ''),
                        "timestamp": m.get('timestamp', '')
                    }
                    for m in unprocessed
                ]

                # Get the starting index for this micro
                start_index = last_index + 1

                summary = self.dm_summary_aggregator.generate_micro_summary(
                    agent_id=self.agent_id,
                    user_id=user_id,  # user_id without dm_ prefix (e.g., "telegram:1001")
                    messages=messages,
                    message_start_index=start_index
                )

                if summary:
                    print(f"[Micro] Created DM micro summary for {effective_group_id}: msgs {summary.message_start}-{summary.message_end}")
        else:
            # Group - use summary_aggregator
            try:
                metrics = self.threshold_manager.get_metrics_debug(self.agent_id, original_group_id)
                total_from_metrics = metrics.get("message_count", 0)
            except:
                total_from_metrics = 0

            last_index = self.group_summary_store.get_last_message_index(original_group_id)
            total_messages = max(total_from_metrics, last_index + 1 + len(unprocessed))
            print(f"[MicroCheck] Group metrics_count={total_from_metrics}, last_index={last_index}, total={total_messages}, threshold=50")

            if self.summary_aggregator.should_generate_micro(original_group_id, total_messages):
                messages = [
                    {
                        "speaker": m.get('username', 'unknown'),
                        "content": m.get('content', ''),
                        "timestamp": m.get('timestamp', '')
                    }
                    for m in unprocessed
                ]

                start_index = last_index + 1

                summary = self.summary_aggregator.generate_micro_summary(
                    agent_id=self.agent_id,
                    group_id=original_group_id,
                    messages=messages,
                    message_start_index=start_index
                )

                if summary:
                    print(f"[Micro] Created group micro summary for {original_group_id}: msgs {summary.message_start}-{summary.message_end}")

    def _process_unprocessed_agent_responses(
        self,
        effective_group_id: str,
        original_group_id: Optional[str],
        platform: str
    ) -> Dict[str, Any]:
        """
        Process unprocessed agent responses from Firestore to LanceDB.

        Extracts metadata using LLM and stores with dual vectors.

        Args:
            effective_group_id: The group ID used in Firestore
            original_group_id: The original group ID (None for DMs)
            platform: The platform

        Returns:
            Dict with processing results
        """
        result = {"processed": 0, "errors": 0}

        if not self.firestore_enabled or not self.firestore:
            return result

        # Initialize ResponseExtractor
        if not hasattr(self, 'response_extractor'):
            self.response_extractor = ResponseExtractor(self.llm_client)

        try:
            # Get recent messages from Firestore
            messages = self.firestore.get_recent(
                self.agent_id,
                effective_group_id,
                limit=50
            )

            # Filter for agent responses that haven't been processed to LanceDB
            agent_responses = [
                m for m in messages
                if m.get('metadata', {}).get('is_agent_response')
                and not m.get('metadata', {}).get('processed_to_lancedb')
            ]

            if not agent_responses:
                return result

            print(f"\n[AgentResponses] Found {len(agent_responses)} unprocessed agent responses")

            from models.group_memory import AgentResponse, ResponseType

            for msg in agent_responses:
                try:
                    metadata = msg.get('metadata', {})
                    platform_identity = msg.get('platform_identity', {})

                    trigger_message = metadata.get('trigger_message', '')
                    response_content = msg.get('content', '')

                    if not trigger_message or not response_content:
                        continue

                    # Extract metadata with LLM
                    extracted = self.response_extractor.extract(
                        trigger_message=trigger_message,
                        response_content=response_content
                    )

                    # Determine scope
                    # If group-specific and mentions user, use "user"
                    # If group general, use "group"
                    # If generic info, use "global"
                    scope = extracted.get('scope', 'user')
                    if original_group_id and scope == 'user':
                        # Check if response addresses a specific user
                        mentioned_user = platform_identity.get('user_id')
                        if not mentioned_user:
                            scope = 'group'

                    # Create AgentResponse
                    response = AgentResponse(
                        agent_id=self.agent_id,
                        scope=scope,
                        user_id=platform_identity.get('user_id'),
                        group_id=original_group_id if original_group_id and not original_group_id.startswith('dm_') else None,
                        trigger_message=trigger_message,
                        content=response_content,
                        summary=extracted.get('summary'),
                        response_type=ResponseType(extracted.get('response_type', 'answer')),
                        topics=extracted.get('topics', []),
                        keywords=extracted.get('keywords', []),
                    )

                    # Store in LanceDB
                    self.unified_store.add_agent_response_with_vectors(response)
                    result["processed"] += 1

                    # Mark as processed to LanceDB
                    self.firestore.mark_processed_to_lancedb(
                        agent_id=self.agent_id,
                        group_id=effective_group_id,
                        doc_id=msg.get('doc_id')
                    )

                except Exception as e:
                    print(f"[AgentResponses] Error processing response: {e}")
                    result["errors"] += 1

            if result["processed"] > 0:
                print(f"[AgentResponses] Processed {result['processed']} agent responses to LanceDB")

        except Exception as e:
            print(f"[AgentResponses] Batch processing error: {e}")
            result["errors"] += 1

        return result

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
        user_id: str = None,
        group_id: Optional[str] = None,
        platform: str = "direct",
        trigger_message: Optional[str] = None,
        trigger_message_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
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
            trigger_message: User message that triggered this response
            trigger_message_id: ID of the trigger message
            timestamp: Optional timestamp override
            metadata: Additional metadata

        Returns:
            True if successful, False otherwise
        """
        if not self.firestore_enabled or not self.firestore:
            return False

        effective_group_id = group_id or f"dm_{user_id}" if user_id else "dm_unknown"

        # Build metadata with trigger info
        response_metadata = {
            "is_agent_response": True,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "processed": True  # Already processed, don't re-process
        }

        # Add trigger info if provided
        if trigger_message:
            response_metadata["trigger_message"] = trigger_message
        if trigger_message_id:
            response_metadata["trigger_message_id"] = trigger_message_id

        # Merge any additional metadata
        if metadata:
            response_metadata.update(metadata)

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
            metadata=response_metadata
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
            "speaker_dm_memories": [],
            "agent_responses": [],
            "user_facts": [],
            "dm_summaries": [],
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
            result["speaker_dm_memories"] = retrieval.get("speaker_dm_memories", [])
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

        # 4. User facts (evidence-based profile)
        if effective_user_id:
            try:
                facts = self.fact_extractor.get_user_context(effective_user_id, query=query)
                if facts.get("facts"):
                    result["user_facts"] = facts["facts"]
                    print(f"[search] User facts: {len(facts['facts'])}")
            except Exception as e:
                print(f"[search] User facts retrieval failed: {e}")

        # 5. DM summaries (for 1-on-1 conversations)
        if group_id is None and effective_user_id:
            dm_id = f"dm_{effective_user_id}"
            try:
                summaries = self.dm_summary_store.get_context_summaries(
                    dm_id, limit_micro=3, limit_chunk=2, limit_block=1
                )
                all_summaries = []
                for level, level_summaries in summaries.items():
                    all_summaries.extend(level_summaries)
                if all_summaries:
                    result["dm_summaries"] = all_summaries
                    print(f"[search] DM summaries: {len(all_summaries)}")
            except Exception as e:
                print(f"[search] DM summaries retrieval failed: {e}")

        # 6. Build formatted context
        sections = []

        if result["recent_messages"]:
            sections.append(f"## Recent Conversation\n{self._format_recent_messages(result['recent_messages'])}")

        # Memory context from retrieval
        if self.use_unified_store and hasattr(self.hybrid_retriever, 'retrieve_for_context'):
            memory_fmt = self.hybrid_retriever._format_multi_table_context(
                {k: result.get(k, []) for k in ["dm_memories", "group_memories", "user_memories",
                                         "interaction_memories", "cross_group_memories", "speaker_dm_memories"]},
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

        # User facts section
        if result["user_facts"]:
            fact_lines = ["## Known Facts About User"]
            for fact in result["user_facts"][:5]:
                fact_lines.append(f"- [{fact.get('type', 'fact')}] {fact.get('content')} (confidence: {fact.get('confidence', 0):.2f})")
            sections.append("\n".join(fact_lines))

        # DM summaries section
        if result["dm_summaries"]:
            summary_lines = ["## Conversation Summaries"]
            for s in result["dm_summaries"][:3]:
                level = s.get("level", "micro")
                summary = s.get("summary", "")[:200]
                summary_lines.append(f"- [{level.upper()}] {summary}")
            sections.append("\n".join(summary_lines))

        result["formatted_context"] = "\n\n".join(sections) if sections else ""

        return result

    def ask(
        self,
        question: str,
        group_id: Optional[str] = None,
        user_id: Optional[str] = None,
        include_firestore_context: bool = True,
        search_results: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Ask question - Q&A interface. Uses search() + answer generator.

        Args:
        - question: User question
        - group_id: Group context for multi-table search
        - user_id: User context for personalized retrieval
        - include_firestore_context: Include recent messages from Firestore
        - search_results: Pre-fetched search results (skip re-running search pipeline)

        Returns:
        - Answer string
        """
        print("\n" + "=" * 60)
        print(f"Question: {question}")
        print("=" * 60)

        if search_results is None:
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
