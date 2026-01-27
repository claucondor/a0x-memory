"""
Memory Builder - Stage 1: Semantic Structured Compression (Section 3.1)

Implements Semantic Structured Compression:
- Entropy-based non-linear filter: Phi_gate (conceptual - filters low-density dialogue)
- De-linearization transformation: F_theta (converts dialogue to atomic entries)
- Generates self-contained Atomic Entries {m_k} via coreference resolution and temporal anchoring

Extended for Unified Memory:
- DMs: Generates MemoryEntry (SimpleMem compatible)
- Groups: Generates GroupMemory, UserMemory, InteractionMemory
"""
from typing import List, Optional, Dict, Any, Tuple, Union
from models.memory_entry import MemoryEntry, Dialogue, MemoryType, PrivacyScope
from models.group_memory import (
    GroupMemory, UserMemory, InteractionMemory,
    MemoryLevel,
    MemoryType as GroupMemoryType,
    PrivacyScope as GroupPrivacyScope
)
from utils.llm_client import LLMClient
from utils.structured_schemas import DM_MEMORY_ENTRIES_SCHEMA, GROUP_MEMORIES_SCHEMA
from core.memory_validator import validate_and_filter_memories
import config
import json
import concurrent.futures
from datetime import datetime, timezone


class MemoryBuilder:
    """
    Memory Builder - Stage 1: Semantic Structured Compression

    Paper Reference: Section 3.1 - Semantic Structured Compression

    Core Functions:
    1. Entropy-based filtering (implicit via window processing)
    2. De-linearization transformation F_theta: Dialogue -> Atomic Entries
    3. Coreference resolution Phi_coref (no pronouns)
    4. Temporal anchoring Phi_time (absolute timestamps)
    5. Generate self-contained Atomic Entries {m_k}

    Extended for Groups:
    6. Generate GroupMemory for group-wide information
    7. Generate UserMemory for user-specific information
    8. Generate InteractionMemory for user-to-user interactions
    """

    def __init__(
        self,
        llm_client: LLMClient,
        unified_store,  # UnifiedMemoryStore or VectorStore (backward compatible)
        window_size: int = None,
        enable_parallel_processing: bool = True,
        max_parallel_workers: int = 3,
        agent_id: Optional[str] = None
    ):
        self.llm_client = llm_client
        self.unified_store = unified_store

        # Backward compatibility: also expose as vector_store
        self.vector_store = unified_store

        self.window_size = window_size or config.WINDOW_SIZE
        self.agent_id = agent_id

        # Use config values as default if not explicitly provided
        self.enable_parallel_processing = enable_parallel_processing if enable_parallel_processing is not None else getattr(config, 'ENABLE_PARALLEL_PROCESSING', True)
        self.max_parallel_workers = max_parallel_workers if max_parallel_workers is not None else getattr(config, 'MAX_PARALLEL_WORKERS', 4)

        # Dialogue buffer
        self.dialogue_buffer: List[Dialogue] = []
        self.processed_count = 0

        # Previous window entries (for context)
        self.previous_entries: List[MemoryEntry] = []

        # Check if unified_store supports group memories
        self._supports_group_memories = hasattr(unified_store, 'add_group_memories_batch')

    def add_dialogue(self, dialogue: Dialogue, auto_process: bool = True):
        """
        Add a dialogue to the buffer
        """
        self.dialogue_buffer.append(dialogue)

        # Auto process
        if auto_process and len(self.dialogue_buffer) >= self.window_size:
            self.process_window()

    def add_dialogues(self, dialogues: List[Dialogue], auto_process: bool = True):
        """
        Batch add dialogues with optional parallel processing
        """
        if self.enable_parallel_processing and len(dialogues) > self.window_size * 2:
            # Use parallel processing for large batches
            self.add_dialogues_parallel(dialogues)
        else:
            # Use sequential processing for smaller batches
            for dialogue in dialogues:
                self.add_dialogue(dialogue, auto_process=False)

            # Process complete windows
            if auto_process:
                while len(self.dialogue_buffer) >= self.window_size:
                    self.process_window()

    def add_dialogues_parallel(self, dialogues: List[Dialogue]):
        """
        Add dialogues using parallel processing for better performance
        """
        try:
            # Add all dialogues to buffer first
            self.dialogue_buffer.extend(dialogues)

            # Group into windows for parallel processing (including remaining dialogues)
            windows_to_process = []
            while len(self.dialogue_buffer) >= self.window_size:
                window = self.dialogue_buffer[:self.window_size]
                self.dialogue_buffer = self.dialogue_buffer[self.window_size:]
                windows_to_process.append(window)

            # Add remaining dialogues as a smaller batch (no need to process separately)
            if self.dialogue_buffer:
                windows_to_process.append(self.dialogue_buffer)
                self.dialogue_buffer = []  # Clear buffer since we're processing all

            if windows_to_process:
                print(f"\n[Parallel Processing] Processing {len(windows_to_process)} batches in parallel with {self.max_parallel_workers} workers")
                print(f"Batch sizes: {[len(w) for w in windows_to_process]}")

                # Process all windows/batches in parallel (including remaining dialogues)
                self._process_windows_parallel(windows_to_process)

        except Exception as e:
            print(f"[Parallel Processing] Failed: {e}. Falling back to sequential processing...")
            # Fallback to sequential processing
            for window in windows_to_process:
                self.dialogue_buffer = window + self.dialogue_buffer
                self.process_window()

    def process_window(self):
        """
        Process current window dialogues - Core logic
        """
        if not self.dialogue_buffer:
            return

        # Extract window
        window = self.dialogue_buffer[:self.window_size]
        self.dialogue_buffer = self.dialogue_buffer[self.window_size:]

        print(f"\nProcessing window: {len(window)} dialogues (processed {self.processed_count} so far)")

        # Detect if this is a group conversation
        is_group = any(d.group_id is not None for d in window)

        if is_group and self._supports_group_memories:
            # Generate group memories (GroupMemory, UserMemory, InteractionMemory)
            group_memories, user_memories, interaction_memories = self._generate_group_memories(window)

            # Store to database
            if group_memories:
                self.unified_store.add_group_memories_batch(group_memories)
            if user_memories:
                self.unified_store.add_user_memories_batch(user_memories)
            if interaction_memories:
                self.unified_store.add_interaction_memories_batch(interaction_memories)

            total = len(group_memories) + len(user_memories) + len(interaction_memories)
            print(f"Generated {total} group memories (group: {len(group_memories)}, user: {len(user_memories)}, interaction: {len(interaction_memories)})")

            self.processed_count += len(window)
        else:
            # Generate DM memories (MemoryEntry)
            entries = self._generate_dm_memories(window)

            # Store to database
            if entries:
                self.unified_store.add_entries(entries)
                self.previous_entries = entries  # Save as context
                self.processed_count += len(window)

            print(f"Generated {len(entries)} memory entries")

    def process_remaining(self):
        """
        Process remaining dialogues (fallback method, normally handled in parallel)
        """
        if self.dialogue_buffer:
            print(f"\nProcessing remaining dialogues: {len(self.dialogue_buffer)} (fallback mode)")

            is_group = any(d.group_id is not None for d in self.dialogue_buffer)

            if is_group and self._supports_group_memories:
                group_memories, user_memories, interaction_memories = self._generate_group_memories(self.dialogue_buffer)

                if group_memories:
                    self.unified_store.add_group_memories_batch(group_memories)
                if user_memories:
                    self.unified_store.add_user_memories_batch(user_memories)
                if interaction_memories:
                    self.unified_store.add_interaction_memories_batch(interaction_memories)

                total = len(group_memories) + len(user_memories) + len(interaction_memories)
                print(f"Generated {total} group memories")
            else:
                entries = self._generate_dm_memories(self.dialogue_buffer)
                if entries:
                    self.unified_store.add_entries(entries)
                print(f"Generated {len(entries)} memory entries")

            self.processed_count += len(self.dialogue_buffer)
            self.dialogue_buffer = []

    def process_dialogues_direct(self, dialogues: List[Dialogue]) -> Dict[str, int]:
        """
        Process dialogues directly without using internal buffer.

        This method is used for stateless processing (Cloud Run) where dialogues
        come from Firestore instead of an in-memory buffer.

        Args:
            dialogues: List of Dialogue objects to process

        Returns:
            Dict with counts of generated memories
        """
        if not dialogues:
            return {"total": 0, "group": 0, "user": 0, "interaction": 0, "dm": 0}

        print(f"\n[Stateless Processing] Processing {len(dialogues)} dialogues directly")

        # Detect if this is a group conversation
        is_group = any(d.group_id is not None for d in dialogues)

        result = {"total": 0, "group": 0, "user": 0, "interaction": 0, "dm": 0}

        if is_group and self._supports_group_memories:
            # Generate group memories (GroupMemory, UserMemory, InteractionMemory)
            group_memories, user_memories, interaction_memories = self._generate_group_memories(dialogues)

            # Store to database
            if group_memories:
                self.unified_store.add_group_memories_batch(group_memories)
                result["group"] = len(group_memories)
            if user_memories:
                self.unified_store.add_user_memories_batch(user_memories)
                result["user"] = len(user_memories)
            if interaction_memories:
                self.unified_store.add_interaction_memories_batch(interaction_memories)
                result["interaction"] = len(interaction_memories)

            result["total"] = result["group"] + result["user"] + result["interaction"]
            print(f"Generated {result['total']} group memories (group: {result['group']}, user: {result['user']}, interaction: {result['interaction']})")
        else:
            # Generate DM memories (MemoryEntry)
            entries = self._generate_dm_memories(dialogues)

            # Store to database
            if entries:
                self.unified_store.add_entries(entries)
                result["dm"] = len(entries)
                result["total"] = len(entries)

            print(f"Generated {result['dm']} DM memory entries")

        self.processed_count += len(dialogues)
        return result

    # ============================================================
    # DM Memory Generation (SimpleMem compatible)
    # ============================================================

    def _generate_dm_memories(self, dialogues: List[Dialogue]) -> List[MemoryEntry]:
        """
        Generate DM memories (MemoryEntry) - SimpleMem compatible.
        """
        return self._generate_memory_entries(dialogues)

    def _generate_memory_entries(self, dialogues: List[Dialogue]) -> List[MemoryEntry]:
        """
        De-linearization Transformation F_theta: W_t -> {m_k}

        Paper Reference: Section 3.1 - Eq. (3)
        Applies composite transformation: F_theta = Phi_time o Phi_coref o Phi_extract
        """
        # Build dialogue text
        dialogue_text = "\n".join([str(d) for d in dialogues])
        dialogue_ids = [d.dialogue_id for d in dialogues]

        # Detect if this is a group conversation
        is_group = any(d.group_id is not None for d in dialogues)
        group_context = self._extract_group_context(dialogues) if is_group else {}

        # Build context
        context = ""
        if self.previous_entries:
            context = "\n[Previous Window Memory Entries (for reference to avoid duplication)]\n"
            for entry in self.previous_entries[:3]:
                context += f"- {entry.lossless_restatement}\n"

        # Build prompt
        prompt = self._build_extraction_prompt(
            dialogue_text, dialogue_ids, context,
            is_group=is_group, group_context=group_context
        )

        # Call LLM
        messages = [
            {
                "role": "system",
                "content": "You are a professional information extraction assistant, skilled at extracting structured, unambiguous information from conversations. You must output valid JSON format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Retry up to 3 times if parsing fails
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm_client.chat_completion(
                    messages,
                    temperature=0.1,
                    response_format=DM_MEMORY_ENTRIES_SCHEMA
                )

                entries = self._parse_llm_response(response, dialogues)

                # Validate and filter memories (reject hallucinations, low quality)
                if entries:
                    dialogue_text = "\n".join([str(d) for d in dialogues])
                    entries = validate_and_filter_memories(
                        entries,
                        dialogue_text,
                        embedding_model=None  # Skip semantic validation for speed
                    )

                return entries

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1}/{max_retries} failed to parse LLM response: {e}")
                    print(f"Retrying...")
                else:
                    print(f"All {max_retries} attempts failed to parse LLM response: {e}")
                    print(f"Raw response: {response[:500] if 'response' in locals() else 'No response'}")
                    return []

    # ============================================================
    # Group Memory Generation
    # ============================================================

    def _generate_group_memories(
        self,
        dialogues: List[Dialogue]
    ) -> Tuple[List[GroupMemory], List[UserMemory], List[InteractionMemory]]:
        """
        Generate group memories from dialogues.

        Returns:
            Tuple of (group_memories, user_memories, interaction_memories)
        """
        # Extract group context
        group_context = self._extract_group_context(dialogues)
        group_id = group_context.get('group_id', '')
        platform = group_context.get('platform', 'direct')

        # Build dialogue text
        dialogue_text = "\n".join([str(d) for d in dialogues])

        # Build prompt for group memory extraction
        prompt = self._build_group_extraction_prompt(dialogue_text, group_context)

        # Call LLM
        messages = [
            {
                "role": "system",
                "content": """You are a memory extraction specialist for group conversations.
Your task is to extract structured memories from conversations and classify them into categories:
1. GROUP_MEMORY: Information relevant to the entire group (announcements, decisions, group events)
2. USER_MEMORY: Information about individual users (expertise, preferences, facts about them)
3. INTERACTION_MEMORY: Notable interactions between users (questions, answers, discussions)

You must output valid JSON format."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Retry up to 3 times
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm_client.chat_completion(
                    messages,
                    temperature=0.1,
                    response_format=GROUP_MEMORIES_SCHEMA
                )

                return self._parse_group_memories_response(response, group_context, dialogues)

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1}/{max_retries} failed to parse group memories: {e}")
                    print(f"Retrying...")
                else:
                    print(f"All {max_retries} attempts failed to parse group memories: {e}")
                    return [], [], []

    def _build_group_extraction_prompt(
        self,
        dialogue_text: str,
        group_context: Dict[str, Any]
    ) -> str:
        """Build prompt for group memory extraction."""
        return f"""
Extract memories from the following group conversation.

[Group Context]
Platform: {group_context.get('platform', 'unknown')}
Group ID: {group_context.get('group_id', 'unknown')}
Participants: {', '.join(group_context.get('usernames', []))}

[Conversation]
{dialogue_text}

[Memory Types to Extract]
1. **Group Memories**: Information relevant to the entire group
   - memory_type: "announcement" | "fact" | "conversation"
   - Examples: "Group meeting scheduled for Friday", "The project deadline is March 15th"

2. **User Memories**: Information about specific users
   - memory_type: "expertise" | "preference" | "fact"
   - Examples: "Alice has 5 years of Solidity experience", "Bob prefers async communication"

3. **Interaction Memories**: Notable interactions between users
   - memory_type: "interaction"
   - Examples: "Alice helped Bob debug his smart contract", "Carol asked David about yield farming"

[Output Format]
Return a JSON object with three arrays:

```json
{{
  "group_memories": [
    {{
      "content": "Complete, unambiguous statement about the group",
      "memory_type": "announcement|fact|conversation",
      "speaker": "username who shared this or null",
      "keywords": ["keyword1", "keyword2"],
      "topics": ["topic1", "topic2"],
      "importance_score": 0.0 to 1.0
    }}
  ],
  "user_memories": [
    {{
      "content": "Complete statement about the user",
      "user_id": "platform:user_id",
      "username": "@username",
      "memory_type": "expertise|preference|fact",
      "keywords": ["keyword1", "keyword2"],
      "topics": ["topic1", "topic2"],
      "importance_score": 0.0 to 1.0
    }}
  ],
  "interaction_memories": [
    {{
      "content": "Description of the interaction",
      "speaker_id": "platform:speaker_id",
      "listener_id": "platform:listener_id",
      "interaction_type": "question|answer|discussion|help",
      "keywords": ["keyword1", "keyword2"],
      "topics": ["topic1", "topic2"],
      "importance_score": 0.0 to 1.0
    }}
  ]
}}
```

[Importance Score Guidelines]
- 0.8-1.0: Critical info (wallet addresses, expertise declarations, important decisions)
- 0.5-0.7: Useful info (preferences, opinions, project updates)
- 0.2-0.4: Nice to know (casual conversation, general questions)
- 0.0-0.2: Trivial (greetings, filler)

[Rules]
1. NO pronouns - use actual names/usernames
2. Each memory must be self-contained and understandable without context
3. Skip trivial greetings and filler
4. Focus on information that would be useful to remember

Return ONLY the JSON object.
"""

    def _parse_group_memories_response(
        self,
        response: str,
        group_context: Dict[str, Any],
        dialogues: List[Dialogue]
    ) -> Tuple[List[GroupMemory], List[UserMemory], List[InteractionMemory]]:
        """Parse LLM response into group memory objects."""
        try:
            data = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            data = self.llm_client.extract_json(response)

        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object but got: {type(data)}")

        group_id = group_context.get('group_id', '')
        platform = group_context.get('platform', 'direct')
        agent_id = self.agent_id or getattr(self.unified_store, 'agent_id', 'unknown')

        now = datetime.now(timezone.utc).isoformat()

        # Parse group memories
        group_memories = []
        for item in data.get('group_memories', []):
            try:
                memory_type_str = item.get('memory_type', 'conversation')
                try:
                    memory_type = GroupMemoryType(memory_type_str)
                except ValueError:
                    memory_type = GroupMemoryType.CONVERSATION

                group_memories.append(GroupMemory(
                    agent_id=agent_id,
                    group_id=group_id,
                    memory_level=MemoryLevel.GROUP,
                    memory_type=memory_type,
                    privacy_scope=GroupPrivacyScope.PUBLIC,
                    content=item['content'],
                    speaker=item.get('speaker'),
                    keywords=item.get('keywords', []),
                    topics=item.get('topics', []),
                    importance_score=item.get('importance_score', 0.5),
                    first_seen=now,
                    last_seen=now,
                    last_updated=now
                ))
            except Exception as e:
                print(f"Warning: Failed to parse group memory: {e}")
                continue

        # Parse user memories
        user_memories = []
        for item in data.get('user_memories', []):
            try:
                memory_type_str = item.get('memory_type', 'fact')
                try:
                    memory_type = GroupMemoryType(memory_type_str)
                except ValueError:
                    memory_type = GroupMemoryType.FACT

                user_id = item.get('user_id', '')
                username = item.get('username', '')

                # Try to find user_id from dialogues if not provided
                if not user_id and username:
                    for d in dialogues:
                        if d.username == username:
                            user_id = d.user_id or ''
                            break

                user_memories.append(UserMemory(
                    agent_id=agent_id,
                    group_id=group_id,
                    user_id=user_id,
                    memory_level=MemoryLevel.INDIVIDUAL,
                    memory_type=memory_type,
                    privacy_scope=GroupPrivacyScope.PROTECTED,
                    content=item['content'],
                    keywords=item.get('keywords', []),
                    topics=item.get('topics', []),
                    importance_score=item.get('importance_score', 0.5),
                    first_seen=now,
                    last_seen=now,
                    last_updated=now,
                    username=username,
                    platform=platform
                ))
            except Exception as e:
                print(f"Warning: Failed to parse user memory: {e}")
                continue

        # Parse interaction memories
        interaction_memories = []
        for item in data.get('interaction_memories', []):
            try:
                interaction_memories.append(InteractionMemory(
                    agent_id=agent_id,
                    group_id=group_id,
                    speaker_id=item.get('speaker_id', ''),
                    listener_id=item.get('listener_id', ''),
                    mentioned_users=[],
                    memory_level=MemoryLevel.INDIVIDUAL,
                    memory_type=GroupMemoryType.INTERACTION,
                    privacy_scope=GroupPrivacyScope.PROTECTED,
                    content=item['content'],
                    keywords=item.get('keywords', []),
                    topics=item.get('topics', []),
                    importance_score=item.get('importance_score', 0.5),
                    first_seen=now,
                    last_seen=now,
                    last_updated=now,
                    interaction_type=item.get('interaction_type', 'discussion')
                ))
            except Exception as e:
                print(f"Warning: Failed to parse interaction memory: {e}")
                continue

        return group_memories, user_memories, interaction_memories

    # ============================================================
    # Helper Methods
    # ============================================================

    def _extract_group_context(self, dialogues: List[Dialogue]) -> Dict[str, Any]:
        """Extract group context from dialogues for prompt enrichment."""
        context = {
            "group_id": None,
            "platform": "direct",
            "users": set(),
            "usernames": set(),
            "has_mentions": False,
            "has_replies": False
        }

        for d in dialogues:
            if d.group_id:
                context["group_id"] = d.group_id
            if d.platform:
                context["platform"] = d.platform
            if d.user_id:
                context["users"].add(d.user_id)
            if d.username:
                context["usernames"].add(d.username)
            if d.mentioned_users:
                context["has_mentions"] = True
            if d.is_reply:
                context["has_replies"] = True

        # Convert sets to lists for serialization
        context["users"] = list(context["users"])
        context["usernames"] = list(context["usernames"])

        return context

    def _build_extraction_prompt(
        self,
        dialogue_text: str,
        dialogue_ids: List[int],
        context: str,
        is_group: bool = False,
        group_context: Dict[str, Any] = None
    ) -> str:
        """
        Build LLM extraction prompt with optional group context.
        """
        # Build group-specific section if applicable
        group_section = ""
        classification_section = ""

        if is_group and group_context:
            group_section = f"""
[Group Context]
This is a GROUP conversation on {group_context.get('platform', 'unknown')} platform.
Group ID: {group_context.get('group_id', 'unknown')}
Participants: {', '.join(group_context.get('usernames', []))}
Has mentions: {group_context.get('has_mentions', False)}
Has replies: {group_context.get('has_replies', False)}
"""
            classification_section = """
5. **Memory Classification** (IMPORTANT for groups):
   - memory_type: Classify each memory as one of:
     * "expertise" - User demonstrates skill/knowledge (e.g., "I've been building smart contracts for 3 years")
     * "preference" - User expresses preference/interest (e.g., "I prefer using Hardhat over Foundry")
     * "fact" - Factual information about user (e.g., "My wallet is 0x123...")
     * "announcement" - Group-wide information (e.g., "The meeting is rescheduled to Friday")
     * "conversation" - General conversation (default)
   - importance_score: 0.0 to 1.0 (how important/memorable is this information?)
     * 0.8-1.0: Critical info (wallet addresses, expertise, important facts)
     * 0.5-0.7: Useful info (preferences, opinions)
     * 0.2-0.4: Nice to know (casual conversation)
     * 0.0-0.2: Trivial (greetings, filler)
"""

        # Build output format based on whether it's a group
        if is_group:
            output_format = """
[Output Format]
Return a JSON array, each element is a memory entry:

```json
[
  {
    "lossless_restatement": "Complete unambiguous restatement",
    "keywords": ["keyword1", "keyword2", ...],
    "timestamp": "YYYY-MM-DDTHH:MM:SS or null",
    "location": "location name or null",
    "persons": ["name1", "name2", ...],
    "entities": ["entity1", "entity2", ...],
    "topic": "topic phrase",
    "memory_type": "expertise|preference|fact|announcement|conversation",
    "importance_score": 0.0 to 1.0
  },
  ...
]
```
"""
        else:
            output_format = """
[Output Format]
Return a JSON array, each element is a memory entry:

```json
[
  {
    "lossless_restatement": "Complete unambiguous restatement (must include all subjects, objects, time, location, etc.)",
    "keywords": ["keyword1", "keyword2", ...],
    "timestamp": "YYYY-MM-DDTHH:MM:SS or null",
    "location": "location name or null",
    "persons": ["name1", "name2", ...],
    "entities": ["entity1", "entity2", ...],
    "topic": "topic phrase"
  },
  ...
]
```
"""

        return f"""
Your task is to extract all valuable information from the following dialogues and convert them into structured memory entries.
{group_section}
{context}

[Current Window Dialogues]
{dialogue_text}

[Requirements]
1. **Complete Coverage**: Generate enough memory entries to ensure ALL information in the dialogues is captured
2. **Force Disambiguation**: Absolutely PROHIBIT using pronouns (he, she, it, they, this, that) and relative time (yesterday, today, last week, tomorrow)
3. **Lossless Information**: Each entry's lossless_restatement must be a complete, independent, understandable sentence
4. **Precise Extraction**:
   - keywords: Core keywords (names, places, entities, topic words)
   - timestamp: Absolute time in ISO 8601 format (if explicit time mentioned in dialogue)
   - location: Specific location name (if mentioned)
   - persons: All person names mentioned
   - entities: Companies, products, organizations, etc.
   - topic: The topic of this information
{classification_section}
{output_format}

[Example]
Dialogues:
[2025-11-15T14:30:00] Alice: Bob, let's meet at Starbucks tomorrow at 2pm to discuss the new product
[2025-11-15T14:31:00] Bob: Okay, I'll prepare the materials

Output:
```json
[
  {{
    "lossless_restatement": "Alice suggested at 2025-11-15T14:30:00 to meet with Bob at Starbucks on 2025-11-16T14:00:00 to discuss the new product.",
    "keywords": ["Alice", "Bob", "Starbucks", "new product", "meeting"],
    "timestamp": "2025-11-16T14:00:00",
    "location": "Starbucks",
    "persons": ["Alice", "Bob"],
    "entities": ["new product"],
    "topic": "Product discussion meeting arrangement"{"," if is_group else ""}
    {"\"memory_type\": \"announcement\"," if is_group else ""}
    {"\"importance_score\": 0.7" if is_group else ""}
  }},
  {{
    "lossless_restatement": "Bob agreed to attend the meeting and committed to prepare relevant materials.",
    "keywords": ["Bob", "prepare materials", "agree"],
    "timestamp": null,
    "location": null,
    "persons": ["Bob"],
    "entities": [],
    "topic": "Meeting preparation confirmation"{"," if is_group else ""}
    {"\"memory_type\": \"conversation\"," if is_group else ""}
    {"\"importance_score\": 0.4" if is_group else ""}
  }}
]
```

Now process the above dialogues. Return ONLY a JSON object with an "entries" key containing the array:
{{"entries": [...]}}
"""

    def _parse_llm_response(
        self,
        response: str,
        dialogues: List[Dialogue]
    ) -> List[MemoryEntry]:
        """
        Parse LLM response to MemoryEntry list, inheriting context from dialogues.
        """
        # Parse JSON - structured outputs return valid JSON directly
        try:
            data = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            data = self.llm_client.extract_json(response)

        # Handle wrapped format {"entries": [...]} from structured outputs
        if isinstance(data, dict) and "entries" in data:
            data = data["entries"]

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array but got: {type(data)}")

        # Extract common context from dialogues
        group_id = None
        platform = "direct"
        user_id = None
        username = None

        for d in dialogues:
            if d.group_id:
                group_id = d.group_id
            if d.platform:
                platform = d.platform
            # Take last user as the primary speaker (could be improved)
            if d.user_id:
                user_id = d.user_id
            if d.username:
                username = d.username

        entries = []
        for item in data:
            # Parse memory_type
            memory_type_str = item.get("memory_type", "conversation")
            try:
                memory_type = MemoryType(memory_type_str)
            except ValueError:
                memory_type = MemoryType.CONVERSATION

            # Determine privacy scope based on group context
            if group_id:
                privacy_scope = PrivacyScope.GROUP_ONLY
            else:
                privacy_scope = PrivacyScope.PRIVATE

            # Create MemoryEntry with all context
            entry = MemoryEntry(
                lossless_restatement=item["lossless_restatement"],
                keywords=item.get("keywords", []),
                timestamp=item.get("timestamp"),
                location=item.get("location"),
                persons=item.get("persons", []),
                entities=item.get("entities", []),
                topic=item.get("topic"),
                # Group context (inherited from dialogues)
                group_id=group_id,
                user_id=user_id,
                username=username,
                platform=platform,
                # Classification (from LLM or defaults)
                memory_type=memory_type,
                privacy_scope=privacy_scope,
                importance_score=item.get("importance_score", 0.5)
            )
            entries.append(entry)

        return entries

    def _process_windows_parallel(self, windows: List[List[Dialogue]]):
        """
        Process multiple windows in parallel using ThreadPoolExecutor
        """
        all_entries = []
        all_group_memories = []
        all_user_memories = []
        all_interaction_memories = []

        # Check if any window has group context
        has_groups = any(
            any(d.group_id is not None for d in window)
            for window in windows
        )

        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
            # Submit all window processing tasks
            future_to_window = {}
            for i, window in enumerate(windows):
                is_group = any(d.group_id is not None for d in window)

                if is_group and self._supports_group_memories:
                    future = executor.submit(self._generate_group_memories_worker, window, i+1)
                else:
                    future = executor.submit(self._generate_dm_memories_worker, window, i+1)

                future_to_window[future] = (window, i+1, is_group)

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_window):
                window, window_num, is_group = future_to_window[future]
                try:
                    result = future.result()

                    if is_group and self._supports_group_memories:
                        group_mems, user_mems, interaction_mems = result
                        all_group_memories.extend(group_mems)
                        all_user_memories.extend(user_mems)
                        all_interaction_memories.extend(interaction_mems)
                        total = len(group_mems) + len(user_mems) + len(interaction_mems)
                        print(f"[Parallel Processing] Window {window_num} completed: {total} group memories")
                    else:
                        all_entries.extend(result)
                        print(f"[Parallel Processing] Window {window_num} completed: {len(result)} entries")

                except Exception as e:
                    print(f"[Parallel Processing] Window {window_num} failed: {e}")

        # Store all entries to database in batch
        if all_entries:
            print(f"\n[Parallel Processing] Storing {len(all_entries)} DM entries to database...")
            self.unified_store.add_entries(all_entries)
            self.previous_entries = all_entries[-10:]

        if all_group_memories:
            print(f"[Parallel Processing] Storing {len(all_group_memories)} group memories...")
            self.unified_store.add_group_memories_batch(all_group_memories)

        if all_user_memories:
            print(f"[Parallel Processing] Storing {len(all_user_memories)} user memories...")
            self.unified_store.add_user_memories_batch(all_user_memories)

        if all_interaction_memories:
            print(f"[Parallel Processing] Storing {len(all_interaction_memories)} interaction memories...")
            self.unified_store.add_interaction_memories_batch(all_interaction_memories)

        self.processed_count += sum(len(window) for window in windows)
        print(f"[Parallel Processing] Completed processing {len(windows)} windows")

    def _generate_dm_memories_worker(self, window: List[Dialogue], window_num: int) -> List[MemoryEntry]:
        """Worker function for parallel processing of DM memories."""
        print(f"[Worker {window_num}] Processing {len(window)} dialogues (DM mode)")
        return self._generate_memory_entries(window)

    def _generate_group_memories_worker(
        self,
        window: List[Dialogue],
        window_num: int
    ) -> Tuple[List[GroupMemory], List[UserMemory], List[InteractionMemory]]:
        """Worker function for parallel processing of group memories."""
        print(f"[Worker {window_num}] Processing {len(window)} dialogues (Group mode)")
        return self._generate_group_memories(window)

    # Backward compatibility method
    def _generate_memory_entries_worker(self, window: List[Dialogue], dialogue_ids: List[int], window_num: int) -> List[MemoryEntry]:
        """Legacy worker function (backward compatible)."""
        return self._generate_dm_memories_worker(window, window_num)
