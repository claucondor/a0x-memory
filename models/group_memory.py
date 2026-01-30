"""
Group Memory Models - Multi-level memory architecture for AI agents

Implements Arch2 (Partitioning) + Arch5 (Hybrid Multi-Level) architecture:
- Partitioned storage by memory type (group, user, interaction, cross_group)
- Multi-level memory (individual, group, cross-group)
- Cross-agent identity linking
- Privacy-scoped access control
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from enum import Enum
import uuid


class MemoryLevel(str, Enum):
    """Memory level in the hierarchy"""
    INDIVIDUAL = "individual"
    GROUP = "group"
    CROSS_GROUP = "cross_group"


class MemoryType(str, Enum):
    """Type of memory content"""
    CONVERSATION = "conversation"
    FACT = "fact"
    PREFERENCE = "preference"
    EXPERTISE = "expertise"
    ANNOUNCEMENT = "announcement"
    INTERACTION = "interaction"


class PrivacyScope(str, Enum):
    """Privacy scope for access control"""
    PUBLIC = "public"  # All group members can see
    PROTECTED = "protected"  # Only specific users can see
    PRIVATE = "private"  # Only agent can see


class GroupMemory(BaseModel):
    """
    Group-level memory - shared knowledge within a group

    Stores:
    - Group decisions and announcements
    - Group culture and norms
    - Shared expertise and knowledge
    - Group-wide events
    """
    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = Field(..., description="Agent this memory belongs to")
    group_id: str = Field(..., description="Group identifier")

    # Memory classification
    memory_level: MemoryLevel = Field(default=MemoryLevel.GROUP)
    memory_type: MemoryType = Field(..., description="Type of memory content")
    privacy_scope: PrivacyScope = Field(default=PrivacyScope.PUBLIC)

    # Shareability - can this memory be shared across contexts?
    is_shareable: bool = Field(default=False, description="Can this memory be shared to other contexts (DM, other groups)?")

    # Content
    content: str = Field(..., description="Memory content")
    speaker: Optional[str] = Field(None, description="Who spoke/created this")
    keywords: List[str] = Field(default_factory=list, description="Keywords for search")
    topics: List[str] = Field(default_factory=list, description="Topics covered")

    # Metadata
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance 0-1")
    evidence_count: int = Field(default=1, ge=1, description="Number of times observed")
    first_seen: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_seen: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Source tracking
    source_message_id: Optional[str] = Field(None, description="Original message ID")
    source_timestamp: Optional[str] = Field(None, description="Original message timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "memory_id": "550e8400-e29b-41d4-a716-446655440000",
                "agent_id": "71f6f657-6800-0892-875f-f26e8c213756",
                "group_id": "telegram_group_123",
                "memory_level": "group",
                "memory_type": "announcement",
                "privacy_scope": "public",
                "is_shareable": False,
                "content": "Group decided to hold weekly DeFi discussions on Fridays",
                "speaker": "admin_user",
                "keywords": ["defi", "weekly", "friday", "discussion"],
                "topics": ["defi", "community"],
                "importance_score": 0.8,
                "evidence_count": 5,
                "first_seen": "2025-01-25T10:00:00Z",
                "last_seen": "2025-01-25T16:00:00Z",
                "last_updated": "2025-01-25T16:00:00Z",
                "source_message_id": "msg_123",
                "source_timestamp": "2025-01-25T10:00:00Z"
            }
        }


class UserMemory(BaseModel):
    """
    User-level memory - user-specific context within a group

    Stores:
    - User actions and contributions
    - User preferences and interests
    - User expertise areas
    - Personal interactions with agent
    """
    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = Field(..., description="Agent this memory belongs to")
    group_id: str = Field(..., description="Group where this was observed")
    user_id: str = Field(..., description="User identifier")

    # Memory classification
    memory_level: MemoryLevel = Field(default=MemoryLevel.INDIVIDUAL)
    memory_type: MemoryType = Field(..., description="Type of memory content")
    privacy_scope: PrivacyScope = Field(default=PrivacyScope.PROTECTED)

    # Shareability - can this memory be shared across contexts?
    is_shareable: bool = Field(default=False, description="Can this memory be shared to other contexts (DM, other groups)?")

    # Content
    content: str = Field(..., description="Memory content")
    keywords: List[str] = Field(default_factory=list, description="Keywords for search")
    topics: List[str] = Field(default_factory=list, description="Topics covered")

    # Metadata
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance 0-1")
    evidence_count: int = Field(default=1, ge=1, description="Number of times observed")
    first_seen: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_seen: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Source tracking
    source_message_id: Optional[str] = Field(None, description="Original message ID")
    source_timestamp: Optional[str] = Field(None, description="Original message timestamp")

    # User-specific metadata
    username: Optional[str] = Field(None, description="Username/handle")
    platform: Optional[str] = Field(None, description="Platform (telegram, twitter, etc)")

    class Config:
        json_schema_extra = {
            "example": {
                "memory_id": "550e8400-e29b-41d4-a716-446655440001",
                "agent_id": "71f6f657-6800-0892-875f-f26e8c213756",
                "group_id": "telegram_group_123",
                "user_id": "telegram_user_456",
                "memory_level": "individual",
                "memory_type": "preference",
                "privacy_scope": "protected",
                "content": "User prefers technical explanations over simplified ones",
                "keywords": ["technical", "explanations", "preference"],
                "topics": ["communication", "preferences"],
                "importance_score": 0.7,
                "evidence_count": 3,
                "first_seen": "2025-01-20T10:00:00Z",
                "last_seen": "2025-01-25T14:00:00Z",
                "last_updated": "2025-01-25T14:00:00Z",
                "source_message_id": "msg_456",
                "source_timestamp": "2025-01-20T10:00:00Z",
                "username": "@alice_web3",
                "platform": "telegram"
            }
        }


class InteractionMemory(BaseModel):
    """
    Interaction memory - conversations between users

    Stores:
    - User-to-user conversations
    - Replies and mentions
    - Collaborative discussions
    """
    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = Field(..., description="Agent this memory belongs to")
    group_id: str = Field(..., description="Group where interaction occurred")

    # Participants
    speaker_id: str = Field(..., description="User who spoke")
    listener_id: str = Field(..., description="User who was addressed")
    mentioned_users: List[str] = Field(default_factory=list, description="Other users mentioned")

    # Memory classification
    memory_level: MemoryLevel = Field(default=MemoryLevel.INDIVIDUAL)
    memory_type: MemoryType = Field(default=MemoryType.INTERACTION)
    privacy_scope: PrivacyScope = Field(default=PrivacyScope.PROTECTED)

    # Shareability - can this memory be shared across contexts?
    is_shareable: bool = Field(default=False, description="Can this memory be shared to other contexts (DM, other groups)?")

    # Content
    content: str = Field(..., description="Interaction content")
    keywords: List[str] = Field(default_factory=list, description="Keywords for search")
    topics: List[str] = Field(default_factory=list, description="Topics covered")

    # Metadata
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance 0-1")
    evidence_count: int = Field(default=1, ge=1, description="Number of similar interactions")
    first_seen: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_seen: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Source tracking
    source_message_id: Optional[str] = Field(None, description="Original message ID")
    source_timestamp: Optional[str] = Field(None, description="Original message timestamp")

    # Interaction metadata
    interaction_type: Optional[str] = Field(None, description="reply, mention, question, etc")

    class Config:
        json_schema_extra = {
            "example": {
                "memory_id": "550e8400-e29b-41d4-a716-446655440002",
                "agent_id": "71f6f657-6800-0892-875f-f26e8c213756",
                "group_id": "telegram_group_123",
                "speaker_id": "telegram_user_456",
                "listener_id": "telegram_user_789",
                "mentioned_users": ["telegram_user_101"],
                "memory_level": "individual",
                "memory_type": "interaction",
                "privacy_scope": "protected",
                "content": "Alice asked Bob about yield farming strategies",
                "keywords": ["yield", "farming", "strategies"],
                "topics": ["defi", "yield"],
                "importance_score": 0.6,
                "evidence_count": 1,
                "first_seen": "2025-01-25T15:00:00Z",
                "last_seen": "2025-01-25T15:00:00Z",
                "last_updated": "2025-01-25T15:00:00Z",
                "source_message_id": "msg_789",
                "source_timestamp": "2025-01-25T15:00:00Z",
                "interaction_type": "question"
            }
        }


class CrossGroupMemory(BaseModel):
    """
    Cross-group memory - consolidated patterns across multiple groups

    Implements Arch5 cross-group consolidation:
    - User identity across groups
    - Universal preferences
    - Expertise areas
    - Cross-group references only (no sensitive data)

    Consolidated from individual/group memories when:
    - Same pattern observed in >= 2 groups
    - High confidence score
    - Clear pattern type identified
    """
    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = Field(..., description="Agent this memory belongs to")

    # Universal user identification
    universal_user_id: str = Field(..., description="platform:specific_id (e.g., telegram:123456789)")
    user_identities: List[str] = Field(default_factory=list, description="All known user_ids across groups")

    # Groups involved
    groups_involved: List[str] = Field(..., description="Group IDs where this pattern was observed")
    group_count: int = Field(..., ge=2, description="Number of groups with this pattern")

    # Memory classification
    memory_level: MemoryLevel = Field(default=MemoryLevel.CROSS_GROUP)
    memory_type: MemoryType = Field(..., description="Type of consolidated pattern")
    privacy_scope: PrivacyScope = Field(default=PrivacyScope.PROTECTED)

    # Content (consolidated, no specific group references)
    content: str = Field(..., description="Consolidated pattern description")
    keywords: List[str] = Field(default_factory=list, description="Keywords for search")
    topics: List[str] = Field(default_factory=list, description="Topics covered")

    # Consolidation metadata
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in consolidation")
    pattern_type: str = Field(..., description="expertise, preference, behavior, etc")
    evidence_count: int = Field(default=0, ge=0, description="Total observations across all groups")
    first_seen: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_seen: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    consolidated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Source tracking (for traceability)
    source_memory_ids: List[str] = Field(default_factory=list, description="Original memory IDs that were consolidated")

    class Config:
        json_schema_extra = {
            "example": {
                "memory_id": "550e8400-e29b-41d4-a716-446655440003",
                "agent_id": "71f6f657-6800-0892-875f-f26e8c213756",
                "universal_user_id": "telegram:123456789",
                "user_identities": ["telegram_user_456", "telegram_user_456"],
                "groups_involved": ["telegram_group_123", "telegram_group_456"],
                "group_count": 2,
                "memory_level": "cross_group",
                "memory_type": "expertise",
                "privacy_scope": "protected",
                "content": "User is a DeFi expert specializing in yield farming and liquid staking",
                "keywords": ["defi", "yield", "farming", "liquid", "staking"],
                "topics": ["defi", "expertise"],
                "confidence_score": 0.85,
                "pattern_type": "expertise",
                "evidence_count": 15,
                "first_seen": "2025-01-15T10:00:00Z",
                "last_seen": "2025-01-25T16:00:00Z",
                "last_updated": "2025-01-25T16:00:00Z",
                "consolidated_at": "2025-01-25T16:00:00Z",
                "source_memory_ids": ["mem_001", "mem_002", "mem_003"]
            }
        }


class CrossAgentLink(BaseModel):
    """
    Cross-agent identity mapping

    Maps the same user across different agents:
    - Links user_id from agent1 to user_id from agent2
    - Maintains universal_user_id for cross-platform identity
    - Stores linking confidence and evidence
    """
    link_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    universal_user_id: str = Field(..., description="Universal user identifier")

    # Agent mappings
    agent_mappings: Dict[str, str] = Field(..., description="agent_id -> user_id mappings")

    # Metadata
    linking_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in linking")
    evidence_count: int = Field(default=1, ge=1, description="Number of linking evidences")
    first_linked: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_verified: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Linking method
    linking_method: str = Field(default="manual", description="How the link was established")
    linking_evidence: List[str] = Field(default_factory=list, description="Evidence for this link")

    class Config:
        json_schema_extra = {
            "example": {
                "link_id": "550e8400-e29b-41d4-a716-446655440004",
                "universal_user_id": "telegram:123456789",
                "agent_mappings": {
                    "jessexbt": "telegram_user_456",
                    "other_agent": "telegram_user_456"
                },
                "linking_confidence": 0.95,
                "evidence_count": 10,
                "first_linked": "2025-01-15T10:00:00Z",
                "last_updated": "2025-01-25T16:00:00Z",
                "last_verified": "2025-01-25T16:00:00Z",
                "linking_method": "wallet_match",
                "linking_evidence": ["Same wallet address", "Same username", "Cross-referenced in messages"]
            }
        }

    def add_agent_mapping(self, agent_id: str, user_id: str):
        """Add or update an agent mapping."""
        self.agent_mappings[agent_id] = user_id
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def get_user_id_for_agent(self, agent_id: str) -> Optional[str]:
        """Get the user_id for a specific agent."""
        return self.agent_mappings.get(agent_id)

    def has_agent(self, agent_id: str) -> bool:
        """Check if this link includes the specified agent."""
        return agent_id in self.agent_mappings


class ResponseType(str, Enum):
    """Type of agent response"""
    GREETING = "greeting"
    ANSWER = "answer"
    CLARIFICATION = "clarification"
    RECOMMENDATION = "recommendation"
    QUESTION = "question"
    ACKNOWLEDGMENT = "acknowledgment"
    OTHER = "other"


class AgentResponse(BaseModel):
    """
    Agent-generated response for avoiding repetition.

    Stores:
    - Complete responses generated by the agent
    - Trigger messages that caused the response
    - Metadata for semantic search and deduplication
    - Used to provide context about previous responses to avoid repetition
    """
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = Field(..., description="Agent this response belongs to")

    # Scope
    scope: str = Field(default="user", description="global | user | group")
    group_id: Optional[str] = Field(None, description="Group identifier (None for DMs)")
    user_id: Optional[str] = Field(None, description="User who triggered this response")

    # Trigger (what caused this response)
    trigger_message: str = Field(..., description="User message that triggered the response")

    # Response
    content: str = Field(..., description="Complete response content")
    summary: Optional[str] = Field(None, description="Short summary for search")

    # Classification
    response_type: ResponseType = Field(default=ResponseType.ANSWER, description="Type of response")
    topics: List[str] = Field(default_factory=list, description="Topics covered")
    keywords: List[str] = Field(default_factory=list, description="Keywords for search")

    # Temporal
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Metrics
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance 0-1")

    # Legacy fields for backward compatibility
    content_hash: Optional[str] = Field(None, description="SHA256 hash for deduplication")
    trigger_message_id: Optional[str] = Field(None, description="Original message ID")
    token_count: int = Field(default=0, ge=0, description="Tokens in response")
    was_repeated: bool = Field(default=False, description="Whether we detected repetition")

    class Config:
        json_schema_extra = {
            "example": {
                "response_id": "550e8400-e29b-41d4-a716-446655440005",
                "agent_id": "71f6f657-6800-0892-875f-f26e8c213756",
                "group_id": "telegram_group_123",
                "user_id": "telegram:123456789",
                "content": "Base has low gas fees, typically under 1 cent for transactions...",
                "content_hash": "a1b2c3d4e5f6",
                "summary": "Explained Base gas fees are under 1 cent",
                "response_type": "answer",
                "topics": ["base", "gas", "fees"],
                "keywords": ["base", "gas", "fees", "cheap", "transaction"],
                "trigger_message": "How much are gas fees on Base?",
                "trigger_message_id": "msg_456",
                "timestamp": "2025-01-26T10:00:00Z",
                "token_count": 45,
                "was_repeated": False,
                "importance_score": 0.7
            }
        }


class GroupTone(str, Enum):
    """Group communication tone"""
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    EDUCATIONAL = "educational"
    MEME_HEAVY = "meme_heavy"
    OTHER = "other"


class SummaryLevel(str, Enum):
    """Summary level based on message volume"""
    MICRO = "micro"    # Every ~50 messages
    CHUNK = "chunk"    # Every 5 micros (~250 messages)
    BLOCK = "block"    # Every 5 chunks (~1250 messages)
    ERA = "era"        # Every 5 blocks (~6250 messages) - long-term archive


class GroupProfile(BaseModel):
    """
    Group profile - cultural and contextual information about a group.

    Generated by LLM from group messages:
    - Group purpose and main topics
    - Communication tone and norms
    - Active participants and culture
    - Expertise level of the group

    Updated incrementally as more messages are processed.
    """
    profile_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = Field(..., description="Agent this profile belongs to")
    group_id: str = Field(..., description="Group identifier")

    # Group identity
    group_name: Optional[str] = Field(None, description="Group name/title")
    platform: str = Field(..., description="Platform: telegram, xmtp, farcaster, twitter")

    # LLM-extracted content
    summary: str = Field(..., description="What this group is about")
    main_topics: List[str] = Field(default_factory=list, description="Main topics discussed")
    group_purpose: str = Field(..., description="Primary purpose of the group")

    # Communication style
    tone: GroupTone = Field(default=GroupTone.CASUAL, description="Group communication tone")
    communication_norms: List[str] = Field(default_factory=list, description="Norms and rules")

    # Group culture
    expertise_level: str = Field(default="intermediate", description="Group expertise level")
    activity_level: str = Field(default="active", description="How active the group is")
    member_count_estimate: Optional[int] = Field(None, description="Estimated active members")

    # Content preferences
    preferred_content_types: List[str] = Field(default_factory=list, description="Types of content preferred")

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

    # Metadata
    total_messages_processed: int = Field(default=0, description="Messages used to build profile")
    last_message_timestamp: Optional[str] = Field(None, description="Last message ISO timestamp")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    profile_version: int = Field(default=1, description="Profile version")

    class Config:
        json_schema_extra = {
            "example": {
                "profile_id": "550e8400-e29b-41d4-a716-446655440006",
                "agent_id": "71f6f657-6800-0892-875f-f26e8c213756",
                "group_id": "telegram_group_123",
                "group_name": "DeFi Degens",
                "platform": "telegram",
                "summary": "Active DeFi community focused on yield farming, liquid staking, and arbitrage opportunities",
                "main_topics": ["yield farming", "liquid staking", "arbitrage", "base chain"],
                "group_purpose": "Discuss DeFi strategies and opportunities",
                "tone": "casual",
                "communication_norms": ["technical discussions welcome", "share opportunities", "no shilling"],
                "expertise_level": "advanced",
                "activity_level": "high",
                "member_count_estimate": 250,
                "preferred_content_types": ["yield strategies", "new protocols", "security alerts"],
                "topic_evolution": [
                    {"period": "2024-W05", "topics": ["grants", "audits"], "activity_score": 0.8}
                ],
                "current_era": {
                    "started": "2024-01-15",
                    "theme": "Base ecosystem grants",
                    "key_topics": ["grants", "audits", "applications"]
                },
                "total_messages_processed": 150,
                "last_message_timestamp": "2025-01-26T10:00:00Z",
                "created_at": "2025-01-15T10:00:00Z",
                "updated_at": "2025-01-26T10:00:00Z",
                "profile_version": 3
            }
        }


class GroupSummary(BaseModel):
    """
    Hierarchical group summary based on message volume.

    Summaries aggregate upward by volume:
    - Micro: Every ~50 messages (raw messages → summary)
    - Chunk: Every 5 micros (~250 messages, micros → chunk)
    - Block: Every 5 chunks (~1250 messages, chunks → block)

    Decay mechanism:
    - decay_score starts at 1.0
    - Decays based on messages_since_created (not time)
    - Old summaries get pruned when decay_score < 0.1
    """
    summary_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = Field(..., description="Agent this summary belongs to")
    group_id: str = Field(..., description="Group identifier")

    # Level definition (volume-based)
    level: SummaryLevel = Field(..., description="Summary level: micro, chunk, block")

    # Message range (primary identifier)
    message_start: int = Field(..., description="First message index in this summary")
    message_end: int = Field(..., description="Last message index in this summary")
    message_count: int = Field(default=0, description="Messages processed in this summary")

    # Temporal context (for display, not triggers)
    time_start: str = Field(..., description="First message timestamp ISO")
    time_end: str = Field(..., description="Last message timestamp ISO")
    duration_hours: float = Field(default=0.0, description="Time span in hours")
    activity_rate: float = Field(default=0.0, description="Messages per hour")

    # Content
    summary: str = Field(..., description="2-5 sentence summary")
    topics: List[str] = Field(default_factory=list, description="3-7 main topics")
    highlights: List[str] = Field(default_factory=list, description="2-4 notable events/discussions")
    active_users: List[str] = Field(default_factory=list, description="Top 5 contributors")

    # Decay (based on new messages, not time)
    decay_score: float = Field(default=1.0, ge=0.0, le=1.0, description="1.0=fresh, 0.0=expired")
    messages_since_created: int = Field(default=0, description="New messages since this summary was created")

    # Hierarchy
    aggregated_from: List[str] = Field(default_factory=list, description="Child summary IDs used")
    is_aggregated: bool = Field(default=False, description="True if already used in higher-level summary")

    # Metadata
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    class Config:
        json_schema_extra = {
            "example": {
                "summary_id": "550e8400-e29b-41d4-a716-446655440008",
                "agent_id": "71f6f657-6800-0892-875f-f26e8c213756",
                "group_id": "telegram_group_123",
                "level": "micro",
                "message_start": 0,
                "message_end": 49,
                "message_count": 50,
                "time_start": "2025-01-27T10:00:00Z",
                "time_end": "2025-01-27T14:30:00Z",
                "duration_hours": 4.5,
                "activity_rate": 11.1,
                "summary": "Group discussed Base chain grant applications and audit processes",
                "topics": ["grants", "audits", "base chain"],
                "highlights": ["New grant round announced", "Security audit discussed"],
                "active_users": ["alice_web3", "bob_builder"],
                "decay_score": 1.0,
                "messages_since_created": 0,
                "aggregated_from": [],
                "is_aggregated": False,
                "created_at": "2025-01-27T14:30:00Z"
            }
        }


class UserInGroupProfile(BaseModel):
    """
    User's profile within a specific group context.

    How a user behaves in a particular group:
    - Their role and participation style
    - Expertise level within this group
    - Relationships with other members
    - Topics they engage with

    Different from global UserProfile - this is group-specific behavior.
    """
    profile_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = Field(..., description="Agent this profile belongs to")

    # Identifiers
    group_id: str = Field(..., description="Group identifier")
    universal_user_id: str = Field(..., description="platform:specific_id")
    username: Optional[str] = Field(None, description="Username/handle")

    # LLM-extracted content
    summary: str = Field(..., description="How this user participates in this group")
    role_in_group: str = Field(..., description="Member role: regular, expert, helper, lurker, leader")

    # Participation patterns
    participation_level: str = Field(default="active", description="How active they are")
    expertise_in_group: str = Field(default="intermediate", description="Expertise level within this group")
    interaction_style: str = Field(default="collaborative", description="How they interact")

    # Topics and interests (group-specific)
    topics_engaged: List[str] = Field(default_factory=list, description="Topics they discuss")
    contributions: List[str] = Field(default_factory=list, description="Types of contributions")

    # Relationships
    frequently_interacts_with: List[str] = Field(default_factory=list, description="Users they engage with")
    mentioned_by: List[str] = Field(default_factory=list, description="Users who mention them")

    # Metadata
    total_messages_in_group: int = Field(default=0, description="Messages by this user in group")
    last_message_timestamp: Optional[str] = Field(None, description="Last message ISO timestamp")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    class Config:
        json_schema_extra = {
            "example": {
                "profile_id": "550e8400-e29b-41d4-a716-446655440007",
                "agent_id": "71f6f657-6800-0892-875f-f26e8c213756",
                "group_id": "telegram_group_123",
                "universal_user_id": "telegram:123456789",
                "username": "@alice_web3",
                "summary": "Active DeFi expert who shares yield strategies and helps newcomers",
                "role_in_group": "expert_helper",
                "participation_level": "high",
                "expertise_in_group": "advanced",
                "interaction_style": "collaborative",
                "topics_engaged": ["yield farming", "liquid staking", "base chain"],
                "contributions": ["explains concepts", "shares opportunities", "answers questions"],
                "frequently_interacts_with": ["telegram:987654321", "telegram:111222333"],
                "mentioned_by": ["telegram:444555666"],
                "total_messages_in_group": 45,
                "last_message_timestamp": "2025-01-26T10:00:00Z",
                "created_at": "2025-01-15T10:00:00Z",
                "updated_at": "2025-01-26T10:00:00Z"
            }
        }


class FactType(str, Enum):
    """Type of user fact"""
    PREFERENCE = "preference"
    EXPERTISE = "expertise"
    BEHAVIOR = "behavior"
    PERSONAL = "personal"
    INTEREST = "interest"
    COMMUNICATION = "communication"


class UserFact(BaseModel):
    """
    A single fact about a user, tracked with evidence.

    Facts are extracted from user messages in groups and DMs.
    They don't decay over time but have confidence scores.
    """
    fact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    user_id: str

    content: str
    fact_type: FactType
    keywords: List[str] = []

    evidence_count: int = 1
    confidence: float = 0.5
    sources: List[str] = []
    source_types: List[str] = []

    first_seen: str
    last_confirmed: str

    is_consolidated: bool = False
    consolidated_from: List[str] = []
    contradicted_by: List[str] = []

    class Config:
        json_schema_extra = {
            "example": {
                "fact_id": "abc123",
                "agent_id": "jessexbt",
                "user_id": "telegram:123456",
                "content": "Expert in DeFi yield farming",
                "fact_type": "expertise",
                "keywords": ["defi", "yield", "farming"],
                "evidence_count": 8,
                "confidence": 0.85,
                "sources": ["group_A", "group_B", "dm_123"],
                "source_types": ["group", "group", "dm"],
                "first_seen": "2025-01-15T10:00:00Z",
                "last_confirmed": "2025-01-28T14:00:00Z",
                "is_consolidated": False
            }
        }
