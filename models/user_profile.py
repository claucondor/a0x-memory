"""
User Profile Model - Complete user information extracted from conversation history

Based on research from SimpleMem, GetProfile, and Mem0 papers.
Profile grows incrementally as more conversations are processed.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid


class TraitScore(BaseModel):
    """A trait with confidence score (0-1)"""
    value: str = Field(..., description="Trait value (e.g., 'advanced', 'technical')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")


class Interest(BaseModel):
    """User interest/keyword with relevance score"""
    keyword: str = Field(..., description="Interest keyword")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score 0-1")


class EntityInfo(BaseModel):
    """Named entity mentioned in conversations"""
    type: str = Field(..., description="Entity type: person, organization, location, etc.")
    name: str = Field(..., description="Entity name")
    context: Optional[str] = Field(None, description="Context where entity was mentioned")


class UserProfile(BaseModel):
    """
    Complete user profile extracted from conversation history

    Based on SimpleMem paper findings:
    - Profile grows incrementally (τcluster=0.85 for clustering)
    - Traits stabilize after ~10 messages
    - Interests evolve continuously

    Fields based on GetProfile research:
    - Summary with confidence scores
    - Cross-platform linking support
    - Semantic search capability
    """
    profile_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Primary identifiers
    agent_id: str = Field(..., description="Agent this profile belongs to")
    universal_user_id: str = Field(..., description="platform:specific_id (e.g., 'telegram:123456789')")
    platform_type: str = Field(..., description="Platform: telegram, twitter, xmtp, farcaster")
    platform_specific_id: str = Field(..., description="Platform-specific user ID")

    # Username/handle for direct lookup
    username: Optional[str] = Field(None, description="Username/handle (e.g., '@alice')")

    # Cross-platform linking
    primary_profile_id: Optional[str] = Field(None, description="Primary profile UUID if this is linked")
    linked_accounts: List[Dict[str, str]] = Field(default_factory=list, description="Other platform accounts")

    # Complete profile data
    summary: str = Field(..., description="User summary based on conversation history")
    traits: Dict[str, TraitScore] = Field(default_factory=dict, description="User traits with confidence")
    interests: List[Interest] = Field(default_factory=list, description="User interests/topics")
    expertise_level: Optional[TraitScore] = Field(None, description="Expertise level (beginner/intermediate/advanced/expert)")
    communication_style: Optional[TraitScore] = Field(None, description="Communication style")

    # Entities mentioned
    entities: List[EntityInfo] = Field(default_factory=list, description="Named entities mentioned")

    # Wallet/Base Name (for Web3 context)
    wallet_address: Optional[str] = Field(None, description="Wallet address if shared")
    basename: Optional[str] = Field(None, description="Base Name (.base.eth) if known")

    # Metadata
    profile_version: int = Field(1, description="Profile version number")
    total_messages_processed: int = Field(0, description="Total messages used to build profile")
    last_message_timestamp: Optional[str] = Field(None, description="Last message ISO timestamp")
    created_at: str = Field(..., description="Profile creation ISO timestamp")
    updated_at: str = Field(..., description="Last update ISO timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "profile_id": "550e8400-e29b-41d4-a716-446655440000",
                "agent_id": "71f6f657-6800-0892-875f-f26e8c213756",
                "universal_user_id": "telegram:123456789",
                "platform_type": "telegram",
                "platform_specific_id": "123456789",
                "username": "@alice_web3",
                "primary_profile_id": None,
                "linked_accounts": [],
                "summary": "Alice is a DeFi developer interested in yield farming and liquid staking derivatives",
                "traits": {
                    "engagement_frequency": {"value": "daily", "confidence": 0.9},
                    "helpfulness": {"value": "high", "confidence": 0.85}
                },
                "interests": [
                    {"keyword": "defi", "score": 0.95},
                    {"keyword": "yield farming", "score": 0.85},
                    {"keyword": "ethereum", "score": 0.8}
                ],
                "expertise_level": {"value": "advanced", "confidence": 0.8},
                "communication_style": {"value": "technical", "confidence": 0.75},
                "entities": [
                    {"type": "person", "name": "Vitalik Buterin", "context": "mentioned in scaling discussion"},
                    {"type": "organization", "name": "Ethereum Foundation", "context": "grant discussion"}
                ],
                "wallet_address": "0x1234567890abcdef1234567890abcdef12345678",
                "basename": "alice.base.eth",
                "profile_version": 1,
                "total_messages_processed": 25,
                "last_message_timestamp": "2025-01-23T14:30:00Z",
                "created_at": "2025-01-15T10:00:00Z",
                "updated_at": "2025-01-23T14:30:00Z"
            }
        }

    def merge_profile(self, other: 'UserProfile') -> 'UserProfile':
        """
        Merge another profile into this one (for cross-platform linking)

        Combines:
        - Messages processed
        - Interests (weighted average)
        - Entities (unique union)
        - Summary (concatenates and re-summaries would be done externally)
        """
        merged_interests = self._merge_interests(self.interests, other.interests)
        merged_entities = self._merge_entities(self.entities, other.entities)

        return UserProfile(
            profile_id=self.profile_id,
            agent_id=self.agent_id,
            universal_user_id=self.universal_user_id,
            platform_type=self.platform_type,
            platform_specific_id=self.platform_specific_id,
            username=self.username,
            primary_profile_id=self.primary_profile_id,
            linked_accounts=self.linked_accounts,
            summary=self.summary,  # Would be regenerated by LLM
            traits=self._merge_traits(self.traits, other.traits),
            interests=merged_interests,
            expertise_level=self._merge_trait_score(self.expertise_level, other.expertise_level),
            communication_style=self._merge_trait_score(self.communication_style, other.communication_style),
            entities=merged_entities,
            wallet_address=self.wallet_address or other.wallet_address,
            basename=self.basename or other.basename,
            profile_version=self.profile_version + 1,
            total_messages_processed=self.total_messages_processed + other.total_messages_processed,
            last_message_timestamp=other.last_message_timestamp or self.last_message_timestamp,
            created_at=self.created_at,
            updated_at=other.updated_at
        )

    def _merge_interests(self, a: List[Interest], b: List[Interest]) -> List[Interest]:
        """Merge interest lists by keyword, averaging scores"""
        interest_map = {}

        for interest in a + b:
            if interest.keyword in interest_map:
                # Weighted average based on message count contribution
                existing = interest_map[interest.keyword]
                interest_map[interest.keyword] = Interest(
                    keyword=interest.keyword,
                    score=(existing.score + interest.score) / 2
                )
            else:
                interest_map[interest.keyword] = interest

        return sorted(interest_map.values(), key=lambda x: x.score, reverse=True)

    def _merge_entities(self, a: List[EntityInfo], b: List[EntityInfo]) -> List[EntityInfo]:
        """Merge entity lists, keeping unique entities"""
        seen = set()
        merged = []

        for entity in a + b:
            key = (entity.type, entity.name)
            if key not in seen:
                seen.add(key)
                merged.append(entity)

        return merged

    def _merge_traits(self, a: Dict[str, TraitScore], b: Dict[str, TraitScore]) -> Dict[str, TraitScore]:
        """Merge trait dictionaries"""
        merged = {}

        for key in set(list(a.keys()) + list(b.keys())):
            if key in a and key in b:
                # Average confidence
                merged[key] = TraitScore(
                    value=a[key].value,  # Keep value from primary
                    confidence=(a[key].confidence + b[key].confidence) / 2
                )
            elif key in a:
                merged[key] = a[key]
            else:
                merged[key] = b[key]

        return merged

    def _merge_trait_score(self, a: Optional[TraitScore], b: Optional[TraitScore]) -> Optional[TraitScore]:
        """Merge two TraitScore objects"""
        if not a:
            return b
        if not b:
            return a
        return TraitScore(
            value=a.value,
            confidence=(a.confidence + b.confidence) / 2
        )
