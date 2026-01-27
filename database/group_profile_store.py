"""
Group Profile Store - LanceDB storage for group profiles

Stores:
- GroupProfile: Culture and context of a group
- UserInGroupProfile: User's behavior within a specific group
"""
import json
import os
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

import lancedb
import pyarrow as pa

from models.group_memory import GroupProfile, UserInGroupProfile, GroupTone
from utils.embedding import EmbeddingModel
import config


# ============================================================
# Structured Output Schemas for OpenRouter
# ============================================================

GROUP_PROFILE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "group_profile",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "summary": {"type": "string"},
                "group_purpose": {"type": "string"},
                "main_topics": {"type": "array", "items": {"type": "string"}},
                "tone": {"type": "string", "enum": ["formal", "casual", "technical", "friendly", "professional", "educational", "meme_heavy", "other"]},
                "expertise_level": {"type": "string", "enum": ["beginner", "intermediate", "advanced", "expert", "mixed"]},
                "activity_level": {"type": "string", "enum": ["low", "moderate", "high", "very_high"]},
                "member_count_estimate": {"type": "integer"},
                "communication_norms": {"type": "array", "items": {"type": "string"}},
                "preferred_content_types": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["reasoning", "summary", "group_purpose", "main_topics", "tone", "expertise_level", "activity_level", "member_count_estimate", "communication_norms", "preferred_content_types"],
            "additionalProperties": False
        }
    }
}

USER_IN_GROUP_PROFILE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "user_in_group_profile",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "summary": {"type": "string"},
                "role_in_group": {"type": "string", "enum": ["expert", "helper", "learner", "leader", "lurker", "regular", "other"]},
                "participation_level": {"type": "string", "enum": ["very_low", "low", "moderate", "high", "very_high"]},
                "expertise_in_group": {"type": "string", "enum": ["beginner", "intermediate", "advanced", "expert"]},
                "interaction_style": {"type": "string", "enum": ["collaborative", "competitive", "supportive", "questioning", "sharing", "other"]},
                "topics_engaged": {"type": "array", "items": {"type": "string"}},
                "contributions": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["reasoning", "summary", "role_in_group", "participation_level", "expertise_in_group", "interaction_style", "topics_engaged", "contributions"],
            "additionalProperties": False
        }
    }
}


class GroupProfileStore:
    """
    LanceDB storage for group profiles and user-in-group profiles.

    Uses same DB as memory entries but separate tables.
    """

    def __init__(
        self,
        db_path: str = None,
        embedding_model: EmbeddingModel = None,
        agent_id: Optional[str] = None
    ):
        self.db_path = db_path or config.LANCEDB_PATH
        self.embedding_model = embedding_model or EmbeddingModel()
        self.agent_id = agent_id

        # LLM client for group profile generation
        from utils.llm_client import LLMClient
        self.llm_client = LLMClient(use_streaming=False)

        # Detect cloud storage
        self._is_cloud_storage = self.db_path.startswith(("gs://", "s3://", "az://"))

        # Connect to database
        if self._is_cloud_storage:
            self.db = lancedb.connect(self.db_path)
        else:
            os.makedirs(self.db_path, exist_ok=True)
            self.db = lancedb.connect(self.db_path)

        self._init_tables()

    def _init_tables(self):
        """Initialize group_profiles and user_in_group_profiles tables."""

        # GROUP_PROFILES TABLE
        group_schema = pa.schema([
            pa.field("profile_id", pa.string()),
            pa.field("agent_id", pa.string()),
            pa.field("group_id", pa.string()),
            pa.field("platform", pa.string()),
            pa.field("profile_data", pa.string()),  # Full GroupProfile as JSON
            pa.field("summary", pa.string()),
            pa.field("main_topics", pa.list_(pa.string())),
            pa.field("tone", pa.string()),
            pa.field("expertise_level", pa.string()),
            pa.field("total_messages", pa.int64()),
            pa.field("created_at", pa.string()),
            pa.field("updated_at", pa.string()),
            pa.field("profile_version", pa.int64()),
            # Semantic search
            pa.field("summary_vector", pa.list_(pa.float32(), list_size=self.embedding_model.dimension)),
        ])

        group_table_name = "group_profiles"
        if group_table_name not in self.db.table_names():
            self.group_profiles_table = self.db.create_table(group_table_name, schema=group_schema)
            print(f"[GroupProfileStore] Created {group_table_name} table")
        else:
            self.group_profiles_table = self.db.open_table(group_table_name)
            print(f"[GroupProfileStore] Opened {group_table_name}")

        # USER_IN_GROUP_PROFILES TABLE
        user_in_group_schema = pa.schema([
            pa.field("profile_id", pa.string()),
            pa.field("agent_id", pa.string()),
            pa.field("group_id", pa.string()),
            pa.field("universal_user_id", pa.string()),
            pa.field("username", pa.string()),
            pa.field("profile_data", pa.string()),  # Full UserInGroupProfile as JSON
            pa.field("role_in_group", pa.string()),
            pa.field("participation_level", pa.string()),
            pa.field("expertise_in_group", pa.string()),
            pa.field("topics_engaged", pa.list_(pa.string())),
            pa.field("total_messages", pa.int64()),
            pa.field("created_at", pa.string()),
            pa.field("updated_at", pa.string()),
            # Semantic search
            pa.field("summary_vector", pa.list_(pa.float32(), list_size=self.embedding_model.dimension)),
        ])

        user_in_group_table_name = "user_in_group_profiles"
        if user_in_group_table_name not in self.db.table_names():
            self.user_in_group_table = self.db.create_table(user_in_group_table_name, schema=user_in_group_schema)
            print(f"[GroupProfileStore] Created {user_in_group_table_name} table")
        else:
            self.user_in_group_table = self.db.open_table(user_in_group_table_name)
            print(f"[GroupProfileStore] Opened {user_in_group_table_name}")

    # ============================================================
    # Group Profile Operations
    # ============================================================

    def upsert_group_profile(self, profile: GroupProfile) -> GroupProfile:
        """Create or update group profile."""
        # Check if exists
        existing = self.get_group_profile(profile.group_id)

        # Generate embedding
        summary_vector = self.embedding_model.encode_single(profile.summary, is_query=False)

        now = datetime.now(timezone.utc).isoformat()

        if existing:
            # Delete old
            agent_filter = f"agent_id = '{self.agent_id}'" if self.agent_id else "TRUE"
            self.group_profiles_table.delete(f"group_id = '{profile.group_id}' AND {agent_filter}")
            profile.updated_at = now
            profile.profile_version += 1
        else:
            profile.created_at = now
            profile.updated_at = now

        data = {
            "profile_id": profile.profile_id,
            "agent_id": profile.agent_id,
            "group_id": profile.group_id,
            "platform": profile.platform,
            "profile_data": profile.model_dump_json(),
            "summary": profile.summary,
            "main_topics": profile.main_topics,
            "tone": profile.tone.value,
            "expertise_level": profile.expertise_level,
            "total_messages": profile.total_messages_processed,
            "created_at": profile.created_at,
            "updated_at": profile.updated_at,
            "profile_version": profile.profile_version,
            "summary_vector": summary_vector.tolist()
        }

        self.group_profiles_table.add([data])
        action = "Updated" if existing else "Created"
        print(f"[GroupProfileStore] {action} group profile for {profile.group_id}")
        return profile

    def get_group_profile(self, group_id: str) -> Optional[GroupProfile]:
        """Get group profile by group_id."""
        agent_filter = f"agent_id = '{self.agent_id}'" if self.agent_id else "TRUE"
        results = self.group_profiles_table.search().where(
            f"group_id = '{group_id}' AND {agent_filter}",
            prefilter=True
        ).to_list()

        if not results:
            return None
        return GroupProfile.model_validate_json(results[0]["profile_data"])

    # ============================================================
    # User In Group Profile Operations
    # ============================================================

    def upsert_user_in_group_profile(self, profile: UserInGroupProfile) -> UserInGroupProfile:
        """Create or update user-in-group profile."""
        # Check if exists
        existing = self.get_user_in_group_profile(profile.group_id, profile.universal_user_id)

        # Generate embedding
        summary_vector = self.embedding_model.encode_single(profile.summary, is_query=False)

        now = datetime.now(timezone.utc).isoformat()

        if existing:
            # Delete old
            agent_filter = f"agent_id = '{self.agent_id}'" if self.agent_id else "TRUE"
            self.user_in_group_table.delete(
                f"group_id = '{profile.group_id}' AND universal_user_id = '{profile.universal_user_id}' AND {agent_filter}"
            )
            profile.updated_at = now
        else:
            profile.created_at = now
            profile.updated_at = now

        data = {
            "profile_id": profile.profile_id,
            "agent_id": profile.agent_id,
            "group_id": profile.group_id,
            "universal_user_id": profile.universal_user_id,
            "username": profile.username or "",
            "profile_data": profile.model_dump_json(),
            "role_in_group": profile.role_in_group,
            "participation_level": profile.participation_level,
            "expertise_in_group": profile.expertise_in_group,
            "topics_engaged": profile.topics_engaged,
            "total_messages": profile.total_messages_in_group,
            "created_at": profile.created_at,
            "updated_at": profile.updated_at,
            "summary_vector": summary_vector.tolist()
        }

        self.user_in_group_table.add([data])
        action = "Updated" if existing else "Created"
        print(f"[GroupProfileStore] {action} user-in-group profile for {profile.universal_user_id} in {profile.group_id}")
        return profile

    def get_user_in_group_profile(self, group_id: str, universal_user_id: str) -> Optional[UserInGroupProfile]:
        """Get user-in-group profile."""
        agent_filter = f"agent_id = '{self.agent_id}'" if self.agent_id else "TRUE"
        results = self.user_in_group_table.search().where(
            f"group_id = '{group_id}' AND universal_user_id = '{universal_user_id}' AND {agent_filter}",
            prefilter=True
        ).to_list()

        if not results:
            return None
        return UserInGroupProfile.model_validate_json(results[0]["profile_data"])

    def get_users_in_group(self, group_id: str) -> List[UserInGroupProfile]:
        """Get all user profiles for a group."""
        agent_filter = f"agent_id = '{self.agent_id}'" if self.agent_id else "TRUE"
        results = self.user_in_group_table.search().where(
            f"group_id = '{group_id}' AND {agent_filter}",
            prefilter=True
        ).to_list()

        return [UserInGroupProfile.model_validate_json(r["profile_data"]) for r in results]

    def get_group_context(self, group_id: str) -> Dict[str, Any]:
        """
        Get comprehensive group context including profile and active users.

        This is the equivalent of Zep's "group context" but richer:
        - Group profile (culture, tone, topics)
        - Top active users in the group
        - User roles and participation levels

        Args:
            group_id: Group identifier

        Returns:
            Dict with:
                - group_summary: str
                - tone: str
                - topics: list
                - active_users: list of dicts with username, role, summary, participation_level
        """
        # Get group profile
        group_profile = self.get_group_profile(group_id)

        if not group_profile:
            return {
                "group_summary": f"Group {group_id}",
                "tone": "casual",
                "topics": [],
                "active_users": []
            }

        # Get users in group, sorted by participation
        users_in_group = self.get_users_in_group(group_id)

        # Sort by total_messages (descending) to get most active users
        users_in_group.sort(key=lambda u: u.total_messages_in_group, reverse=True)

        # Take top 5-10 active users
        top_active = users_in_group[:10]

        # Format active users
        active_users = []
        for user_profile in top_active:
            active_users.append({
                "username": user_profile.username or user_profile.universal_user_id,
                "universal_user_id": user_profile.universal_user_id,
                "role": user_profile.role_in_group,
                "summary": user_profile.summary,
                "participation_level": user_profile.participation_level,
                "expertise_in_group": user_profile.expertise_in_group,
                "total_messages": user_profile.total_messages_in_group
            })

        return {
            "group_summary": group_profile.summary,
            "group_purpose": group_profile.group_purpose,
            "tone": group_profile.tone.value if hasattr(group_profile.tone, 'value') else group_profile.tone,
            "topics": group_profile.main_topics,
            "expertise_level": group_profile.expertise_level,
            "activity_level": group_profile.activity_level,
            "communication_norms": group_profile.communication_norms,
            "preferred_content_types": group_profile.preferred_content_types,
            "member_count_estimate": group_profile.member_count_estimate,
            "active_users": active_users
        }

    # ============================================================
    # LLM-Based Profile Generation
    # ============================================================

    def generate_group_profile_from_messages(
        self,
        agent_id: str,
        group_id: str,
        platform: str,
        group_name: Optional[str],
        messages: List[str],
        usernames: Optional[List[str]] = None
    ) -> GroupProfile:
        """
        Generate group profile using LLM from group messages.

        Uses llama-3.1-8b-instruct to extract:
        - Summary and purpose
        - Main topics
        - Communication tone
        - Expertise level
        - Activity level
        """
        # Combine messages
        full_text = "\n".join(messages)

        # Improved prompt with example and strict constraints
        prompt = f"""Analyze this group conversation and extract a profile.

[Context]
Platform: {platform}
Group Name: {group_name or 'Unknown'}
Total Messages: {len(messages)}

[Conversation Sample]
{full_text[:4000]}

[Instructions]
- summary: 1-2 sentences describing what this group is about. Be specific, not generic.
- group_purpose: The primary reason this group exists.
- main_topics: 3-5 specific topics actually discussed in the messages (e.g. "Uniswap V3 liquidity", not just "DeFi").
- tone: Pick ONE from: formal, casual, technical, friendly, professional, educational, meme_heavy, other.
- expertise_level: Pick ONE from: beginner, intermediate, advanced, expert, mixed.
- activity_level: Based on {len(messages)} messages - low (<50), moderate (50-200), high (200-500), very_high (>500).
- member_count_estimate: Estimate unique participants from the conversation.
- communication_norms: 2-4 observed norms (e.g. "members share links with context", "questions get answered quickly").
- preferred_content_types: 3-5 content types (e.g. "code snippets", "news links", "memes", "technical discussions").

[Example Output]
{{"summary": "A Telegram group focused on Base L2 development, where builders share code and discuss smart contract patterns.", "group_purpose": "Technical collaboration for Base L2 developers", "main_topics": ["Solidity gas optimization", "Base bridge integration", "ERC-4337 account abstraction"], "tone": "technical", "expertise_level": "advanced", "activity_level": "high", "member_count_estimate": 45, "communication_norms": ["Members share code with explanations", "New questions get responses within hours"], "preferred_content_types": ["code snippets", "GitHub links", "technical discussions", "deployment guides"]}}"""

        try:
            messages_list = [
                {"role": "system", "content": "You are a group analysis expert. Analyze the conversation and extract the group profile."},
                {"role": "user", "content": prompt}
            ]

            # Use structured outputs for reliable JSON
            response = self.llm_client.chat_completion(
                messages_list,
                temperature=0.1,
                response_format=GROUP_PROFILE_SCHEMA
            )

            # With structured outputs, response is valid JSON directly
            try:
                data = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                # Fallback to extract_json if direct parse fails
                data = self.llm_client.extract_json(response)

            # Normalize: LLM sometimes returns a list instead of a dict
            if isinstance(data, list):
                data = data[0] if data else {}
            if not isinstance(data, dict):
                print(f"[GroupProfileStore] Unexpected LLM response type: {type(data)}, falling back to empty dict")
                data = {}

            # Map tone to enum
            tone_map = {
                "formal": GroupTone.FORMAL,
                "casual": GroupTone.CASUAL,
                "technical": GroupTone.TECHNICAL,
                "friendly": GroupTone.FRIENDLY,
                "professional": GroupTone.PROFESSIONAL,
                "educational": GroupTone.EDUCATIONAL,
                "meme_heavy": GroupTone.MEME_HEAVY,
            }
            tone = tone_map.get(data.get("tone", "casual").lower(), GroupTone.CASUAL)

            # Create profile
            existing = self.get_group_profile(group_id)

            profile = GroupProfile(
                agent_id=agent_id,
                group_id=group_id,
                group_name=group_name,
                platform=platform,
                summary=data.get("summary", f"Group with {len(messages)} messages"),
                main_topics=data.get("main_topics", []),
                group_purpose=data.get("group_purpose", "General discussion"),
                tone=tone,
                communication_norms=data.get("communication_norms", []),
                expertise_level=data.get("expertise_level", "intermediate"),
                activity_level=data.get("activity_level", "active"),
                member_count_estimate=data.get("member_count_estimate"),
                preferred_content_types=data.get("preferred_content_types", []),
                total_messages_processed=existing.total_messages_processed + len(messages) if existing else len(messages),
                profile_version=existing.profile_version + 1 if existing else 1
            )

            return self.upsert_group_profile(profile)

        except Exception as e:
            print(f"[GroupProfileStore] LLM extraction failed: {e}")
            # Fallback
            existing = self.get_group_profile(group_id)
            profile = GroupProfile(
                agent_id=agent_id,
                group_id=group_id,
                group_name=group_name,
                platform=platform,
                summary=f"Group with {len(messages)} messages",
                main_topics=[],
                group_purpose="Discussion",
                total_messages_processed=existing.total_messages_processed + len(messages) if existing else len(messages)
            )
            return self.upsert_group_profile(profile)

    def generate_user_in_group_profile(
        self,
        agent_id: str,
        group_id: str,
        universal_user_id: str,
        username: Optional[str],
        user_messages: List[str],
        context_messages: Optional[List[str]] = None
    ) -> UserInGroupProfile:
        """
        Generate user-in-group profile using LLM.

        Analyzes how a specific user behaves within a group context.
        """
        user_text = "\n".join(user_messages)

        prompt = f"""Analyze how this user participates in the group based on their actual messages.

[Context]
Group: {group_id}
Username: {username or 'Unknown'}
User Messages: {len(user_messages)}

[User's Messages]
{user_text[:3000]}

[Instructions]
- summary: 1 sentence describing how this user participates, based on evidence from their messages. Be specific.
- role_in_group: Pick ONE from: expert, helper, learner, leader, lurker, regular, other. Base this on what they actually do in their messages.
- participation_level: Based on {len(user_messages)} messages - very_low (<5), low (5-15), moderate (15-40), high (40-80), very_high (>80).
- expertise_in_group: Pick ONE from: beginner, intermediate, advanced, expert. Judge from the complexity and accuracy of their messages.
- interaction_style: Pick ONE from: collaborative, competitive, supportive, questioning, sharing, other.
- topics_engaged: 3-5 specific topics they actually discuss in their messages (use concrete terms from their messages).
- contributions: 2-4 types of contributions (e.g. "answers technical questions about Solidity", "shares project updates", "asks clarifying questions").

[Example Output]
{{"summary": "Active contributor who frequently answers questions about smart contract development and shares code examples.", "role_in_group": "helper", "participation_level": "high", "expertise_in_group": "advanced", "interaction_style": "supportive", "topics_engaged": ["Solidity inheritance patterns", "gas optimization", "OpenZeppelin contracts"], "contributions": ["answers technical questions with code examples", "reviews others' code snippets", "shares relevant documentation links"]}}"""

        try:
            messages_list = [
                {"role": "system", "content": "You are a user behavior analyst. Analyze the user's messages and extract their profile."},
                {"role": "user", "content": prompt}
            ]

            # Use structured outputs for reliable JSON
            response = self.llm_client.chat_completion(
                messages_list,
                temperature=0.1,
                response_format=USER_IN_GROUP_PROFILE_SCHEMA
            )

            # With structured outputs, response is valid JSON directly
            try:
                data = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                # Fallback to extract_json if direct parse fails
                data = self.llm_client.extract_json(response)

            # Normalize: LLM sometimes returns a list instead of a dict
            if isinstance(data, list):
                data = data[0] if data else {}
            if not isinstance(data, dict):
                print(f"[GroupProfileStore] Unexpected LLM response type: {type(data)}, falling back to empty dict")
                data = {}

            existing = self.get_user_in_group_profile(group_id, universal_user_id)

            profile = UserInGroupProfile(
                agent_id=agent_id,
                group_id=group_id,
                universal_user_id=universal_user_id,
                username=username,
                summary=data.get("summary", f"User with {len(user_messages)} messages"),
                role_in_group=data.get("role_in_group", "regular"),
                participation_level=data.get("participation_level", "moderate"),
                expertise_in_group=data.get("expertise_in_group", "intermediate"),
                interaction_style=data.get("interaction_style", "collaborative"),
                topics_engaged=data.get("topics_engaged", []),
                contributions=data.get("contributions", []),
                total_messages_in_group=existing.total_messages_in_group + len(user_messages) if existing else len(user_messages)
            )

            return self.upsert_user_in_group_profile(profile)

        except Exception as e:
            print(f"[GroupProfileStore] User-in-group LLM extraction failed: {e}")
            existing = self.get_user_in_group_profile(group_id, universal_user_id)
            profile = UserInGroupProfile(
                agent_id=agent_id,
                group_id=group_id,
                universal_user_id=universal_user_id,
                username=username,
                summary=f"User with {len(user_messages)} messages in group",
                role_in_group="regular",
                total_messages_in_group=existing.total_messages_in_group + len(user_messages) if existing else len(user_messages)
            )
            return self.upsert_user_in_group_profile(profile)
