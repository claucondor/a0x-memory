"""
User Profile Store - LanceDB-based storage for user profiles

Same LanceDB file, separate table from memory entries.
Multi-tenant support via agent_id filtering.

Features:
- Scalar indices for fast username/wallet lookups
- Semantic search with 4096D embeddings (OpenRouter Qwen)
- Cross-platform linking via primary_profile_id
- Version tracking for profile evolution
"""
import json
import os
import uuid
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone

import lancedb
import pyarrow as pa

from models.user_profile import UserProfile, TraitScore, Interest, EntityInfo
from utils.embedding import EmbeddingModel
import config


class UserProfileStore:
    """
    LanceDB storage for user profiles

    Uses same DB as memory entries but separate table.
    Follows VectorStore pattern for consistency.
    """

    def __init__(
        self,
        db_path: str = None,
        embedding_model: EmbeddingModel = None,
        table_name: str = "user_profiles",
        storage_options: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None
    ):
        self.db_path = db_path or config.LANCEDB_PATH
        self.embedding_model = embedding_model or EmbeddingModel()
        self.table_name = table_name
        self.table = None
        self._indices_created = False

        # Multi-tenant context
        self.agent_id = agent_id

        # LLM client for profile extraction (summary + interests)
        from utils.llm_client import LLMClient
        self.profile_llm = LLMClient(use_streaming=False)

        # Detect cloud storage
        self._is_cloud_storage = self.db_path.startswith(("gs://", "s3://", "az://"))

        # Connect to database
        if self._is_cloud_storage:
            self.db = lancedb.connect(self.db_path, storage_options=storage_options)
        else:
            os.makedirs(self.db_path, exist_ok=True)
            self.db = lancedb.connect(self.db_path)

        self._init_table()

    def _init_table(self):
        """Initialize user_profiles table with 4096D embeddings."""
        schema = pa.schema([
            # Primary identifiers
            pa.field("id", pa.string()),
            pa.field("agent_id", pa.string()),

            # Universal ID pattern
            pa.field("universal_user_id", pa.string()),    # "telegram:123456789"
            pa.field("platform_type", pa.string()),        # "telegram" | "twitter" | "xmtp"
            pa.field("platform_specific_id", pa.string()), # "123456789"

            # Username for direct search
            pa.field("username", pa.string()),             # "@alice"

            # Cross-platform linking
            pa.field("primary_profile_id", pa.string()),   # UUID of primary profile if linked
            pa.field("linked_accounts", pa.string()),      # JSON array of connected accounts

            # Complete profile in JSON
            pa.field("profile_data", pa.string()),         # Full UserProfile as JSON
            pa.field("profile_version", pa.int64()),
            pa.field("total_messages", pa.int64()),
            pa.field("created_at", pa.string()),
            pa.field("updated_at", pa.string()),

            # Semantic search (dimension from config: 384D for a0x, 4096D for OpenRouter)
            pa.field("summary_vector", pa.list_(pa.float32(), list_size=self.embedding_model.dimension)),

            # FTS searchable text
            pa.field("searchable_text", pa.string()),      # For BM25 search
        ])

        if self.table_name not in self.db.table_names():
            self.table = self.db.create_table(self.table_name, schema=schema)
            print(f"[UserProfileStore] Created new table: {self.table_name}")
        else:
            self.table = self.db.open_table(self.table_name)
            print(f"[UserProfileStore] Opened existing table: {self.table_name}")

    def _create_indices(self):
        """Create scalar indices for fast lookups."""
        if self._indices_created:
            return

        try:
            # Index on username for fast lookup
            self.table.create_scalar_index("username", replace=True)
            # Index on universal_user_id for exact match
            self.table.create_scalar_index("universal_user_id", replace=True)
            # Index on agent_id for tenant filtering
            self.table.create_scalar_index("agent_id", replace=True)

            self._indices_created = True
            print("[UserProfileStore] Scalar indices created")
        except Exception as e:
            print(f"[UserProfileStore] Warning: Index creation skipped: {e}")

    def _init_fts_index(self):
        """Initialize Full-Text Search index."""
        try:
            if self._is_cloud_storage:
                self.table.create_fts_index(
                    "searchable_text",
                    use_tantivy=False,
                    replace=True
                )
            else:
                self.table.create_fts_index(
                    "searchable_text",
                    use_tantivy=True,
                    tokenizer_name="en_stem",
                    replace=True
                )
            print("[UserProfileStore] FTS index created")
        except Exception as e:
            print(f"[UserProfileStore] Warning: FTS index creation skipped: {e}")

    def _profile_to_dict(self, profile: UserProfile) -> dict:
        """Convert UserProfile to dict for insertion."""
        # Generate embedding for summary
        summary_vector = self.embedding_model.encode_single(profile.summary, is_query=True)

        # Create searchable text for FTS
        searchable_parts = [
            profile.username or "",
            profile.platform_type,
            profile.summary,
            " ".join([i.keyword for i in profile.interests]),
            profile.wallet_address or "",
            profile.basename or ""
        ]
        searchable_text = " ".join([p for p in searchable_parts if p])

        return {
            "id": profile.profile_id,
            "agent_id": profile.agent_id,
            "universal_user_id": profile.universal_user_id,
            "platform_type": profile.platform_type,
            "platform_specific_id": profile.platform_specific_id,
            "username": profile.username or "",
            "primary_profile_id": profile.primary_profile_id or "",
            "linked_accounts": json.dumps(profile.linked_accounts),
            "profile_data": profile.model_dump_json(),
            "profile_version": profile.profile_version,
            "total_messages": profile.total_messages_processed,
            "created_at": profile.created_at,
            "updated_at": profile.updated_at,
            "summary_vector": summary_vector.tolist(),
            "searchable_text": searchable_text
        }

    def _dict_to_profile(self, row: dict) -> UserProfile:
        """Convert LanceDB row to UserProfile."""
        return UserProfile.model_validate_json(row["profile_data"])

    def create_profile(self, profile: UserProfile) -> UserProfile:
        """Create a new user profile."""
        # Check if profile exists
        existing = self.get_profile_by_universal_id(profile.universal_user_id)
        if existing:
            raise ValueError(f"Profile already exists for {profile.universal_user_id}")

        data = self._profile_to_dict(profile)
        self.table.add([data])

        # Create indices after first insert
        if not self._indices_created:
            self._create_indices()
            self._init_fts_index()

        print(f"[UserProfileStore] Created profile for {profile.universal_user_id}")
        return profile

    def get_profile(self, profile_id: str) -> Optional[UserProfile]:
        """Get profile by ID."""
        agent_filter = f"agent_id = '{self.agent_id}'" if self.agent_id else "TRUE"
        results = self.table.search().where(
            f"id = '{profile_id}' AND {agent_filter}",
            prefilter=True
        ).to_list()

        if not results:
            return None
        return self._dict_to_profile(results[0])

    def get_profile_by_universal_id(self, universal_user_id: str) -> Optional[UserProfile]:
        """Get profile by universal_user_id (exact match)."""
        agent_filter = f"agent_id = '{self.agent_id}'" if self.agent_id else "TRUE"
        results = self.table.search().where(
            f"universal_user_id = '{universal_user_id}' AND {agent_filter}",
            prefilter=True
        ).to_list()

        if not results:
            return None
        return self._dict_to_profile(results[0])

    def get_profile_by_username(self, username: str) -> Optional[UserProfile]:
        """Get profile by username (exact match)."""
        agent_filter = f"agent_id = '{self.agent_id}'" if self.agent_id else "TRUE"
        results = self.table.search().where(
            f"username = '{username}' AND {agent_filter}",
            prefilter=True
        ).to_list()

        if not results:
            return None
        return self._dict_to_profile(results[0])

    def find_by_wallet(self, wallet_address: str) -> List[UserProfile]:
        """Find profiles by wallet address."""
        agent_filter = f"agent_id = '{self.agent_id}'" if self.agent_id else "TRUE"
        results = self.table.search().where(
            f"agent_id = '{self.agent_id if self.agent_id else ''}'",
            prefilter=True
        ).to_list()

        # Filter by wallet in profile_data JSON
        matching = []
        for r in results:
            profile = self._dict_to_profile(r)
            if profile.wallet_address and profile.wallet_address.lower() == wallet_address.lower():
                matching.append(profile)

        return matching

    def search_similar_users(self, query: str, top_k: int = 5) -> List[Tuple[UserProfile, float]]:
        """Find users similar to query using semantic search."""
        if self.table.count_rows() == 0:
            return []

        query_vector = self.embedding_model.encode_single(query, is_query=True)
        search = self.table.search(query_vector.tolist())

        # Apply agent filter
        if self.agent_id:
            search = search.where(f"agent_id = '{self.agent_id}'", prefilter=True)

        results = search.limit(top_k).to_list()

        similar = []
        for r in results:
            distance = r.get("_distance", 1.0)
            score = 1.0 - distance  # Convert to similarity
            profile = self._dict_to_profile(r)
            similar.append((profile, score))

        return similar

    def update_profile(self, profile: UserProfile) -> UserProfile:
        """Update existing profile."""
        # Check if exists
        existing = self.get_profile(profile.profile_id)
        if not existing:
            raise ValueError(f"Profile not found: {profile.profile_id}")

        # LanceDB doesn't support in-place updates, so we delete and re-insert
        agent_filter = f"agent_id = '{self.agent_id}'" if self.agent_id else "TRUE"
        self.table.delete(f"id = '{profile.profile_id}' AND {agent_filter}")

        # Update timestamp
        profile.updated_at = datetime.now(timezone.utc).isoformat()
        profile.profile_version += 1

        data = self._profile_to_dict(profile)
        self.table.add([data])

        print(f"[UserProfileStore] Updated profile {profile.profile_id} (v{profile.profile_version})")
        return profile

    def link_accounts(self, primary_profile_id: str, secondary_profile_id: str) -> UserProfile:
        """
        Link two profiles as the same person across platforms.

        Returns the merged primary profile.
        """
        primary = self.get_profile(primary_profile_id)
        secondary = self.get_profile(secondary_profile_id)

        if not primary or not secondary:
            raise ValueError("Both profiles must exist")

        # Merge profiles
        merged = primary.merge_profile(secondary)

        # Update secondary to point to primary
        secondary.primary_profile_id = primary_profile_id
        secondary.linked_accounts.append({
            "platform": secondary.platform_type,
            "id": secondary.platform_specific_id,
            "username": secondary.username
        })

        # Update both
        self.update_profile(merged)
        self.update_profile(secondary)

        print(f"[UserProfileStore] Linked {secondary.universal_user_id} -> {primary.universal_user_id}")
        return merged

    def get_linked_profiles(self, profile_id: str) -> List[UserProfile]:
        """Get all profiles linked to this one."""
        profiles = []
        primary = self.get_profile(profile_id)

        if not primary:
            return profiles

        # Add primary
        profiles.append(primary)

        # Find all profiles that have this as primary_profile_id
        agent_filter = f"agent_id = '{self.agent_id}'" if self.agent_id else "TRUE"
        results = self.table.search().where(
            f"primary_profile_id = '{profile_id}' AND {agent_filter}",
            prefilter=True
        ).to_list()

        for r in results:
            profiles.append(self._dict_to_profile(r))

        return profiles

    def list_profiles(self, platform_type: Optional[str] = None) -> List[UserProfile]:
        """List all profiles for agent, optionally filtered by platform."""
        agent_filter = f"agent_id = '{self.agent_id}'" if self.agent_id else None

        if agent_filter and platform_type:
            where_clause = f"{agent_filter} AND platform_type = '{platform_type}'"
            results = self.table.search().where(where_clause, prefilter=True).to_list()
        elif agent_filter:
            results = self.table.search().where(agent_filter, prefilter=True).to_list()
        elif platform_type:
            results = self.table.search().where(f"platform_type = '{platform_type}'", prefilter=True).to_list()
        else:
            results = self.table.to_arrow().to_pylist()

        return [self._dict_to_profile(r) for r in results]

    def count_profiles(self) -> int:
        """Count profiles for agent."""
        if not self.agent_id:
            return self.table.count_rows()

        results = self.table.search().where(
            f"agent_id = '{self.agent_id}'",
            prefilter=True
        ).to_list()
        return len(results)

    def delete_profile(self, profile_id: str) -> bool:
        """Delete a profile."""
        agent_filter = f"agent_id = '{self.agent_id}'" if self.agent_id else "TRUE"
        self.table.delete(f"id = '{profile_id}' AND {agent_filter}")
        print(f"[UserProfileStore] Deleted profile {profile_id}")
        return True

    def clear(self):
        """Clear all profiles for agent (or all if no agent set)."""
        if self.agent_id:
            self.table.delete(f"agent_id = '{self.agent_id}'")
            print(f"[UserProfileStore] Cleared profiles for agent {self.agent_id}")
        else:
            self.db.drop_table(self.table_name)
            self._indices_created = False
            self._init_table()
            print("[UserProfileStore] Cleared all profiles")

    def get_relevant_profiles(
        self,
        query: str,
        group_id: str = None,
        top_k: int = 5
    ) -> List[UserProfile]:
        """
        Search for profiles relevant to the query using semantic similarity.

        This gives us "related entities" - people relevant to the current query,
        similar to Zep's knowledge graph entities but simpler.

        Args:
            query: Search query
            group_id: Optional group ID to filter by (requires filtering after search)
            top_k: Number of profiles to return

        Returns:
            List of UserProfile objects most relevant to the query
        """
        if self.table.count_rows() == 0:
            return []

        # Generate query embedding
        query_vector = self.embedding_model.encode_single(query, is_query=True)

        # Vector search
        search = self.table.search(query_vector.tolist())

        # Apply agent filter
        if self.agent_id:
            search = search.where(f"agent_id = '{self.agent_id}'", prefilter=True)

        # Get results
        results = search.limit(top_k * 2).to_list()  # Get more for post-filtering

        if not results:
            return []

        # Convert to profiles
        profiles = [self._dict_to_profile(r) for r in results]

        # Optional: Post-filter by group_id
        # This requires checking which users have memories in the specified group
        if group_id:
            profiles = self._filter_profiles_by_group(profiles, group_id, top_k)

        # Return top_k
        return profiles[:top_k]

    def _filter_profiles_by_group(
        self,
        profiles: List[UserProfile],
        group_id: str,
        top_k: int
    ) -> List[UserProfile]:
        """
        Filter profiles to only include users who have memories in the specified group.

        For simplicity, this skips the actual query and returns all profiles.
        A full implementation would:
        1. Query the user_memories table for the group_id
        2. Extract unique user_ids
        3. Filter profiles by universal_user_id

        For now, we return profiles as-is since this is optional functionality.
        """
        # TODO: Implement group filtering by querying user_memories table
        # For now, return all profiles (no filtering)
        return profiles

    # ============================================================
    # Auto-Generation from Messages (a0x-models API)
    # ============================================================

    def generate_profile_from_messages(
        self,
        agent_id: str,
        platform_type: str,
        platform_specific_id: str,
        messages: List[str],
        username: Optional[str] = None,
        wallet_address: Optional[str] = None,
        basename: Optional[str] = None
    ) -> UserProfile:
        """
        Generate or update user profile from conversation messages.

        Single LLM call extracts everything: summary, interests, expertise,
        communication style, domains, and entities.

        Args:
            agent_id: Agent ID
            platform_type: Platform (telegram, twitter, xmtp, farcaster)
            platform_specific_id: Platform-specific user ID
            messages: List of user messages
            username: Optional username
            wallet_address: Optional wallet address
            basename: Optional Base Name

        Returns:
            Created or updated UserProfile
        """
        import json as json_module

        # Combine messages for analysis
        full_text = "\n".join(messages)
        universal_user_id = f"{platform_type}:{platform_specific_id}"

        # Single LLM call for full profile extraction (replaces 1 LLM + 3 /classify + 1 /ner)
        from utils.structured_schemas import USER_PROFILE_EXTRACTION_SCHEMA
        try:
            profile_prompt = f"""Analyze the following user messages and extract a complete profile.

MESSAGES:
{full_text[:6000]}

INSTRUCTIONS:
1. "summary": 1-2 sentence summary of who this user is based ONLY on evidence. Include name/handle, what they do, main focus areas. No repeated phrases.
2. "interests": 5-10 specific interest keywords (e.g. "Solidity development", "MEV strategies", "NFT art"). Score each 0.0-1.0 by evidence strength.
3. "expertise_level": One of "beginner", "intermediate", "advanced", "expert" based on demonstrated knowledge.
4. "communication_style": One of "formal", "casual", "technical", "conversational" based on how they write.
5. "domains": Which domains they engage with. For each, score 0.0-1.0 by relevance. Only include domains with score > 0.3.
6. "entities": Named entities mentioned (projects, protocols, tokens, tools, people, organizations). Include context for each.

Return ONLY the JSON."""
            extraction_response = self.profile_llm.chat_completion(
                messages=[{"role": "user", "content": profile_prompt}],
                temperature=0.1,
                response_format=USER_PROFILE_EXTRACTION_SCHEMA,
            )
            extracted = json_module.loads(extraction_response)

            summary = extracted["summary"]
            interests = []
            for item in extracted.get("interests", [])[:10]:
                interests.append(Interest(keyword=item["keyword"], score=item["score"]))

            expertise = TraitScore(
                value=extracted.get("expertise_level", "intermediate"),
                confidence=0.8
            )
            communication_style = TraitScore(
                value=extracted.get("communication_style", "conversational"),
                confidence=0.8
            )
            domains_data = {
                "labels": [d["name"] for d in extracted.get("domains", [])],
                "scores": [d["score"] for d in extracted.get("domains", [])]
            }
            entities = []
            for ent in extracted.get("entities", []):
                entities.append(EntityInfo(
                    type=ent.get("type", "other"),
                    name=ent.get("name", ""),
                    context=ent.get("context", "")
                ))

        except Exception as e:
            print(f"[UserProfileStore] Profile extraction failed: {e}")
            summary = f"User with {len(messages)} messages"
            interests = []
            expertise = TraitScore(value="intermediate", confidence=0.5)
            communication_style = TraitScore(value="conversational", confidence=0.5)
            domains_data = {"labels": [], "scores": []}
            entities = []

        # Build traits
        traits = {
            "engagement_level": TraitScore(value="active" if len(messages) >= 10 else "casual", confidence=0.7),
        }

        # Create or update profile
        existing = self.get_profile_by_universal_id(universal_user_id)
        now = datetime.now(timezone.utc).isoformat()

        if existing:
            # Update existing profile
            existing.summary = summary
            existing.traits = traits
            existing.interests = interests
            existing.expertise_level = expertise
            existing.communication_style = communication_style
            existing.entities = entities
            existing.total_messages_processed += len(messages)
            existing.last_message_timestamp = now
            existing.updated_at = now

            if wallet_address:
                existing.wallet_address = wallet_address
            if basename:
                existing.basename = basename
            if username:
                existing.username = username

            profile = self.update_profile(existing)
            print(f"[UserProfileStore] Updated profile for {universal_user_id} (+{len(messages)} messages)")
            return profile
        else:
            # Create new profile
            profile = UserProfile(
                agent_id=agent_id,
                universal_user_id=universal_user_id,
                platform_type=platform_type,
                platform_specific_id=platform_specific_id,
                username=username,
                summary=summary,
                traits=traits,
                interests=interests,
                expertise_level=expertise,
                communication_style=communication_style,
                entities=entities,
                wallet_address=wallet_address,
                basename=basename,
                total_messages_processed=len(messages),
                last_message_timestamp=now,
                created_at=now,
                updated_at=now
            )
            profile = self.create_profile(profile)
            print(f"[UserProfileStore] Created profile for {universal_user_id} from {len(messages)} messages")
            return profile
