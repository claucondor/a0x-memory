"""
FactExtractor - Extracts user facts from messages.

Works on both group messages and DM messages.
Identifies preferences, expertise, behaviors, etc.

Also auto-creates facts from shareable memories.
"""
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone

from models.group_memory import UserFact, FactType, UserMemory, GroupMemory, InteractionMemory
from database.user_fact_store import UserFactStore
from utils.llm_client import LLMClient


FACT_EXTRACTION_PROMPT = """Extract facts about the user from these messages.

User: {user_id}
Context: {context_type} ({context_id})

Messages:
{messages}

Extract any facts about the user including:
- Preferences (communication style, content preferences, tool choices)
- Expertise (skills, knowledge areas, technical proficiency)
- Personal info (job, location, projects they're building)
- Interests (topics they engage with, what they want to learn)
- Behaviors (patterns, habits, decision-making style)
- Communication (how they express themselves, language preference)

IMPORTANT - Preserve Specific Details:
- Include ALL numbers, percentages, and configurations (e.g., "30% allocation", "4-year vesting")
- Include ALL technical decisions and their parameters (e.g., "using Gitcoin Passport with score 20")
- Include project names and specific features being built
- Do NOT generalize into vague statements - keep specifics

Return JSON:
{{
  "facts": [
    {{
      "content": "Clear, specific statement about the user WITH exact numbers/configs if mentioned",
      "fact_type": "preference|expertise|personal|interest|behavior|communication",
      "keywords": ["keyword1", "keyword2", "include_numbers_too"],
      "confidence": 0.5-0.9
    }}
  ]
}}

Examples of GOOD facts (specific):
- "User is building Baseswap DEX with 30% team allocation and 4-year vesting"
- "User plans to use Gitcoin Passport with minimum score of 20 for sybil prevention"
- "User prefers permissionless protocols without KYC"

Examples of BAD facts (too vague):
- "User is interested in tokenomics" (missing specifics)
- "User is concerned about security" (missing what they're doing about it)

Only include facts you're confident about. Skip if uncertain."""


class FactExtractor:
    """Extracts and manages user facts from conversations."""

    def __init__(
        self,
        fact_store: UserFactStore,
        llm_client: Optional[LLMClient] = None
    ):
        self.fact_store = fact_store
        self.llm_client = llm_client or LLMClient(use_streaming=False)

    def extract_from_messages(
        self,
        agent_id: str,
        user_id: str,
        messages: List[Dict[str, Any]],
        context_type: str,  # "group" or "dm"
        context_id: str
    ) -> List[UserFact]:
        """Extract facts from a batch of messages."""
        if not messages:
            return []

        # Format messages
        messages_text = "\n".join([
            f"[{m.get('speaker', 'unknown')}]: {m.get('content', '')}"
            for m in messages
        ])

        prompt = FACT_EXTRACTION_PROMPT.format(
            user_id=user_id,
            context_type=context_type,
            context_id=context_id,
            messages=messages_text
        )

        # Extract facts via LLM
        try:
            response = self.llm_client.chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=0.3
            )
            data = self.llm_client.extract_json(response)
        except Exception as e:
            print(f"[FactExtractor] Failed to extract facts: {e}")
            return []

        # Create UserFact objects and add to store
        extracted_facts = []
        now = datetime.now(timezone.utc).isoformat()

        for fact_data in data.get("facts", []):
            try:
                fact = UserFact(
                    agent_id=agent_id,
                    user_id=user_id,
                    content=fact_data["content"],
                    fact_type=FactType(fact_data["fact_type"]),
                    keywords=fact_data.get("keywords", []),
                    confidence=fact_data.get("confidence", 0.5),
                    sources=[context_id],
                    source_types=[context_type],
                    first_seen=now,
                    last_confirmed=now
                )

                fact_id = self.fact_store.add_fact(fact)
                extracted_facts.append(fact)

                print(f"[FactExtractor] Extracted fact: {fact.content[:50]}... (type: {fact.fact_type.value})")

            except Exception as e:
                print(f"[FactExtractor] Failed to create fact: {e}")
                continue

        return extracted_facts

    def get_user_context(
        self,
        user_id: str,
        query: str = None,
        min_confidence: float = 0.3
    ) -> Dict[str, Any]:
        """Get user facts for context building."""
        if query:
            facts = self.fact_store.search_facts(user_id, query, min_confidence=min_confidence)
        else:
            facts = self.fact_store.get_user_facts(user_id, min_confidence=min_confidence)

        # Group by fact type for better context
        facts_by_type = {}
        for fact in facts:
            fact_type = fact.fact_type.value
            if fact_type not in facts_by_type:
                facts_by_type[fact_type] = []
            facts_by_type[fact_type].append({
                "content": fact.content,
                "confidence": fact.confidence,
                "evidence": fact.evidence_count,
                "sources": fact.sources
            })

        return {
            "user_id": user_id,
            "facts_by_type": facts_by_type,
            "total_facts": len(facts),
            "high_confidence_facts": len([f for f in facts if f.confidence >= 0.7])
        }

    def consolidate_if_needed(self, user_id: str):
        """Check if user has enough facts to consolidate."""
        fact_count = self.fact_store.count_facts(user_id)

        # Consolidate if user has more than threshold
        if fact_count >= 10:  # Same as CONFIDENCE_CONFIG["consolidation_threshold"]
            print(f"[FactExtractor] User {user_id} has {fact_count} facts, consolidating...")
            self.fact_store.consolidate_facts(user_id)

    def extract_and_consolidate(
        self,
        agent_id: str,
        user_id: str,
        messages: List[Dict[str, Any]],
        context_type: str,
        context_id: str
    ) -> Dict[str, Any]:
        """
        Extract facts from messages and consolidate if needed.

        Returns summary of extraction results.
        """
        # Extract facts
        facts = self.extract_from_messages(
            agent_id=agent_id,
            user_id=user_id,
            messages=messages,
            context_type=context_type,
            context_id=context_id
        )

        # Check if consolidation is needed
        self.consolidate_if_needed(user_id)

        # Get updated context
        context = self.get_user_context(user_id)

        return {
            "extracted_count": len(facts),
            "total_facts": context["total_facts"],
            "high_confidence_count": context["high_confidence_facts"],
            "facts_by_type": list(context["facts_by_type"].keys())
        }

    def extract_from_shareable_memory(
        self,
        memory: Union[UserMemory, GroupMemory, InteractionMemory],
        agent_id: str,
        user_id: str = None
    ) -> Optional[UserFact]:
        """
        Convert a shareable memory to a global UserFact.

        When a memory is marked as is_shareable=true, it represents information
        about the user that can be shared across contexts. This method converts
        such memories into UserFacts for cross-context availability.

        Args:
            memory: The shareable memory to convert
            agent_id: Agent ID
            user_id: Optional user ID (inferred from memory if not provided)

        Returns:
            Created UserFact or None if memory is not shareable
        """
        # Check if memory is shareable
        if not getattr(memory, 'is_shareable', False):
            return None

        # Infer user_id from memory if not provided
        if not user_id:
            user_id = getattr(memory, 'user_id', None)

        if not user_id:
            print(f"[FactExtractor] Cannot extract fact: no user_id in memory")
            return None

        # Infer fact type from memory
        fact_type = self._infer_fact_type_from_memory(memory)

        # Get source type
        group_id = getattr(memory, 'group_id', '')
        if group_id and group_id.startswith('dm_'):
            source_type = "dm"
        else:
            source_type = "group"

        # Determine confidence based on memory importance
        importance_score = getattr(memory, 'importance_score', 0.5)
        confidence = 0.5 + (importance_score * 0.3)  # Map 0-1 to 0.5-0.8

        now = datetime.now(timezone.utc).isoformat()

        fact = UserFact(
            fact_id=f"shareable_{memory.memory_id}",
            agent_id=agent_id,
            user_id=user_id,
            content=memory.content,
            fact_type=fact_type,
            keywords=getattr(memory, 'keywords', []),
            evidence_count=1,
            confidence=confidence,
            sources=[group_id] if group_id else ["unknown"],
            source_types=[source_type],
            first_seen=now,
            last_confirmed=now
        )

        # Add to fact store
        try:
            fact_id = self.fact_store.add_fact(fact)
            print(f"[FactExtractor] Auto-created fact from shareable memory: {fact.content[:50]}...")
            return fact
        except Exception as e:
            print(f"[FactExtractor] Failed to auto-create fact: {e}")
            return None

    def _infer_fact_type_from_memory(
        self,
        memory: Union[UserMemory, GroupMemory, InteractionMemory]
    ) -> FactType:
        """
        Infer the most appropriate FactType from a memory's type and content.

        Args:
            memory: The memory to analyze

        Returns:
            Inferred FactType
        """
        memory_type = getattr(memory, 'memory_type', None)

        # Direct mapping for known types
        if memory_type:
            memory_type_str = memory_type.value if hasattr(memory_type, 'value') else memory_type
            type_mapping = {
                "expertise": FactType.EXPERTISE,
                "preference": FactType.PREFERENCE,
                "fact": FactType.PERSONAL,
                "announcement": FactType.PERSONAL,
                "conversation": FactType.PERSONAL,
                "interaction": FactType.COMMUNICATION
            }
            if memory_type_str in type_mapping:
                return type_mapping[memory_type_str]

        # Fallback: analyze content keywords
        content = memory.content.lower()

        if any(word in content for word in ["expert", "skilled", "proficient", "experience", "specialist"]):
            return FactType.EXPERTISE
        elif any(word in content for word in ["prefer", "like", "enjoy", "love", "hate"]):
            return FactType.PREFERENCE
        elif any(word in content for word in ["project", "work", "job", "company"]):
            return FactType.PERSONAL
        elif any(word in content for word in ["said", "asked", "replied", "discussed"]):
            return FactType.COMMUNICATION
        else:
            return FactType.PERSONAL
