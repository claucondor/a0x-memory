"""
FactExtractor - Extracts user facts from messages.

Works on both group messages and DM messages.
Identifies preferences, expertise, behaviors, etc.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from models.group_memory import UserFact, FactType
from database.user_fact_store import UserFactStore
from utils.llm_client import LLMClient


FACT_EXTRACTION_PROMPT = """Extract facts about the user from these messages.

User: {user_id}
Context: {context_type} ({context_id})

Messages:
{messages}

Extract any facts about the user including:
- Preferences (communication style, content preferences)
- Expertise (skills, knowledge areas)
- Personal info (job, location, projects)
- Interests (topics they engage with)
- Behaviors (patterns, habits)
- Communication (how they express themselves)

Return JSON:
{{
  "facts": [
    {{
      "content": "Clear statement about the user",
      "fact_type": "preference|expertise|personal|interest|behavior|communication",
      "keywords": ["keyword1", "keyword2"],
      "confidence": 0.5-0.9
    }}
  ]
}}

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
