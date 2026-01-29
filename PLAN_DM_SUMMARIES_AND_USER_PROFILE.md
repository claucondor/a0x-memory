# Plan: DM Summaries and Evidence-Based User Profiles

## Overview

Implement two features:
1. **DM Summaries**: Volume-based summarization for 1-on-1 conversations (lower thresholds than groups)
2. **User Profile Enhancement**: Evidence-based fact tracking with cross-context consolidation

---

## Part 1: DM Summaries

### 1.1 Configuration

Add DM-specific thresholds to `database/group_summary_store.py`:

```python
# Add after SUMMARY_CONFIG
DM_SUMMARY_CONFIG = {
    "micro": {
        "message_threshold": 20,       # Lower than group (50)
        "aggregate_count": 5,          # 5 micros → 1 chunk
        "decay_start_messages": 100,
        "max_messages": 200,
    },
    "chunk": {
        "message_threshold": 100,      # 5 micros
        "aggregate_count": 5,          # 5 chunks → 1 block
        "decay_start_messages": 500,
        "max_messages": 1000,
    },
    "block": {
        "message_threshold": 500,      # 5 chunks
        "aggregate_count": 5,
        "decay_start_messages": 2500,
        "max_messages": 5000,
    }
}
```

### 1.2 DM Summary Store

Create `database/dm_summary_store.py`:

```python
"""
DMSummaryStore - LanceDB storage for DM conversation summaries.

Lower thresholds than group summaries because DMs have less volume.
Uses same GroupSummary model but with DM_SUMMARY_CONFIG.
"""
from database.group_summary_store import GroupSummaryStore, DM_SUMMARY_CONFIG

class DMSummaryStore(GroupSummaryStore):
    """Storage for DM summaries with lower volume thresholds."""

    def __init__(self, db_path=None, embedding_model=None, agent_id=None):
        super().__init__(db_path, embedding_model, agent_id)
        # Override config with DM thresholds
        self.config = DM_SUMMARY_CONFIG

    def _init_table(self):
        """Initialize dm_summaries table."""
        # Same schema as group_summaries
        # Table name: "dm_summaries"
        pass
```

### 1.3 DM Summary Aggregator

Create `services/dm_summary_aggregator.py`:

```python
"""
DMSummaryAggregator - Summarizes 1-on-1 conversations.

Same hierarchy as groups but lower thresholds:
- Micro: every 20 messages
- Chunk: every 100 messages (5 micros)
- Block: every 500 messages (5 chunks)
"""
from services.summary_aggregator import SummaryAggregator
from database.dm_summary_store import DMSummaryStore, DM_SUMMARY_CONFIG

DM_MICRO_PROMPT = """Summarize this 1-on-1 conversation segment.

User: {user_id}
Messages {msg_start} to {msg_end} ({msg_count} messages)
Time span: {time_start} to {time_end}

Conversation:
{messages}

Return JSON:
{{
  "summary": "2-3 sentences summarizing the conversation",
  "topics": ["topic1", "topic2", ...],
  "user_requests": ["what the user asked for"],
  "agent_actions": ["what the agent did/provided"],
  "user_sentiment": "positive|neutral|negative|mixed"
}}"""

class DMSummaryAggregator(SummaryAggregator):
    """Aggregator for DM conversations with lower thresholds."""

    def __init__(self, summary_store: DMSummaryStore, llm_client=None):
        super().__init__(summary_store, llm_client)
        self.config = DM_SUMMARY_CONFIG
```

---

## Part 2: Evidence-Based User Profile

### 2.1 UserFact Model

Add to `models/group_memory.py`:

```python
class FactType(str, Enum):
    """Type of user fact"""
    PREFERENCE = "preference"      # "Prefers technical explanations"
    EXPERTISE = "expertise"        # "Expert in DeFi"
    BEHAVIOR = "behavior"          # "Usually active mornings"
    PERSONAL = "personal"          # "Works at Company X"
    INTEREST = "interest"          # "Interested in NFTs"
    COMMUNICATION = "communication" # "Prefers short responses"


class UserFact(BaseModel):
    """
    A single fact about a user, tracked with evidence.

    Facts are extracted from user messages in groups and DMs.
    They don't decay over time but have confidence scores.
    """
    fact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    user_id: str  # universal_user_id (platform:id)

    # Content
    content: str  # "Prefers technical explanations"
    fact_type: FactType
    keywords: List[str] = []

    # Evidence tracking
    evidence_count: int = 1
    confidence: float = 0.5  # 0.0 - 1.0
    sources: List[str] = []  # ["group_123", "dm_456"]
    source_types: List[str] = []  # ["group", "dm", "group"]

    # Temporal
    first_seen: str
    last_confirmed: str

    # Consolidation
    is_consolidated: bool = False
    consolidated_from: List[str] = []  # fact_ids that were merged
    contradicted_by: List[str] = []  # fact_ids that contradict

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
```

### 2.2 UserFactStore

Create `database/user_fact_store.py`:

```python
"""
UserFactStore - LanceDB storage for evidence-based user facts.

Facts are extracted from both groups and DMs.
No decay - facts remain valid until contradicted.
Confidence increases with more evidence from diverse sources.
"""
import lancedb
import pyarrow as pa
from typing import List, Optional
from models.group_memory import UserFact, FactType

CONFIDENCE_CONFIG = {
    "base_confidence": 0.5,
    "evidence_boost": 0.05,      # +0.05 per additional evidence
    "source_diversity_boost": 0.1,  # +0.1 per unique source type
    "max_confidence": 0.95,
    "consolidation_threshold": 10,  # Merge after 10 similar facts
    "similarity_threshold": 0.85,   # For deduplication
}


class UserFactStore:
    """LanceDB storage for user facts with evidence tracking."""

    def __init__(self, db_path=None, embedding_model=None, agent_id=None):
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.agent_id = agent_id
        self.db = lancedb.connect(db_path)
        self._init_table()

    def _init_table(self):
        """Initialize user_facts table."""
        schema = pa.schema([
            pa.field("fact_id", pa.string()),
            pa.field("agent_id", pa.string()),
            pa.field("user_id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("fact_type", pa.string()),
            pa.field("keywords", pa.list_(pa.string())),
            pa.field("evidence_count", pa.int32()),
            pa.field("confidence", pa.float32()),
            pa.field("sources", pa.list_(pa.string())),
            pa.field("source_types", pa.list_(pa.string())),
            pa.field("first_seen", pa.string()),
            pa.field("last_confirmed", pa.string()),
            pa.field("is_consolidated", pa.bool_()),
            pa.field("consolidated_from", pa.list_(pa.string())),
            pa.field("contradicted_by", pa.list_(pa.string())),
            pa.field("fact_vector", pa.list_(pa.float32(), 384)),
        ])
        # Create or open table
        pass

    def add_fact(self, fact: UserFact) -> str:
        """Add a new fact or update existing if similar."""
        # Check for similar existing fact
        similar = self._find_similar_fact(fact.user_id, fact.content)
        if similar:
            return self._merge_fact(similar, fact)
        # Add new fact
        pass

    def _find_similar_fact(self, user_id: str, content: str) -> Optional[UserFact]:
        """Find existing fact with similar content."""
        # Vector search with similarity threshold
        pass

    def _merge_fact(self, existing: UserFact, new: UserFact) -> str:
        """Merge new evidence into existing fact."""
        existing.evidence_count += 1
        existing.last_confirmed = new.last_confirmed

        # Add source if new
        if new.sources[0] not in existing.sources:
            existing.sources.append(new.sources[0])
            existing.source_types.append(new.source_types[0])

        # Recalculate confidence
        existing.confidence = self._calculate_confidence(existing)

        # Update in DB
        pass

    def _calculate_confidence(self, fact: UserFact) -> float:
        """Calculate confidence based on evidence and source diversity."""
        config = CONFIDENCE_CONFIG

        confidence = config["base_confidence"]

        # Evidence boost
        evidence_boost = (fact.evidence_count - 1) * config["evidence_boost"]
        confidence += evidence_boost

        # Source diversity boost
        unique_source_types = len(set(fact.source_types))
        diversity_boost = (unique_source_types - 1) * config["source_diversity_boost"]
        confidence += diversity_boost

        return min(confidence, config["max_confidence"])

    def get_user_facts(self, user_id: str, min_confidence: float = 0.3) -> List[UserFact]:
        """Get all facts for a user above confidence threshold."""
        pass

    def search_facts(self, user_id: str, query: str, limit: int = 5) -> List[UserFact]:
        """Semantic search for relevant facts."""
        pass

    def consolidate_facts(self, user_id: str):
        """
        Consolidate similar facts into general statements.

        Called when a user has many facts (>consolidation_threshold).
        Groups similar facts and creates consolidated fact.
        """
        facts = self.get_user_facts(user_id)
        if len(facts) < CONFIDENCE_CONFIG["consolidation_threshold"]:
            return

        # Group similar facts
        # Create consolidated fact
        # Mark original facts as consolidated
        pass

    def add_contradiction(self, fact_id: str, contradicting_fact_id: str):
        """Mark two facts as contradicting each other."""
        pass
```

### 2.3 Fact Extractor Service

Create `services/fact_extractor.py`:

```python
"""
FactExtractor - Extracts user facts from messages.

Works on both group messages and DM messages.
Identifies preferences, expertise, behaviors, etc.
"""
from typing import List, Dict, Any, Optional
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

    def __init__(self, fact_store: UserFactStore, llm_client: LLMClient = None):
        self.fact_store = fact_store
        self.llm_client = llm_client or LLMClient()

    def extract_from_messages(
        self,
        agent_id: str,
        user_id: str,
        messages: List[Dict[str, Any]],
        context_type: str,  # "group" or "dm"
        context_id: str
    ) -> List[UserFact]:
        """Extract facts from a batch of messages."""
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
        response = self.llm_client.chat_completion(
            [{"role": "user", "content": prompt}],
            temperature=0.3
        )
        data = self.llm_client.extract_json(response)

        # Create UserFact objects and add to store
        extracted_facts = []
        for fact_data in data.get("facts", []):
            fact = UserFact(
                agent_id=agent_id,
                user_id=user_id,
                content=fact_data["content"],
                fact_type=FactType(fact_data["fact_type"]),
                keywords=fact_data.get("keywords", []),
                confidence=fact_data.get("confidence", 0.5),
                sources=[context_id],
                source_types=[context_type],
                first_seen=messages[-1].get("timestamp", ""),
                last_confirmed=messages[-1].get("timestamp", "")
            )

            self.fact_store.add_fact(fact)
            extracted_facts.append(fact)

        return extracted_facts

    def get_user_context(self, user_id: str, query: str = None) -> Dict[str, Any]:
        """Get user facts for context building."""
        if query:
            facts = self.fact_store.search_facts(user_id, query)
        else:
            facts = self.fact_store.get_user_facts(user_id)

        return {
            "user_id": user_id,
            "facts": [
                {
                    "content": f.content,
                    "type": f.fact_type.value,
                    "confidence": f.confidence,
                    "evidence": f.evidence_count
                }
                for f in facts
            ],
            "total_facts": len(facts)
        }
```

---

## Commit Plan

### Commit 1: DM Summary Config and Store
```
feat(dm-summaries): add DM summary store with lower thresholds

- Add DM_SUMMARY_CONFIG (20/100/500 vs group 50/250/1250)
- Create DMSummaryStore extending GroupSummaryStore
- Lower thresholds appropriate for 1-on-1 conversations
```

### Commit 2: DM Summary Aggregator
```
feat(dm-summaries): add DM summary aggregator

- Create DMSummaryAggregator with DM-specific prompts
- Include user_requests, agent_actions, user_sentiment
- Same micro/chunk/block hierarchy with lower thresholds
```

### Commit 3: UserFact Model
```
feat(user-profile): add evidence-based UserFact model

- Add FactType enum (preference, expertise, behavior, etc.)
- Add UserFact model with evidence tracking
- Track sources, confidence, consolidation status
```

### Commit 4: UserFactStore
```
feat(user-profile): add UserFactStore with evidence tracking

- Implement fact storage with deduplication
- Add confidence calculation based on evidence + source diversity
- Add consolidation for users with many facts
- Add contradiction tracking
```

### Commit 5: Fact Extractor Service
```
feat(user-profile): add FactExtractor service

- Extract facts from group and DM messages
- LLM-based extraction with structured output
- Automatic merging with existing facts
- Context builder for RAG
```

### Commit 6: Integration
```
feat(user-profile): integrate fact extraction into message flow

- Call FactExtractor during message processing
- Add facts to context retrieval
- Update API endpoints for fact queries
```

---

## Testing

Create `tests/test_dm_summaries.py`:
```python
# Test DM summary generation with lower thresholds
# Test decay calculation
# Test aggregation flow
```

Create `tests/test_user_facts.py`:
```python
# Test fact extraction from messages
# Test evidence merging
# Test confidence calculation
# Test consolidation
# Test cross-context (group + DM) fact building
```

---

## Files to Create/Modify

### New Files
- `database/dm_summary_store.py`
- `services/dm_summary_aggregator.py`
- `database/user_fact_store.py`
- `services/fact_extractor.py`
- `tests/test_dm_summaries.py`
- `tests/test_user_facts.py`

### Modified Files
- `models/group_memory.py` - Add FactType, UserFact
- `database/group_summary_store.py` - Add DM_SUMMARY_CONFIG
- `main.py` - Integrate fact extraction
- `api.py` - Add fact query endpoints
