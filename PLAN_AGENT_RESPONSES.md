# Plan: Agent Responses System

## Objective
Implement a system to store and retrieve agent responses with dual-vector search (trigger + response) to:
1. Avoid repetition ("Did I already say this?")
2. Find previous responses to similar questions ("Did someone ask this before?")
3. Track what was said to each user and group

## Base Commit
```
dc66df6 fix(store): implement search_agent_responses delegation
```

## Current State
- ✅ `agent_responses` table exists in LanceDB (single vector)
- ✅ Firestore stores responses with `trigger_message` field
- ✅ `AgentResponsesTable.search_semantic()` exists
- ❌ Only single vector (response), no trigger vector
- ❌ No scope field (global/user/group)
- ❌ No batch processing from Firestore to LanceDB
- ❌ No LLM extraction for summary/topics

## Architecture

### Single Table with Dual Vectors and Scope
```
agent_responses (LanceDB)
├── response_id: str (UUID)
├── agent_id: str
├── scope: str ("global" | "user" | "group")
├── user_id: Optional[str] (null if global)
├── group_id: Optional[str] (null if global/user-dm)
│
├── trigger_message: str (user's question)
├── trigger_vector: List[float] (embedding of trigger - for "similar question?" search)
│
├── response_content: str (full agent response)
├── response_summary: str (LLM-generated short summary)
├── response_vector: List[float] (embedding of response - for "similar answer?" search)
│
├── response_type: str ("greeting" | "explanation" | "recommendation" | "opinion" | "action")
├── topics: List[str] (["base", "grants"])
├── keywords: List[str] (for FTS)
│
├── timestamp: str
├── importance_score: float
```

### Search Modes
| Search | Vector Used | Filter | Use Case |
|--------|-------------|--------|----------|
| "Similar question asked before?" | trigger_vector | scope=global OR group_id=X | Reuse explanations |
| "Did I tell THIS user this?" | response_vector | user_id=Y | Avoid repetition to same person |
| "Did I say this in THIS group?" | response_vector | group_id=X | Avoid repetition in group |

### Scope Logic
- `scope=global`: Generic explanations, FAQs, reusable info
- `scope=user`: Said specifically to one user (DM or group mention)
- `scope=group`: Said to a group (not targeting specific user)

## Implementation Steps

### Step 1: Update AgentResponsesTable Schema
**File:** `database/tables/agent_responses.py`

Add new fields to schema:
```python
schema = pa.schema([
    pa.field("response_id", pa.string()),
    pa.field("agent_id", pa.string()),
    # Scope
    pa.field("scope", pa.string()),  # NEW: "global" | "user" | "group"
    pa.field("user_id", pa.string()),
    pa.field("group_id", pa.string()),
    # Trigger (user's message)
    pa.field("trigger_message", pa.string()),  # NEW
    pa.field("trigger_vector", pa.list_(pa.float32(), EMBEDDING_DIM)),  # NEW
    # Response
    pa.field("response_content", pa.string()),  # renamed from "content"
    pa.field("response_summary", pa.string()),  # renamed from "summary"
    pa.field("response_vector", pa.list_(pa.float32(), EMBEDDING_DIM)),  # renamed from "vector"
    # Classification
    pa.field("response_type", pa.string()),
    pa.field("topics", pa.list_(pa.string())),
    pa.field("keywords", pa.list_(pa.string())),
    # Metadata
    pa.field("timestamp", pa.string()),
    pa.field("importance_score", pa.float32()),
])
```

Create indices:
```python
# Scalar indices for filtering
self._create_scalar_index(table, "scope")
self._create_scalar_index(table, "user_id")
self._create_scalar_index(table, "group_id")

# Vector indices for ANN search
table.create_index(metric="cosine", vector_column_name="trigger_vector")
table.create_index(metric="cosine", vector_column_name="response_vector")
```

**Commit:** `feat(tables): add dual-vector schema to agent_responses`

### Step 2: Update AgentResponse Model
**File:** `models/group_memory.py`

Update `AgentResponse` class:
```python
class AgentResponse(BaseModel):
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str

    # Scope
    scope: str = Field(default="user", description="global | user | group")
    user_id: Optional[str] = None
    group_id: Optional[str] = None

    # Trigger
    trigger_message: str = Field(..., description="User message that triggered response")

    # Response
    response_content: str = Field(..., description="Full agent response")
    response_summary: Optional[str] = Field(None, description="Short summary")

    # Classification
    response_type: ResponseType = Field(default=ResponseType.ANSWER)
    topics: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)

    # Metadata
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    importance_score: float = Field(default=0.5)
```

**Commit:** `feat(models): update AgentResponse with scope and trigger fields`

### Step 3: Add Search Methods
**File:** `database/tables/agent_responses.py`

```python
def search_by_trigger(
    self,
    query_vector,
    scope: str = None,
    group_id: str = None,
    limit: int = 5
) -> List[dict]:
    """Search for similar questions asked before."""
    search = self.table.search(query_vector, vector_column_name="trigger_vector")

    conditions = [f"agent_id = '{self.agent_id}'"]
    if scope:
        conditions.append(f"scope = '{scope}'")
    if group_id:
        conditions.append(f"(scope = 'global' OR group_id = '{group_id}')")

    where_clause = " AND ".join(conditions)
    return search.where(where_clause, prefilter=True).limit(limit).to_list()

def search_by_response(
    self,
    query_vector,
    user_id: str = None,
    group_id: str = None,
    limit: int = 5
) -> List[dict]:
    """Search for similar responses given before."""
    search = self.table.search(query_vector, vector_column_name="response_vector")

    conditions = [f"agent_id = '{self.agent_id}'"]
    if user_id:
        conditions.append(f"user_id = '{user_id}'")
    if group_id:
        conditions.append(f"group_id = '{group_id}'")

    where_clause = " AND ".join(conditions)
    return search.where(where_clause, prefilter=True).limit(limit).to_list()
```

**Commit:** `feat(tables): add dual-vector search methods for agent_responses`

### Step 4: Add Method to Store Response with Dual Vectors
**File:** `database/tables/agent_responses.py`

```python
def add_with_vectors(self, response: AgentResponse) -> str:
    """Add response with both trigger and response vectors."""
    trigger_vector = self.embedding_model.encode_single(response.trigger_message, is_query=False)
    response_vector = self.embedding_model.encode_single(
        response.response_summary or response.response_content,
        is_query=False
    )

    data = {
        "response_id": response.response_id,
        "agent_id": response.agent_id,
        "scope": response.scope,
        "user_id": response.user_id,
        "group_id": response.group_id,
        "trigger_message": response.trigger_message,
        "trigger_vector": trigger_vector.tolist(),
        "response_content": response.response_content,
        "response_summary": response.response_summary,
        "response_vector": response_vector.tolist(),
        "response_type": response.response_type.value,
        "topics": response.topics,
        "keywords": response.keywords,
        "timestamp": response.timestamp,
        "importance_score": response.importance_score,
    }

    self.table.add([data])
    return response.response_id
```

**Commit:** `feat(tables): add dual-vector storage method`

### Step 5: Update MemoryStore Facade
**File:** `database/stores/memory_store.py`

Add delegation methods:
```python
def search_agent_responses_by_trigger(
    self,
    query: str,
    scope: str = None,
    group_id: str = None,
    limit: int = 5
) -> List[dict]:
    """Search for similar questions asked before."""
    query_vector = self.embedding_model.encode_single(query, is_query=True)
    return self.agent_responses.search_by_trigger(query_vector, scope, group_id, limit)

def search_agent_responses_by_response(
    self,
    query: str,
    user_id: str = None,
    group_id: str = None,
    limit: int = 5
) -> List[dict]:
    """Search for similar responses given before."""
    query_vector = self.embedding_model.encode_single(query, is_query=True)
    return self.agent_responses.search_by_response(query_vector, user_id, group_id, limit)

def add_agent_response_with_vectors(self, response: AgentResponse) -> str:
    """Add response with dual vectors."""
    return self.agent_responses.add_with_vectors(response)
```

**Commit:** `feat(store): add dual-vector agent response methods`

### Step 6: LLM Extraction for Summary/Topics
**File:** `core/response_extractor.py` (NEW)

```python
class ResponseExtractor:
    """Extract metadata from agent responses using LLM."""

    EXTRACTION_PROMPT = """Analyze this agent response and extract:
1. A 1-sentence summary
2. Main topics (max 5)
3. Keywords (max 10)
4. Response type: greeting | explanation | recommendation | opinion | action
5. Scope: global (reusable FAQ), user (specific to user), group (specific to group context)

User's question: {trigger_message}
Agent's response: {response_content}

Return JSON:
{
  "summary": "...",
  "topics": ["..."],
  "keywords": ["..."],
  "response_type": "...",
  "scope": "..."
}"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def extract(self, trigger_message: str, response_content: str) -> dict:
        """Extract metadata from response."""
        # Call LLM and parse JSON
        ...
```

**Commit:** `feat(core): add ResponseExtractor for LLM metadata extraction`

### Step 7: Batch Processing from Firestore
**File:** `main.py`

Add method to process agent responses from Firestore:
```python
def _process_unprocessed_agent_responses(self, group_id: str) -> Dict[str, Any]:
    """Process agent responses from Firestore to LanceDB."""
    result = {"processed": 0}

    # Get unprocessed agent responses from Firestore
    messages = self.firestore.get_recent_messages(self.agent_id, group_id, limit=50)
    agent_responses = [m for m in messages if m.get('metadata', {}).get('is_agent_response')]

    for msg in agent_responses:
        if msg.get('metadata', {}).get('processed_to_lancedb'):
            continue

        trigger_message = msg.get('metadata', {}).get('trigger_message', '')
        response_content = msg.get('content', '')

        if not trigger_message or not response_content:
            continue

        # Extract metadata with LLM
        extracted = self.response_extractor.extract(trigger_message, response_content)

        # Create AgentResponse
        response = AgentResponse(
            agent_id=self.agent_id,
            scope=extracted.get('scope', 'user'),
            user_id=msg.get('metadata', {}).get('user_id'),
            group_id=group_id if not group_id.startswith('dm_') else None,
            trigger_message=trigger_message,
            response_content=response_content,
            response_summary=extracted.get('summary'),
            response_type=ResponseType(extracted.get('response_type', 'answer')),
            topics=extracted.get('topics', []),
            keywords=extracted.get('keywords', []),
        )

        # Store in LanceDB
        self.unified_store.add_agent_response_with_vectors(response)
        result["processed"] += 1

    return result
```

**Commit:** `feat(main): add batch processing for agent responses`

### Step 8: Integration in Retrieval
**File:** `core/hybrid_retriever.py`

Add agent responses to context retrieval:
```python
def retrieve_for_context(self, query: str, context: dict, ...) -> dict:
    # ... existing code ...

    # Search agent responses
    agent_context = {
        "similar_questions": [],  # "Someone asked this before, I said..."
        "said_to_user": [],       # "I already told you..."
        "said_in_group": [],      # "I mentioned in this group..."
    }

    # Search by trigger (similar questions)
    similar_qs = self.unified_store.search_agent_responses_by_trigger(
        query,
        group_id=group_id,
        limit=3
    )
    agent_context["similar_questions"] = similar_qs

    # Search by response (what I said to this user)
    if user_id:
        said_to_user = self.unified_store.search_agent_responses_by_response(
            query,
            user_id=user_id,
            limit=3
        )
        agent_context["said_to_user"] = said_to_user

    # Search by response (what I said in this group)
    if group_id:
        said_in_group = self.unified_store.search_agent_responses_by_response(
            query,
            group_id=group_id,
            limit=3
        )
        agent_context["said_in_group"] = said_in_group

    result["agent_responses"] = agent_context
    return result
```

**Commit:** `feat(retriever): integrate agent responses in context retrieval`

### Step 9: Update Stats Endpoint
**File:** `api.py`

Include agent_responses in stats.

**Commit:** `feat(api): include agent_responses in stats endpoint`

## Summary of Commits (in order)

1. `feat(tables): add dual-vector schema to agent_responses`
2. `feat(models): update AgentResponse with scope and trigger fields`
3. `feat(tables): add dual-vector search methods for agent_responses`
4. `feat(tables): add dual-vector storage method`
5. `feat(store): add dual-vector agent response methods`
6. `feat(core): add ResponseExtractor for LLM metadata extraction`
7. `feat(main): add batch processing for agent responses`
8. `feat(retriever): integrate agent responses in context retrieval`
9. `feat(api): include agent_responses in stats endpoint`

## Files to Modify
- `database/tables/agent_responses.py` - Schema, search, storage
- `database/stores/memory_store.py` - Facade methods
- `models/group_memory.py` - AgentResponse model
- `core/response_extractor.py` - NEW: LLM extraction
- `main.py` - Batch processing
- `core/hybrid_retriever.py` - Retrieval integration
- `api.py` - Stats

## Testing
After implementation, test with:
```bash
# Add a response
curl -X POST http://136.118.160.81:8080/v1/memory/test_agent:user123/add-response \
  -H "Content-Type: application/json" \
  -d '{
    "response": "Base is a Layer 2 built by Coinbase...",
    "trigger_message": "What is Base?",
    "trigger_message_id": "msg_001"
  }'

# Trigger batch processing
curl -X POST "http://136.118.160.81:8080/v1/memory/process-pending?agent_id=test_agent&group_id=telegram_-100001"

# Check stats
curl http://136.118.160.81:8080/v1/memory/stats/test_agent | jq .memory_breakdown.agent_responses

# Search context (should include agent_responses)
curl -X POST http://136.118.160.81:8080/v1/memory/context \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "test_agent",
    "query": "Tell me about Base",
    "platform_identity": {"platform": "telegram", "chatId": "-100001"}
  }' | jq .agent_responses
```

## Notes
- LanceDB supports multiple vector columns with `vector_column_name` parameter
- Create separate indices for each vector column
- Use `prefilter=True` for efficient filtered search
- Batch processing should run after regular message processing
