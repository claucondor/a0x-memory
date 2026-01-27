# Group Memory - Architecture Proposals

**Date:** January 2026
**Purpose:** Design and test 5 different architectures for group memories in A0X platform

---

## Executive Summary

This document presents 5 distinct architectures for implementing group memories in the A0X platform, where AI agents interact in Telegram groups with 50+ users. Each architecture is based on research findings from:

- **Transactive Memory Theory** - Groups need "who knows what" metaknowledge
- **Social Identity Theory** - Groups form shared representations
- **Distributed Cognition** - Agent is one node in a larger cognitive system
- **Multi-level Memory** - Individual, Group, Cross-group layers

---

## Architecture 1: Triple-Tenant Hierarchy

### Overview

Extend the existing multi-tenant pattern (agent_id, user_id) to add group_id as a first-class tenant. Simple, backwards-compatible, follows existing patterns.

### Schema Design

```python
schema = pa.schema([
    # Triple-tenant hierarchy
    pa.field("agent_id", pa.string()),      # Which agent
    pa.field("group_id", pa.string()),       # Which group (null = DM)
    pa.field("user_id", pa.string()),        # Which user (null = group-level memory)

    # Memory classification
    pa.field("memory_type", pa.string()),    # "group" | "user" | "interaction"
    pa.field("privacy_scope", pa.string()),  # "public" | "protected" | "private"

    # Content
    pa.field("content", pa.string()),
    pa.field("speaker", pa.string()),        # Who said it
    pa.field("timestamp", pa.string()),

    # Existing fields
    pa.field("entry_id", pa.string()),
    pa.field("lossless_restatement", pa.string()),
    pa.field("keywords", pa.list_(pa.string())),
    pa.field("entities", pa.list_(pa.string())),
    pa.field("vector", pa.list_(pa.float32(), 384))
])
```

### Memory Types

| Type | group_id | user_id | Use Case |
|------|----------|---------|----------|
| `group` | set | null | Group-wide decisions, announcements |
| `user` | set | set | User-specific context within group |
| `interaction` | set | set | User-to-user conversations observed by agent |

### Retrieval Strategy

```python
def get_group_context(group_id: str, user_id: str = None):
    # 1. Group-level memories (shared with everyone)
    group_memories = search(
        where=f"group_id = '{group_id}' AND memory_type = 'group'"
    )

    # 2. User-specific memories (if user provided)
    user_memories = []
    if user_id:
        user_memories = search(
            where=f"group_id = '{group_id}' AND user_id = '{user_id}'"
        )

    return {
        "group_context": group_memories,
        "user_context": user_memories
    }
```

### Privacy Model

- **public**: All group members can see (group-level memories)
- **protected**: Only that user can see (user-level memories)
- **private**: Only agent can see (internal reasoning)

### Pros/Cons

| Pros | Cons |
|------|------|
| Simple extension of existing pattern | May not scale to 1000+ groups |
| Backwards compatible (group_id=null for DMs) | Single table could become large |
| Easy to understand and maintain | Limited granularity in access control |
| Fast queries with proper indexing | Complex queries require multiple filters |

---

## Architecture 2: Memory Type Partitioning

### Overview

Partition memories by type into separate physical tables, each optimized for its access pattern. Unified retrieval layer merges results.

### Schema Design

```python
# Three separate tables
group_memories_schema = pa.schema([
    pa.field("agent_id", pa.string()),
    pa.field("group_id", pa.string()),
    pa.field("content", pa.string()),
    pa.field("timestamp", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), 384))
])

user_memories_schema = pa.schema([
    pa.field("agent_id", pa.string()),
    pa.field("group_id", pa.string()),
    pa.field("user_id", pa.string()),
    pa.field("content", pa.string()),
    pa.field("timestamp", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), 384))
])

interaction_memories_schema = pa.schema([
    pa.field("agent_id", pa.string()),
    pa.field("group_id", pa.string()),
    pa.field("speaker_id", pa.string()),
    pa.field("listener_id", pa.string()),
    pa.field("content", pa.string()),
    pa.field("timestamp", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), 384))
])
```

### Retrieval Strategy

```python
def unified_search(group_id: str, user_id: str, query: str):
    results = []

    # Parallel search across partitions
    group_results = group_table.search(query).where(f"group_id = '{group_id}'")
    user_results = user_table.search(query).where(f"group_id = '{group_id}' AND user_id = '{user_id}'")
    interaction_results = interaction_table.search(query).where(
        f"group_id = '{group_id}' AND (speaker_id = '{user_id}' OR listener_id = '{user_id}')"
    )

    # Merge and re-rank
    merged = merge_and_rerank([group_results, user_results, interaction_results])
    return merged
```

### Pros/Cons

| Pros | Cons |
|------|------|
| Optimized indexes per memory type | More complex storage layer |
| Better query performance (smaller tables) | Need to manage 3x tables |
| Can scale partitions independently | Cross-type queries are complex |
| Natural isolation of concerns | Migration complexity |

---

## Architecture 3: Graph-Based Group Memory

### Overview

Implement Transactive Memory Theory using a graph structure where nodes are users/messages and edges represent relationships. Track "who knows what" metaknowledge.

### Schema Design

```python
# Nodes table (users, messages, concepts)
nodes_schema = pa.schema([
    pa.field("node_id", pa.string()),          # Unique identifier
    pa.field("node_type", pa.string()),        # "user" | "message" | "concept"
    pa.field("agent_id", pa.string()),
    pa.field("group_id", pa.string()),
    pa.field("properties", pa.string()),       # JSON blob
    pa.field("vector", pa.list_(pa.float32(), 384))
])

# Edges table (relationships)
edges_schema = pa.schema([
    pa.field("edge_id", pa.string()),
    pa.field("source_node", pa.string()),
    pa.field("target_node", pa.string()),
    pa.field("edge_type", pa.string()),        # "said" | "knows_about" | "replied_to"
    pa.field("weight", pa.float32()),          # Strength of relationship
    pa.field("timestamp", pa.string())
])
```

### Metaknowledge Tracking

```python
# Who knows what
def get_experts(group_id: str, topic: str):
    # Find users who have discussed this topic
    experts = query("""
        SELECT u.node_id, u.properties, COUNT(e.weight) as expertise_score
        FROM nodes u
        JOIN edges e ON u.node_id = e.source_node
        WHERE u.group_id = ? AND u.node_type = 'user'
        AND e.edge_type = 'knows_about'
        AND EXISTS (
            SELECT 1 FROM nodes t
            WHERE t.node_id = e.target_node
            AND t.node_type = 'concept'
            AND t.properties LIKE ?
        )
        GROUP BY u.node_id
        ORDER BY expertise_score DESC
    """, (group_id, f'%{topic}%'))

    return experts
```

### Retrieval Strategy

```python
def graph_search(group_id: str, user_id: str, query: str):
    # 1. Find relevant concept nodes
    concepts = search_nodes(query, node_type="concept")

    # 2. Traverse graph to find related users and messages
    results = []
    for concept in concepts:
        # Users who know about this
        users = traverse_edges(concept, edge_type="knows_about", direction="incoming")

        # Messages about this
        messages = traverse_edges(concept, edge_type="mentions", direction="incoming")

        results.extend(users + messages)

    # 3. Rank by relevance and relationship strength
    return rank_by_proximity(results, user_id)
```

### Pros/Cons

| Pros | Cons |
|------|------|
| Natural fit for Transactive Memory Theory | Requires graph query engine |
| Tracks relationships and expertise | More complex data model |
| Enables "who knows what" queries | Slower for simple lookups |
| Handles multi-hop reasoning | Graph traversal can be expensive |

---

## Architecture 4: Privacy-Scoped Memories

### Overview

Focus on privacy as the primary organizing principle. Three concentric scopes with asymmetric access control. Based on "Multi-User Memory Sharing" research.

### Schema Design

```python
schema = pa.schema([
    # Tenant
    pa.field("agent_id", pa.string()),
    pa.field("group_id", pa.string()),
    pa.field("user_id", pa.string()),

    # Privacy scope (primary organizing principle)
    pa.field("privacy_scope", pa.string()),  # "public" | "protected" | "private"

    # Access control list (for protected memories)
    pa.field("acl", pa.list_(pa.string())),  # List of user_ids who can access

    # Content
    pa.field("content", pa.string()),
    pa.field("keywords", pa.list_(pa.string())),
    pa.field("timestamp", pa.string()),

    # Vector search
    pa.field("vector", pa.list_(pa.float32(), 384))
])
```

### Privacy Model

```
┌─────────────────────────────────────────┐
│           PRIVATE (Agent Only)           │
│  - Internal reasoning                    │
│  - User classifications                  │
│  - Strategic observations                │
│                                         │
│  ┌─────────────────────────────────────┐│
│  │     PROTECTED (User + Agent)         ││
│  │  - User preferences                  ││
│  │  - 1:1 conversations                ││
│  │  - Personal information              ││
│  │                                     ││
│  │  ┌─────────────────────────────────┐││
│  │  │     PUBLIC (All Group Members)   │││
│  │  │  - Group decisions               │││
│  │  │  - Announcements                 │││
│  │  │  - Shared knowledge              │││
│  │  └─────────────────────────────────┘││
│  └─────────────────────────────────────┘│
└─────────────────────────────────────────┘
```

### Access Control

```python
def can_access(memory, user_id, group_id):
    if memory.privacy_scope == "public":
        # Anyone in group can access
        return is_group_member(user_id, group_id)

    elif memory.privacy_scope == "protected":
        # Only specific users in ACL
        return user_id in memory.acl or user_id == memory.user_id

    else:  # private
        # Only agent
        return False  # Never returned to users
```

### Retrieval Strategy

```python
def privacy_aware_search(group_id: str, user_id: str, query: str):
    results = []

    # Search all memories
    all_results = vector_search(query)

    # Filter by access control
    for result in all_results:
        if can_access(result, user_id, group_id):
            # Strip private memories before returning
            if result.privacy_scope == "private":
                result.content = "[Agent internal memory]"
            results.append(result)

    return results
```

### Pros/Cons

| Pros | Cons |
|------|------|
| Privacy-first design | Complex access control logic |
| Clear security boundaries | Overhead for checking permissions |
| Supports regulatory compliance | May limit context sharing |
| Asymmetric access control | Requires careful ACL management |

---

## Architecture 5: Hybrid Multi-Level Memory

### Overview

Combine best elements from all approaches. Multi-level memory (individual, group, cross-group) with adaptive retrieval based on query complexity.

### Schema Design

```python
# Unified schema with multi-level indicators
schema = pa.schema([
    # Tenant
    pa.field("agent_id", pa.string()),
    pa.field("group_id", pa.string()),
    pa.field("user_id", pa.string()),

    # Multi-level classification
    pa.field("memory_level", pa.string()),    # "individual" | "group" | "cross_group"
    pa.field("memory_type", pa.string()),     # "conversation" | "fact" | "preference" | "expertise"
    pa.field("privacy_scope", pa.string()),   # "public" | "protected" | "private"

    # Content
    pa.field("content", pa.string()),
    pa.field("speaker", pa.string()),
    pa.field("timestamp", pa.string()),

    # Metadata for structured access
    pa.field("importance_score", pa.float32()),  # 0-1, for prioritization
    pa.field("access_count", pa.int32()),        # For decay/forgetting
    pa.field("last_accessed", pa.string()),

    # Vector search
    pa.field("vector", pa.list_(pa.float32(), 384))
])
```

### Multi-Level Memory

```
┌──────────────────────────────────────────────────────────┐
│                    CROSS-GROUP LEVEL                      │
│  - User identity across groups                           │
│  - Universal preferences (language, timezone)            │
│  - Expertise areas                                       │
│  - Cross-group references only!                          │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │                  GROUP LEVEL                        │  │
│  │  - Group context and culture                        │  │
│  │  - Group decisions and norms                        │  │
│  │  - Who knows what (transactive memory)             │  │
│  │  - Shared knowledge                                │  │
│  │                                                    │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │            INDIVIDUAL LEVEL                   │  │  │
│  │  │  - User-specific interactions                │  │  │
│  │  │  - Personal context                          │  │  │
│  │  │  - 1:1 conversations                         │  │  │
│  │  │  - Private observations                      │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### Adaptive Retrieval

```python
def adaptive_search(group_id: str, user_id: str, query: str):
    # Estimate query complexity
    complexity = estimate_complexity(query)

    # Low complexity: simple keyword match
    if complexity < 0.3:
        return keyword_search(query, limit=5)

    # Medium complexity: semantic search at individual level
    elif complexity < 0.7:
        return semantic_search(query, level="individual", limit=10)

    # High complexity: multi-level search with consolidation
    else:
        individual = semantic_search(query, level="individual", limit=10)
        group = semantic_search(query, level="group", limit=5)
        cross_group = semantic_search(query, level="cross_group", limit=3)

        # Consolidate and remove redundancy
        return consolidate_memories([individual, group, cross_group])
```

### Consolidation Strategy

```python
def consolidate_memories(memory_lists):
    """
    Implement "Recursive Consolidation" from SimpleMem paper.
    Merge redundant memories across levels.
    """
    seen = set()
    consolidated = []

    for memories in memory_lists:
        for memory in memories:
            # Create semantic signature
            signature = semantic_hash(memory.content)

            if signature not in seen:
                seen.add(signature)
                consolidated.append(memory)
            else:
                # Merge with existing
                existing = find_by_signature(consolidated, signature)
                existing.confidence = max(existing.confidence, memory.confidence)
                existing.access_count += memory.access_count

    return consolidated
```

### Pros/Cons

| Pros | Cons |
|------|------|
| Most comprehensive solution | Most complex to implement |
| Adaptive to query complexity | Requires careful tuning |
| Combines strengths of all approaches | Higher computational overhead |
| Supports long-term evolution | May be overkill for simple use cases |

---

## Comparison Summary

### Complexity Ranking (1-5, 5=most complex)

| Architecture | Implementation | Query Complexity | Storage Overhead |
|--------------|----------------|------------------|------------------|
| Arch 1: Triple-Tenant | 2 | 2 | 1 |
| Arch 2: Partitioning | 3 | 3 | 3 |
| Arch 3: Graph-Based | 5 | 5 | 4 |
| Arch 4: Privacy-Scoped | 3 | 4 | 2 |
| Arch 5: Hybrid | 5 | 5 | 3 |

### Use Case Fit

| Use Case | Best Architecture | Reasoning |
|----------|-------------------|-----------|
| Simple groups with basic memory | Arch 1 | Easiest to implement, sufficient |
| High-volume groups with distinct patterns | Arch 2 | Optimized for scale |
| Expertise location & "who knows what" | Arch 3 | Natural graph representation |
| Privacy-critical environments | Arch 4 | Security-first design |
| Production-grade with future growth | Arch 5 | Most comprehensive |

### Performance Predictions

| Metric | Arch 1 | Arch 2 | Arch 3 | Arch 4 | Arch 5 |
|--------|--------|--------|--------|--------|--------|
| Insert Latency | ~5ms | ~5ms | ~10ms | ~6ms | ~8ms |
| Query Latency | ~20ms | ~15ms | ~50ms | ~25ms | ~30ms |
| Storage Efficiency | High | Medium | Low | High | Medium |
| Scalability (100 groups) | Good | Excellent | Poor | Good | Excellent |
| Scalability (1000 groups) | Medium | Excellent | Poor | Medium | Good |

---

## Test Plan

Each architecture will be tested with:

1. **Test Data Generator**
   - 50 users per group
   - 500 messages (100 group, 300 user, 100 interaction)
   - Realistic patterns (mentions, replies, announcements)

2. **Query Scenarios**
   - Agent mentioned directly
   - User-to-user conversation retrieval
   - Group-wide announcements
   - Cross-context queries

3. **Metrics**
   - Insert latency (batch and single)
   - Query latency (p50, p95, p99)
   - Recall@k (accuracy)
   - Relevance scores
   - Storage efficiency

4. **Load Testing**
   - Single group (baseline)
   - 10 groups (medium scale)
   - 100 groups (high scale)

---

## Next Steps

1. Implement test data generator
2. Implement each architecture
3. Run comprehensive tests
4. Compare results
5. Select and recommend best architecture

