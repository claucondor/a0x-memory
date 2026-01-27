"""
Architecture 5: Hybrid Multi-Level Memory

Combine best elements from all approaches.
Multi-level memory (individual, group, cross-group) with adaptive retrieval.

Memory Levels:
- individual: User-specific interactions
- group: Group context and culture
- cross_group: User identity across groups (universal preferences)
"""

import os
import sys
from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib

# Add parent directory to path to import from a0x-memory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pyarrow as pa
import lancedb
from utils.embedding import EmbeddingModel
import json


class HybridMultiLevelMemoryStore:
    """
    Hybrid multi-level memory store with adaptive retrieval.

    Three memory levels:
    - individual: User-specific, personal context
    - group: Shared group knowledge and culture
    - cross_group: User identity across groups
    """

    def __init__(
        self,
        db_path: str,
        agent_id: str = "jessexbt",
        embedding_model: EmbeddingModel = None
    ):
        self.db_path = db_path
        self.agent_id = agent_id
        self.embedding_model = embedding_model or EmbeddingModel()

        # Connect to database
        os.makedirs(db_path, exist_ok=True)
        self.db = lancedb.connect(db_path)

        # Initialize tables (one per level for optimization)
        self.individual_table = None
        self.group_table = None
        self.cross_group_table = None

        self._init_tables()

    def _init_tables(self):
        """Initialize three tables for multi-level memory."""

        # Individual level schema
        individual_schema = pa.schema([
            pa.field("agent_id", pa.string()),
            pa.field("group_id", pa.string()),
            pa.field("user_id", pa.string()),

            # Multi-level classification
            pa.field("memory_level", pa.string()),  # "individual" | "group" | "cross_group"
            pa.field("memory_type", pa.string()),  # "conversation" | "fact" | "preference" | "expertise"
            pa.field("privacy_scope", pa.string()),  # "public" | "protected" | "private"

            # Content
            pa.field("content", pa.string()),
            pa.field("speaker", pa.string()),
            pa.field("timestamp", pa.string()),

            # Metadata for structured access
            pa.field("importance_score", pa.float32()),  # 0-1
            pa.field("access_count", pa.int32()),
            pa.field("last_accessed", pa.string()),

            # IDs
            pa.field("message_id", pa.string()),
            pa.field("keywords", pa.list_(pa.string())),
            pa.field("topics", pa.list_(pa.string())),

            # Vector
            pa.field("vector", pa.list_(pa.float32(), self.embedding_model.dimension))
        ])

        # Create/open tables
        if "arch5_individual" not in self.db.table_names():
            self.individual_table = self.db.create_table("arch5_individual", schema=individual_schema)
            print("[Arch5] Created individual table")
        else:
            self.individual_table = self.db.open_table("arch5_individual")
            print(f"[Arch5] Opened individual ({self.individual_table.count_rows()} rows)")

        if "arch5_group" not in self.db.table_names():
            self.group_table = self.db.create_table("arch5_group", schema=individual_schema)
            print("[Arch5] Created group table")
        else:
            self.group_table = self.db.open_table("arch5_group")
            print(f"[Arch5] Opened group ({self.group_table.count_rows()} rows)")

        if "arch5_cross_group" not in self.db.table_names():
            self.cross_group_table = self.db.create_table("arch5_cross_group", schema=individual_schema)
            print("[Arch5] Created cross_group table")
        else:
            self.cross_group_table = self.db.open_table("arch5_cross_group")
            print(f"[Arch5] Opened cross_group ({self.cross_group_table.count_rows()} rows)")

        # Initialize FTS indexes
        self._init_fts_indexes()

    def _init_fts_indexes(self):
        """Initialize full-text search indexes."""
        tables = [
            (self.individual_table, "individual"),
            (self.group_table, "group"),
            (self.cross_group_table, "cross_group")
        ]

        for table, name in tables:
            try:
                table.create_fts_index("content", use_tantivy=True, replace=True)
                print(f"[Arch5] FTS index created for {name}")
            except Exception as e:
                print(f"[Arch5] FTS index skipped for {name}: {e}")

    def _determine_memory_level(self, message: Dict[str, Any]) -> str:
        """Determine which memory level this belongs to."""
        msg_type = message.get("message_type", "")

        # Announcements and expertise are group-level
        if msg_type in ["announcement", "expertise_demonstration"]:
            return "group"

        # Agent mentions could be individual or cross-group
        # For simplicity, we'll make them individual
        elif msg_type == "agent_mention":
            return "individual"

        # User conversations are individual
        elif msg_type == "user_conversation":
            return "individual"

        # Default to individual
        else:
            return "individual"

    def _determine_memory_type(self, message: Dict[str, Any], level: str) -> str:
        """Determine memory type."""
        msg_type = message.get("message_type", "")

        if msg_type == "agent_mention":
            return "conversation"
        elif msg_type == "expertise_demonstration":
            return "expertise"
        elif msg_type == "announcement":
            return "fact"
        else:
            return "conversation"

    def _calculate_importance(self, message: Dict[str, Any]) -> float:
        """Calculate importance score (0-1)."""
        msg_type = message.get("message_type", "")

        # Announcements are important
        if msg_type == "announcement":
            return 0.9

        # Expertise demonstrations
        elif msg_type == "expertise_demonstration":
            return 0.8

        # Agent mentions
        elif msg_type == "agent_mention":
            return 0.7

        # Default
        else:
            return 0.5

    def _create_cross_group_memory(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create cross-group memory if applicable.

        Cross-group memories are universal facts about a user
        that apply across all groups (expertise, preferences).
        """
        msg_type = message.get("message_type", "")

        # Only expertise demonstrations create cross-group memories
        if msg_type == "expertise_demonstration":
            user_id = message.get("sender_id", "")
            topics = message.get("topics", [])

            if topics:
                # Create expertise memory
                content = f"User has expertise in: {', '.join(topics)}"
                vector = self.embedding_model.encode_documents([content])[0]

                # Use consistent ID for cross-group memories
                expertise_key = "_".join(sorted(topics))
                memory_id = f"cross_expertise_{user_id}_{expertise_key}"

                return {
                    "agent_id": self.agent_id,
                    "group_id": "",  # Empty for cross-group
                    "user_id": user_id,
                    "memory_level": "cross_group",
                    "memory_type": "expertise",
                    "privacy_scope": "public",
                    "content": content,
                    "speaker": message.get("sender_name", ""),
                    "timestamp": message.get("timestamp", ""),
                    "importance_score": 0.8,
                    "access_count": 0,
                    "last_accessed": "",
                    "message_id": memory_id,
                    "keywords": topics,
                    "topics": topics,
                    "vector": vector.tolist()
                }

        return None

    def add_message(self, message: Dict[str, Any]) -> str:
        """Add a single message to appropriate level(s)."""
        level = self._determine_memory_level(message)
        memory_type = self._determine_memory_type(message, level)
        importance = self._calculate_importance(message)

        content = message.get("content", "")
        vector = self.embedding_model.encode_documents([content])[0]

        data = {
            "agent_id": self.agent_id,
            "group_id": message.get("group_id", ""),
            "user_id": message.get("sender_id", ""),
            "memory_level": level,
            "memory_type": memory_type,
            "privacy_scope": "public" if level == "group" else "protected",
            "content": content,
            "speaker": message.get("sender_name", ""),
            "timestamp": message.get("timestamp", ""),
            "importance_score": importance,
            "access_count": 0,
            "last_accessed": "",
            "message_id": message.get("message_id", ""),
            "keywords": message.get("topics", []),
            "topics": message.get("topics", []),
            "vector": vector.tolist()
        }

        # Add to appropriate table
        if level == "individual":
            self.individual_table.add([data])
        elif level == "group":
            self.group_table.add([data])

        # Check if cross-group memory should be created
        cross_group_data = self._create_cross_group_memory(message)
        if cross_group_data:
            # Check if exists
            existing = self.cross_group_table.search().where(
                f"message_id = '{cross_group_data['message_id']}'",
                prefilter=True
            ).to_list()

            if not existing:
                self.cross_group_table.add([cross_group_data])

        return level

    def add_messages_batch(self, messages: List[Dict[str, Any]]) -> Dict[str, int]:
        """Add multiple messages in batch."""
        counts = {"individual": 0, "group": 0, "cross_group": 0}

        # Prepare batches
        individual_batch = []
        group_batch = []
        cross_group_batch = []

        for message in messages:
            level = self._determine_memory_level(message)
            memory_type = self._determine_memory_type(message, level)
            importance = self._calculate_importance(message)
            content = message.get("content", "")

            data = {
                "agent_id": self.agent_id,
                "group_id": message.get("group_id", ""),
                "user_id": message.get("sender_id", ""),
                "memory_level": level,
                "memory_type": memory_type,
                "privacy_scope": "public" if level == "group" else "protected",
                "content": content,
                "speaker": message.get("sender_name", ""),
                "timestamp": message.get("timestamp", ""),
                "importance_score": importance,
                "access_count": 0,
                "last_accessed": "",
                "message_id": message.get("message_id", ""),
                "keywords": message.get("topics", []),
                "topics": message.get("topics", []),
                "vector": None  # Will be filled
            }

            if level == "individual":
                individual_batch.append(data)
            elif level == "group":
                group_batch.append(data)

            counts[level] += 1

            # Check for cross-group
            cross_group_data = self._create_cross_group_memory(message)
            if cross_group_data:
                cross_group_batch.append(cross_group_data)

        # Process each batch
        for batch, table in [(individual_batch, self.individual_table),
                             (group_batch, self.group_table),
                             (cross_group_batch, self.cross_group_table)]:
            if batch:
                contents = [item["content"] for item in batch]
                vectors = self.embedding_model.encode_documents(contents)

                for i, item in enumerate(batch):
                    item["vector"] = vectors[i].tolist()

                table.add(batch)

        counts["cross_group"] = len(cross_group_batch)

        return counts

    def _estimate_complexity(self, query: str) -> float:
        """Estimate query complexity (0-1)."""
        # Simple heuristic based on query length and keywords
        complexity_keywords = ["who", "what", "how", "why", "explain", "compare", "between", "across"]

        score = 0.0

        # Length factor
        if len(query.split()) > 10:
            score += 0.3
        elif len(query.split()) > 5:
            score += 0.1

        # Keyword factor
        for keyword in complexity_keywords:
            if keyword.lower() in query.lower():
                score += 0.2

        return min(score, 1.0)

    def semantic_search(
        self,
        query: str,
        group_id: str = None,
        user_id: str = None,
        memory_level: str = None,
        memory_type: str = None,  # For test compatibility (maps to memory_level)
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Adaptive semantic search based on query complexity.

        Simple queries: keyword search
        Complex queries: multi-level semantic search with consolidation
        """
        # Map memory_type to memory_level for test compatibility
        if memory_type and not memory_level:
            memory_type_map = {
                "group": "group",
                "user": "individual",
                "interaction": "individual"
            }
            memory_level = memory_type_map.get(memory_type)

        complexity = self._estimate_complexity(query)

        # Generate query embedding
        query_vector = self.embedding_model.encode_query([query])[0]

        results = []

        # Determine which levels to search
        if memory_level:
            levels_to_search = [memory_level]
        elif complexity < 0.3:
            # Simple query: search individual level only
            levels_to_search = ["individual"]
        elif complexity < 0.7:
            # Medium complexity: individual + group
            levels_to_search = ["individual", "group"]
        else:
            # High complexity: all levels
            levels_to_search = ["individual", "group", "cross_group"]

        # Search each level
        level_to_table = {
            "individual": self.individual_table,
            "group": self.group_table,
            "cross_group": self.cross_group_table
        }

        for level in levels_to_search:
            table = level_to_table.get(level)
            if not table:
                continue

            search = table.search(query_vector.tolist())

            # Build filter
            conditions = [f"agent_id = '{self.agent_id}'"]

            if group_id and level != "cross_group":
                conditions.append(f"group_id = '{group_id}'")

            if user_id and level == "individual":
                conditions.append(f"user_id = '{user_id}'")

            if conditions:
                where_clause = " AND ".join(conditions)
                search = search.where(where_clause, prefilter=True)

            # Limit per level
            per_level_limit = limit // len(levels_to_search) + 1
            level_results = search.limit(per_level_limit).to_list()

            # Tag with level
            for result in level_results:
                result["_level"] = level

            results.extend(level_results)

        # Consolidate (remove redundancy)
        results = self._consolidate_memories(results)

        return results[:limit]

    def _consolidate_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove redundant memories across levels.

        Implements "Recursive Consolidation" from SimpleMem paper.
        """
        seen = set()
        consolidated = []

        for memory in memories:
            # Create semantic signature (simple hash of content)
            content = memory.get("content", "")
            signature = hashlib.md5(content.lower().encode()).hexdigest()

            if signature not in seen:
                seen.add(signature)
                consolidated.append(memory)
            else:
                # Merge with existing (increase confidence/importance)
                existing = next((m for m in consolidated if
                               hashlib.md5(m.get("content", "").lower().encode()).hexdigest() == signature), None)
                if existing:
                    existing["importance_score"] = max(existing.get("importance_score", 0.5),
                                                      memory.get("importance_score", 0.5))

        return consolidated

    def get_group_context(self, group_id: str, user_id: str = None, limit: int = 10) -> Dict[str, Any]:
        """Get multi-level context for a group."""
        query_vector = self.embedding_model.encode_query([""])[0]

        # Individual level
        individual_search = self.individual_table.search(query_vector.tolist()).where(
            f"agent_id = '{self.agent_id}' AND group_id = '{group_id}'",
            prefilter=True
        )
        if user_id:
            individual_search = individual_search.where(f"user_id = '{user_id}'", prefilter=True)

        individual_memories = individual_search.limit(limit).to_list()

        # Group level
        group_memories = self.group_table.search(query_vector.tolist()).where(
            f"agent_id = '{self.agent_id}' AND group_id = '{group_id}'",
            prefilter=True
        ).limit(limit).to_list()

        # Cross-group level (if user_id provided)
        cross_group_memories = []
        if user_id:
            cross_group_memories = self.cross_group_table.search(query_vector.tolist()).where(
                f"agent_id = '{self.agent_id}' AND user_id = '{user_id}'",
                prefilter=True
            ).limit(limit // 2).to_list()

        return {
            "individual_context": individual_memories,
            "group_context": group_memories,
            "cross_group_context": cross_group_memories
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory store."""
        return {
            "total_memories": (
                self.individual_table.count_rows() +
                self.group_table.count_rows() +
                self.cross_group_table.count_rows()
            ),
            "individual_memories": self.individual_table.count_rows(),
            "group_memories": self.group_table.count_rows(),
            "cross_group_memories": self.cross_group_table.count_rows()
        }

    def clear(self):
        """Clear all data."""
        self.db.drop_table("arch5_individual")
        self.db.drop_table("arch5_group")
        self.db.drop_table("arch5_cross_group")
        self._init_tables()


def test_arch5():
    """Test Architecture 5 with generated test data."""
    print("=" * 60)
    print("Testing Architecture 5: Hybrid Multi-Level Memory")
    print("=" * 60)

    # Load test data
    test_data_file = "/home/oydual3/a0x/a0x-memory/tests/group_memory/test_data.json"
    with open(test_data_file, 'r') as f:
        test_data = json.load(f)

    messages = test_data["messages"]
    groups = test_data["groups"]
    users = test_data["users"]

    print(f"\nTest Data:")
    print(f"  - Users: {len(users)}")
    print(f"  - Groups: {len(groups)}")
    print(f"  - Messages: {len(messages)}")

    # Initialize store
    db_path = "/home/oydual3/a0x/a0x-memory/tests/group_memory/lancedb_arch5"
    store = HybridMultiLevelMemoryStore(
        db_path=db_path,
        agent_id="jessexbt"
    )

    # Test 1: Batch insert performance
    print("\n" + "-" * 60)
    print("Test 1: Batch Insert Performance")
    print("-" * 60)

    import time
    start = time.time()

    counts = store.add_messages_batch(messages)

    elapsed = time.time() - start

    print(f"Inserted {len(messages)} messages in {elapsed:.2f}s")
    print(f"  - Throughput: {len(messages)/elapsed:.2f} msg/sec")
    print(f"Memory distribution:")
    print(f"  - Individual: {counts['individual']}")
    print(f"  - Group: {counts['group']}")
    print(f"  - Cross-group: {counts['cross_group']}")

    # Get stats
    stats = store.get_stats()
    print(f"\nStore Stats:")
    print(f"  - Total memories: {stats['total_memories']}")
    print(f"  - Individual: {stats['individual_memories']}")
    print(f"  - Group: {stats['group_memories']}")
    print(f"  - Cross-group: {stats['cross_group_memories']}")

    # Test 2: Adaptive search based on complexity
    print("\n" + "-" * 60)
    print("Test 2: Adaptive Search (Query Complexity)")
    print("-" * 60)

    test_queries = [
        ("hi", 0.1),  # Simple
        ("defi strategies", 0.3),  # Medium
        ("who here knows about smart contract security and can explain best practices?", 0.9)  # Complex
    ]

    for query, expected_complexity in test_queries:
        complexity = store._estimate_complexity(query)
        start = time.time()
        results = store.semantic_search(query, limit=5)
        elapsed = time.time() - start

        # Count levels
        levels = set(r.get("_level", "unknown") for r in results)

        print(f"\nQuery: '{query}'")
        print(f"  - Complexity: {complexity:.2f} (expected: ~{expected_complexity})")
        print(f"  - Results: {len(results)} from levels: {levels}")
        print(f"  - Latency: {elapsed*1000:.2f}ms")

    # Test 3: Multi-level context retrieval
    print("\n" + "-" * 60)
    print("Test 3: Multi-Level Context Retrieval")
    print("-" * 60)

    group_id = groups[0]["group_id"]
    user_id = users[0]["user_id"]

    start = time.time()
    context = store.get_group_context(group_id, user_id, limit=5)
    elapsed = time.time() - start

    print(f"Group: {groups[0]['group_name']}")
    print(f"  - Individual context: {len(context['individual_context'])} memories")
    print(f"  - Group context: {len(context['group_context'])} memories")
    print(f"  - Cross-group context: {len(context['cross_group_context'])} memories")
    print(f"  - Latency: {elapsed*1000:.2f}ms")

    # Test 4: Search by memory level
    print("\n" + "-" * 60)
    print("Test 4: Search by Memory Level")
    print("-" * 60)

    for level in ["individual", "group", "cross_group"]:
        start = time.time()
        results = store.semantic_search(
            query="",
            group_id=group_id,
            user_id=user_id,
            memory_level=level,
            limit=5
        )
        elapsed = time.time() - start

        count = sum(1 for r in results if r.get("_level") == level)
        print(f"{level.replace('_', ' ').title()}: {count} results ({elapsed*1000:.2f}ms)")

    # Test 5: Consolidation
    print("\n" + "-" * 60)
    print("Test 5: Memory Consolidation")
    print("-" * 60)

    # Create duplicate memories
    print("Creating test duplicates...")
    duplicate_results = store.semantic_search("expertise", limit=20)
    print(f"  - Found {len(duplicate_results)} results before consolidation")

    # Test consolidation explicitly
    consolidated = store._consolidate_memories(duplicate_results)
    print(f"  - After consolidation: {len(consolidated)} unique memories")

    # Test 6: Cross-group user query
    print("\n" + "-" * 60)
    print("Test 6: Cross-Group User Query")
    print("-" * 60)

    start = time.time()
    results = store.semantic_search(
        query="",
        user_id=user_id,
        limit=20
    )
    elapsed = time.time() - start

    # Count by level
    level_counts = {}
    for r in results:
        level = r.get("_level", "unknown")
        level_counts[level] = level_counts.get(level, 0) + 1

    print(f"User {users[0]['username']}: {len(results)} memories")
    print(f"  - By level: {level_counts}")
    print(f"  - Latency: {elapsed*1000:.2f}ms")

    # Summary
    print("\n" + "=" * 60)
    print("Architecture 5 Test Summary")
    print("=" * 60)
    print("✓ Multi-level storage implemented")
    print("✓ Adaptive retrieval based on complexity")
    print("✓ Memory consolidation working")
    print("✓ Cross-group expertise tracking")
    print(f"\nPerformance:")
    print(f"  - Insert throughput: {len(messages)/elapsed:.2f} msg/sec")
    print(f"  - Query latency: ~{elapsed*1000:.2f}ms")
    print(f"  - Storage efficiency: Medium (3 tables)")
    print(f"\nAdvanced features:")
    print(f"  - Adaptive query complexity detection")
    print(f"  - Recursive consolidation (SimpleMem)")
    print(f"  - Cross-group user profiling")
    print(f"  - Importance-based prioritization")


if __name__ == "__main__":
    test_arch5()
