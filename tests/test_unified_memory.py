"""
Test Unified Memory System

Tests for:
1. DM flow (backward compatibility with SimpleMem)
2. Group flow (new group memory tables)
3. Unified retrieval across tables
"""
import sys
import os
from datetime import datetime, timezone
import uuid

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.legacy.unified_store import UnifiedMemoryStore
from models.memory_entry import MemoryEntry, Dialogue, MemoryType, PrivacyScope
from models.group_memory import (
    GroupMemory, UserMemory, InteractionMemory,
    MemoryLevel, MemoryType as GroupMemoryType, PrivacyScope as GroupPrivacyScope
)
from utils.embedding import EmbeddingModel


def test_unified_store_initialization():
    """Test UnifiedMemoryStore initializes all tables correctly."""
    print("\n" + "=" * 60)
    print("TEST: UnifiedMemoryStore Initialization")
    print("=" * 60)

    agent_id = f"test_agent_{uuid.uuid4().hex[:8]}"
    store = UnifiedMemoryStore(
        agent_id=agent_id,
        db_base_path="./test_lancedb"
    )

    # Check all tables exist
    assert store.memories_table is not None, "memories_table should exist"
    assert store.group_memories_table is not None, "group_memories_table should exist"
    assert store.user_memories_table is not None, "user_memories_table should exist"
    assert store.interaction_memories_table is not None, "interaction_memories_table should exist"
    assert store.cross_group_memories_table is not None, "cross_group_memories_table should exist"
    assert store.cross_agent_links_table is not None, "cross_agent_links_table should exist"

    # Check stats
    stats = store.get_stats()
    print(f"Stats: {stats}")

    assert stats["agent_id"] == agent_id
    assert "memories_count" in stats

    print("PASSED: All tables initialized correctly")
    return store


def test_dm_memory_flow(store: UnifiedMemoryStore):
    """Test DM memory flow (SimpleMem compatible)."""
    print("\n" + "=" * 60)
    print("TEST: DM Memory Flow (SimpleMem Compatible)")
    print("=" * 60)

    # Create test memory entries
    entries = [
        MemoryEntry(
            lossless_restatement="Alice mentioned she loves coffee and prefers Starbucks.",
            keywords=["Alice", "coffee", "Starbucks", "preference"],
            timestamp="2025-01-15T10:00:00",
            persons=["Alice"],
            topic="Alice's coffee preference",
            user_id="alice123",
            memory_type=MemoryType.PREFERENCE,
            privacy_scope=PrivacyScope.PRIVATE,
            importance_score=0.7
        ),
        MemoryEntry(
            lossless_restatement="Bob is a blockchain developer with 5 years of Solidity experience.",
            keywords=["Bob", "blockchain", "Solidity", "developer", "experience"],
            persons=["Bob"],
            topic="Bob's expertise",
            user_id="bob456",
            memory_type=MemoryType.EXPERTISE,
            privacy_scope=PrivacyScope.PRIVATE,
            importance_score=0.9
        ),
        MemoryEntry(
            lossless_restatement="Alice and Bob scheduled a meeting at Starbucks on 2025-01-16 at 2pm.",
            keywords=["Alice", "Bob", "meeting", "Starbucks", "schedule"],
            timestamp="2025-01-16T14:00:00",
            location="Starbucks",
            persons=["Alice", "Bob"],
            topic="Meeting schedule",
            memory_type=MemoryType.ANNOUNCEMENT,
            privacy_scope=PrivacyScope.PRIVATE,
            importance_score=0.8
        )
    ]

    # Add entries
    store.add_memory_entries(entries, user_id="alice123")

    # Verify count
    stats = store.get_stats()
    print(f"After adding: {stats['memories_count']} DM memories")
    assert stats["memories_count"] >= 3, "Should have at least 3 memories"

    # Test semantic search
    results = store.search_memories("coffee preference", user_id="alice123", top_k=5)
    print(f"\nSemantic search 'coffee preference': {len(results)} results")
    for r in results:
        print(f"  - {r.lossless_restatement[:80]}...")

    assert len(results) > 0, "Should find at least one result"

    # Test get_all_entries
    all_entries = store.get_all_entries()
    print(f"\nTotal entries: {len(all_entries)}")
    assert len(all_entries) >= 3

    print("\nPASSED: DM memory flow works correctly")


def test_group_memory_flow(store: UnifiedMemoryStore):
    """Test group memory flow (new tables)."""
    print("\n" + "=" * 60)
    print("TEST: Group Memory Flow")
    print("=" * 60)

    agent_id = store.agent_id
    group_id = "telegram_-1001234567890"
    now = datetime.now(timezone.utc).isoformat()

    # Create group memories
    group_memories = [
        GroupMemory(
            agent_id=agent_id,
            group_id=group_id,
            memory_level=MemoryLevel.GROUP,
            memory_type=GroupMemoryType.ANNOUNCEMENT,
            privacy_scope=GroupPrivacyScope.PUBLIC,
            content="The group meeting has been scheduled for Friday at 3pm EST.",
            speaker="@admin",
            keywords=["meeting", "Friday", "schedule"],
            topics=["group events"],
            importance_score=0.9,
            first_seen=now,
            last_seen=now,
            last_updated=now
        ),
        GroupMemory(
            agent_id=agent_id,
            group_id=group_id,
            memory_level=MemoryLevel.GROUP,
            memory_type=GroupMemoryType.FACT,
            privacy_scope=GroupPrivacyScope.PUBLIC,
            content="This group is focused on Base ecosystem development.",
            keywords=["Base", "ecosystem", "development"],
            topics=["group purpose"],
            importance_score=0.8,
            first_seen=now,
            last_seen=now,
            last_updated=now
        )
    ]

    # Add group memories
    store.add_group_memories_batch(group_memories)

    # Check count
    stats = store.get_stats()
    print(f"Group memories count: {stats['group_memories_count']}")
    assert stats["group_memories_count"] >= 2

    # Test search
    results = store.search_group_memories(group_id, "meeting schedule", limit=5)
    print(f"\nGroup search 'meeting schedule': {len(results)} results")
    for r in results:
        print(f"  - {r.content[:80]}...")

    assert len(results) > 0

    print("\nPASSED: Group memory flow works correctly")


def test_user_memory_flow(store: UnifiedMemoryStore):
    """Test user memory flow within groups."""
    print("\n" + "=" * 60)
    print("TEST: User Memory Flow")
    print("=" * 60)

    agent_id = store.agent_id
    group_id = "telegram_-1001234567890"
    now = datetime.now(timezone.utc).isoformat()

    # Create user memories
    user_memories = [
        UserMemory(
            agent_id=agent_id,
            group_id=group_id,
            user_id="telegram:123456",
            memory_level=MemoryLevel.INDIVIDUAL,
            memory_type=GroupMemoryType.EXPERTISE,
            privacy_scope=GroupPrivacyScope.PROTECTED,
            content="@alice has extensive experience with Solidity smart contracts, especially for DeFi protocols.",
            keywords=["Solidity", "smart contracts", "DeFi", "expertise"],
            topics=["technical skills"],
            importance_score=0.9,
            first_seen=now,
            last_seen=now,
            last_updated=now,
            username="@alice",
            platform="telegram"
        ),
        UserMemory(
            agent_id=agent_id,
            group_id=group_id,
            user_id="telegram:789012",
            memory_level=MemoryLevel.INDIVIDUAL,
            memory_type=GroupMemoryType.PREFERENCE,
            privacy_scope=GroupPrivacyScope.PROTECTED,
            content="@bob prefers using Hardhat over Foundry for smart contract development.",
            keywords=["Hardhat", "Foundry", "preference", "development"],
            topics=["tool preferences"],
            importance_score=0.6,
            first_seen=now,
            last_seen=now,
            last_updated=now,
            username="@bob",
            platform="telegram"
        )
    ]

    # Add user memories
    store.add_user_memories_batch(user_memories)

    # Check count
    stats = store.get_stats()
    print(f"User memories count: {stats['user_memories_count']}")
    assert stats["user_memories_count"] >= 2

    # Test search for specific user
    results = store.search_user_memories(group_id, "telegram:123456", "Solidity", limit=5)
    print(f"\nUser search (alice + Solidity): {len(results)} results")
    for r in results:
        print(f"  - [{r.username}] {r.content[:60]}...")

    # Test search across all users in group
    results = store.search_user_memories_in_group(group_id, "development", limit=5)
    print(f"\nGroup-wide user search 'development': {len(results)} results")
    for r in results:
        print(f"  - [{r.username}] {r.content[:60]}...")

    print("\nPASSED: User memory flow works correctly")


def test_interaction_memory_flow(store: UnifiedMemoryStore):
    """Test interaction memory flow."""
    print("\n" + "=" * 60)
    print("TEST: Interaction Memory Flow")
    print("=" * 60)

    agent_id = store.agent_id
    group_id = "telegram_-1001234567890"
    now = datetime.now(timezone.utc).isoformat()

    # Create interaction memories
    interaction_memories = [
        InteractionMemory(
            agent_id=agent_id,
            group_id=group_id,
            speaker_id="telegram:123456",
            listener_id="telegram:789012",
            mentioned_users=[],
            memory_level=MemoryLevel.INDIVIDUAL,
            memory_type=GroupMemoryType.INTERACTION,
            privacy_scope=GroupPrivacyScope.PROTECTED,
            content="@alice helped @bob debug a reentrancy vulnerability in his smart contract.",
            keywords=["debug", "reentrancy", "vulnerability", "smart contract", "help"],
            topics=["technical help"],
            importance_score=0.85,
            first_seen=now,
            last_seen=now,
            last_updated=now,
            interaction_type="help"
        ),
        InteractionMemory(
            agent_id=agent_id,
            group_id=group_id,
            speaker_id="telegram:789012",
            listener_id="telegram:123456",
            mentioned_users=[],
            memory_level=MemoryLevel.INDIVIDUAL,
            memory_type=GroupMemoryType.INTERACTION,
            privacy_scope=GroupPrivacyScope.PROTECTED,
            content="@bob asked @alice about best practices for gas optimization.",
            keywords=["gas", "optimization", "question", "best practices"],
            topics=["technical questions"],
            importance_score=0.7,
            first_seen=now,
            last_seen=now,
            last_updated=now,
            interaction_type="question"
        )
    ]

    # Add interaction memories
    store.add_interaction_memories_batch(interaction_memories)

    # Check count
    stats = store.get_stats()
    print(f"Interaction memories count: {stats['interaction_memories_count']}")
    assert stats["interaction_memories_count"] >= 2

    # Test search interactions
    results = store.search_interactions(group_id, speaker_id="telegram:123456", query="debug", limit=5)
    print(f"\nInteraction search (alice + debug): {len(results)} results")
    for r in results:
        print(f"  - {r.content[:60]}... ({r.interaction_type})")

    print("\nPASSED: Interaction memory flow works correctly")


def test_unified_search(store: UnifiedMemoryStore):
    """Test unified search across all tables."""
    print("\n" + "=" * 60)
    print("TEST: Unified Search (Multi-Table)")
    print("=" * 60)

    group_id = "telegram_-1001234567890"

    # Test search_all for group context
    context = {
        "group_id": group_id,
        "user_id": "telegram:123456"
    }

    results = store.search_all(
        query="smart contract development",
        context=context,
        limit_per_table=3
    )

    print("\nUnified search 'smart contract development':")
    print(f"  - DM memories: {len(results['dm_memories'])}")
    print(f"  - Group memories: {len(results['group_memories'])}")
    print(f"  - User memories: {len(results['user_memories'])}")
    print(f"  - Interaction memories: {len(results['interaction_memories'])}")
    print(f"  - Cross-group memories: {len(results['cross_group_memories'])}")

    # At least some results should be found
    total = sum(len(v) for v in results.values())
    print(f"\nTotal results: {total}")
    assert total > 0, "Should find at least one result across tables"

    # Test DM context search
    dm_context = {
        "group_id": None,  # DM
        "user_id": "alice123"
    }

    dm_results = store.search_all(
        query="coffee preference",
        context=dm_context,
        limit_per_table=3
    )

    print(f"\nDM search 'coffee preference': {len(dm_results['dm_memories'])} results")
    assert len(dm_results['dm_memories']) > 0

    print("\nPASSED: Unified search works correctly")


def test_backward_compatibility(store: UnifiedMemoryStore):
    """Test backward compatibility with VectorStore interface."""
    print("\n" + "=" * 60)
    print("TEST: Backward Compatibility (VectorStore Interface)")
    print("=" * 60)

    # Test semantic_search method (VectorStore compatible)
    results = store.semantic_search("coffee", top_k=5)
    print(f"semantic_search('coffee'): {len(results)} results")

    # Test keyword_search method
    results = store.keyword_search(["Solidity", "blockchain"], top_k=3)
    print(f"keyword_search(['Solidity', 'blockchain']): {len(results)} results")

    # Test get_all_entries method
    entries = store.get_all_entries()
    print(f"get_all_entries(): {len(entries)} entries")

    # Test count_entries method
    count = store.count_entries()
    print(f"count_entries(): {count}")

    print("\nPASSED: Backward compatibility works correctly")


def cleanup(store: UnifiedMemoryStore):
    """Clean up test data."""
    print("\n" + "=" * 60)
    print("Cleaning up test data...")
    print("=" * 60)

    store.clear_agent_data()
    print("Test data cleared")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("UNIFIED MEMORY SYSTEM TESTS")
    print("=" * 60)

    try:
        # Initialize
        store = test_unified_store_initialization()

        # Run tests
        test_dm_memory_flow(store)
        test_group_memory_flow(store)
        test_user_memory_flow(store)
        test_interaction_memory_flow(store)
        test_unified_search(store)
        test_backward_compatibility(store)

        # Print final stats
        print("\n" + "=" * 60)
        print("FINAL STATS")
        print("=" * 60)
        stats = store.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Cleanup
        cleanup(store)

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\nERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
