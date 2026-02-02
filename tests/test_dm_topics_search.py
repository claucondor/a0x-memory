#!/usr/bin/env python3
"""
Test script for DM topics search functionality.

Tests the new dm_topics table and search_by_topic method with:
1. Topic auto-creation when DM memories are added
2. Semantic topic search
3. Parallel retrieval (topics + memories)

Run: python3 tests/test_dm_topics_search.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from models.memory_entry import MemoryEntry, MemoryType, PrivacyScope
from database.tables.dm_memories import DMMemoriesTable
from database.tables.dm_topics import DMTopicsTable
from utils.embedding import EmbeddingModel

AGENT_ID = "test_dm_topics_search"
USER_ID = "test_user_telegram"
STORAGE_PATH = "./data/test_dm_topics"


def test_topic_creation():
    """Test that topics are auto-created when DM memories are added."""
    print("\n" + "=" * 60)
    print("TEST 1: DM Topic Auto-Creation")
    print("=" * 60)

    storage_options = {"allow_create": True}
    memories_table = DMMemoriesTable(
        agent_id=AGENT_ID,
        storage_options=storage_options
    )

    # Add DM memories with different topics
    test_entries = [
        MemoryEntry(
            entry_id="entry_1",
            lossless_restatement="Alice is an expert in DeFi protocols and yield farming on Base",
            keywords=["defi", "yield", "farming", "base", "expert"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            topic="defi",
            user_id=USER_ID,
            username="alice",
            platform="telegram",
            memory_type=MemoryType.CONVERSATION,
            privacy_scope=PrivacyScope.PRIVATE,
            importance_score=0.8,
        ),
        MemoryEntry(
            entry_id="entry_2",
            lossless_restatement="Bob actively trades crypto on Base using Uniswap V4",
            keywords=["trading", "crypto", "base", "uniswap"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            topic="trading",
            user_id=USER_ID,
            username="bob",
            platform="telegram",
            memory_type=MemoryType.FACT,
            privacy_scope=PrivacyScope.PRIVATE,
            importance_score=0.7,
        ),
        MemoryEntry(
            entry_id="entry_3",
            lossless_restatement="Carol is building NFT projects using ERC-721 and OpenZeppelin",
            keywords=["nft", "erc721", "openzeppelin", "contracts"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            topic="nft",
            user_id=USER_ID,
            username="carol",
            platform="telegram",
            memory_type=MemoryType.CONVERSATION,
            privacy_scope=PrivacyScope.PRIVATE,
            importance_score=0.75,
        ),
        MemoryEntry(
            entry_id="entry_4",
            lossless_restatement="Dave knows a lot about Solidity smart contract development",
            keywords=["solidity", "smart", "contracts", "development"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            topic="defi",  # Same topic as entry_1, should increment count
            user_id=USER_ID,
            username="dave",
            platform="telegram",
            memory_type=MemoryType.EXPERTISE,
            privacy_scope=PrivacyScope.PRIVATE,
            importance_score=0.9,
        ),
    ]

    memories_table.add_batch(test_entries, user_id=USER_ID)

    # Check that topics were created
    topics = memories_table.topics_table.get_topics_for_user(USER_ID)

    print(f"\n  Created {len(topics)} topics:")
    for topic in sorted(topics, key=lambda x: x["memory_count"], reverse=True):
        print(f"    - {topic['name']}: {topic['memory_count']} memories")

    # Verify expected topics exist
    topic_names = {t["name"] for t in topics}
    expected_topics = {"defi", "trading", "nft"}
    missing = expected_topics - topic_names

    if missing:
        print(f"\n  ❌ FAIL: Missing topics: {missing}")
        return False

    # Verify defi has 2 memories
    defi_topic = next((t for t in topics if t["name"] == "defi"), None)
    if defi_topic and defi_topic["memory_count"] == 2:
        print(f"\n  ✓ PASS: All expected topics created with correct counts")
        return True, topics, memories_table
    else:
        print(f"\n  ❌ FAIL: Topic count incorrect (defi should have 2, has {defi_topic['memory_count'] if defi_topic else 0})")
        return False


def test_semantic_topic_search(topics, memories_table):
    """Test semantic search within DM topics."""
    print("\n" + "=" * 60)
    print("TEST 2: Semantic DM Topic Search")
    print("=" * 60)

    embedding_model = EmbeddingModel()

    # Test semantic queries
    queries = [
        ("decentralized finance", "defi"),
        ("crypto trading markets", "trading"),
        ("digital collectibles art", "nft"),
    ]

    for query, expected_topic in queries:
        query_vector = embedding_model.encode_single(query, is_query=True)
        results = memories_table.topics_table.search_semantic(
            USER_ID, query_vector, limit=3
        )

        print(f"\n  Query: '{query}'")
        print(f"  Top 3 topics:")
        found_expected = False
        for r in results[:3]:
            match = " ✓" if r["name"] == expected_topic else ""
            print(f"    - {r['name']}: similarity={r.get('_distance', 0):.3f}{match}")
            if r["name"] == expected_topic:
                found_expected = True

        if not found_expected:
            print(f"    ⚠ Expected '{expected_topic}' not in top results")

    print(f"\n  ✓ PASS: Semantic topic search working")
    return True


def test_parallel_search_by_topic(memories_table):
    """Test parallel search_by_topic method."""
    print("\n" + "=" * 60)
    print("TEST 3: Parallel DM Search by Topic")
    print("=" * 60)

    import time

    # Test parallel search
    queries = [
        "What does Alice know about?",
        "Tell me about trading activity",
        "Who is building NFTs?",
    ]

    print("\n  Testing parallel retrieval (topics + memories):")
    for query in queries:
        start = time.time()
        results = memories_table.search_by_topic(
            user_id=USER_ID,
            query=query,
            limit=5
        )
        elapsed = time.time() - start

        print(f"\n  Query: '{query}'")
        print(f"  Time: {elapsed*1000:.1f}ms")
        print(f"  Results: {len(results['topic_results'])} topics, {len(results['memory_results'])} memories")

        # Show top topics
        if results["topic_results"]:
            print(f"  Top topics:")
            for t in results["topic_results"][:3]:
                print(f"    - {t['name']}: {t['memory_count']} memories")

    print(f"\n  ✓ PASS: Parallel search working")
    return True


def test_filter_by_specific_topic(memories_table):
    """Test filtering by specific topic name."""
    print("\n" + "=" * 60)
    print("TEST 4: Filter by Specific Topic")
    print("=" * 60)

    results = memories_table.search_by_topic(
        user_id=USER_ID,
        query="blockchain development",
        topic_name="defi",
        limit=10
    )

    print(f"\n  Filtered by topic: 'defi'")
    print(f"  Found {len(results['topic_results'])} matching topics:")
    for t in results["topic_results"]:
        print(f"    - {t['name']}: {t['memory_count']} memories")

    print(f"\n  Found {len(results['memory_results'])} matching memories:")
    for m in results["memory_results"]:
        print(f"    - [{m.username}]: {m.lossless_restatement[:60]}...")
        print(f"      topic: {m.topic}")

    # Verify all returned memories have the specified topic
    all_match = all(m.topic == "defi" for m in results["memory_results"])

    if all_match:
        print(f"\n  ✓ PASS: Topic filter working correctly")
        return True
    else:
        print(f"\n  ❌ FAIL: Some memories don't match topic filter")
        return False


def cleanup():
    """Clean up test data."""
    print("\n" + "=" * 60)
    print("CLEANUP")
    print("=" * 60)

    try:
        import shutil
        if os.path.exists(STORAGE_PATH):
            shutil.rmtree(STORAGE_PATH)
            print("  Test data cleaned up")
    except Exception as e:
        print(f"  Cleanup error: {e}")


def run_test():
    """Run all tests."""
    print("=" * 60)
    print("DM TOPICS SEARCH FUNCTIONALITY TEST")
    print("=" * 60)
    print(f"Agent ID: {AGENT_ID}")
    print(f"User ID: {USER_ID}")

    try:
        # Test 1: Topic creation
        result1 = test_topic_creation()
        if isinstance(result1, tuple):
            success, topics, memories_table = result1
        else:
            success = result1
            if not success:
                return False
            memories_table = DMMemoriesTable(
                agent_id=AGENT_ID,
                storage_options={"allow_create": True}
            )
            topics = memories_table.topics_table.get_topics_for_user(USER_ID)

        if not success:
            return False

        # Test 2: Semantic topic search
        if not test_semantic_topic_search(topics, memories_table):
            return False

        # Test 3: Parallel search
        if not test_parallel_search_by_topic(memories_table):
            return False

        # Test 4: Filter by specific topic
        if not test_filter_by_specific_topic(memories_table):
            return False

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup()


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
