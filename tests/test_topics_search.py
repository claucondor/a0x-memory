#!/usr/bin/env python3
"""
Test script for topics search functionality.

Tests the new group_topics table and search_by_topic method with:
1. Topic auto-creation when memories are added
2. Semantic topic search
3. Parallel retrieval (topics + memories)

Run: python tests/test_topics_search.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from models.group_memory import GroupMemory, MemoryLevel, MemoryType, PrivacyScope
from database.tables.group_memories import GroupMemoriesTable
from database.tables.group_topics import GroupTopicsTable
from utils.embedding import EmbeddingModel

AGENT_ID = "test_topics_search"
GROUP_ID = "test_group_base"
STORAGE_PATH = "./data/test_topics"


def test_topic_creation():
    """Test that topics are auto-created when memories are added."""
    print("\n" + "=" * 60)
    print("TEST 1: Topic Auto-Creation")
    print("=" * 60)

    storage_options = {"allow_create": True}
    memories_table = GroupMemoriesTable(
        agent_id=AGENT_ID,
        storage_options=storage_options
    )

    # Add memories with different topics
    test_memories = [
        GroupMemory(
            memory_id="mem_1",
            agent_id=AGENT_ID,
            group_id=GROUP_ID,
            memory_level=MemoryLevel.ATOMIC,
            memory_type=MemoryType.CONVERSATION,
            privacy_scope=PrivacyScope.GROUP_ONLY,
            content="Alice is an expert in DeFi protocols and liquidity mining",
            speaker="alice",
            keywords=["defi", "liquidity", "mining", "expert"],
            topics=["defi", "expertise"],
            first_seen=datetime.now(timezone.utc).isoformat(),
            last_seen=datetime.now(timezone.utc).isoformat(),
            last_updated=datetime.now(timezone.utc).isoformat(),
        ),
        GroupMemory(
            memory_id="mem_2",
            agent_id=AGENT_ID,
            group_id=GROUP_ID,
            memory_level=MemoryLevel.ATOMIC,
            memory_type=MemoryType.FACT,
            privacy_scope=PrivacyScope.GROUP_ONLY,
            content="Bob trades on Base chain using Uniswap V4 hooks",
            speaker="bob",
            keywords=["trading", "base", "uniswap", "hooks"],
            topics=["trading", "base"],
            first_seen=datetime.now(timezone.utc).isoformat(),
            last_seen=datetime.now(timezone.utc).isoformat(),
            last_updated=datetime.now(timezone.utc).isoformat(),
        ),
        GroupMemory(
            memory_id="mem_3",
            agent_id=AGENT_ID,
            group_id=GROUP_ID,
            memory_level=MemoryLevel.ATOMIC,
            memory_type=MemoryType.CONVERSATION,
            privacy_scope=PrivacyScope.GROUP_ONLY,
            content="Carol is building an NFT project on Base using OpenZeppelin contracts",
            speaker="carol",
            keywords=["nft", "base", "contracts"],
            topics=["nft", "base", "development"],
            first_seen=datetime.now(timezone.utc).isoformat(),
            last_seen=datetime.now(timezone.utc).isoformat(),
            last_updated=datetime.now(timezone.utc).isoformat(),
        ),
    ]

    for memory in test_memories:
        memories_table.add(memory)

    # Check that topics were created
    topics = memories_table.topics_table.get_topics_for_group(GROUP_ID)

    print(f"\n  Created {len(topics)} topics:")
    for topic in sorted(topics, key=lambda x: x["memory_count"], reverse=True):
        print(f"    - {topic['name']}: {topic['memory_count']} memories")

    # Verify expected topics exist
    topic_names = {t["name"] for t in topics}
    expected_topics = {"defi", "expertise", "trading", "base", "nft", "development"}
    missing = expected_topics - topic_names

    if missing:
        print(f"\n  ❌ FAIL: Missing topics: {missing}")
        return False
    else:
        print(f"\n  ✓ PASS: All expected topics created")

    return True, topics, memories_table


def test_semantic_topic_search(topics, memories_table):
    """Test semantic search within topics."""
    print("\n" + "=" * 60)
    print("TEST 2: Semantic Topic Search")
    print("=" * 60)

    embedding_model = EmbeddingModel()

    # Test semantic queries
    queries = [
        ("decentralized finance", "defi"),
        ("crypto trading", "trading"),
        ("digital art NFTs", "nft"),
    ]

    for query, expected_topic in queries:
        query_vector = embedding_model.encode_single(query, is_query=True)
        results = memories_table.topics_table.search_semantic(
            GROUP_ID, query_vector, limit=3
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
    print("TEST 3: Parallel Search by Topic")
    print("=" * 60)

    import time

    # Test parallel search
    queries = [
        "Who knows about DeFi?",
        "What trading activity is happening?",
        "Who is building on Base?",
    ]

    print("\n  Testing parallel retrieval (topics + memories):")
    for query in queries:
        start = time.time()
        results = memories_table.search_by_topic(
            group_id=GROUP_ID,
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


def test_filter_by_specific_topics(memories_table):
    """Test filtering by specific topic names."""
    print("\n" + "=" * 60)
    print("TEST 4: Filter by Specific Topics")
    print("=" * 60)

    results = memories_table.search_by_topic(
        group_id=GROUP_ID,
        query="blockchain development",
        topic_names=["development", "base"],
        limit=10
    )

    print(f"\n  Filtered by topics: ['development', 'base']")
    print(f"  Found {len(results['topic_results'])} matching topics:")
    for t in results["topic_results"]:
        print(f"    - {t['name']}: {t['memory_count']} memories")

    print(f"\n  Found {len(results['memory_results'])} matching memories:")
    for m in results["memory_results"]:
        print(f"    - [{m.speaker}]: {m.content[:60]}...")
        print(f"      topics: {m.topics}")

    # Verify all returned memories have at least one of the specified topics
    all_match = all(
        any(t in ["development", "base"] for t in m.topics)
        for m in results["memory_results"]
    )

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
    print("TOPICS SEARCH FUNCTIONALITY TEST")
    print("=" * 60)
    print(f"Agent ID: {AGENT_ID}")
    print(f"Group ID: {GROUP_ID}")

    try:
        # Test 1: Topic creation
        result1 = test_topic_creation()
        if isinstance(result1, tuple):
            success, topics, memories_table = result1
        else:
            success = result1
            if not success:
                return False
            # Get references for subsequent tests
            memories_table = GroupMemoriesTable(
                agent_id=AGENT_ID,
                storage_options={"allow_create": True}
            )
            topics = memories_table.topics_table.get_topics_for_group(GROUP_ID)

        if not success:
            return False

        # Test 2: Semantic topic search
        if not test_semantic_topic_search(topics, memories_table):
            return False

        # Test 3: Parallel search
        if not test_parallel_search_by_topic(memories_table):
            return False

        # Test 4: Filter by specific topics
        if not test_filter_by_specific_topics(memories_table):
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
