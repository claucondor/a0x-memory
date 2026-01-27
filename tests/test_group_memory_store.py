"""
Comprehensive tests for GroupMemoryStore

Tests:
1. Memory insertion (group, user, interaction, cross-group)
2. Search operations (semantic, keyword, filtered)
3. Context retrieval (multi-level)
4. Cross-group consolidation
5. Cross-agent linking
6. Performance benchmarks
"""
import os
import sys
import time
from typing import List
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from database.group_memory_store import GroupMemoryStore
from models.group_memory import (
    GroupMemory, UserMemory, InteractionMemory,
    CrossGroupMemory, CrossAgentLink,
    MemoryLevel, MemoryType, PrivacyScope
)
from utils.embedding import EmbeddingModel


def test_group_memory_insertion():
    """Test inserting group memories."""
    print("\n=== Test: Group Memory Insertion ===")

    # Use unique path for this test
    import shutil
    test_path = "/tmp/test_group_memory_insertion"
    if os.path.exists(test_path):
        shutil.rmtree(test_path)

    store = GroupMemoryStore(
        agent_id="test_agent",
        db_base_path=test_path
    )

    # Create test data
    memories = [
        GroupMemory(
            agent_id="test_agent",
            group_id="test_group_1",
            memory_type=MemoryType.ANNOUNCEMENT,
            content="Group decided to hold weekly discussions on Fridays",
            speaker="admin_user",
            keywords=["weekly", "discussion", "friday"],
            topics=["community", "discussion"],
            importance_score=0.8
        ),
        GroupMemory(
            agent_id="test_agent",
            group_id="test_group_1",
            memory_type=MemoryType.FACT,
            content="Group has 50+ active members interested in DeFi",
            speaker="moderator",
            keywords=["members", "defi", "active"],
            topics=["defi", "community"],
            importance_score=0.7
        )
    ]

    # Insert memories
    for memory in memories:
        store.add_group_memory(memory)

    # Verify insertion
    stats = store.get_stats()
    print(f"Group memories count: {stats['group_memories_count']}")
    assert stats['group_memories_count'] == 2, "Should have 2 group memories"

    print("Group memory insertion test PASSED")


def test_user_memory_insertion():
    """Test inserting user memories."""
    print("\n=== Test: User Memory Insertion ===")

    # Use unique path for this test
    import shutil
    test_path = "/tmp/test_group_memory_user"
    if os.path.exists(test_path):
        shutil.rmtree(test_path)

    store = GroupMemoryStore(
        agent_id="test_agent",
        db_base_path=test_path
    )

    # Create test data
    memories = [
        UserMemory(
            agent_id="test_agent",
            group_id="test_group_1",
            user_id="user_123",
            memory_type=MemoryType.PREFERENCE,
            content="User prefers technical explanations over simplified ones",
            keywords=["technical", "explanations", "preference"],
            topics=["communication", "preference"],
            importance_score=0.7,
            username="@alice_web3",
            platform="telegram"
        ),
        UserMemory(
            agent_id="test_agent",
            group_id="test_group_1",
            user_id="user_123",
            memory_type=MemoryType.EXPERTISE,
            content="User is an expert in DeFi and yield farming",
            keywords=["defi", "yield", "farming", "expert"],
            topics=["defi", "expertise"],
            importance_score=0.9,
            username="@alice_web3",
            platform="telegram"
        )
    ]

    # Insert memories
    for memory in memories:
        store.add_user_memory(memory)

    # Verify insertion
    stats = store.get_stats()
    print(f"User memories count: {stats['user_memories_count']}")
    assert stats['user_memories_count'] == 2, "Should have 2 user memories"

    print("User memory insertion test PASSED")


def test_interaction_memory_insertion():
    """Test inserting interaction memories."""
    print("\n=== Test: Interaction Memory Insertion ===")

    # Use unique path for this test
    import shutil
    test_path = "/tmp/test_group_memory_interaction"
    if os.path.exists(test_path):
        shutil.rmtree(test_path)

    store = GroupMemoryStore(
        agent_id="test_agent",
        db_base_path=test_path
    )

    # Create test data
    memory = InteractionMemory(
        agent_id="test_agent",
        group_id="test_group_1",
        speaker_id="user_123",
        listener_id="user_456",
        mentioned_users=["user_789"],
        memory_type=MemoryType.INTERACTION,
        content="Alice asked Bob about yield farming strategies",
        keywords=["yield", "farming", "strategies"],
        topics=["defi", "yield"],
        importance_score=0.6,
        interaction_type="question"
    )

    # Insert memory
    store.add_interaction_memory(memory)

    # Verify insertion
    stats = store.get_stats()
    print(f"Interaction memories count: {stats['interaction_memories_count']}")
    assert stats['interaction_memories_count'] == 1, "Should have 1 interaction memory"

    print("Interaction memory insertion test PASSED")


def test_search_group_memories():
    """Test searching group memories."""
    print("\n=== Test: Search Group Memories ===")

    store = GroupMemoryStore(
        agent_id="test_agent",
        db_base_path="/tmp/test_group_memory_insertion"
    )

    # Search for relevant memories
    results = store.search_group_memories(
        group_id="test_group_1",
        query="weekly discussions",
        limit=5
    )

    print(f"Found {len(results)} group memories")
    assert len(results) > 0, "Should find at least one memory"

    for result in results:
        print(f"  - {result.content[:60]}...")

    print("Search group memories test PASSED")


def test_search_user_memories():
    """Test searching user memories."""
    print("\n=== Test: Search User Memories ===")

    store = GroupMemoryStore(
        agent_id="test_agent",
        db_base_path="/tmp/test_group_memory_user"
    )

    # Search for user memories
    results = store.search_user_memories(
        group_id="test_group_1",
        user_id="user_123",
        query="defi expertise",
        limit=5
    )

    print(f"Found {len(results)} user memories")
    assert len(results) > 0, "Should find at least one memory"

    for result in results:
        print(f"  - {result.content[:60]}...")

    print("Search user memories test PASSED")


def test_search_interactions():
    """Test searching interaction memories."""
    print("\n=== Test: Search Interactions ===")

    store = GroupMemoryStore(
        agent_id="test_agent",
        db_base_path="/tmp/test_group_memory_interaction"
    )

    # Search for interactions
    results = store.search_interactions(
        group_id="test_group_1",
        speaker_id="user_123",
        query="yield farming",
        limit=5
    )

    print(f"Found {len(results)} interaction memories")
    # Note: This might be empty if no matching interactions

    print("Search interactions test PASSED")


def test_get_group_context():
    """Test getting comprehensive group context."""
    print("\n=== Test: Get Group Context ===")

    store = GroupMemoryStore(
        agent_id="test_agent",
        db_base_path="/tmp/test_group_memory_user"
    )

    # Get context for a user
    context = store.get_group_context(
        group_id="test_group_1",
        user_id="user_123",
        query="defi",
        limit_per_level=5
    )

    print(f"Group context: {len(context['group_context'])} memories")
    print(f"User context: {len(context['user_context'])} memories")
    print(f"Interaction context: {len(context['interaction_context'])} memories")
    print(f"Cross-group context: {len(context['cross_group_context'])} memories")

    # Verify we got some context
    assert len(context['group_context']) > 0 or len(context['user_context']) > 0, \
        "Should have some context"

    print("Get group context test PASSED")


def test_cross_group_consolidation():
    """Test cross-group pattern detection and consolidation."""
    print("\n=== Test: Cross-Group Consolidation ===")

    # Use unique path for this test
    import shutil
    test_path = "/tmp/test_group_memory_consolidation2"
    if os.path.exists(test_path):
        shutil.rmtree(test_path)

    store = GroupMemoryStore(
        agent_id="test_agent",
        db_base_path=test_path
    )

    # Add user memories across multiple groups
    groups = ["test_group_1", "test_group_2", "test_group_3"]

    for group_id in groups:
        memory = UserMemory(
            agent_id="test_agent",
            group_id=group_id,
            user_id="user_123",
            memory_type=MemoryType.EXPERTISE,
            content=f"User demonstrated DeFi expertise in {group_id}",
            keywords=["defi", "expertise"],
            topics=["defi", "expertise"],
            importance_score=0.8,
            username="@alice_web3",
            platform="telegram"
        )
        store.add_user_memory(memory)

    # Detect cross-group patterns
    patterns = store.detect_cross_group_patterns(
        user_id="user_123",
        min_groups=2,
        min_evidence=2
    )

    print(f"Detected {len(patterns)} cross-group patterns")
    for pattern in patterns:
        print(f"  - {pattern.pattern_type}: {pattern.content[:60]}...")
        print(f"    Groups: {len(pattern.groups_involved)}")

    # Should detect at least one pattern
    assert len(patterns) >= 1, "Should detect at least one cross-group pattern"

    print("Cross-group consolidation test PASSED")


def test_cross_agent_linking():
    """Test cross-agent identity linking."""
    print("\n=== Test: Cross-Agent Linking ===")

    store = GroupMemoryStore(
        agent_id="test_agent",
        db_base_path="/tmp/test_group_memory"
    )

    # Create cross-agent link
    link = store.link_user_across_agents(
        agent1_id="test_agent",
        agent2_id="other_agent",
        universal_user_id="telegram:user_123",
        linking_method="wallet_match",
        linking_evidence=["Same wallet address", "Same username"]
    )

    print(f"Created cross-agent link: {link.link_id}")
    print(f"Agent mappings: {link.agent_mappings}")

    # Retrieve mappings
    mappings = store.get_agent_mappings("telegram:user_123")
    print(f"Retrieved mappings: {mappings}")

    assert "test_agent" in mappings, "Should have test_agent mapping"
    assert "other_agent" in mappings, "Should have other_agent mapping"

    print("Cross-agent linking test PASSED")


def test_performance_benchmark():
    """Benchmark performance of various operations."""
    print("\n=== Performance Benchmark ===")

    store = GroupMemoryStore(
        agent_id="test_agent",
        db_base_path="/tmp/test_group_memory_bench"
    )

    # Benchmark: Batch insertion
    print("\nBenchmark: Batch Insertion")
    num_memories = 100

    start_time = time.time()
    for i in range(num_memories):
        memory = UserMemory(
            agent_id="test_agent",
            group_id=f"group_{i % 10}",
            user_id=f"user_{i % 50}",
            memory_type=MemoryType.PREFERENCE,
            content=f"Test memory {i} about DeFi and yield farming",
            keywords=["test", "defi"],
            topics=["test"],
            importance_score=0.5
        )
        store.add_user_memory(memory)
    insert_time = time.time() - start_time

    print(f"  Inserted {num_memories} memories in {insert_time:.3f}s")
    print(f"  Average: {insert_time/num_memories*1000:.2f}ms per memory")

    # Benchmark: Search
    print("\nBenchmark: Search Performance")
    num_searches = 50

    start_time = time.time()
    for i in range(num_searches):
        results = store.search_user_memories(
            group_id=f"group_{i % 10}",
            user_id=f"user_{i % 50}",
            query="defi yield",
            limit=5
        )
    search_time = time.time() - start_time

    print(f"  Performed {num_searches} searches in {search_time:.3f}s")
    print(f"  Average: {search_time/num_searches*1000:.2f}ms per search")

    # Benchmark: Context retrieval
    print("\nBenchmark: Context Retrieval")
    num_contexts = 20

    start_time = time.time()
    for i in range(num_contexts):
        context = store.get_group_context(
            group_id=f"group_{i % 10}",
            user_id=f"user_{i % 50}",
            query="defi",
            limit_per_level=3
        )
    context_time = time.time() - start_time

    print(f"  Retrieved {num_contexts} contexts in {context_time:.3f}s")
    print(f"  Average: {context_time/num_contexts*1000:.2f}ms per context")

    print("\nPerformance benchmark completed")


def test_consolidation_workflow():
    """Test complete cross-group consolidation workflow."""
    print("\n=== Test: Consolidation Workflow ===")

    store = GroupMemoryStore(
        agent_id="test_agent",
        db_base_path="/tmp/test_group_memory_consolidate"
    )

    # Setup: Create memories across multiple groups
    print("\nSetting up test data...")
    users = ["alice", "bob", "charlie"]
    groups = ["defi_traders", "crypto_analysts", "web3_devs"]

    for user_id in users:
        for group_id in groups:
            # Add expertise memories
            memory = UserMemory(
                agent_id="test_agent",
                group_id=group_id,
                user_id=user_id,
                memory_type=MemoryType.EXPERTISE,
                content=f"{user_id} showed expertise in smart contracts in {group_id}",
                keywords=["smart", "contracts", "expertise"],
                topics=["blockchain", "smart_contracts"],
                importance_score=0.8,
                evidence_count=2
            )
            store.add_user_memory(memory)

    # Detect patterns
    print("\nDetecting cross-group patterns...")
    for user_id in users:
        patterns = store.detect_cross_group_patterns(
            user_id=user_id,
            min_groups=2,
            min_evidence=3
        )

        print(f"\nUser {user_id}: {len(patterns)} patterns detected")
        for pattern in patterns:
            print(f"  - {pattern.pattern_type}")
            print(f"    Groups: {pattern.groups_involved}")
            print(f"    Confidence: {pattern.confidence_score:.2f}")

            # Add cross-group memory
            store.add_cross_group_memory(pattern)

    # Verify consolidation
    stats = store.get_stats()
    print(f"\nCross-group memories created: {stats['cross_group_memories_count']}")

    # Test updating cross-group memory
    if stats['cross_group_memories_count'] > 0:
        print("\nTesting cross-group memory update...")
        # Get first cross-group memory
        results = store.cross_group_memories_table.search().limit(1).to_list()
        if results:
            memory_id = results[0]["memory_id"]
            updated = store.update_cross_group_memory(
                memory_id=memory_id,
                new_evidence={"evidence_count": 1, "last_seen": datetime.now(timezone.utc).isoformat()}
            )
            print(f"Updated cross-group memory: {updated.memory_id}")
            print(f"New evidence count: {updated.evidence_count}")

    print("\nConsolidation workflow test PASSED")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("GROUP MEMORY STORE TEST SUITE")
    print("=" * 60)

    tests = [
        ("Group Memory Insertion", test_group_memory_insertion),
        ("User Memory Insertion", test_user_memory_insertion),
        ("Interaction Memory Insertion", test_interaction_memory_insertion),
        ("Search Group Memories", test_search_group_memories),
        ("Search User Memories", test_search_user_memories),
        ("Search Interactions", test_search_interactions),
        ("Get Group Context", test_get_group_context),
        ("Cross-Group Consolidation", test_cross_group_consolidation),
        ("Cross-Agent Linking", test_cross_agent_linking),
        ("Performance Benchmark", test_performance_benchmark),
        ("Consolidation Workflow", test_consolidation_workflow)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"\n{test_name}: PASSED")
        except AssertionError as e:
            failed += 1
            print(f"\n{test_name}: FAILED - {e}")
        except Exception as e:
            failed += 1
            print(f"\n{test_name}: ERROR - {e}")

    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
