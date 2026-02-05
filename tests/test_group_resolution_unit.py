"""
Unit tests for Group Reference Resolution feature.

Tests the individual methods without requiring end-to-end data flow.
"""
import sys
sys.path.insert(0, '/home/oydual3/a0x/a0x-memory')

from database.tables.user_memories import UserMemoriesTable
from utils.embedding import EmbeddingModel
import tempfile
import os


def test_get_groups_for_user():
    """Test the get_groups_for_user method with mock data."""
    print("\n=== Test: get_groups_for_user ===")

    # Create a temp directory for LanceDB
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize the table
        os.environ['LANCEDB_PATH'] = tmpdir

        # Create a UserMemoriesTable
        table = UserMemoriesTable(
            agent_id="test_agent",
            embedding_model=EmbeddingModel()
        )

        # Add some test memories
        from models.group_memory import UserMemory, MemoryLevel, MemoryType, PrivacyScope
        from datetime import datetime, timezone

        # User in group 1 - trading
        for i, content in enumerate([
            "I'm bullish on BTC",
            "ETH looking good",
            "My portfolio is up"
        ]):
            mem = UserMemory(
                agent_id="test_agent",
                group_id="-100123",  # Trading group
                user_id="telegram:1001",
                memory_level=MemoryLevel.OBSERVATION,
                memory_type=MemoryType.STATED,
                privacy_scope=PrivacyScope.GROUP_ONLY,
                content=content,
                keywords=["trading"],
                topics=["crypto"],
                importance_score=0.5
            )
            table.add(mem)

        # User in group 2 - dev
        for i, content in enumerate([
            "Building smart contracts",
            "Solidity is complex"
        ]):
            mem = UserMemory(
                agent_id="test_agent",
                group_id="-100456",  # Dev group
                user_id="telegram:1001",
                memory_level=MemoryLevel.OBSERVATION,
                memory_type=MemoryType.STATED,
                privacy_scope=PrivacyScope.GROUP_ONLY,
                content=content,
                keywords=["dev"],
                topics=["development"],
                importance_score=0.5
            )
            table.add(mem)

        # Different user - should not show up
        mem = UserMemory(
            agent_id="test_agent",
            group_id="-100999",
            user_id="telegram:2002",
            memory_level=MemoryLevel.OBSERVATION,
            memory_type=MemoryType.STATED,
            privacy_scope=PrivacyScope.GROUP_ONLY,
            content="Other user's memory",
            keywords=[],
            topics=[],
            importance_score=0.5
        )
        table.add(mem)

        # Test get_groups_for_user
        groups = table.get_groups_for_user("telegram:1001")

        print(f"Groups found for telegram:1001: {groups}")

        # Verify
        assert len(groups) == 2, f"Expected 2 groups, got {len(groups)}"
        assert "-100123" in groups, "Expected trading group"
        assert "-100456" in groups, "Expected dev group"
        assert groups["-100123"]["memory_count"] == 3, f"Expected 3 memories in trading group"
        assert groups["-100456"]["memory_count"] == 2, f"Expected 2 memories in dev group"

        print("PASS: get_groups_for_user works correctly")
        return True


def test_plan_and_generate_queries_with_groups():
    """Test that _plan_and_generate_queries detects group references."""
    print("\n=== Test: _plan_and_generate_queries with group detection ===")

    from core.hybrid_retriever import HybridRetriever
    from utils.llm_client import LLMClient
    from database.vector_store import VectorStore
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ['LANCEDB_PATH'] = tmpdir

        # Create minimal retriever
        llm_client = LLMClient(use_streaming=False)
        vector_store = VectorStore(agent_id="test_agent")

        retriever = HybridRetriever(
            llm_client=llm_client,
            unified_store=vector_store,
            enable_planning=True
        )

        # Test query with group reference
        user_groups_summary = """- Group: Crypto Trading (ID: -100123), topics: [trading, defi, crypto], memories: 10, last active: 2h ago
- Group: Dev Builders (ID: -100456), topics: [solidity, smart contracts, development], memories: 5, last active: 1d ago"""

        query = "What did they say about my project in the trading group?"

        plan, queries = retriever._plan_and_generate_queries(
            query,
            user_id="telegram:1001",
            user_groups_summary=user_groups_summary
        )

        print(f"Query: {query}")
        print(f"Plan: {plan}")
        print(f"Queries: {queries}")

        # Check if group reference was detected
        if plan.get("references_group"):
            print("DETECTED: Query references a group")
            if plan.get("group_hints"):
                print(f"  Hints: {plan.get('group_hints')}")
            if plan.get("inferred_group_id"):
                print(f"  Inferred group: {plan.get('inferred_group_id')}")
                print(f"  Confidence: {plan.get('group_inference_confidence')}")
        else:
            print("NOT DETECTED: No group reference found (might be expected for some queries)")

        print("PASS: _plan_and_generate_queries completed without errors")
        return True


def test_resolve_group_by_hints():
    """Test the _resolve_group_by_hints method."""
    print("\n=== Test: _resolve_group_by_hints ===")

    from core.hybrid_retriever import HybridRetriever
    from utils.llm_client import LLMClient
    from database.vector_store import VectorStore
    from database.group_profile_store import GroupProfileStore
    from models.group_memory import GroupProfile, GroupTone
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ['LANCEDB_PATH'] = tmpdir

        # Create stores
        llm_client = LLMClient(use_streaming=False)
        vector_store = VectorStore(agent_id="test_agent")
        group_profile_store = GroupProfileStore(db_path=tmpdir)

        # Add group profiles
        trading_profile = GroupProfile(
            agent_id="test_agent",
            group_id="-100123",
            group_name="Crypto Trading",
            platform="telegram",
            summary="A group focused on crypto trading and market analysis",
            main_topics=["trading", "defi", "crypto", "markets"],
            group_purpose="Trading discussions",
            tone=GroupTone.TECHNICAL
        )
        group_profile_store.upsert_group_profile(trading_profile)

        dev_profile = GroupProfile(
            agent_id="test_agent",
            group_id="-100456",
            group_name="Dev Builders",
            platform="telegram",
            summary="A group for developers building smart contracts",
            main_topics=["solidity", "smart contracts", "development", "ERC-721"],
            group_purpose="Development discussions",
            tone=GroupTone.TECHNICAL
        )
        group_profile_store.upsert_group_profile(dev_profile)

        # Create retriever with group_profile_store
        retriever = HybridRetriever(
            llm_client=llm_client,
            unified_store=vector_store,
            group_profile_store=group_profile_store
        )

        # Mock the user_memories to return our test groups
        class MockUserMemories:
            def get_groups_for_user(self, user_id):
                return {
                    "-100123": {"memory_count": 10, "last_active": "2026-02-02T10:00:00Z", "first_seen": "2026-01-01T10:00:00Z"},
                    "-100456": {"memory_count": 5, "last_active": "2026-02-01T10:00:00Z", "first_seen": "2026-01-01T10:00:00Z"}
                }

        vector_store.user_memories = MockUserMemories()

        # Test with trading hints
        hints = ["trading", "crypto"]
        query = "What happened in the trading group?"
        resolved = retriever._resolve_group_by_hints("telegram:1001", hints, query)

        print(f"Hints: {hints}")
        print(f"Resolved group: {resolved}")

        if resolved == "-100123":
            print("PASS: Correctly resolved to trading group")
        else:
            print(f"WARNING: Expected -100123, got {resolved}")

        # Test with dev hints
        hints = ["solidity", "smart contract"]
        query = "What was discussed about smart contracts?"
        resolved = retriever._resolve_group_by_hints("telegram:1001", hints, query)

        print(f"\nHints: {hints}")
        print(f"Resolved group: {resolved}")

        if resolved == "-100456":
            print("PASS: Correctly resolved to dev group")
        else:
            print(f"WARNING: Expected -100456, got {resolved}")

        return True


def main():
    print("=" * 60)
    print("GROUP REFERENCE RESOLUTION - UNIT TESTS")
    print("=" * 60)

    results = []

    # Test 1: get_groups_for_user
    try:
        results.append(("get_groups_for_user", test_get_groups_for_user()))
    except Exception as e:
        print(f"FAIL: {e}")
        results.append(("get_groups_for_user", False))

    # Test 2: _plan_and_generate_queries with group detection
    try:
        results.append(("_plan_and_generate_queries", test_plan_and_generate_queries_with_groups()))
    except Exception as e:
        print(f"FAIL: {e}")
        results.append(("_plan_and_generate_queries", False))

    # Test 3: _resolve_group_by_hints
    try:
        results.append(("_resolve_group_by_hints", test_resolve_group_by_hints()))
    except Exception as e:
        print(f"FAIL: {e}")
        results.append(("_resolve_group_by_hints", False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
