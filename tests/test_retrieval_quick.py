"""
Quick Retrieval Test - Validates the intelligent retrieval pipeline across all tables.

Tests:
1. Inserts ~20 messages (mix of DM and group) using add_dialogue with use_stateless_processing=True
2. Tests 3 queries using system.ask()
3. Prints full context and answer for each

Run: USE_LOCAL_STORAGE=true python tests/test_retrieval_quick.py
"""
import sys
import os
import time
import shutil

# Use local storage (no Firestore network calls)
os.environ["USE_LOCAL_STORAGE"] = "true"

# Add parent to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from main import SimpleMemSystem
from models.memory_entry import MemoryEntry, MemoryType, PrivacyScope
from models.group_memory import (
    GroupMemory, UserMemory, InteractionMemory,
    MemoryLevel, MemoryType as GMemoryType, PrivacyScope as GPrivacyScope
)
from datetime import datetime, timezone, timedelta
import uuid

# ============================================================
# Config
# ============================================================
AGENT_ID = "test_retrieval"
GROUP_ID = "-100099"
DB_PATH = "./test_lancedb_retrieval"

# Clean up any previous test data
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)


def create_system():
    """Create SimpleMemSystem for testing."""
    return SimpleMemSystem(
        agent_id=AGENT_ID,
        db_path=DB_PATH,
        clear_db=True,
        enable_firestore=True,
        use_unified_store=True,
        enable_planning=True,
        enable_reflection=False,  # Disable reflection for speed
        enable_parallel_retrieval=True,
    )


def insert_dm_messages(system: SimpleMemSystem):
    """Insert DM messages directly into the memory store."""
    print("\n" + "=" * 60)
    print("INSERTING DM MEMORIES (direct to store)")
    print("=" * 60)

    base_time = datetime.now(timezone.utc) - timedelta(days=3)

    dm_entries = [
        MemoryEntry(
            lossless_restatement="Alice is a senior Solidity developer with 5 years of experience building DeFi protocols on Ethereum and Base.",
            keywords=["Alice", "Solidity", "DeFi", "developer", "Base", "Ethereum"],
            timestamp=(base_time + timedelta(hours=1)).isoformat(),
            persons=["Alice"],
            topic="Alice's expertise",
            user_id="alice_123",
            memory_type=MemoryType.EXPERTISE,
            privacy_scope=PrivacyScope.PRIVATE,
            importance_score=0.9
        ),
        MemoryEntry(
            lossless_restatement="Bob mentioned he is working on a Telegram trading bot that uses the Uniswap V3 API for automated swaps.",
            keywords=["Bob", "Telegram", "trading bot", "Uniswap V3", "automated swaps"],
            timestamp=(base_time + timedelta(hours=2)).isoformat(),
            persons=["Bob"],
            topic="Bob's project",
            user_id="bob_456",
            memory_type=MemoryType.CONVERSATION,
            privacy_scope=PrivacyScope.PRIVATE,
            importance_score=0.8
        ),
        MemoryEntry(
            lossless_restatement="Carol asked about the Base grants program and whether NFT projects are eligible for funding.",
            keywords=["Carol", "Base grants", "NFT", "funding", "eligible"],
            timestamp=(base_time + timedelta(hours=3)).isoformat(),
            persons=["Carol"],
            topic="Base grants inquiry",
            user_id="carol_789",
            memory_type=MemoryType.CONVERSATION,
            privacy_scope=PrivacyScope.PRIVATE,
            importance_score=0.7
        ),
        MemoryEntry(
            lossless_restatement="Alice shared that she deployed a new lending protocol on Base mainnet with $2M TVL in the first week.",
            keywords=["Alice", "lending protocol", "Base mainnet", "TVL", "deployment"],
            timestamp=(base_time + timedelta(hours=5)).isoformat(),
            persons=["Alice"],
            topic="Alice's deployment",
            user_id="alice_123",
            memory_type=MemoryType.ANNOUNCEMENT,
            privacy_scope=PrivacyScope.PRIVATE,
            importance_score=0.9
        ),
        MemoryEntry(
            lossless_restatement="David said he prefers using Hardhat over Foundry for smart contract testing because of better plugin ecosystem.",
            keywords=["David", "Hardhat", "Foundry", "smart contract", "testing", "plugin"],
            timestamp=(base_time + timedelta(hours=6)).isoformat(),
            persons=["David"],
            topic="Development tools preference",
            user_id="david_101",
            memory_type=MemoryType.PREFERENCE,
            privacy_scope=PrivacyScope.PRIVATE,
            importance_score=0.6
        ),
    ]

    system.unified_store.add_memory_entries(dm_entries, user_id="alice_123")
    print(f"Inserted {len(dm_entries)} DM memories")


def insert_group_memories(system: SimpleMemSystem):
    """Insert group memories directly into the store."""
    print("\n" + "=" * 60)
    print("INSERTING GROUP MEMORIES (direct to store)")
    print("=" * 60)

    base_time = datetime.now(timezone.utc) - timedelta(days=2)
    now = datetime.now(timezone.utc).isoformat()

    # Group memories (shared knowledge)
    group_mems = [
        GroupMemory(
            agent_id=AGENT_ID,
            group_id=GROUP_ID,
            memory_level=MemoryLevel.GROUP,
            memory_type=GMemoryType.FACT,
            privacy_scope=GPrivacyScope.PUBLIC,
            content="The group is focused on building DeFi applications on Base chain, with several members working on lending and DEX protocols.",
            speaker="alice_dev",
            keywords=["DeFi", "Base", "lending", "DEX"],
            topics=["DeFi development"],
            importance_score=0.9,
            first_seen=now, last_seen=now, last_updated=now,
        ),
        GroupMemory(
            agent_id=AGENT_ID,
            group_id=GROUP_ID,
            memory_level=MemoryLevel.GROUP,
            memory_type=GMemoryType.FACT,
            privacy_scope=GPrivacyScope.PUBLIC,
            content="The group decided to use Foundry as the standard testing framework for all shared smart contracts.",
            speaker="henry_auditor",
            keywords=["Foundry", "testing", "smart contracts", "standard"],
            topics=["Development tools"],
            importance_score=0.8,
            first_seen=now, last_seen=now, last_updated=now,
        ),
        GroupMemory(
            agent_id=AGENT_ID,
            group_id=GROUP_ID,
            memory_level=MemoryLevel.GROUP,
            memory_type=GMemoryType.ANNOUNCEMENT,
            privacy_scope=GPrivacyScope.PUBLIC,
            content="Base hackathon starts next Monday with $50K in prizes. Submissions due in 2 weeks.",
            speaker="david_founder",
            keywords=["Base", "hackathon", "prizes", "submissions"],
            topics=["Events"],
            importance_score=0.95,
            first_seen=now, last_seen=now, last_updated=now,
        ),
        GroupMemory(
            agent_id=AGENT_ID,
            group_id=GROUP_ID,
            memory_level=MemoryLevel.GROUP,
            memory_type=GMemoryType.FACT,
            privacy_scope=GPrivacyScope.PUBLIC,
            content="Useful resource shared: Base documentation for gas optimization at docs.base.org/gas-optimization.",
            speaker="emma_researcher",
            keywords=["Base", "documentation", "gas optimization", "resource"],
            topics=["Resources"],
            importance_score=0.7,
            first_seen=now, last_seen=now, last_updated=now,
        ),
    ]
    system.unified_store.add_group_memories_batch(group_mems)
    print(f"Inserted {len(group_mems)} group memories")

    # User memories (about individual users in the group)
    user_mems = [
        UserMemory(
            agent_id=AGENT_ID,
            group_id=GROUP_ID,
            user_id="alice_dev",
            memory_level=MemoryLevel.INDIVIDUAL,
            memory_type=GMemoryType.EXPERTISE,
            privacy_scope=GPrivacyScope.PUBLIC,
            content="alice_dev is the most experienced Solidity developer in the group, often helps others debug smart contracts.",
            keywords=["alice_dev", "Solidity", "debugging", "experienced"],
            topics=["Expertise"],
            importance_score=0.9,
            first_seen=now, last_seen=now, last_updated=now,
            username="alice_dev", platform="telegram",
        ),
        UserMemory(
            agent_id=AGENT_ID,
            group_id=GROUP_ID,
            user_id="bob_trader",
            memory_level=MemoryLevel.INDIVIDUAL,
            memory_type=GMemoryType.EXPERTISE,
            privacy_scope=GPrivacyScope.PUBLIC,
            content="bob_trader focuses on MEV strategies and has built multiple arbitrage bots for Base chain.",
            keywords=["bob_trader", "MEV", "arbitrage", "bots", "Base"],
            topics=["Trading"],
            importance_score=0.8,
            first_seen=now, last_seen=now, last_updated=now,
            username="bob_trader", platform="telegram",
        ),
        UserMemory(
            agent_id=AGENT_ID,
            group_id=GROUP_ID,
            user_id="frank_newbie",
            memory_level=MemoryLevel.INDIVIDUAL,
            memory_type=GMemoryType.PREFERENCE,
            privacy_scope=GPrivacyScope.PUBLIC,
            content="frank_newbie is a complete beginner who just started learning Solidity last week. Asks many basic questions.",
            keywords=["frank_newbie", "beginner", "Solidity", "learning"],
            topics=["Onboarding"],
            importance_score=0.6,
            first_seen=now, last_seen=now, last_updated=now,
            username="frank_newbie", platform="telegram",
        ),
    ]
    system.unified_store.add_user_memories_batch(user_mems)
    print(f"Inserted {len(user_mems)} user memories")

    # Interaction memories
    interaction_mems = [
        InteractionMemory(
            agent_id=AGENT_ID,
            group_id=GROUP_ID,
            speaker_id="alice_dev",
            listener_id="frank_newbie",
            memory_level=MemoryLevel.GROUP,
            memory_type=GMemoryType.INTERACTION,
            privacy_scope=GPrivacyScope.PUBLIC,
            content="alice_dev helped frank_newbie understand how to write a basic ERC-20 token contract, spending 30 minutes explaining.",
            keywords=["ERC-20", "token", "teaching", "help"],
            topics=["Mentoring"],
            importance_score=0.8,
            first_seen=now, last_seen=now, last_updated=now,
            interaction_type="mentoring",
        ),
        InteractionMemory(
            agent_id=AGENT_ID,
            group_id=GROUP_ID,
            speaker_id="bob_trader",
            listener_id="alice_dev",
            memory_level=MemoryLevel.GROUP,
            memory_type=GMemoryType.INTERACTION,
            privacy_scope=GPrivacyScope.PUBLIC,
            content="bob_trader and alice_dev debated the security implications of flash loan attacks on lending protocols.",
            keywords=["flash loan", "security", "lending", "debate"],
            topics=["Security"],
            importance_score=0.85,
            first_seen=now, last_seen=now, last_updated=now,
            interaction_type="debate",
        ),
        InteractionMemory(
            agent_id=AGENT_ID,
            group_id=GROUP_ID,
            speaker_id="david_founder",
            listener_id="bob_trader",
            memory_level=MemoryLevel.GROUP,
            memory_type=GMemoryType.INTERACTION,
            privacy_scope=GPrivacyScope.PUBLIC,
            content="david_founder proposed a collaboration with bob_trader to build a MEV protection feature for his DeFi protocol.",
            keywords=["collaboration", "MEV protection", "DeFi", "proposal"],
            topics=["Collaboration"],
            importance_score=0.9,
            first_seen=now, last_seen=now, last_updated=now,
            interaction_type="collaboration",
        ),
    ]
    system.unified_store.add_interaction_memories_batch(interaction_mems)
    print(f"Inserted {len(interaction_mems)} interaction memories")


def test_queries(system: SimpleMemSystem):
    """Test 3 queries with ask() and print full context."""
    queries = [
        {
            "query": "Who in the group knows about Solidity and can help with smart contracts?",
            "group_id": GROUP_ID,
            "user_id": "frank_newbie",
            "description": "Group query - should find user memories about alice_dev + interactions"
        },
        {
            "query": "What is happening with the Base hackathon?",
            "group_id": GROUP_ID,
            "user_id": "david_founder",
            "description": "Group query - should find group event memory about hackathon"
        },
        {
            "query": "What projects is Alice working on?",
            "group_id": None,
            "user_id": "alice_123",
            "description": "DM query - should use full retrieve() pipeline for DM memories"
        },
    ]

    results = []
    for i, q in enumerate(queries, 1):
        print("\n" + "#" * 70)
        print(f"# QUERY {i}: {q['description']}")
        print(f"# Q: {q['query']}")
        print(f"# group_id={q['group_id']}, user_id={q['user_id']}")
        print("#" * 70)

        start = time.time()
        answer = system.ask(
            question=q["query"],
            group_id=q["group_id"],
            user_id=q["user_id"],
            include_firestore_context=False  # Skip firestore for speed
        )
        elapsed = time.time() - start

        results.append({
            "query": q["query"],
            "description": q["description"],
            "answer": answer,
            "time": elapsed
        })

        print(f"\n[Time: {elapsed:.1f}s]")

    return results


def print_summary(results):
    """Print summary of all test results."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_time = 0
    for i, r in enumerate(results, 1):
        print(f"\nQuery {i}: {r['description']}")
        print(f"  Q: {r['query']}")
        print(f"  A: {r['answer'][:200]}...")
        print(f"  Time: {r['time']:.1f}s")
        total_time += r["time"]

    print(f"\nTotal time: {total_time:.1f}s")
    print("=" * 70)


def main():
    start = time.time()

    print("Creating SimpleMemSystem...")
    system = create_system()

    # Insert test data directly into stores (fast, no LLM processing)
    insert_dm_messages(system)
    insert_group_memories(system)

    # Test queries
    results = test_queries(system)
    print_summary(results)

    total = time.time() - start
    print(f"\nTotal test time: {total:.1f}s")

    # Cleanup
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print("Cleaned up test data.")


if __name__ == "__main__":
    main()
