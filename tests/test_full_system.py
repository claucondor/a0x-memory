"""
Full System Test - Unified Memory with Firestore Stateless Processing

Tests:
1. Firestore as buffer (stateless processing for Cloud Run)
2. 150+ messages from different users, groups, DMs
3. Same user in multiple groups
4. Search across all tables
5. OpenRouter LLM integration

Run: python3 tests/test_full_system.py
"""
import sys
import os
import random
import time
from datetime import datetime, timezone, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import SimpleMemSystem
import config

# Test configuration
TEST_AGENT_ID = "test_agent_full_system"
TEST_LANCEDB_PATH = "./test_lancedb_full"

# Simulated users
USERS = [
    {"id": "telegram:111111", "username": "@alice_dev", "name": "Alice"},
    {"id": "telegram:222222", "username": "@bob_trader", "name": "Bob"},
    {"id": "telegram:333333", "username": "@charlie_builder", "name": "Charlie"},
    {"id": "telegram:444444", "username": "@diana_analyst", "name": "Diana"},
    {"id": "telegram:555555", "username": "@eve_founder", "name": "Eve"},
    {"id": "xmtp:0xaaa111", "username": "alice.eth", "name": "Alice"},  # Same person, different platform
    {"id": "xmtp:0xbbb222", "username": "bob.base.eth", "name": "Bob"},
    {"id": "direct:user_001", "username": "dm_user_1", "name": "DirectUser1"},
    {"id": "direct:user_002", "username": "dm_user_2", "name": "DirectUser2"},
]

# Simulated groups
GROUPS = [
    {"id": "telegram_-100001", "name": "Base Builders", "platform": "telegram"},
    {"id": "telegram_-100002", "name": "DeFi Alpha", "platform": "telegram"},
    {"id": "telegram_-100003", "name": "NFT Collectors", "platform": "telegram"},
    {"id": "xmtp_group_001", "name": "XMTP Devs", "platform": "xmtp"},
]

# Message templates for different contexts
EXPERTISE_MESSAGES = [
    "I've been building smart contracts with Solidity for {years} years now",
    "My experience with {tech} has taught me that security audits are essential",
    "As a {role}, I focus on {focus}",
    "I specialize in {specialty} and have deployed {count} contracts",
    "Been working on {project_type} projects since {year}",
]

PREFERENCE_MESSAGES = [
    "I prefer {tool_a} over {tool_b} for {use_case}",
    "For testing, I always use {testing_tool}",
    "My go-to framework is {framework}",
    "I like to deploy on {chain} because of {reason}",
    "For frontend I always choose {frontend}",
]

FACT_MESSAGES = [
    "My wallet is {wallet}",
    "You can find my work at {url}",
    "I'm based in {location}",
    "My Twitter is @{handle}",
    "I work at {company}",
]

ANNOUNCEMENT_MESSAGES = [
    "Everyone, we're launching {project} next week!",
    "Important: The grant deadline is {date}",
    "Meeting scheduled for {day} at {time}",
    "New partnership with {partner} announced",
    "V2 is live on {chain}!",
]

CONVERSATION_MESSAGES = [
    "What do you think about {topic}?",
    "Has anyone tried {tool}?",
    "Looking for feedback on my {project_type}",
    "Can someone help with {issue}?",
    "Just deployed my first {contract_type}!",
    "gm everyone!",
    "Thoughts on the latest {event}?",
    "Anyone going to {conference}?",
    "This is bullish for {token}",
    "Check out this new {thing}",
]

INTERACTION_MESSAGES = [
    "@{user} I agree with your point about {topic}",
    "@{user} can you explain more about {thing}?",
    "Thanks @{user} for the help with {issue}!",
    "@{user} your {project} is amazing",
    "Hey @{user}, let's collaborate on {idea}",
]


def generate_message(msg_type: str, user: dict, mentioned_user: dict = None) -> str:
    """Generate a realistic message based on type."""
    templates = {
        "expertise": EXPERTISE_MESSAGES,
        "preference": PREFERENCE_MESSAGES,
        "fact": FACT_MESSAGES,
        "announcement": ANNOUNCEMENT_MESSAGES,
        "conversation": CONVERSATION_MESSAGES,
        "interaction": INTERACTION_MESSAGES,
    }

    template = random.choice(templates.get(msg_type, CONVERSATION_MESSAGES))

    # Fill in placeholders
    replacements = {
        "years": str(random.randint(1, 10)),
        "tech": random.choice(["Solidity", "Rust", "Move", "Cairo", "Vyper"]),
        "role": random.choice(["smart contract developer", "security researcher", "DeFi architect", "protocol engineer"]),
        "focus": random.choice(["gas optimization", "security", "scalability", "UX"]),
        "specialty": random.choice(["DeFi protocols", "NFT marketplaces", "DAOs", "bridges", "oracles"]),
        "count": str(random.randint(5, 50)),
        "project_type": random.choice(["DeFi", "NFT", "gaming", "social", "infrastructure"]),
        "year": str(random.randint(2019, 2024)),
        "tool_a": random.choice(["Hardhat", "Foundry", "Truffle", "Brownie"]),
        "tool_b": random.choice(["Remix", "Hardhat", "Truffle", "Brownie"]),
        "use_case": random.choice(["testing", "deployment", "debugging", "auditing"]),
        "testing_tool": random.choice(["Foundry", "Hardhat", "Echidna", "Slither"]),
        "framework": random.choice(["Next.js", "Remix", "Vite", "Astro"]),
        "chain": random.choice(["Base", "Optimism", "Arbitrum", "Polygon", "Ethereum"]),
        "reason": random.choice(["low fees", "fast finality", "great tooling", "strong community"]),
        "frontend": random.choice(["React", "Vue", "Svelte", "SolidJS"]),
        "wallet": f"0x{random.randint(1000000, 9999999):07x}...{random.randint(1000, 9999):04x}",
        "url": f"https://github.com/{user['username'].replace('@', '')}",
        "location": random.choice(["San Francisco", "NYC", "Berlin", "Singapore", "Remote"]),
        "handle": user['username'].replace('@', ''),
        "company": random.choice(["Coinbase", "a16z", "Paradigm", "OpenSea", "Uniswap Labs"]),
        "project": random.choice(["TokenSwap", "NFTMarket", "YieldFarm", "Bridge", "DAO"]),
        "date": f"2025-0{random.randint(1,9)}-{random.randint(10,28)}",
        "day": random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
        "time": f"{random.randint(9,17)}:00 UTC",
        "partner": random.choice(["Chainlink", "The Graph", "Alchemy", "Infura"]),
        "topic": random.choice(["L2 scaling", "account abstraction", "intent-based trading", "restaking"]),
        "tool": random.choice(["Tenderly", "Dune", "Nansen", "DeBank"]),
        "issue": random.choice(["gas optimization", "reentrancy", "frontrunning", "MEV"]),
        "contract_type": random.choice(["ERC-20", "ERC-721", "ERC-1155", "ERC-4626"]),
        "event": random.choice(["ETH ETF approval", "Base launch", "Dencun upgrade"]),
        "conference": random.choice(["ETHDenver", "Devcon", "ETHGlobal", "Consensus"]),
        "token": random.choice(["ETH", "BASE", "OP", "ARB"]),
        "thing": random.choice(["protocol", "dApp", "tool", "framework"]),
        "user": mentioned_user['username'] if mentioned_user else "@someone",
        "idea": random.choice(["DEX", "lending protocol", "NFT collection", "DAO"]),
    }

    for key, value in replacements.items():
        template = template.replace("{" + key + "}", value)

    return template


def generate_test_messages(count: int = 150) -> list:
    """Generate test messages for various scenarios."""
    messages = []
    base_time = datetime.now(timezone.utc) - timedelta(hours=2)

    # Distribution of message types
    type_weights = {
        "conversation": 40,
        "expertise": 15,
        "preference": 10,
        "fact": 10,
        "announcement": 10,
        "interaction": 15,
    }

    for i in range(count):
        # Pick random message type based on weights
        msg_type = random.choices(
            list(type_weights.keys()),
            weights=list(type_weights.values())
        )[0]

        # Pick random user
        user = random.choice(USERS)

        # Pick context (group or DM)
        is_dm = random.random() < 0.25  # 25% DMs

        if is_dm:
            group = None
            platform = user['id'].split(':')[0]
        else:
            group = random.choice(GROUPS)
            platform = group['platform']

        # For interactions, pick a mentioned user
        mentioned_user = None
        mentioned_users = []
        if msg_type == "interaction":
            mentioned_user = random.choice([u for u in USERS if u != user])
            mentioned_users = [mentioned_user['username']]

        # Generate message content
        content = generate_message(msg_type, user, mentioned_user)

        # Generate timestamp (spread over 2 hours)
        timestamp = (base_time + timedelta(seconds=i * 48)).isoformat()

        messages.append({
            "speaker": user['name'],
            "content": content,
            "timestamp": timestamp,
            "platform": platform,
            "group_id": group['id'] if group else None,
            "user_id": user['id'],
            "username": user['username'],
            "message_id": f"msg_{i:04d}",
            "is_reply": msg_type == "interaction",
            "mentioned_users": mentioned_users,
            "expected_type": msg_type,
        })

    return messages


def run_test():
    """Run the full system test."""
    print("=" * 70)
    print("FULL SYSTEM TEST - Unified Memory with Stateless Processing")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  - Agent ID: {TEST_AGENT_ID}")
    print(f"  - LanceDB Path: {TEST_LANCEDB_PATH}")
    print(f"  - Firestore Collection: {config.FIRESTORE_COLLECTION_PREFIX}")
    print(f"  - LLM Model: {config.LLM_MODEL}")
    print(f"  - Batch Trigger: {config.RECENT_BATCH_TRIGGER} messages")

    # Clean up previous test data
    import shutil
    if os.path.exists(TEST_LANCEDB_PATH):
        print(f"\nCleaning up previous test data...")
        shutil.rmtree(TEST_LANCEDB_PATH)

    # Initialize system
    print("\n" + "=" * 70)
    print("INITIALIZING SYSTEM")
    print("=" * 70)

    system = SimpleMemSystem(
        agent_id=TEST_AGENT_ID,
        db_path=TEST_LANCEDB_PATH,
        use_unified_store=True,
        enable_firestore=True
    )

    # Check Firestore connection
    if not system.firestore_enabled:
        print("\n⚠️  WARNING: Firestore not available. Test will use in-memory buffer.")
        print("    Set up Firebase credentials to test stateless processing.")

    # Generate test messages
    print("\n" + "=" * 70)
    print("GENERATING TEST MESSAGES")
    print("=" * 70)

    messages = generate_test_messages(160)  # Generate 160 messages

    # Count by type
    type_counts = {}
    group_counts = {}
    user_counts = {}
    for msg in messages:
        t = msg['expected_type']
        type_counts[t] = type_counts.get(t, 0) + 1

        g = msg['group_id'] or 'DM'
        group_counts[g] = group_counts.get(g, 0) + 1

        u = msg['username']
        user_counts[u] = user_counts.get(u, 0) + 1

    print(f"\nGenerated {len(messages)} messages:")
    print(f"\nBy type:")
    for t, c in sorted(type_counts.items()):
        print(f"  - {t}: {c}")

    print(f"\nBy group/context:")
    for g, c in sorted(group_counts.items()):
        print(f"  - {g}: {c}")

    print(f"\nBy user:")
    for u, c in sorted(user_counts.items()):
        print(f"  - {u}: {c}")

    # Send messages
    print("\n" + "=" * 70)
    print("SENDING MESSAGES (with stateless processing)")
    print("=" * 70)

    total_processed = 0
    total_memories = 0
    start_time = time.time()

    for i, msg in enumerate(messages):
        # Retry logic for transient failures
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = system.add_dialogue(
                    speaker=msg['speaker'],
                    content=msg['content'],
                    timestamp=msg['timestamp'],
                    platform=msg['platform'],
                    group_id=msg['group_id'],
                    user_id=msg['user_id'],
                    username=msg['username'],
                    message_id=msg['message_id'],
                    is_reply=msg['is_reply'],
                    mentioned_users=msg['mentioned_users'],
                    use_stateless_processing=True
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  Retry {attempt + 1}/{max_retries} after error: {str(e)[:50]}...")
                    time.sleep(2)
                else:
                    print(f"  Failed after {max_retries} attempts: {str(e)[:50]}...")
                    result = {"added": False, "processed": False, "memories_created": 0}

        if result.get('processed'):
            total_processed += 1
            total_memories += result.get('memories_created', 0)
            print(f"  [{i+1}/{len(messages)}] Batch processed! Memories created: {result.get('memories_created', 0)}")

        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(messages)} messages sent...")

    elapsed = time.time() - start_time
    print(f"\n✓ Sent {len(messages)} messages in {elapsed:.2f}s")
    print(f"  - Batch processing triggered: {total_processed} times")
    print(f"  - Total memories created: {total_memories}")

    # Process any remaining messages
    print("\n" + "=" * 70)
    print("PROCESSING REMAINING MESSAGES")
    print("=" * 70)

    # Process remaining for each group
    for group in GROUPS:
        result = system.process_pending(group_id=group['id'])
        if result.get('processed'):
            print(f"  Processed remaining for {group['name']}: {result.get('memories_created', 0)} memories")

    # Process remaining DMs
    for user in USERS:
        if 'direct' in user['id'] or 'telegram' in user['id']:
            result = system.process_pending(user_id=user['id'])
            if result.get('processed'):
                print(f"  Processed remaining DMs for {user['username']}: {result.get('memories_created', 0)} memories")

    # Get final stats
    print("\n" + "=" * 70)
    print("FINAL STATS")
    print("=" * 70)

    if hasattr(system.unified_store, 'get_stats'):
        stats = system.unified_store.get_stats()
        print(f"\nLanceDB Stats:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")

    # Test searches
    print("\n" + "=" * 70)
    print("TESTING SEARCH")
    print("=" * 70)

    search_queries = [
        ("smart contract expertise", None, None),
        ("DeFi protocols", GROUPS[0]['id'], None),
        ("wallet address", None, USERS[0]['id']),
        ("Base deployment", GROUPS[1]['id'], None),
        ("testing framework preference", None, None),
    ]

    for query, group_id, user_id in search_queries:
        print(f"\n📍 Query: '{query}'")
        if group_id:
            print(f"   Context: group={group_id}")
        if user_id:
            print(f"   Context: user={user_id}")

        # Use ask() which includes retrieval
        try:
            answer = system.ask(
                question=query,
                group_id=group_id,
                user_id=user_id,
                include_firestore_context=True
            )
            print(f"   Answer: {answer[:200]}..." if len(answer) > 200 else f"   Answer: {answer}")
        except Exception as e:
            print(f"   Error: {e}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    run_test()
