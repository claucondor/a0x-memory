"""
Test script for Group Reference Resolution feature.

Tests that when a user in DM asks about "the group" without specifying which,
the system can infer which group they mean based on context.
"""
import requests
import json
import time
import random
import string

BASE_URL = "http://136.118.160.81:8080"

def random_id(prefix="test"):
    """Generate a random ID for testing."""
    return f"{prefix}_{''.join(random.choices(string.ascii_lowercase + string.digits, k=8))}"


def test_setup_groups_and_memories():
    """
    Setup test data:
    1. Create a user who participates in two groups
    2. Add memories about different topics in each group
    3. Test that DM queries can infer the correct group
    """
    agent_id = random_id("agent")
    user_id = f"telegram:{random.randint(10000, 99999)}"

    # Two groups with different topics
    trading_group = f"-100{random.randint(100000, 999999)}"
    dev_group = f"-100{random.randint(100000, 999999)}"

    print(f"=== Test Setup ===")
    print(f"Agent: {agent_id}")
    print(f"User: {user_id}")
    print(f"Trading Group: {trading_group}")
    print(f"Dev Group: {dev_group}")

    # Add messages to trading group
    print(f"\n=== Adding messages to Trading Group ===")
    trading_messages = [
        "I think BTC will hit 100k soon",
        "The ETH/BTC ratio is looking bullish",
        "Anyone shorting SOL here?",
        "Great alpha about trading strategies today",
        "My portfolio is up 20% this week",
    ]

    for msg in trading_messages:
        response = requests.post(
            f"{BASE_URL}/v1/memory/passive",
            json={
                "agent_id": agent_id,
                "message": msg,
                "platform_identity": {
                    "platform": "telegram",
                    "telegramId": int(user_id.split(":")[1]),
                    "username": "trader_alice",
                    "chatId": trading_group
                },
                "speaker": "trader_alice"
            }
        )
        print(f"  Added: {msg[:40]}... - {response.status_code}")

    # Add messages to dev group
    print(f"\n=== Adding messages to Dev Group ===")
    dev_messages = [
        "I'm building a smart contract for NFTs",
        "The Solidity code compiles now",
        "Anyone know how to optimize gas in Foundry?",
        "My project uses ERC-721A for batch minting",
        "The frontend is almost done, using React",
    ]

    for msg in dev_messages:
        response = requests.post(
            f"{BASE_URL}/v1/memory/passive",
            json={
                "agent_id": agent_id,
                "message": msg,
                "platform_identity": {
                    "platform": "telegram",
                    "telegramId": int(user_id.split(":")[1]),
                    "username": "dev_alice",
                    "chatId": dev_group
                },
                "speaker": "dev_alice"
            }
        )
        print(f"  Added: {msg[:40]}... - {response.status_code}")

    # Wait for batch processing
    print("\n=== Waiting for batch processing (10s) ===")
    time.sleep(10)

    return agent_id, user_id, trading_group, dev_group


def test_dm_group_reference_trading(agent_id: str, user_id: str, expected_group: str):
    """
    Test: User asks about trading in DM without specifying group.
    Expected: System infers the trading group.
    """
    print("\n=== Test: DM query about trading ===")

    query = "What did they say about my portfolio in the group?"

    response = requests.post(
        f"{BASE_URL}/v1/memory/context",
        json={
            "agent_id": agent_id,
            "query": query,
            "platform_identity": {
                "platform": "telegram",
                "telegramId": int(user_id.split(":")[1]),
                "username": "alice"
                # Note: No chatId = DM context
            }
        }
    )

    print(f"Query: {query}")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Formatted context preview:")
        formatted = data.get("formatted_context", "")[:500]
        print(formatted)

        # Check if trading-related content was retrieved
        has_trading_content = any(
            term in formatted.lower()
            for term in ["btc", "eth", "trading", "portfolio", "bullish"]
        )
        print(f"\nContains trading content: {has_trading_content}")
        return has_trading_content
    else:
        print(f"Error: {response.text}")
        return False


def test_dm_group_reference_dev(agent_id: str, user_id: str, expected_group: str):
    """
    Test: User asks about development in DM without specifying group.
    Expected: System infers the dev group.
    """
    print("\n=== Test: DM query about development ===")

    query = "What was discussed about my smart contract in the group?"

    response = requests.post(
        f"{BASE_URL}/v1/memory/context",
        json={
            "agent_id": agent_id,
            "query": query,
            "platform_identity": {
                "platform": "telegram",
                "telegramId": int(user_id.split(":")[1]),
                "username": "alice"
            }
        }
    )

    print(f"Query: {query}")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Formatted context preview:")
        formatted = data.get("formatted_context", "")[:500]
        print(formatted)

        # Check if dev-related content was retrieved
        has_dev_content = any(
            term in formatted.lower()
            for term in ["solidity", "smart contract", "nft", "erc", "foundry"]
        )
        print(f"\nContains dev content: {has_dev_content}")
        return has_dev_content
    else:
        print(f"Error: {response.text}")
        return False


def test_dm_ambiguous_reference(agent_id: str, user_id: str):
    """
    Test: User asks about "the group" without any context hints.
    Expected: System should still function (may return mixed or no results).
    """
    print("\n=== Test: Ambiguous group reference ===")

    query = "What happened in the group yesterday?"

    response = requests.post(
        f"{BASE_URL}/v1/memory/context",
        json={
            "agent_id": agent_id,
            "query": query,
            "platform_identity": {
                "platform": "telegram",
                "telegramId": int(user_id.split(":")[1]),
                "username": "alice"
            }
        }
    )

    print(f"Query: {query}")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Formatted context preview:")
        formatted = data.get("formatted_context", "")[:500]
        print(formatted)
        return True
    else:
        print(f"Error: {response.text}")
        return False


def main():
    print("=" * 60)
    print("GROUP REFERENCE RESOLUTION TEST")
    print("=" * 60)

    # Check health first
    health = requests.get(f"{BASE_URL}/health")
    if health.status_code != 200:
        print(f"Service not healthy: {health.status_code}")
        return

    print(f"Service healthy: {health.json()}")

    # Setup test data
    agent_id, user_id, trading_group, dev_group = test_setup_groups_and_memories()

    # Run tests
    results = []

    # Test trading reference
    results.append(("Trading reference", test_dm_group_reference_trading(agent_id, user_id, trading_group)))

    # Test dev reference
    results.append(("Dev reference", test_dm_group_reference_dev(agent_id, user_id, dev_group)))

    # Test ambiguous reference
    results.append(("Ambiguous reference", test_dm_ambiguous_reference(agent_id, user_id)))

    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
