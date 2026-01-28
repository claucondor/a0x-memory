#!/usr/bin/env python3
"""
E2E test script for a0x-memory deployed API.
Sends 22 DM messages from carlos_dev, then tests all read endpoints.
"""

import requests
import json
import time

BASE_URL = "https://a0x-memory-679925931457.us-west1.run.app"

AGENT_ID = "test-e2e-v2"
PLATFORM_IDENTITY = {
    "platform": "telegram",
    "telegramId": 9001,
    "username": "carlos_dev"
}

MESSAGES = [
    "Hey! I'm Carlos, been building DeFi protocols on Base for about 8 months now",
    "Just deployed my first Uniswap V4 hook yesterday - custom fee logic based on volatility",
    "I mostly write Solidity but my off-chain stuff is all TypeScript with Hardhat",
    "Working on a lending aggregator that routes between Aave and Compound for best rates",
    "Gas optimization is my obsession lately - managed to cut 40% off our swap contract",
    "Our team is 4 devs, 2 focused on smart contracts, 2 on the frontend/backend",
    "We're targeting mainnet launch on Base by end of Q2, testnet is live already",
    "The Uniswap V4 hooks are game-changing, you can customize literally everything about a pool",
    "I've been using Foundry for fuzzing tests but Hardhat for deployment scripts",
    "Thinking about adding flash loan functionality to the aggregator, Aave V3 makes it pretty easy",
    "Had a crazy week debugging a reentrancy issue in our yield vault - classic mistake",
    "By the way, do you know any good Base hackathons coming up? Want to showcase the hooks",
    "Our aggregator compares APY across Aave, Compound, and Morpho Blue in real-time",
    "I prefer TypeScript over Python for backend stuff - better type safety for DeFi",
    "Just integrated Chainlink price feeds for our volatility-based fee hook",
    "The Base ecosystem is growing fast, gas fees are insanely low compared to mainnet",
    "We use a diamond proxy pattern for upgradeability - EIP-2535 is underrated",
    "My co-founder handles the tokenomics side, I focus purely on smart contract architecture",
    "Been experimenting with Solidity inline assembly for the gas-critical paths",
    "Our test coverage is at 95% - I'm paranoid about security after the reentrancy bug",
    "Planning to get an audit from Trail of Bits before mainnet, already in their queue",
    "Anyway, gotta run - have a standup in 5. Talk later!"
]

def send_passive_message(msg: str, idx: int) -> dict:
    """Send a single passive memory message."""
    payload = {
        "agent_id": AGENT_ID,
        "message": msg,
        "platform_identity": PLATFORM_IDENTITY,
        "speaker": "carlos_dev"
    }
    start = time.time()
    resp = requests.post(f"{BASE_URL}/v1/memory/passive", json=payload, timeout=30)
    elapsed = time.time() - start
    data = resp.json()
    processed = data.get("processed", False)
    batch_tag = " *** BATCH PROCESSED ***" if processed else ""
    print(f"  [{idx+1:2d}/22] {resp.status_code} | {elapsed:.2f}s | msg_count={data.get('message_count','?')} | processed={processed}{batch_tag}")
    if processed:
        print(f"         Memories created: {data.get('memories_created', '?')}, Profile updated: {data.get('profile_updated', '?')}")
    return data

def test_endpoint(method: str, path: str, label: str, json_body=None, params=None):
    """Test an endpoint, print full JSON response."""
    url = f"{BASE_URL}{path}"
    print(f"\n{'='*70}")
    print(f"TEST: {label}")
    print(f"{method} {url}")
    if json_body:
        print(f"Body: {json.dumps(json_body, indent=2)}")
    print("-"*70)

    start = time.time()
    try:
        if method == "GET":
            resp = requests.get(url, params=params, timeout=30)
        else:
            resp = requests.post(url, json=json_body, params=params, timeout=30)
        elapsed = time.time() - start
        print(f"Status: {resp.status_code} | Time: {elapsed:.2f}s")
        try:
            data = resp.json()
            print(json.dumps(data, indent=2, default=str))
        except Exception:
            print(resp.text[:2000])
    except Exception as e:
        elapsed = time.time() - start
        print(f"ERROR after {elapsed:.2f}s: {e}")


def main():
    print("=" * 70)
    print("a0x-memory E2E Test v2")
    print(f"Target: {BASE_URL}")
    print(f"Agent: {AGENT_ID} | User: carlos_dev (telegram:9001)")
    print("=" * 70)

    # --- Phase 1: Send 22 messages ---
    print("\n--- PHASE 1: Sending 22 passive messages ---\n")
    total_start = time.time()

    for i, msg in enumerate(MESSAGES):
        send_passive_message(msg, i)
        if i == 10:
            print("\n  >>> Pausing 5 seconds after message 11 (let batch processing finish) <<<\n")
            time.sleep(5)

    total_elapsed = time.time() - total_start
    print(f"\n  All 22 messages sent in {total_elapsed:.1f}s")

    # --- Phase 2: Wait for processing ---
    print("\n--- Waiting 10 seconds for background processing to complete ---\n")
    time.sleep(10)

    # --- Phase 3: Test all read endpoints ---
    print("\n--- PHASE 2: Testing all read endpoints ---")

    # 1. User profile
    test_endpoint("GET", "/v1/profiles/user/telegram:9001", "User Profile")

    # 2. Full context with profiles
    test_endpoint("GET", f"/v1/memory/context/{AGENT_ID}:telegram:9001", "Full Context (GET)")

    # 3. Context query via POST
    test_endpoint("POST", "/v1/memory/context", "Context Query (POST)", json_body={
        "agent_id": AGENT_ID,
        "query": "What does carlos work on?",
        "platform_identity": PLATFORM_IDENTITY
    })

    # 4. Ask endpoint
    test_endpoint("POST", f"/memories/{AGENT_ID}:telegram:9001/ask", "Ask Question", json_body={
        "question": "What projects is carlos building?"
    })

    # 5. Search endpoint
    test_endpoint("POST", f"/memories/{AGENT_ID}:telegram:9001/search", "Search Memories",
                  params={"query": "DeFi"})

    # 6. Stats
    test_endpoint("GET", f"/v1/memory/stats/{AGENT_ID}", "Agent Stats")

    print("\n" + "=" * 70)
    print("E2E Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
