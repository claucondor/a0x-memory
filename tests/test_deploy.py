#!/usr/bin/env python3
"""
Test script for deployed a0x-memory API.
Tests all endpoints and prints full responses including errors.
"""

import requests
import json
import time

BASE_URL = "https://a0x-memory-679925931457.us-west1.run.app"
AGENT_ID = "test-agent-deploy-v1"
SPEAKER = "carlos_dev"
PLATFORM_IDENTITY = {
    "platform": "telegram",
    "telegramId": 9001,
    "chatId": "chat_9001",
    "groupId": None,
    "username": "carlos_dev"
}

MESSAGES = [
    "Hey! I'm Carlos, been building in DeFi for about 2 years now. Currently working on a new protocol on Base.",
    "We're building a yield aggregator that uses Uniswap V4 hooks to optimize swap routing. Pretty excited about the architecture.",
    "My background is mostly in smart contracts - been writing Solidity since 2021. Also do a lot of TypeScript for the frontend and tooling.",
    "For our dev setup we use Hardhat with TypeScript. Tried Foundry for a bit but the team is more comfortable with Hardhat honestly.",
    "The protocol is called BaseFi - it aggregates yield across Aave, Compound, and a few other lending markets on Base.",
    "One thing I've been deep diving into is gas optimization. On Base fees are low but we still want to be efficient since we're doing a lot of batch operations.",
    "We've been looking at how Uniswap V4 hooks work - the beforeSwap and afterSwap hooks are really powerful for custom logic.",
    "Our team is 5 people - 2 smart contract devs, 2 frontend devs, and a designer. We're all remote, mostly in Latin America.",
    "Timeline wise we're aiming to launch a testnet in about 6 weeks, then mainnet by end of Q2.",
    "I've been studying how Aave V3's isolation mode works. Want to implement something similar for our risk management.",
    "Also been contributing to some open source Solidity libraries. Submitted a PR to OpenZeppelin last month for a gas optimization in their ERC20 implementation.",
    "What do you think about the current state of DeFi on Base? I feel like there's a lot of room for innovation with the lower fees.",
    "We're using Chainlink oracles for price feeds. Considered using Pyth but Chainlink has better coverage for the assets we need.",
    "One challenge we're facing is MEV protection. Looking into using Flashbots Protect or similar solutions on Base.",
    "I prefer TypeScript over JavaScript for everything. The type safety is worth the extra setup time, especially for blockchain interactions.",
    "Been thinking about adding a governance token but honestly I'm not sure it's necessary. What's your take on token launches these days?",
    "Our smart contracts are about 80% done. Main thing left is the vault strategy contracts and the hook integration with Uni V4.",
    "For testing we use a combination of unit tests in Hardhat and fork tests against Base mainnet. Coverage is around 95%.",
    "Had a fun weekend hacking on a side project - built a Telegram bot that tracks whale movements on Base using The Graph.",
    "Security is top priority. We have an audit scheduled with Trail of Bits for next month. Also running Slither and Mythril locally.",
    "Just deployed our first test contracts on Base Sepolia. The hook registration with Uniswap V4 PoolManager was tricky but we got it working.",
    "Looking forward to connecting more! Always happy to chat about DeFi, Solidity, or anything crypto really.",
]


def timed_request(method, url, **kwargs):
    """Make a request and return (response, elapsed) - never returns None for response."""
    start = time.time()
    try:
        resp = requests.request(method, url, timeout=60, **kwargs)
        elapsed = time.time() - start
        return resp, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"  EXCEPTION after {elapsed:.2f}s: {type(e).__name__}: {e}")
        return None, elapsed


def print_response(label, resp, elapsed):
    """Pretty print a response."""
    status_emoji = "OK" if resp.status_code < 400 else "ERR"
    print(f"\n{'='*80}")
    print(f"  [{status_emoji}] {label}")
    print(f"  Status: {resp.status_code} | Time: {elapsed:.2f}s")
    print(f"{'='*80}")
    try:
        data = resp.json()
        text = json.dumps(data, indent=2, default=str)
        if len(text) > 4000:
            print(text[:4000])
            print(f"\n  ... (truncated, total {len(text)} chars)")
        else:
            print(text)
    except Exception:
        body = resp.text[:2000]
        print(body if body else "(empty body)")
    print()


def main():
    memory_id = None
    total_start = time.time()

    # ==================================================================
    # STEP 1: Health Check
    # ==================================================================
    print("\n" + "#" * 80)
    print("  STEP 1: Health Check")
    print("#" * 80)
    resp, elapsed = timed_request("GET", f"{BASE_URL}/health")
    if resp is not None:
        print_response("GET /health", resp, elapsed)
    else:
        print("Health check failed completely - API is unreachable. Aborting.")
        return

    # ==================================================================
    # STEP 2: Send 22 DM messages via POST /v1/memory/passive
    # ==================================================================
    print("\n" + "#" * 80)
    print("  STEP 2: Sending 22 messages via POST /v1/memory/passive")
    print("#" * 80)
    print()

    success_count = 0
    fail_count = 0

    for i, msg in enumerate(MESSAGES):
        payload = {
            "agent_id": AGENT_ID,
            "message": msg,
            "speaker": SPEAKER,
            "platform_identity": PLATFORM_IDENTITY,
        }
        resp, elapsed = timed_request("POST", f"{BASE_URL}/v1/memory/passive", json=payload)
        if resp is not None:
            status = resp.status_code
            try:
                body = resp.json()
                short = json.dumps(body)[:200]
            except Exception:
                short = resp.text[:200] if resp.text else "(empty)"
            
            marker = "OK" if status < 400 else "FAIL"
            print(f"  [{i+1:2d}/22] [{marker}] {status} ({elapsed:.2f}s) | {short}")

            if status < 400:
                success_count += 1
                try:
                    data = resp.json()
                    if isinstance(data, dict):
                        mid = data.get("memory_id") or data.get("memoryId") or data.get("id")
                        if mid and not memory_id:
                            memory_id = mid
                            print(f"           -> Captured memory_id: {memory_id}")
                except Exception:
                    pass
            else:
                fail_count += 1
                # Print full error for first failure
                if fail_count == 1:
                    print(f"           (first error - full response above)")
        else:
            fail_count += 1
            print(f"  [{i+1:2d}/22] [EXCEPTION] Request failed")

        # Pause after batches
        if i == 9:
            print(f"\n  --- Pausing 6s after message 10 (success={success_count}, fail={fail_count}) ---\n")
            time.sleep(6)
        elif i == 19:
            print(f"\n  --- Pausing 6s after message 20 (success={success_count}, fail={fail_count}) ---\n")
            time.sleep(6)
        else:
            time.sleep(0.3)

    print(f"\n  Message sending complete: {success_count} success, {fail_count} failed")

    # ==================================================================
    # STEP 3: Wait for processing
    # ==================================================================
    print("\n" + "#" * 80)
    print("  STEP 3: Waiting 12 seconds for backend processing...")
    print("#" * 80)
    time.sleep(12)

    # ==================================================================
    # STEP 4: Test all read endpoints
    # ==================================================================
    print("\n" + "#" * 80)
    print("  STEP 4: Testing all read endpoints")
    print("#" * 80)

    # 4a. POST /v1/memory/context - search "DeFi"
    print("\n--- 4a: POST /v1/memory/context (query: DeFi) ---")
    payload = {
        "agent_id": AGENT_ID,
        "query": "DeFi",
        "platform_identity": PLATFORM_IDENTITY,
    }
    resp, elapsed = timed_request("POST", f"{BASE_URL}/v1/memory/context", json=payload)
    if resp is not None:
        print_response("POST /v1/memory/context (DeFi)", resp, elapsed)
        if resp.status_code == 200:
            try:
                data = resp.json()
                if not memory_id:
                    memory_id = data.get("memory_id")
                    if memory_id:
                        print(f"  -> Got memory_id: {memory_id}")
            except Exception:
                pass

    # 4b. POST /v1/memory/context - search "smart contracts"
    print("\n--- 4b: POST /v1/memory/context (query: smart contracts) ---")
    payload = {
        "agent_id": AGENT_ID,
        "query": "smart contracts",
        "platform_identity": PLATFORM_IDENTITY,
    }
    resp, elapsed = timed_request("POST", f"{BASE_URL}/v1/memory/context", json=payload)
    if resp is not None:
        print_response("POST /v1/memory/context (smart contracts)", resp, elapsed)

    # 4c. GET /v1/memory/context/{memory_id}
    test_memory_id = memory_id or f"{AGENT_ID}:telegram:9001"
    print(f"\n--- 4c: GET /v1/memory/context/{test_memory_id} ---")
    resp, elapsed = timed_request("GET", f"{BASE_URL}/v1/memory/context/{test_memory_id}")
    if resp is not None:
        print_response(f"GET /v1/memory/context/{test_memory_id}", resp, elapsed)

    # Also try just agent_id
    if memory_id != AGENT_ID:
        print(f"\n--- 4c alt: GET /v1/memory/context/{AGENT_ID} ---")
        resp, elapsed = timed_request("GET", f"{BASE_URL}/v1/memory/context/{AGENT_ID}")
        if resp is not None:
            print_response(f"GET /v1/memory/context/{AGENT_ID}", resp, elapsed)

    # 4d. GET /v1/profiles/user/telegram:9001
    print("\n--- 4d: GET /v1/profiles/user/telegram:9001 ---")
    resp, elapsed = timed_request("GET", f"{BASE_URL}/v1/profiles/user/telegram:9001")
    if resp is not None:
        print_response("GET /v1/profiles/user/telegram:9001", resp, elapsed)

    # 4e. POST /memories/{memory_id}/search
    search_id = memory_id or AGENT_ID
    print(f"\n--- 4e: POST /memories/{search_id}/search ---")
    payload = {"query": "Uniswap V4 hooks"}
    resp, elapsed = timed_request("POST", f"{BASE_URL}/memories/{search_id}/search", json=payload)
    if resp is not None:
        print_response(f"POST /memories/{search_id}/search (Uniswap V4 hooks)", resp, elapsed)

    # 4f. POST /memories/{memory_id}/ask
    print(f"\n--- 4f: POST /memories/{search_id}/ask ---")
    payload = {"question": "What does carlos work on?"}
    resp, elapsed = timed_request("POST", f"{BASE_URL}/memories/{search_id}/ask", json=payload)
    if resp is not None:
        print_response(f"POST /memories/{search_id}/ask (What does carlos work on?)", resp, elapsed)

    # 4g. GET /v1/memory/stats/{agent_id}
    print(f"\n--- 4g: GET /v1/memory/stats/{AGENT_ID} ---")
    resp, elapsed = timed_request("GET", f"{BASE_URL}/v1/memory/stats/{AGENT_ID}")
    if resp is not None:
        print_response(f"GET /v1/memory/stats/{AGENT_ID}", resp, elapsed)

    # Bonus endpoints
    print(f"\n--- Bonus: GET /tenants ---")
    resp, elapsed = timed_request("GET", f"{BASE_URL}/tenants")
    if resp is not None:
        print_response("GET /tenants", resp, elapsed)

    print(f"\n--- Bonus: GET /instances ---")
    resp, elapsed = timed_request("GET", f"{BASE_URL}/instances")
    if resp is not None:
        print_response("GET /instances", resp, elapsed)

    # POST /v1/memory/process-pending
    print(f"\n--- Bonus: POST /v1/memory/process-pending ---")
    resp, elapsed = timed_request("POST", f"{BASE_URL}/v1/memory/process-pending?agent_id={AGENT_ID}")
    if resp is not None:
        print_response(f"POST /v1/memory/process-pending?agent_id={AGENT_ID}", resp, elapsed)

    total_elapsed = time.time() - total_start
    print("\n" + "#" * 80)
    print(f"  TEST RUN COMPLETE  (total time: {total_elapsed:.1f}s)")
    print(f"  Memory ID captured: {memory_id or 'NONE'}")
    print("#" * 80)


if __name__ == "__main__":
    main()
