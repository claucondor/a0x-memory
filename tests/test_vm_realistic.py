#!/usr/bin/env python3
"""
VM Realistic Scenario Test - HTTP API Performance
Tests the VM deployment with realistic multi-user, multi-group scenarios.

Run: python tests/test_vm_realistic.py
"""
import requests
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# VM endpoint
BASE_URL = "http://136.118.160.81:8080"

AGENT_ID = "test_vm_realistic"

# 10 test users
USERS = [
    {"id": 1001, "username": "alice_dev"},
    {"id": 1002, "username": "bob_trader"},
    {"id": 1003, "username": "carol_artist"},
    {"id": 1004, "username": "david_founder"},
    {"id": 1005, "username": "emma_researcher"},
    {"id": 1006, "username": "frank_newbie"},
    {"id": 1007, "username": "grace_pm"},
    {"id": 1008, "username": "henry_auditor"},
    {"id": 1009, "username": "ivy_marketer"},
    {"id": 1010, "username": "jack_whale"},
]

# DM messages per user
DM_MESSAGES = {
    "alice_dev": [
        "Hey, I've been building smart contracts for 5 years now",
        "Mostly worked on Uniswap forks and lending protocols",
        "I'm interested in the Base chain grants program",
        "My specialty is gas optimization and security",
    ],
    "bob_trader": [
        "What's up, I trade mostly on Base now",
        "I use a combination of TA and on-chain analysis",
        "I've been profitable for 3 months straight",
        "Risk management is key, I never risk more than 2%",
    ],
    "carol_artist": [
        "Hi! I'm new to the crypto space",
        "I'm an artist and want to explore NFTs",
        "I create generative art using Processing and p5.js",
        "I have a collection of 1000 unique pieces ready",
    ],
    "david_founder": [
        "Hey, I'm building a startup on Base",
        "We're creating a decentralized identity solution",
        "Our MVP is almost ready for testnet",
        "Team of 4 engineers, all remote",
    ],
    "emma_researcher": [
        "Hello, I'm researching L2 scaling solutions",
        "Working on a paper comparing Base, Optimism, Arbitrum",
        "I'm affiliated with Stanford blockchain club",
        "Published 3 papers on consensus mechanisms",
    ],
    "frank_newbie": [
        "Hi, I'm completely new to all this",
        "I don't really understand what L2 means",
        "How do I even get started?",
        "What wallet should I use?",
    ],
    "grace_pm": [
        "Hey, I'm a PM at a DeFi protocol",
        "We're expanding to Base next month",
        "We're bringing over $10M in TVL initially",
        "Our protocol focuses on yield aggregation",
    ],
    "henry_auditor": [
        "Hi, I do smart contract security audits",
        "Certified by Code4rena and Sherlock",
        "Found critical bugs in 5 major protocols",
        "I use Slither, Mythril, and manual review",
    ],
    "ivy_marketer": [
        "Hey! I do marketing for crypto projects",
        "I've built communities of 50k+ members",
        "Twitter and Discord are my main channels",
        "Have connections with major crypto influencers",
    ],
    "jack_whale": [
        "Hello, I manage a crypto fund",
        "Typical check size is $500k-2M",
        "Interested in infrastructure plays",
        "Our thesis is L2 infrastructure dominance",
    ],
}

# Group messages
GROUPS = [
    {"id": "-100001", "name": "Base Builders", "members": ["alice_dev", "david_founder", "henry_auditor", "emma_researcher"]},
    {"id": "-100002", "name": "DeFi Discussion", "members": ["bob_trader", "grace_pm", "jack_whale", "alice_dev"]},
    {"id": "-100003", "name": "NFT Creators", "members": ["carol_artist", "ivy_marketer", "frank_newbie", "grace_pm"]},
]

GROUP_MESSAGES = {
    "Base Builders": [
        {"user": "alice_dev", "msg": "Just deployed my new contract on Base testnet!"},
        {"user": "david_founder", "msg": "Nice! What are you building?"},
        {"user": "alice_dev", "msg": "A new AMM with concentrated liquidity"},
        {"user": "henry_auditor", "msg": "Happy to take a look at the security when ready"},
        {"user": "emma_researcher", "msg": "Concentrated liquidity on L2s has different dynamics"},
        {"user": "david_founder", "msg": "Our identity solution could help with KYC"},
    ],
    "DeFi Discussion": [
        {"user": "bob_trader", "msg": "Anyone else seeing the WETH premium on Base?"},
        {"user": "jack_whale", "msg": "Yeah, about 0.1% right now, arb opportunity"},
        {"user": "grace_pm", "msg": "Our protocol is showing good yields, 8% APY on stables"},
        {"user": "alice_dev", "msg": "Depends on the source of yield, what's the mechanism?"},
        {"user": "bob_trader", "msg": "Made 15% last week on the ETH/USDC pool"},
        {"user": "jack_whale", "msg": "Institutional money prefers 80% stables minimum"},
    ],
    "NFT Creators": [
        {"user": "carol_artist", "msg": "Just finished my new generative collection!"},
        {"user": "ivy_marketer", "msg": "Congrats! Can I see some previews?"},
        {"user": "carol_artist", "msg": "Sure, it's geometric patterns inspired by nature"},
        {"user": "frank_newbie", "msg": "How do you even create generative art?"},
        {"user": "grace_pm", "msg": "We're exploring NFT-gated features for our protocol"},
        {"user": "ivy_marketer", "msg": "Pro tip: build community before launch"},
    ],
}

def get_user(username):
    for u in USERS:
        if u["username"] == username:
            return u
    return None


def send_dm(username, message, idx):
    """Send a DM message"""
    user = get_user(username)
    payload = {
        "agent_id": AGENT_ID,
        "message": message,
        "platform_identity": {
            "platform": "telegram",
            "telegramId": user["id"],
            "username": username,
        },
        "speaker": username,
    }
    start = time.time()
    resp = requests.post(f"{BASE_URL}/v1/memory/passive", json=payload, timeout=120)
    elapsed = time.time() - start
    return {"type": "dm", "user": username, "idx": idx, "time": elapsed, "status": resp.status_code, "data": resp.json()}


def send_group(group_name, group_id, username, message, idx):
    """Send a group message"""
    user = get_user(username)
    payload = {
        "agent_id": AGENT_ID,
        "message": message,
        "platform_identity": {
            "platform": "telegram",
            "telegramId": user["id"],
            "username": username,
            "chatId": group_id,  # Negative chatId = Telegram group
        },
        "speaker": username,
    }
    start = time.time()
    resp = requests.post(f"{BASE_URL}/v1/memory/passive", json=payload, timeout=120)
    elapsed = time.time() - start
    return {"type": "group", "group": group_name, "user": username, "idx": idx, "time": elapsed, "status": resp.status_code, "data": resp.json()}


def search_context(query, group_id=None):
    """Search for context"""
    user = USERS[0]
    platform_identity = {
        "platform": "telegram",
        "telegramId": user["id"],
        "username": user["username"],
    }
    if group_id:
        platform_identity["chatId"] = group_id  # Negative = group context

    payload = {
        "agent_id": AGENT_ID,
        "query": query,
        "platform_identity": platform_identity,
    }

    start = time.time()
    resp = requests.post(f"{BASE_URL}/v1/memory/context", json=payload, timeout=60)
    elapsed = time.time() - start
    return {"query": query, "time": elapsed, "status": resp.status_code, "data": resp.json()}


def run_test():
    print("=" * 70)
    print(f"VM REALISTIC SCENARIO TEST")
    print(f"Target: {BASE_URL}")
    print("=" * 70)

    # Health check
    resp = requests.get(f"{BASE_URL}/health")
    print(f"\nHealth: {resp.json()}")

    all_times = []

    # Phase 1: Send DM messages (sequential)
    print("\n" + "=" * 50)
    print("PHASE 1: DM Messages (sequential)")
    print("=" * 50)

    dm_times = []
    total_dms = sum(len(msgs) for msgs in DM_MESSAGES.values())
    dm_count = 0

    for username, messages in DM_MESSAGES.items():
        for idx, msg in enumerate(messages):
            dm_count += 1
            result = send_dm(username, msg, idx)
            dm_times.append(result["time"])
            all_times.append(result["time"])
            processed = result["data"].get("processed", False)
            batch_tag = " [BATCH]" if processed else ""
            print(f"  [{dm_count}/{total_dms}] {username}: {result['time']:.3f}s | {result['status']}{batch_tag}")

    avg_dm = sum(dm_times) / len(dm_times)
    print(f"\n  DM Summary: {len(dm_times)} messages, avg {avg_dm:.3f}s, total {sum(dm_times):.1f}s")

    # Phase 2: Send Group messages (sequential)
    print("\n" + "=" * 50)
    print("PHASE 2: Group Messages (sequential)")
    print("=" * 50)

    group_times = []
    total_groups = sum(len(msgs) for msgs in GROUP_MESSAGES.values())
    group_count = 0

    for group in GROUPS:
        group_name = group["name"]
        group_id = group["id"]
        messages = GROUP_MESSAGES.get(group_name, [])

        print(f"\n  --- {group_name} ---")
        for idx, msg_data in enumerate(messages):
            group_count += 1
            result = send_group(group_name, group_id, msg_data["user"], msg_data["msg"], idx)
            group_times.append(result["time"])
            all_times.append(result["time"])
            processed = result["data"].get("processed", False)
            batch_tag = " [BATCH]" if processed else ""
            print(f"  [{group_count}/{total_groups}] {msg_data['user']}: {result['time']:.3f}s | {result['status']}{batch_tag}")

    avg_group = sum(group_times) / len(group_times)
    print(f"\n  Group Summary: {len(group_times)} messages, avg {avg_group:.3f}s, total {sum(group_times):.1f}s")

    # Phase 3: Concurrent ingestion test
    print("\n" + "=" * 50)
    print("PHASE 3: Concurrent Ingestion (10 parallel)")
    print("=" * 50)

    concurrent_times = []
    test_messages = [
        ("alice_dev", "Testing concurrent message 1"),
        ("bob_trader", "Testing concurrent message 2"),
        ("carol_artist", "Testing concurrent message 3"),
        ("david_founder", "Testing concurrent message 4"),
        ("emma_researcher", "Testing concurrent message 5"),
        ("frank_newbie", "Testing concurrent message 6"),
        ("grace_pm", "Testing concurrent message 7"),
        ("henry_auditor", "Testing concurrent message 8"),
        ("ivy_marketer", "Testing concurrent message 9"),
        ("jack_whale", "Testing concurrent message 10"),
    ]

    start_concurrent = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(send_dm, user, msg, 99) for user, msg in test_messages]
        for future in as_completed(futures):
            result = future.result()
            concurrent_times.append(result["time"])
            print(f"  {result['user']}: {result['time']:.3f}s")

    total_concurrent = time.time() - start_concurrent
    print(f"\n  Concurrent Summary: 10 messages in {total_concurrent:.2f}s wall time")
    print(f"  Individual avg: {sum(concurrent_times)/len(concurrent_times):.3f}s")

    # Phase 4: Search/Context queries
    print("\n" + "=" * 50)
    print("PHASE 4: Context Queries")
    print("=" * 50)

    queries = [
        ("Who knows about smart contracts?", None),
        ("What projects are being built?", None),
        ("Who is interested in NFTs?", None),
        ("What's being discussed in Base Builders?", "-100001"),
        ("Trading activity in DeFi Discussion?", "-100002"),
    ]

    search_times = []
    for query, group_id in queries:
        result = search_context(query, group_id)
        search_times.append(result["time"])
        memories = len(result["data"].get("relevant_memories", []))
        recent = len(result["data"].get("recent_messages", []))
        print(f"  [{result['time']:.3f}s] {query[:40]}... → {memories} memories, {recent} recent")

    avg_search = sum(search_times) / len(search_times)
    print(f"\n  Search Summary: {len(search_times)} queries, avg {avg_search:.3f}s")

    # Final Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    all_sorted = sorted(all_times)
    p50 = all_sorted[len(all_sorted)//2]
    p95_idx = int(len(all_sorted) * 0.95)
    p95 = all_sorted[p95_idx] if p95_idx < len(all_sorted) else all_sorted[-1]

    print(f"\n  Total messages: {len(all_times)}")
    print(f"  Average latency: {sum(all_times)/len(all_times):.3f}s")
    print(f"  Median (P50): {p50:.3f}s")
    print(f"  P95: {p95:.3f}s")
    print(f"  Min: {all_sorted[0]:.3f}s")
    print(f"  Max: {all_sorted[-1]:.3f}s")
    print(f"\n  Search avg: {avg_search:.3f}s")
    print(f"  Concurrent throughput: {10/total_concurrent:.1f} msg/s")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_test()
