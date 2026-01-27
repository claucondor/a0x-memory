#!/usr/bin/env python3
"""
Quick test for Architecture 5 only
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from arch5_hybrid_multi_level import HybridMultiLevelMemoryStore
import json

# Load test data
with open("/home/oydual3/a0x/a0x-memory/tests/group_memory/test_data.json") as f:
    test_data = json.load(f)

messages = test_data["messages"]

print("=== Testing Architecture 5: Hybrid Multi-Level ===\n")

db_path = "/home/oydual3/a0x/a0x-memory/tests/group_memory/lancedb_arch5"
store = HybridMultiLevelMemoryStore(db_path=db_path, agent_id="jessexbt")

# Test 1: Batch insert
print("[Test 1] Batch Insert...")
import time
start = time.time()
counts = store.add_messages_batch(messages)
elapsed = time.time() - start
print(f"  - Total: {len(messages)} messages")
print(f"  - Time: {elapsed:.2f}s")
print(f"  - Throughput: {len(messages)/elapsed:.2f} msg/sec")
print(f"  - Distribution: {counts}")

# Test 2: Query performance
print("\n[Test 2] Query Performance...")
test_queries = [
    "yield farming strategies",
    "NFT collecting tips",
    "smart contract security",
]

for query in test_queries:
    start = time.time()
    results = store.semantic_search(query, limit=10)
    elapsed = (time.time() - start) * 1000
    print(f"  - '{query}': {elapsed:.2f}ms ({len(results)} results)")

# Test 3: Filtered search (with memory_type)
print("\n[Test 3] Filtered Search (with memory_type)...")
for memory_type in ["group", "user", "interaction"]:
    start = time.time()
    results = store.semantic_search(query="", memory_type=memory_type, limit=5)
    elapsed = (time.time() - start) * 1000
    print(f"  - {memory_type}: {elapsed:.2f}ms ({len(results)} results)")

print("\n=== Architecture 5 Test Complete ===")
