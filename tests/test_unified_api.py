"""
Test Unified Memory API Endpoints

Tests the new unified API endpoints:
- POST /v1/memory/passive
- POST /v1/memory/active
- POST /v1/memory/context
- POST /v1/memory/process-pending
- GET /v1/memory/stats/{agent_id}

Run: python3 tests/test_unified_api.py
"""
import sys
import os

# Add parent to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the app directly from api.py (not api/ folder)
import importlib.util
spec = importlib.util.spec_from_file_location("api_module", os.path.join(parent_dir, "api.py"))
api_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api_module)
app = api_module.app

from fastapi.testclient import TestClient

# Test client
client = TestClient(app)

TEST_AGENT_ID = "test_unified_api"


def test_passive_memory():
    """Test adding passive memory (background processing)"""
    print("\n" + "=" * 60)
    print("TEST: POST /v1/memory/passive")
    print("=" * 60)

    response = client.post("/v1/memory/passive", json={
        "agent_id": TEST_AGENT_ID,
        "message": "I've been working with Solidity for 5 years, mainly on DeFi protocols",
        "platform_identity": {
            "platform": "telegram",
            "chatId": "-100123456",  # Negative = group
            "telegramId": 111111,
            "username": "alice_dev"
        },
        "speaker": "Alice",
        "metadata": {
            "message_id": "msg_001",
            "timestamp": "2025-01-26T10:00:00Z"
        }
    })

    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert data["is_group"] == True
    assert "telegram_" in (data["group_id"] or "")

    return data


def test_active_memory():
    """Test adding active memory with context return"""
    print("\n" + "=" * 60)
    print("TEST: POST /v1/memory/active")
    print("=" * 60)

    response = client.post("/v1/memory/active", json={
        "agent_id": TEST_AGENT_ID,
        "message": "@bot what smart contract experience do people here have?",
        "platform_identity": {
            "platform": "telegram",
            "chatId": "-100123456",
            "telegramId": 222222,
            "username": "bob_trader"
        },
        "speaker": "Bob",
        "return_context": True,
        "context_limit": 10
    })

    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Success: {data.get('success')}")
    print(f"Added: {data.get('added')}")
    print(f"Processed: {data.get('processed')}")
    print(f"Memories created: {data.get('memories_created')}")
    print(f"Context returned: {'Yes' if data.get('context') else 'No'}")

    assert response.status_code == 200
    assert data["success"] == True

    return data


def test_context_retrieval():
    """Test context retrieval only"""
    print("\n" + "=" * 60)
    print("TEST: POST /v1/memory/context")
    print("=" * 60)

    response = client.post("/v1/memory/context", json={
        "agent_id": TEST_AGENT_ID,
        "query": "Who has experience with smart contracts?",
        "platform_identity": {
            "platform": "telegram",
            "chatId": "-100123456",
            "telegramId": 222222,
            "username": "bob_trader"
        },
        "include_recent": True,
        "recent_limit": 10,
        "memory_limit": 5
    })

    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Success: {data.get('success')}")
    print(f"Recent messages: {len(data.get('recent_messages', []))}")
    print(f"Formatted context length: {len(data.get('formatted_context', ''))}")

    if data.get('formatted_context'):
        print(f"\nFormatted context preview:")
        print(data['formatted_context'][:500])

    assert response.status_code == 200
    assert data["success"] == True

    return data


def test_dm_flow():
    """Test DM flow (no group_id)"""
    print("\n" + "=" * 60)
    print("TEST: DM Flow (passive + active)")
    print("=" * 60)

    # Add passive DM
    response = client.post("/v1/memory/passive", json={
        "agent_id": TEST_AGENT_ID,
        "message": "Hey, I'm interested in learning about Base chain grants",
        "platform_identity": {
            "platform": "telegram",
            "chatId": "333333",  # Positive = DM
            "telegramId": 333333,
            "username": "charlie_builder"
        },
        "speaker": "Charlie"
    })

    print(f"Passive DM - Status: {response.status_code}")
    data = response.json()
    print(f"  is_group: {data.get('is_group')}")
    print(f"  group_id: {data.get('group_id')}")

    assert response.status_code == 200
    assert data["is_group"] == False

    # Add active DM with context
    response = client.post("/v1/memory/active", json={
        "agent_id": TEST_AGENT_ID,
        "message": "Can you tell me more about the grant application process?",
        "platform_identity": {
            "platform": "telegram",
            "chatId": "333333",
            "telegramId": 333333,
            "username": "charlie_builder"
        },
        "speaker": "Charlie",
        "return_context": True
    })

    print(f"Active DM - Status: {response.status_code}")
    data = response.json()
    print(f"  Context returned: {'Yes' if data.get('context') else 'No'}")

    assert response.status_code == 200

    return data


def test_stats():
    """Test stats endpoint"""
    print("\n" + "=" * 60)
    print("TEST: GET /v1/memory/stats/{agent_id}")
    print("=" * 60)

    response = client.get(f"/v1/memory/stats/{TEST_AGENT_ID}")

    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Stats: {data}")

    assert response.status_code == 200
    assert data.get("agent_id") == TEST_AGENT_ID

    return data


def test_process_pending():
    """Test manual process pending"""
    print("\n" + "=" * 60)
    print("TEST: POST /v1/memory/process-pending")
    print("=" * 60)

    response = client.post(
        "/v1/memory/process-pending",
        params={
            "agent_id": TEST_AGENT_ID,
            "group_id": "telegram_-100123456"
        }
    )

    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {data}")

    assert response.status_code == 200

    return data


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("UNIFIED MEMORY API TESTS")
    print("=" * 70)

    try:
        test_passive_memory()
        test_active_memory()
        test_context_retrieval()
        test_dm_flow()
        test_stats()
        test_process_pending()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED!")
        print("=" * 70)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
