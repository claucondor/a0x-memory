"""
Test a0x-memory API endpoints

Tests:
1. Health check
2. Passive memory endpoint (simulate group message without mention)
3. Active memory endpoint (simulate group message with mention)
4. Context retrieval
"""
import os
import sys
import tempfile
import shutil
import requests
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import this BEFORE uvicorn to avoid conflicts
os.environ['FIRESTORE_COLLECTION_PREFIX'] = 'agents_test'

from api.endpoints import app
from services.group_memory_manager import GroupMemoryManager


class MemoryAPITest:
    """Test the a0x-memory HTTP API"""

    def __init__(self):
        self.test_db_path = tempfile.mkdtemp(prefix='a0x-api-test-')
        self.api_url = "http://localhost:8080"
        self.agent_id = "test_agent_api"

    def setup(self):
        print("=" * 80)
        print("A0X-MEMORY API TEST")
        print("=" * 80)

        # Set environment for test
        os.environ['DB_PATH'] = self.test_db_path
        os.environ['PROJECT_ID'] = 'a0x-co'
        os.environ['FIRESTORE_COLLECTION_PREFIX'] = 'agents_test'

        # Pre-create manager to initialize DB
        manager = GroupMemoryManager(
            agent_id=self.agent_id,
            db_base_path=self.test_db_path
        )
        print(f"Test DB: {self.test_db_path}")

    def teardown(self):
        # Cleanup test DB
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)
        print("\nCleaned up test DB")

    def test_health_check(self):
        """Test 1: Health check endpoint"""
        print("\n" + "=" * 80)
        print("TEST 1: Health Check")
        print("=" * 80)

        try:
            # Check if server is running
            response = requests.get(f"{self.api_url}/health", timeout=2)
            print(f"  Status: {response.status_code}")
            print(f"  Response: {response.json()}")
            return True
        except requests.exceptions.ConnectionError:
            print("  ⚠ Server not running - start with: python api/endpoints.py")
            return False
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False

    def test_passive_memory(self):
        """Test 2: Passive memory endpoint"""
        print("\n" + "=" * 80)
        print("TEST 2: Passive Memory (Group Message WITHOUT Mention)")
        print("=" * 80)

        request_data = {
            "agent_id": self.agent_id,
            "message": "I'm working on DeFi lending protocols",
            "platform_identity": {
                "platform": "telegram",
                "chatId": "-1001234567890"
            },
            "username": "alice_defi",
            "metadata": {"message_id": "passive_1"}
        }

        try:
            response = requests.post(
                f"{self.api_url}/v1/memory/passive",
                json=request_data,
                timeout=10
            )

            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"  Success: {data['success']}")
                print(f"  Message: {data['message']}")
                print(f"  Recent context: {len(data['recent_context'])} messages")
                print(f"  Window size: {data['window_size']}")
                return True
            else:
                print(f"  Error: {response.text}")
                return False

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False

    def test_active_memory(self):
        """Test 3: Active memory endpoint"""
        print("\n" + "=" * 80)
        print("TEST 3: Active Memory (Group Message WITH Mention)")
        print("=" * 80)

        request_data = {
            "agent_id": self.agent_id,
            "message": "I'm a DeFi expert with 5 years experience",
            "platform_identity": {
                "platform": "telegram",
                "chatId": "-1001234567890"
            },
            "username": "bob_defi",
            "metadata": {"message_id": "active_1"},
            "generate_response": False
        }

        try:
            response = requests.post(
                f"{self.api_url}/v1/memory/active",
                json=request_data,
                timeout=30
            )

            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"  Success: {data['success']}")
                print(f"  Message: {data['message']}")
                print(f"  Memories created: {data['memories_created']}")
                print(f"  Recent context: {len(data['recent_context'])} messages")
                return True
            else:
                print(f"  Error: {response.text}")
                return False

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False

    def test_context_retrieval(self):
        """Test 4: Context retrieval endpoint"""
        print("\n" + "=" * 80)
        print("TEST 4: Context Retrieval")
        print("=" * 80)

        # First, add some messages
        self.test_passive_memory()
        time.sleep(1)
        self.test_active_memory()
        time.sleep(1)

        request_data = {
            "agent_id": self.agent_id,
            "group_id": "telegram_group_-1001234567890",
            "limit": 10
        }

        try:
            response = requests.post(
                f"{self.api_url}/v1/memory/context",
                json=request_data,
                timeout=10
            )

            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"  Success: {data['success']}")
                context = data['context']
                print(f"  Is group: {context.get('is_group')}")
                print(f"  Group context: {len(context.get('group_context', []))} items")
                print(f"  User context: {len(context.get('user_context', []))} items")
                print(f"  Recent messages: {len(context.get('recent_messages', []))} items")
                return True
            else:
                print(f"  Error: {response.text}")
                return False

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False

    def run_all(self):
        """Run all tests"""
        try:
            self.setup()

            print("\n" + "=" * 80)
            print("RUNNING API TESTS")
            print("=" * 80)
            print("\nNOTE: Start server first with: python api/endpoints.py")
            print("Then run this test in another terminal\n")

            # Test 1: Health check
            if not self.test_health_check():
                print("\n⚠ Server not running - tests aborted")
                return False

            # Test 2: Passive memory
            self.test_passive_memory()

            # Test 3: Active memory
            self.test_active_memory()

            # Test 4: Context retrieval
            self.test_context_retrieval()

            print("\n" + "=" * 80)
            print("API TESTS COMPLETE")
            print("=" * 80 + "\n")

            return True

        except Exception as e:
            print(f"\nERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            self.teardown()


if __name__ == "__main__":
    tester = MemoryAPITest()
    success = tester.run_all()
    sys.exit(0 if success else 1)
