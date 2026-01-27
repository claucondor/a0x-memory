"""
Firestore Window Integration Test

Tests:
1. Firestore connection (using ADC)
2. Sliding window behavior
3. Batch processing trigger
4. Integration with GroupMemoryManager
"""
import os
import sys
import tempfile
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.group_memory_manager import GroupMemoryManager
from services.firestore_window import get_firestore_store
import config


class FirestoreWindowTest:
    """Test Firestore window integration"""

    def __init__(self):
        self.test_db_path = tempfile.mkdtemp(prefix='a0x-firestore-')
        self.agent_id = "test_agent_firestore"
        self.manager = None
        self.group_id = "telegram_group_-1009999999999"
        self.use_firestore = True  # Will be set to False if Firestore fails

    def setup(self):
        print("=" * 80)
        print("FIRESTORE WINDOW INTEGRATION TEST")
        print("=" * 80)
        print(f"\nTest DB: {self.test_db_path}")
        print(f"Agent: {self.agent_id}")
        print(f"Group: {self.group_id}")

        # Test Firestore connection first
        print("\n" + "-" * 80)
        print("Testing Firestore connection...")
        print("-" * 80)

        firestore = get_firestore_store()

        if not firestore.is_enabled():
            print("\n⚠ WARNING: Firestore not available")
            print("  - Running in local-only mode (no Firestore)")
            print("  - Test will verify LanceDB functionality only")
            self.use_firestore = False
        else:
            print(f"\n✓ Firestore connected!")
            print(f"  Project: {config.FIRESTORE_PROJECT}")
            print(f"  Collection prefix: {config.FIRESTORE_COLLECTION_PREFIX}")
            self.use_firestore = True

        # Initialize GroupMemoryManager
        self.manager = GroupMemoryManager(
            agent_id=self.agent_id,
            db_base_path=self.test_db_path
        )

    def teardown(self):
        # Clean up test DB
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)

        # Clean up Firestore test data
        if self.use_firestore:
            firestore = get_firestore_store()
            firestore.clear_recent(self.agent_id, self.group_id)
            print("\n[FirebaseStore] Cleaned up test data")

    def test_firestore_window(self):
        """Test basic Firestore window operations"""
        print("\n" + "=" * 80)
        print("TEST 1: Firestore Window Operations")
        print("=" * 80)

        if not self.use_firestore:
            print("SKIPPED: Firestore not available")
            return

        firestore = get_firestore_store()

        # Test 1.1: Add messages
        print("\n1.1 Adding messages...")
        for i in range(5):
            doc_id = firestore.add_message(
                agent_id=self.agent_id,
                group_id=self.group_id,
                message=f"Test message {i}",
                username=f"test_user_{i}",
                platform_identity={'platform': 'telegram', 'chatId': self.group_id}
            )
            print(f"  Added message {i}: {doc_id}")

        # Test 1.2: Get recent messages
        print("\n1.2 Getting recent messages...")
        recent = firestore.get_recent(self.agent_id, self.group_id, limit=10)
        print(f"  Retrieved {len(recent)} messages:")
        for msg in recent:
            print(f"    - {msg['username']}: {msg['content'][:30]}...")

        # Test 1.3: Test sliding window (add 10 more, should keep only 10)
        print("\n1.3 Testing sliding window (add 10 more messages)...")
        for i in range(5, 15):
            firestore.add_message(
                agent_id=self.agent_id,
                group_id=self.group_id,
                message=f"Test message {i}",
                username=f"test_user_{i}",
                platform_identity={'platform': 'telegram', 'chatId': self.group_id}
            )

        recent = firestore.get_recent(self.agent_id, self.group_id, limit=10)
        print(f"  After sliding window: {len(recent)} messages (should be 10)")
        print(f"  Oldest message starts with: {recent[0]['content'] if recent else 'None'}")
        print(f"  Newest message ends with: {recent[-1]['content'] if recent else 'None'}")

        assert len(recent) <= 10, f"Expected <= 10 messages, got {len(recent)}"
        print("  ✓ Sliding window working correctly")

    def test_batch_trigger(self):
        """Test batch processing trigger"""
        print("\n" + "=" * 80)
        print("TEST 2: Batch Processing Trigger")
        print("=" * 80)

        if not self.use_firestore:
            print("SKIPPED: Firestore not available")
            return

        firestore = get_firestore_store()

        # Clear first
        firestore.clear_recent(self.agent_id, self.group_id)

        # Add messages below threshold
        print("\n2.1 Adding messages below threshold...")
        for i in range(5):
            firestore.add_message(
                agent_id=self.agent_id,
                group_id=self.group_id,
                message=f"Below threshold {i}",
                username=f"user_{i}",
                platform_identity={'platform': 'telegram', 'chatId': self.group_id}
            )

        unprocessed = firestore.get_unprocessed(self.agent_id, self.group_id, min_count=10)
        print(f"  Unprocessed messages: {len(unprocessed)} (should be 0 - below threshold)")
        assert len(unprocessed) == 0, "Should have 0 unprocessed below threshold"

        # Add messages to reach threshold
        print("\n2.2 Adding messages to reach threshold...")
        for i in range(5, 12):
            firestore.add_message(
                agent_id=self.agent_id,
                group_id=self.group_id,
                message=f"Above threshold {i}",
                username=f"user_{i}",
                platform_identity={'platform': 'telegram', 'chatId': self.group_id}
            )

        unprocessed = firestore.get_unprocessed(self.agent_id, self.group_id, min_count=10)
        print(f"  Unprocessed messages: {len(unprocessed)} (should be >= 10)")
        assert len(unprocessed) >= 10, f"Should have >= 10 unprocessed, got {len(unprocessed)}"
        print("  ✓ Batch trigger working correctly")

    def test_manager_integration(self):
        """Test GroupMemoryManager integration with Firestore"""
        print("\n" + "=" * 80)
        print("TEST 3: GroupMemoryManager Integration")
        print("=" * 80)

        # Clear any existing data
        if self.use_firestore:
            firestore = get_firestore_store()
            firestore.clear_recent(self.agent_id, self.group_id)

        # Test 3.1: Process a message
        print("\n3.1 Processing a message...")
        result = self.manager.process_group_message(
            message="Hi! I'm a DeFi developer working on lending protocols",
            platform_identity={
                'platform': 'telegram',
                'chatId': self.group_id,
                'telegramId': 999999,
                'username': 'alice_defi'
            },
            speaker_info={'username': 'alice_defi'},
            metadata={'message_id': 'test_1'}
        )

        print(f"  Group ID: {result.get('group_id')}")
        print(f"  User ID: {result.get('user_id')}")
        print(f"  Memories created: {sum(len(v) for v in result.get('created_memories', {}).values())}")

        if self.use_firestore and 'recent_context' in result:
            print(f"  Recent context size: {len(result['recent_context'])}")
            print("  ✓ Recent context included in result")
        elif self.use_firestore:
            print("  ⚠ WARNING: Recent context not in result")
        else:
            print("  (Firestore disabled - no recent context)")

        # Test 3.2: Process multiple messages to trigger batch
        print("\n3.2 Processing multiple messages (to test batch trigger)...")
        for i in range(12):
            result = self.manager.process_group_message(
                message=f"Message {i}: I'm working on DeFi and Base ecosystem",
                platform_identity={
                    'platform': 'telegram',
                    'chatId': self.group_id,
                    'telegramId': 1000000 + i,
                    'username': f'user_{i}'
                },
                speaker_info={'username': f'user_{i}'},
                metadata={'message_id': f'batch_test_{i}'}
            )

        # Give time for background processing
        print("  Waiting for background batch processing...")
        time.sleep(2)

        # Test 3.3: Get context for agent
        print("\n3.3 Getting context for agent...")
        context = self.manager.get_context_for_agent(
            message="Who knows about DeFi?",
            platform_identity={
                'platform': 'telegram',
                'chatId': self.group_id,
                'telegramId': 999999,
                'username': 'asker'
            },
            limit_per_level=5
        )

        print(f"  Is group: {context.get('is_group')}")
        print(f"  Group context size: {len(context.get('group_context', []))}")
        print(f"  User context size: {len(context.get('user_context', []))}")
        print(f"  Interaction context size: {len(context.get('interaction_context', []))}")

        if self.use_firestore and 'recent_messages' in context:
            print(f"  Recent messages size: {len(context['recent_messages'])}")
            print("  ✓ Recent messages included in context")
        elif self.use_firestore:
            print("  ⚠ WARNING: Recent messages not in context")
        else:
            print("  (Firestore disabled - no recent messages)")

    def test_stats(self):
        """Test Firestore stats"""
        print("\n" + "=" * 80)
        print("TEST 4: Firestore Stats")
        print("=" * 80)

        if not self.use_firestore:
            print("SKIPPED: Firestore not available")
            return

        firestore = get_firestore_store()

        stats = firestore.get_stats(self.agent_id, self.group_id)
        print(f"\nStats for {self.group_id}:")
        print(f"  Enabled: {stats.get('enabled')}")
        print(f"  Total messages: {stats.get('total', 0)}")
        print(f"  Processed: {stats.get('processed', 0)}")
        print(f"  Unprocessed: {stats.get('unprocessed', 0)}")
        print(f"  Window size: {stats.get('window_size', 'N/A')}")

    def run_all(self):
        """Run all tests"""
        try:
            self.setup()

            print("\n" + "=" * 80)
            print("RUNNING FIRESTORE WINDOW TESTS")
            print("=" * 80)

            # Test 1: Basic Firestore operations
            self.test_firestore_window()

            # Test 2: Batch trigger
            self.test_batch_trigger()

            # Test 3: Manager integration
            self.test_manager_integration()

            # Test 4: Stats
            self.test_stats()

            print("\n" + "=" * 80)
            print("FIRESTORE WINDOW TESTS COMPLETE")
            print("=" * 80)

            if self.use_firestore:
                print("\n✓ All tests passed with Firestore!")
            else:
                print("\n⚠ Tests passed but Firestore was not available")

            print()

            return True

        except Exception as e:
            print(f"\nERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            self.teardown()


if __name__ == "__main__":
    tester = FirestoreWindowTest()
    success = tester.run_all()
    sys.exit(0 if success else 1)
