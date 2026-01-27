"""
Integration Test - Full System Group Memory with a0x Agents

This test simulates the complete flow of group memory integration:
1. Receive a message from Telegram (group context)
2. Process the message and create memories
3. Retrieve context with multi-level search
4. Test cross-group consolidation
5. Verify backward compatibility with VectorStore

Prerequisites:
- LanceDB accessible at /tmp/a0x-memory-test
- Embedding model configured in config.py
"""
import os
import sys
import tempfile
import shutil
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from services.group_memory_manager import GroupMemoryManager
from models.group_memory import MemoryType, PrivacyScope
from database.group_memory_store import GroupMemoryStore
from database.vector_store import VectorStore
from utils.embedding import EmbeddingModel


class TestFullSystemIntegration:
    """Integration tests for the complete group memory system"""

    def __init__(self):
        self.test_db_path = tempfile.mkdtemp(prefix='a0x-memory-test-')
        self.agent_id = "test-agent-123"
        self.manager = None

    def setup(self):
        """Setup test environment"""
        print(f"\n{'='*60}")
        print("Setting up test environment...")
        print(f"Test DB path: {self.test_db_path}")
        print(f"Agent ID: {self.agent_id}")
        print(f"{'='*60}\n")

        # Initialize manager
        self.manager = GroupMemoryManager(
            agent_id=self.agent_id,
            db_base_path=self.test_db_path
        )

    def teardown(self):
        """Cleanup test environment"""
        print(f"\n{'='*60}")
        print("Cleaning up test environment...")
        print(f"Removing: {self.test_db_path}")
        print(f"{'='*60}\n")

        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)

    # ==========================================================================
    # TEST 1: Telegram Group Message Processing
    # ==========================================================================

    def test_telegram_group_message(self):
        """Test processing a Telegram group message"""
        print("\n--- Test 1: Telegram Group Message Processing ---\n")

        # Simulate a Telegram group message
        platform_identity = {
            'platform': 'telegram',
            'chatId': '-1001234567890',  # Negative = group
            'telegramId': 123456789,
            'username': 'alice_builder'
        }

        message = "Hey everyone! I've decided to specialize in DeFi protocols on Base. Anyone working on lending protocols?"

        speaker_info = {
            'username': 'alice_builder',
            'first_name': 'Alice',
            'last_name': 'Builder'
        }

        metadata = {
            'message_id': 'msg_001',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'is_reply': False,
            'mentioned_users': []
        }

        # Process the message
        result = self.manager.process_group_message(
            message=message,
            platform_identity=platform_identity,
            speaker_info=speaker_info,
            metadata=metadata
        )

        print(f"Message processed:")
        print(f"  Group ID: {result['group_id']}")
        print(f"  User ID: {result['user_id']}")
        print(f"  Memories created:")
        print(f"    Group: {len(result['created_memories']['group'])}")
        print(f"    User: {len(result['created_memories']['user'])}")
        print(f"    Interaction: {len(result['created_memories']['interaction'])}")

        # Verify results
        assert result['group_id'] == 'telegram_group_-1001234567890'
        assert result['user_id'] == 'telegram_123456789'
        # Should create user memory (expertise)
        assert len(result['created_memories']['user']) > 0

        print("  PASSED\n")
        return result

    # ==========================================================================
    # TEST 2: Context Retrieval for Agent
    # ==========================================================================

    def test_context_retrieval(self):
        """Test getting context for agent response"""
        print("\n--- Test 2: Context Retrieval for Agent ---\n")

        platform_identity = {
            'platform': 'telegram',
            'chatId': '-1001234567890',
            'telegramId': 123456789,
            'username': 'alice_builder'
        }

        current_message = "Can you recommend some Base DeFi resources?"

        # Get context
        context = self.manager.get_context_for_agent(
            message=current_message,
            platform_identity=platform_identity,
            limit_per_level=5,
            include_user_profile=True
        )

        print(f"Context retrieved:")
        print(f"  Is group: {context['is_group']}")
        print(f"  Group ID: {context['group_id']}")
        print(f"  User ID: {context['user_id']}")
        print(f"  Retrieved at: {context['retrieved_at']}")

        if 'group_context' in context:
            print(f"  Group context: {len(context['group_context'])} memories")
        if 'user_context' in context:
            print(f"  User context: {len(context['user_context'])} memories")
        if 'interaction_context' in context:
            print(f"  Interaction context: {len(context['interaction_context'])} memories")

        # Verify
        assert context['is_group'] == True
        assert context['group_id'] == 'telegram_group_-1001234567890'

        print("  PASSED\n")
        return context

    # ==========================================================================
    # TEST 3: Multiple Messages from Different Users
    # ==========================================================================

    def test_multiple_users(self):
        """Test processing messages from multiple users"""
        print("\n--- Test 3: Multiple Users in Group ---\n")

        users = [
            {'telegramId': 111, 'username': 'bob_dev', 'name': 'Bob'},
            {'telegramId': 222, 'username': 'carol_design', 'name': 'Carol'},
            {'telegramId': 333, 'username': 'dave_pm', 'name': 'Dave'}
        ]

        messages = [
            ("I'm interested in UX design for DeFi apps", 'preference'),
            ("I have experience with Solidity smart contracts", 'expertise'),
            ("I prefer to work on consumer-facing applications", 'preference')
        ]

        platform_identity_base = {
            'platform': 'telegram',
            'chatId': '-1001234567890'
        }

        for i, (user, (msg, msg_type)) in enumerate(zip(users, messages)):
            platform_identity = {
                **platform_identity_base,
                'telegramId': user['telegramId'],
                'username': user['username']
            }

            result = self.manager.process_group_message(
                message=msg,
                platform_identity=platform_identity,
                speaker_info={'username': user['username'], 'first_name': user['name']},
                metadata={'message_id': f'msg_00{i+2}'}
            )

            print(f"  User {user['username']}: {len(result['created_memories']['user'])} user memories")

        print("  PASSED\n")

    # ==========================================================================
    # TEST 4: Direct Message (Backward Compatibility)
    # ==========================================================================

    def test_direct_message(self):
        """Test DM processing with VectorStore"""
        print("\n--- Test 4: Direct Message (VectorStore) ---\n")

        platform_identity = {
            'platform': 'direct',
            'clientId': 'client_001',
            'username': 'dm_user'
        }

        message = "Tell me about Base grants"

        result = self.manager.process_group_message(
            message=message,
            platform_identity=platform_identity,
            metadata={'message_id': 'dm_001'}
        )

        print(f"DM processed:")
        print(f"  Group ID: {result['group_id']} (should be None)")
        print(f"  User ID: {result['user_id']}")
        print(f"  Vector store entries: {result['created_memories'].get('vector_store', [])}")

        # Verify
        assert result['group_id'] is None
        assert result['user_id'] == 'direct_client_001'

        print("  PASSED\n")

    # ==========================================================================
    # TEST 5: Cross-Group Consolidation
    # ==========================================================================

    def test_cross_group_consolidation(self):
        """Test cross-group pattern detection"""
        print("\n--- Test 5: Cross-Group Consolidation ---\n")

        # Add memories for same user in multiple groups
        user_id = 'telegram_123456789'

        # Simulate user being in multiple groups
        groups = ['-100111111111', '-100222222222', '-100333333333']

        for group_id in groups:
            platform_identity = {
                'platform': 'telegram',
                'chatId': group_id,
                'telegramId': 123456789,
                'username': 'alice_builder'
            }

            # User expresses expertise in DeFi across groups
            message = "I've been working on DeFi protocols for 2 years"

            self.manager.process_group_message(
                message=message,
                platform_identity=platform_identity,
                speaker_info={'username': 'alice_builder'},
                metadata={'message_id': f'group_{group_id}_msg'}
            )

        # Consolidate patterns
        patterns = self.manager.consolidate_cross_group_patterns(
            user_id=user_id,
            min_groups=2,
            min_evidence=1
        )

        print(f"Cross-group consolidation:")
        print(f"  User ID: {user_id}")
        print(f"  Patterns detected: {len(patterns)}")

        for pattern in patterns:
            print(f"    - {pattern.pattern_type}: {pattern.content[:50]}...")
            print(f"      Groups: {pattern.groups_involved}")
            print(f"      Confidence: {pattern.confidence_score}")

        print("  PASSED\n")

    # ==========================================================================
    # TEST 6: Context with Cross-Group Patterns
    # ==========================================================================

    def test_context_with_cross_group(self):
        """Test context retrieval includes cross-group patterns"""
        print("\n--- Test 6: Context with Cross-Group Patterns ---\n")

        platform_identity = {
            'platform': 'telegram',
            'chatId': '-1001234567890',
            'telegramId': 123456789,
            'username': 'alice_builder'
        }

        context = self.manager.get_context_for_agent(
            message="What are my expertise areas?",
            platform_identity=platform_identity,
            limit_per_level=3
        )

        print(f"Context with cross-group patterns:")
        if 'cross_group_context' in context:
            print(f"  Cross-group context: {len(context['cross_group_context'])} patterns")
            for pattern in context['cross_group_context']:
                print(f"    - {pattern.pattern_type}: {pattern.content[:50]}...")

        print("  PASSED\n")

    # ==========================================================================
    # TEST 7: Different Platforms
    # ==========================================================================

    def test_different_platforms(self):
        """Test messages from different platforms"""
        print("\n--- Test 7: Different Platforms ---\n")

        platforms = [
            {
                'name': 'XMTP',
                'identity': {
                    'platform': 'xmtp',
                    'conversationId': 'groups/123',
                    'walletAddress': '0x1234567890abcdef'
                }
            },
            {
                'name': 'Farcaster',
                'identity': {
                    'platform': 'farcaster',
                    'fid': 12345,
                    'channel': 'base',
                    'username': 'alice'
                }
            },
            {
                'name': 'Twitter',
                'identity': {
                    'platform': 'twitter',
                    'username': 'alice_builder'
                }
            }
        ]

        for platform in platforms:
            result = self.manager.process_group_message(
                message=f"Hello from {platform['name']}!",
                platform_identity=platform['identity'],
                metadata={'message_id': f"{platform['name']}_msg"}
            )

            is_group = result['group_id'] is not None
            print(f"  {platform['name']}: group={is_group}, group_id={result['group_id']}")

        print("  PASSED\n")

    # ==========================================================================
    # RUN ALL TESTS
    # ==========================================================================

    def run_all_tests(self):
        """Run all integration tests"""
        print("\n" + "="*60)
        print("FULL SYSTEM INTEGRATION TEST")
        print("="*60)

        try:
            self.setup()

            # Run tests
            self.test_telegram_group_message()
            self.test_context_retrieval()
            self.test_multiple_users()
            self.test_direct_message()
            self.test_cross_group_consolidation()
            self.test_context_with_cross_group()
            self.test_different_platforms()

            print("\n" + "="*60)
            print("ALL TESTS PASSED!")
            print("="*60 + "\n")

            return True

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"TEST FAILED: {str(e)}")
            print(f"{'='*60}\n")
            import traceback
            traceback.print_exc()
            return False

        finally:
            self.teardown()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    tester = TestFullSystemIntegration()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
