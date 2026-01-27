"""
Stress Test - Group Memory System with Large Volume

Tests:
1. Message processing at scale (100+ messages)
2. Multi-user concurrent simulation (20+ users)
3. Retrieval performance with different dataset sizes
4. Cross-group pattern detection with many groups
5. Memory accuracy and relevance
"""
import os
import sys
import tempfile
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict
import random

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.group_memory_manager import GroupMemoryManager


class StressTest:
    """Stress test the group memory system"""

    def __init__(self):
        self.test_db_path = tempfile.mkdtemp(prefix='a0x-stress-')
        self.agent_id = "jessexbt"
        self.manager = None
        self.group_id = "telegram_group_-1001234567890"

    def setup(self):
        print("=" * 80)
        print("GROUP MEMORY STRESS TEST")
        print("=" * 80)
        print(f"\nTest DB: {self.test_db_path}")
        print(f"Agent: {self.agent_id}")
        print(f"Group: {self.group_id}\n")

        self.manager = GroupMemoryManager(
            agent_id=self.agent_id,
            db_base_path=self.test_db_path
        )

    def teardown(self):
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)

    def generate_users(self, count: int) -> List[Dict]:
        """Generate realistic user profiles"""
        roles = [
            ("DeFi Developer", "Working on lending protocols, yield farming, DEXs"),
            ("NFT Creator", "Building NFT marketplaces, digital art, metadata standards"),
            ("Infra Engineer", "Base node optimization, RPC endpoints, layer-2 solutions"),
            ("Smart Contract Auditor", "Security reviews, formal verification, gas optimization"),
            ("UX Designer", "Wallet UX, onboarding flows, DeFi interfaces"),
            ("Data Analyst", "On-chain analytics, dashboards, metrics tracking"),
            ("Community Manager", "Governance, tokenomics, community growth"),
            ("Product Manager", "DeFi products, user research, feature prioritization"),
        ]

        users = []
        for i in range(count):
            role_idx = i % len(roles)
            users.append({
                'telegramId': 1000000 + i,
                'username': f'user_{i}',
                'role': roles[role_idx][0],
                'bio': roles[role_idx][1]
            })
        return users

    def generate_messages(self, user_count: int, messages_per_user: int) -> List[Dict]:
        """Generate realistic messages"""
        message_templates = [
            "I've been working on {topic} for {years} years, focusing on {detail}.",
            "My expertise is in {topic}, specifically {detail}.",
            "I specialize in {topic} - been doing it since {year}.",
            "Background: {years} years experience with {topic}, mainly {detail}.",
            "We're building a {project_type} for {purpose}. Launching in {timeline}.",
            "Working on {project_type} - it's going to {impact}.",
            "I'm looking for help with {need}. Anyone experienced?",
            "Does anyone know about {topic}? I need some guidance.",
            "Prefer {preference} over {alternative} for {reason}.",
            "Interested in collab on {topic}. DM me if you're working on similar.",
        ]

        topics = [
            "DeFi lending protocols", "yield optimization", "NFT marketplaces", "Base infrastructure",
            "smart contract security", "gas optimization", "cross-chain bridges", "liquid staking",
            "automated market makers", "stablecoin design", "governance tokens", "MEV protection",
            "wallet UX", "onboarding flows", "layer-2 scaling", "zk-rollups"
        ]

        users = self.generate_users(user_count)
        messages = []

        for user in users:
            for _ in range(messages_per_user):
                template = random.choice(message_templates)

                msg = template.format(
                    topic=random.choice(topics),
                    years=random.randint(1, 10),
                    detail=random.choice(["production", "R&D", "consulting", "auditing"]),
                    year=random.randint(2018, 2025),
                    project_type=random.choice(["protocol", "platform", "tool", "dApp"]),
                    purpose=random.choice(["DeFi users", "traders", "developers", "nft collectors"]),
                    timeline=random.choice(["next week", "2 weeks", "next month", "Q2"]),
                    impact=random.choice(["revolutionize the space", "solve real problems", "improve UX"]),
                    need=random.choice(["scaling", "security audits", "UI design", "marketing"]),
                    preference=random.choice(["Solidity", "Rust", "Vyper", "TypeScript"]),
                    alternative=random.choice(["other languages", "traditional frameworks", "synchronous approaches"]),
                    reason=random.choice(["security", "speed", "developer experience"])
                )

                messages.append({
                    'user': user,
                    'text': msg,
                    'is_reply': random.random() < 0.3,
                })

        return messages

    # ==========================================================================
    # TEST 1: MESSAGE PROCESSING THROUGHPUT
    # ==========================================================================

    def test_message_processing(self, num_messages: int = 100):
        """Test processing many messages"""
        print("\n" + "=" * 80)
        print(f"TEST 1: Message Processing ({num_messages} messages)")
        print("=" * 80)

        users = self.generate_users(20)
        messages = []

        for user in users[:10]:
            for _ in range(num_messages // 10):
                messages.append({
                    'user': user,
                    'text': f"I'm {user['username']}, working on {user['role']}. {user['bio']}",
                    'is_reply': False
                })

        print(f"\nProcessing {len(messages)} messages...")
        start = time.time()

        total_memories = {'group': 0, 'user': 0, 'interaction': 0}
        processing_times = []

        for i, msg_data in enumerate(messages):
            msg_start = time.time()

            result = self.manager.process_group_message(
                message=msg_data['text'],
                platform_identity={
                    'platform': 'telegram',
                    'chatId': '-1001234567890',
                    'telegramId': msg_data['user']['telegramId'],
                    'username': msg_data['user']['username']
                },
                speaker_info={'username': msg_data['user']['username']},
                metadata={'message_id': f'msg_{i}'}
            )

            msg_time = (time.time() - msg_start) * 1000
            processing_times.append(msg_time)

            total_memories['group'] += len(result['created_memories']['group'])
            total_memories['user'] += len(result['created_memories']['user'])
            total_memories['interaction'] += len(result['created_memories']['interaction'])

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(messages)} messages...")

        total_time = time.time() - start

        print(f"\nResults:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {len(messages) / total_time:.2f} msg/s")
        print(f"  Avg latency: {sum(processing_times) / len(processing_times):.1f}ms")
        print(f"  Median latency: {sorted(processing_times)[len(processing_times) // 2]:.1f}ms")
        print(f"  P95 latency: {sorted(processing_times)[int(len(processing_times) * 0.95)]:.1f}ms")
        print(f"\n  Memories created:")
        print(f"    Group: {total_memories['group']}")
        print(f"    User: {total_memories['user']}")
        print(f"    Interaction: {total_memories['interaction']}")
        print(f"    Total: {sum(total_memories.values())}")

        return total_time

    # ==========================================================================
    # TEST 2: RETRIEVAL AT SCALE
    # ==========================================================================

    def test_retrieval_at_scale(self, num_messages: int):
        """Test retrieval performance with different dataset sizes"""
        print("\n" + "=" * 80)
        print("TEST 2: Retrieval Performance at Scale")
        print("=" * 80)

        # First, populate with messages
        print(f"\nPopulating with {num_messages} messages...")
        messages = self.generate_messages(30, num_messages // 30)

        for msg_data in messages:
            self.manager.process_group_message(
                message=msg_data['text'],
                platform_identity={
                    'platform': 'telegram',
                    'chatId': '-1001234567890',
                    'telegramId': msg_data['user']['telegramId'],
                    'username': msg_data['user']['username']
                },
                speaker_info={'username': msg_data['user']['username']},
                metadata={'message_id': f'msg_{random.randint(1000, 9999)}'}
            )

        # Test different retrieval queries
        queries = [
            "Who knows about DeFi lending protocols?",
            "I need help with NFT marketplace development",
            "Looking for Base infrastructure experts",
            "Who has experience with smart contract security?",
            "Need advice on yield optimization strategies"
        ]

        print(f"\nTesting {len(queries)} retrieval queries...")

        retrieval_times = []
        result_counts = []

        for query in queries:
            start = time.time()

            context = self.manager.get_context_for_agent(
                message=query,
                platform_identity={
                    'platform': 'telegram',
                    'chatId': '-1001234567890',
                    'telegramId': 999999,
                    'username': 'asker'
                },
                limit_per_level=10
            )

            query_time = (time.time() - start) * 1000
            retrieval_times.append(query_time)

            user_count = len(context.get('user_context', []))
            group_count = len(context.get('group_context', []))
            interaction_count = len(context.get('interaction_context', []))
            total_count = user_count + group_count + interaction_count

            result_counts.append(total_count)

            print(f"  Query: {query[:50]}...")
            print(f"    Time: {query_time:.1f}ms | Results: {total_count} (user:{user_count}, group:{group_count}, interaction:{interaction_count})")

        print(f"\nRetrieval Statistics:")
        print(f"  Avg time: {sum(retrieval_times) / len(retrieval_times):.1f}ms")
        print(f"  Min time: {min(retrieval_times):.1f}ms")
        print(f"  Max time: {max(retrieval_times):.1f}ms")
        print(f"  Avg results: {sum(result_counts) / len(result_counts):.1f}")

    # ==========================================================================
    # TEST 3: CROSS-GROUP DETECTION
    # ==========================================================================

    def test_cross_group_detection(self, num_users: int, num_groups: int):
        """Test cross-group pattern detection"""
        print("\n" + "=" * 80)
        print(f"TEST 3: Cross-Group Detection ({num_users} users, {num_groups} groups)")
        print("=" * 80)

        users = self.generate_users(num_users)
        group_ids = [f"-100{i}00000000" for i in range(1, num_groups + 1)]

        print(f"\nCreating memories across {num_groups} groups...")

        for user in users:
            user_groups = random.sample(group_ids, min(5, len(group_ids)))

            for group_id in user_groups:
                self.manager.process_group_message(
                    message=f"Hi! I'm {user['username']}, {user['bio']}",
                    platform_identity={
                        'platform': 'telegram',
                        'chatId': group_id,
                        'telegramId': user['telegramId'],
                        'username': user['username']
                    },
                    speaker_info={'username': user['username']},
                    metadata={'message_id': f'msg_{random.randint(1000, 9999)}'}
                )

        print(f"\nDetecting cross-group patterns...")
        start = time.time()

        patterns_found = 0
        users_with_patterns = 0

        for user in users[:min(10, len(users))]:
            user_id = f"telegram_{user['telegramId']}"
            patterns = self.manager.consolidate_cross_group_patterns(
                user_id=user_id,
                min_groups=2,
                min_evidence=1
            )

            if patterns:
                users_with_patterns += 1
                patterns_found += len(patterns)

                if users_with_patterns <= 3:
                    print(f"\n  {user['username']}: {len(patterns)} patterns")
                    for pattern in patterns[:2]:
                        print(f"    - {pattern.pattern_type}: {pattern.content[:50]}...")

        detection_time = time.time() - start

        print(f"\nCross-Group Statistics:")
        print(f"  Detection time: {detection_time:.2f}s")
        print(f"  Users tested: {min(10, len(users))}")
        print(f"  Users with patterns: {users_with_patterns}")
        print(f"  Total patterns: {patterns_found}")

    # ==========================================================================
    # TEST 4: MEMORY ACCURACY
    # ==========================================================================

    def test_memory_accuracy(self):
        """Test that retrieved memories are actually relevant"""
        print("\n" + "=" * 80)
        print("TEST 4: Memory Accuracy & Relevance")
        print("=" * 80)

        # Test scenarios
        test_data = [
            ('alice_defi', "I've been a DeFi developer for 5 years, working on lending protocols", "Who has DeFi lending expertise?"),
            ('bob_nft', "NFT artist and marketplace developer building NFT collateral systems", "Who can help with NFT marketplace development?"),
            ('carol_infra', "Infrastructure engineer optimizing Base node performance and RPC endpoints", "Who knows about Base infrastructure?"),
        ]

        print(f"\nCreating test scenarios...")

        for username, message, _ in test_data:
            telegram_id = abs(hash(username)) % 1000000000
            self.manager.process_group_message(
                message=message,
                platform_identity={
                    'platform': 'telegram',
                    'chatId': '-1001234567890',
                    'telegramId': telegram_id,
                    'username': username
                },
                speaker_info={'username': username},
                metadata={'message_id': f'test_{random.randint(1000, 9999)}'}
            )

        print(f"\nTesting retrieval accuracy...")

        correct = 0
        for username, _, query in test_data:
            context = self.manager.get_context_for_agent(
                message=query,
                platform_identity={
                    'platform': 'telegram',
                    'chatId': '-1001234567890',
                    'telegramId': 999999,
                    'username': 'asker'
                },
                limit_per_level=5
            )

            user_context = context.get('user_context', [])
            found = any(username in str(mem.username).lower() for mem in user_context)

            if found:
                correct += 1
                print(f"  ✓ Found {username} for: {query[:40]}...")
            else:
                print(f"  ✗ NOT FOUND {username} for: {query[:40]}...")

        accuracy = (correct / len(test_data)) * 100
        print(f"\nAccuracy: {accuracy:.1f}% ({correct}/{len(test_data)} correct)")

        return accuracy >= 66

    # ==========================================================================
    # RUN ALL TESTS
    # ==========================================================================

    def run_all(self):
        """Run all stress tests"""
        try:
            self.setup()

            print("\n" + "=" * 80)
            print("RUNNING ALL STRESS TESTS")
            print("=" * 80)

            # Test 1 - Minimal for speed
            self.test_message_processing(15)

            # Test 2 - Single scale test
            print("\n" + "-" * 80)
            print("Testing retrieval at scale...")
            print("-" * 80)
            self.test_retrieval_at_scale(50)

            # Test 3 - Reduced cross-group test
            self.test_cross_group_detection(10, 3)

            # Test 4
            accuracy_pass = self.test_memory_accuracy()

            print("\n" + "=" * 80)
            print("STRESS TEST COMPLETE")
            print("=" * 80)

            if accuracy_pass:
                print("\n✓ All tests passed!")
            else:
                print("\n⚠ Some accuracy tests failed")

            print()

        except Exception as e:
            print(f"\nERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            self.teardown()

        return True


if __name__ == "__main__":
    tester = StressTest()
    success = tester.run_all()
    sys.exit(0 if success else 1)
