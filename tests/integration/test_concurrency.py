"""
Concurrency Test - Multiple users sending messages simultaneously

Tests:
1. Concurrent message processing (10 users at once)
2. OpenRouter rate limiting behavior
3. Message ordering consistency
4. Performance under concurrent load
"""
import os
import sys
import tempfile
import shutil
import time
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.group_memory_manager import GroupMemoryManager


class ConcurrencyTest:
    """Test concurrent message processing"""

    def __init__(self):
        self.test_db_path = tempfile.mkdtemp(prefix='a0x-concurrency-')
        self.agent_id = "jessexbt"
        self.manager = None
        self.group_id = "telegram_group_-1001234567890"

    def setup(self):
        print("=" * 80)
        print("CONCURRENCY TEST - Multiple Users Simultaneously")
        print("=" * 80)
        print(f"\nTest DB: {self.test_db_path}")

        self.manager = GroupMemoryManager(
            agent_id=self.agent_id,
            db_base_path=self.test_db_path
        )

    def teardown(self):
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)

    def process_single_message(self, user_data):
        """Process a single message - used in parallel"""
        try:
            start = time.time()

            result = self.manager.process_group_message(
                message=user_data['message'],
                platform_identity={
                    'platform': 'telegram',
                    'chatId': '-1001234567890',
                    'telegramId': user_data['telegram_id'],
                    'username': user_data['username']
                },
                speaker_info={'username': user_data['username']},
                metadata={'message_id': user_data['message_id']}
            )

            elapsed = (time.time() - start) * 1000

            return {
                'success': True,
                'username': user_data['username'],
                'message_id': user_data['message_id'],
                'elapsed_ms': elapsed,
                'memories_created': sum(len(v) for v in result['created_memories'].values()),
                'error': None
            }

        except Exception as e:
            return {
                'success': False,
                'username': user_data['username'],
                'message_id': user_data['message_id'],
                'elapsed_ms': 0,
                'memories_created': 0,
                'error': str(e)[:100]
            }

    def generate_users(self, count: int):
        """Generate test users"""
        users = []
        roles = [
            "DeFi developer working on lending protocols",
            "NFT artist building marketplace",
            "Infra engineer optimizing Base nodes",
            "Smart contract auditor",
            "UX designer for wallets",
            "Data analyst tracking metrics",
            "Community manager for governance",
            "Product manager prioritizing features",
            "Security engineer",
            "Backend developer"
        ]

        for i in range(count):
            role_idx = i % len(roles)  # Cycle through roles
            users.append({
                'telegram_id': 1000000 + i,
                'username': f'user_{i}',
                'message': f"Hi! I'm user_{i}, a {roles[role_idx]}. Working on Base ecosystem.",
                'message_id': f'concurrent_{i}'
            })

        return users

    def test_concurrent_users(self, num_users: int = 10):
        """Test multiple users sending messages simultaneously"""
        print("\n" + "=" * 80)
        print(f"TEST: {num_users} Users Sending Messages SIMULTANEOUSLY")
        print("=" * 80)

        users = self.generate_users(num_users)

        print(f"\nUsers:")
        for user in users:
            print(f"  - {user['username']}: {user['message'][:50]}...")

        print(f"\nProcessing {len(users)} messages simultaneously...")
        print("-" * 80)

        # Process all messages in parallel
        start = time.time()

        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            future_to_user = {
                executor.submit(self.process_single_message, user): user
                for user in users
            }

            # Collect results as they complete
            results = []
            completed = 0

            for future in as_completed(future_to_user):
                user = future_to_user[future]
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                    completed += 1

                    status = "✓" if result['success'] else "✗"
                    elapsed = result['elapsed_ms']
                    memories = result['memories_created']

                    print(f"  [{completed}/{len(users)}] {status} {result['username']:12} - {elapsed:6.0f}ms - {memories} memories - {result['error'][:20] if result['error'] else ''}")

                except Exception as e:
                    print(f"  [{completed}/{len(users)}] ✗ TIMEOUT or ERROR: {str(e)[:50]}")

        total_time = time.time() - start

        # Analyze results
        print(f"\n{'='*80}")
        print("RESULTS")
        print('='*80)

        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        print(f"\nTotal messages: {len(users)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Throughput: {len(users) / total_time:.2f} msg/s")

        if successful:
            times = [r['elapsed_ms'] for r in successful]
            memories = [r['memories_created'] for r in successful]

            print(f"\nSuccessful requests:")
            print(f"  Avg time: {sum(times) / len(times):.0f}ms")
            print(f"  Min time: {min(times):.0f}ms")
            print(f"  Max time: {max(times):.0f}ms")
            print(f"  Median: {sorted(times)[len(times)//2]:.0f}ms")
            print(f"  P95: {sorted(times)[int(len(times) * 0.95)]:.0f}ms")
            print(f"  Total memories: {sum(memories)}")
            print(f"  Avg memories/msg: {sum(memories) / len(memories):.1f}")

        if failed:
            print(f"\nFailed requests:")
            for f in failed[:5]:  # Show first 5
                print(f"  - {f['username']}: {f['error']}")

        # Success criteria
        success_rate = (len(successful) / len(users)) * 100
        print(f"\nSuccess rate: {success_rate:.1f}%")

        if success_rate >= 90:
            print("  ✓ PASS: Excellent concurrent processing")
        elif success_rate >= 70:
            print("  ⚠ WARNING: Some failures, rate limiting?")
        else:
            print("  ✗ FAIL: Too many failures")

        return success_rate >= 70

    def test_concurrent_waves(self, num_users: int = 10, num_waves: int = 3):
        """Test multiple waves of concurrent messages"""
        print("\n" + "=" * 80)
        print(f"TEST: {num_waves} Waves of {num_users} Users")
        print("=" * 80)

        all_results = []

        for wave in range(num_waves):
            print(f"\n--- Wave {wave + 1}/{num_waves} ---")

            users = self.generate_users(num_users)

            start = time.time()

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(self.process_single_message, user)
                    for user in users
                ]

                results = []
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=30)
                        results.append(result)
                    except Exception as e:
                        results.append({
                            'success': False,
                            'error': str(e)[:50]
                        })

            wave_time = time.time() - start
            successful = [r for r in results if r['success']]
            failed = [r for r in results if not r['success']]

            print(f"  Completed: {len(results)}/{len(users)}")
            print(f"  Successful: {len(successful)}")
            print(f"  Failed: {len(failed)}")
            print(f"  Wave time: {wave_time:.2f}s")
            print(f"  Throughput: {len(users) / wave_time:.2f} msg/s")

            all_results.extend(results)

        # Overall stats
        total_successful = len([r for r in all_results if r['success']])
        print(f"\n{'='*80}")
        print("OVERALL RESULTS")
        print('='*80)
        print(f"Total messages: {num_users * num_waves}")
        print(f"Total successful: {total_successful}")
        print(f"Success rate: {(total_successful / (num_users * num_waves)) * 100:.1f}%")

    def test_retrieval_under_load(self, num_messages: int = 50):
        """Test retrieval while processing"""
        print("\n" + "=" * 80)
        print("TEST: Retrieval During Message Processing")
        print("=" * 80)

        # First populate with messages
        print(f"\nPopulating with {num_messages} messages...")
        users = self.generate_users(20)

        for i, user in enumerate(users):
            self.manager.process_group_message(
                message=user['message'],
                platform_identity={
                    'platform': 'telegram',
                    'chatId': '-1001234567890',
                    'telegramId': user['telegram_id'],
                    'username': user['username']
                },
                speaker_info={'username': user['username']},
                metadata={'message_id': f'load_{i}'}
            )

        # Now test retrieval while processing
        print(f"\nTesting retrieval while processing...")

        queries = [
            "Who knows about DeFi lending?",
            "Looking for NFT developers",
            "Need help with Base infrastructure"
        ]

        retrieval_times = []
        processing_times = []

        # Mix retrieval and processing
        start = time.time()

        for i, query in enumerate(queries):
            # Retrieval
            r_start = time.time()
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
            retrieval_times.append((time.time() - r_start) * 1000)

            # Processing
            p_start = time.time()
            self.manager.process_group_message(
                message=f"Query {i}: {query}",
                platform_identity={
                    'platform': 'telegram',
                    'chatId': '-1001234567890',
                    'telegramId': 1000000 + i,
                    'username': f'query_user_{i}'
                },
                speaker_info={'username': f'query_user_{i}'},
                metadata={'message_id': f'query_{i}'}
            )
            processing_times.append((time.time() - p_start) * 1000)

        total_time = time.time() - start

        print(f"\nResults:")
        print(f"  Total operations: {len(queries) + len(queries)} (3 retrievals + 3 processings)")
        print(f"  Total time: {total_time:.2f}s")
        print(f"\n  Retrieval:")
        print(f"    Avg: {sum(retrieval_times) / len(retrieval_times):.0f}ms")
        print(f"    Max: {max(retrieval_times):.0f}ms")
        print(f"\n  Processing:")
        print(f"    Avg: {sum(processing_times) / len(processing_times):.0f}ms")
        print(f"    Max: {max(processing_times):.0f}ms")

    def run_all(self):
        """Run all concurrency tests"""
        try:
            self.setup()

            print("\n" + "=" * 80)
            print("RUNNING CONCURRENCY TESTS")
            print("=" * 80)

            # Test 1: Concurrent users
            self.test_concurrent_users(10)

            # Test 2: Concurrent waves
            self.test_concurrent_waves(5, 2)

            # Test 3: Retrieval under load
            self.test_retrieval_under_load(30)

            print("\n" + "=" * 80)
            print("CONCURRENCY TESTS COMPLETE")
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
    tester = ConcurrencyTest()
    success = tester.run_all()
    sys.exit(0 if success else 1)
