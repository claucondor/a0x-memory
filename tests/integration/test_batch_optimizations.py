"""
Batch Optimizations Test - Verify parallel processing improvements

Tests:
1. Batch embeddings performance
2. Parallel cross-group consolidation
3. Comparison: sequential vs batch processing
"""
import os
import sys
import tempfile
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.group_memory_manager import GroupMemoryManager
from models.group_memory import GroupMemory, UserMemory, MemoryType, PrivacyScope


class BatchOptimizationsTest:
    """Test batch processing optimizations"""

    def __init__(self):
        self.test_db_path = tempfile.mkdtemp(prefix='a0x-batch-')
        self.agent_id = "jessexbt"
        self.manager = None

    def setup(self):
        print("=" * 80)
        print("BATCH OPTIMIZATIONS TEST")
        print("=" * 80)
        print(f"\nTest DB: {self.test_db_path}")

        self.manager = GroupMemoryManager(
            agent_id=self.agent_id,
            db_base_path=self.test_db_path
        )

    def teardown(self):
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)

    def test_batch_vs_sequential(self, num_memories: int = 20):
        """Compare batch vs sequential memory insertion"""
        print("\n" + "=" * 80)
        print(f"TEST: Batch vs Sequential ({num_memories} memories)")
        print("=" * 80)

        # Create test memories
        user_memories = []
        for i in range(num_memories):
            user_memories.append(UserMemory(
                agent_id=self.agent_id,
                group_id="telegram_group_-1001234567890",
                user_id=f"telegram_{1000000 + i}",
                username=f"user_{i}",
                platform="telegram",
                memory_type=MemoryType.EXPERTISE,
                content=f"User {i} is an expert in DeFi lending protocols and yield optimization",
                topics=["DeFi", "lending", "yield"],
                keywords=["expert", "protocols", "optimization"],
                importance_score=0.8,
                privacy_scope=PrivacyScope.PUBLIC
            ))

        # Test sequential (using individual add)
        print("\n1. Sequential insertion (individual adds)...")
        sequential_store = self.manager.group_store
        start = time.time()

        for mem in user_memories[:num_memories//2]:
            # Create new memories with unique IDs
            new_mem = UserMemory(
                agent_id=mem.agent_id,
                group_id=mem.group_id,
                user_id=mem.user_id + "_seq",
                username=mem.username + "_seq",
                platform=mem.platform,
                memory_type=mem.memory_type,
                content=mem.content,
                topics=mem.topics,
                keywords=mem.keywords,
                importance_score=mem.importance_score,
                privacy_scope=mem.privacy_scope
            )
            sequential_store.add_user_memory(new_mem)

        sequential_time = time.time() - start
        print(f"   Time: {sequential_time*1000:.0f}ms")
        print(f"   Avg per memory: {(sequential_time/(num_memories//2))*1000:.0f}ms")

        # Test batch (using batch add)
        print("\n2. Batch insertion (single batch call)...")
        start = time.time()

        batch_memories = []
        for mem in user_memories[num_memories//2:]:
            new_mem = UserMemory(
                agent_id=mem.agent_id,
                group_id=mem.group_id,
                user_id=mem.user_id + "_batch",
                username=mem.username + "_batch",
                platform=mem.platform,
                memory_type=mem.memory_type,
                content=mem.content,
                topics=mem.topics,
                keywords=mem.keywords,
                importance_score=mem.importance_score,
                privacy_scope=mem.privacy_scope
            )
            batch_memories.append(new_mem)

        sequential_store.add_user_memories_batch(batch_memories)
        batch_time = time.time() - start
        print(f"   Time: {batch_time*1000:.0f}ms")
        print(f"   Avg per memory: {(batch_time/len(batch_memories))*1000:.0f}ms")

        # Calculate speedup
        speedup = sequential_time / batch_time if batch_time > 0 else 1
        print(f"\n3. Results:")
        print(f"   Sequential: {sequential_time*1000:.0f}ms")
        print(f"   Batch: {batch_time*1000:.0f}ms")
        print(f"   Speedup: {speedup:.1f}x")

        if speedup >= 1.5:
            print("   ✓ PASS: Batch processing is significantly faster")
        elif speedup >= 1.1:
            print("   ⚠ INFO: Batch processing is slightly faster")
        else:
            print("   ⚠ INFO: Batch processing similar to sequential (cache warm?)")

        return speedup

    def test_parallel_consolidation(self, num_users: int = 10):
        """Test parallel cross-group consolidation"""
        print("\n" + "=" * 80)
        print(f"TEST: Parallel Cross-Group Consolidation ({num_users} users)")
        print("=" * 80)

        # First, create memories across multiple groups for each user
        print("\nPopulating test data...")
        group_ids = [f"-100{i}00000000" for i in range(1, 6)]  # 5 groups

        for i in range(num_users):
            user_id = f"telegram_{1000000 + i}"
            username = f"user_{i}"

            # Add memories to 3-4 random groups per user
            for group_id in group_ids[:3]:
                self.manager.process_group_message(
                    message=f"Hi! I'm {username}, working on DeFi and Base ecosystem",
                    platform_identity={
                        'platform': 'telegram',
                        'chatId': group_id,
                        'telegramId': 1000000 + i,
                        'username': username
                    },
                    speaker_info={'username': username},
                    metadata={'message_id': f'test_{i}_{group_id}'}
                )

        user_ids = [f"telegram_{1000000 + i}" for i in range(num_users)]

        # Test sequential consolidation
        print(f"\n1. Sequential consolidation...")
        start = time.time()

        sequential_results = {}
        for user_id in user_ids[:num_users//2]:
            patterns = self.manager.consolidate_cross_group_patterns(
                user_id=user_id,
                min_groups=2,
                min_evidence=1
            )
            sequential_results[user_id] = patterns

        sequential_time = time.time() - start
        print(f"   Time: {sequential_time*1000:.0f}ms")
        print(f"   Users processed: {len(user_ids[:num_users//2])}")

        # Test parallel consolidation
        print(f"\n2. Parallel consolidation...")
        start = time.time()

        parallel_results = self.manager.consolidate_cross_group_patterns_parallel(
            user_ids=user_ids[num_users//2:],
            min_groups=2,
            min_evidence=1,
            max_workers=5
        )

        parallel_time = time.time() - start
        print(f"   Time: {parallel_time*1000:.0f}ms")
        print(f"   Users processed: {len(user_ids[num_users//2:])}")

        # Calculate speedup
        seq_per_user = sequential_time / (num_users // 2)
        par_per_user = parallel_time / (num_users // 2)
        speedup = seq_per_user / par_per_user if par_per_user > 0 else 1

        print(f"\n3. Results:")
        print(f"   Sequential: {seq_per_user*1000:.0f}ms per user")
        print(f"   Parallel: {par_per_user*1000:.0f}ms per user")
        print(f"   Speedup: {speedup:.1f}x")

        if speedup >= 2.0:
            print("   ✓ PASS: Parallel processing is significantly faster")
        elif speedup >= 1.5:
            print("   ⚠ INFO: Parallel processing is moderately faster")
        else:
            print("   ⚠ INFO: Limited speedup (small dataset or overhead)")

        return speedup

    def test_end_to_end_batch(self, num_messages: int = 10):
        """Test end-to-end message processing with batch optimizations"""
        print("\n" + "=" * 80)
        print(f"TEST: End-to-End Batch Processing ({num_messages} messages)")
        print("=" * 80)

        messages = []
        for i in range(num_messages):
            messages.append({
                'message': f"I'm user_{i}, specializing in DeFi protocols and Base ecosystem development",
                'telegram_id': 2000000 + i,
                'username': f'batch_user_{i}'
            })

        print(f"\nProcessing {num_messages} messages...")
        start = time.time()

        total_memories = {'group': 0, 'user': 0, 'interaction': 0}
        processing_times = []

        for i, msg in enumerate(messages):
            msg_start = time.time()

            result = self.manager.process_group_message(
                message=msg['message'],
                platform_identity={
                    'platform': 'telegram',
                    'chatId': '-1001234567890',
                    'telegramId': msg['telegram_id'],
                    'username': msg['username']
                },
                speaker_info={'username': msg['username']},
                metadata={'message_id': f'batch_test_{i}'}
            )

            msg_time = (time.time() - msg_start) * 1000
            processing_times.append(msg_time)

            total_memories['group'] += len(result['created_memories']['group'])
            total_memories['user'] += len(result['created_memories']['user'])
            total_memories['interaction'] += len(result['created_memories']['interaction'])

        total_time = time.time() - start

        print(f"\nResults:")
        print(f"  Total time: {total_time*1000:.0f}ms")
        print(f"  Avg per message: {(total_time/num_messages)*1000:.0f}ms")
        print(f"  Min: {min(processing_times):.0f}ms")
        print(f"  Max: {max(processing_times):.0f}ms")
        print(f"  Median: {sorted(processing_times)[len(processing_times)//2]:.0f}ms")
        print(f"\n  Memories created:")
        print(f"    Group: {total_memories['group']}")
        print(f"    User: {total_memories['user']}")
        print(f"    Interaction: {total_memories['interaction']}")
        print(f"    Total: {sum(total_memories.values())}")

        return total_time / num_messages

    def run_all(self):
        """Run all optimization tests"""
        try:
            self.setup()

            print("\n" + "=" * 80)
            print("RUNNING BATCH OPTIMIZATION TESTS")
            print("=" * 80)

            # Test 1: Batch vs Sequential
            batch_speedup = self.test_batch_vs_sequential(20)

            # Test 2: Parallel Consolidation
            parallel_speedup = self.test_parallel_consolidation(10)

            # Test 3: End-to-End
            avg_msg_time = self.test_end_to_end_batch(10)

            print("\n" + "=" * 80)
            print("OPTIMIZATION TESTS COMPLETE")
            print("=" * 80)

            print("\n" + "=" * 80)
            print("SUMMARY")
            print("=" * 80)
            print(f"\n1. Batch Embeddings:")
            print(f"   Speedup: {batch_speedup:.1f}x vs sequential")

            print(f"\n2. Parallel Consolidation:")
            print(f"   Speedup: {parallel_speedup:.1f}x vs sequential")

            print(f"\n3. End-to-End Processing:")
            print(f"   Avg time per message: {avg_msg_time*1000:.0f}ms")

            print("\n" + "=" * 80 + "\n")

            return True

        except Exception as e:
            print(f"\nERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            self.teardown()


if __name__ == "__main__":
    tester = BatchOptimizationsTest()
    success = tester.run_all()
    sys.exit(0 if success else 1)
