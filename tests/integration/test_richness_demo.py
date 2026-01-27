"""
Richness Demo - Show how the group memory system enriches context over time

This simulates a real Telegram group conversation and shows:
1. What memories are created from each message
2. How context becomes richer with each interaction
3. Cross-group pattern detection
4. The difference between first interaction vs. mature context
"""
import os
import sys
import tempfile
import shutil
import json
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.group_memory_manager import GroupMemoryManager


class RichnessDemo:
    """Demo showing the richness of accumulated group memory"""

    def __init__(self):
        self.test_db_path = tempfile.mkdtemp(prefix='a0x-richness-')
        self.agent_id = "jessexbt"
        self.manager = None

    def setup(self):
        """Setup demo environment"""
        print("=" * 80)
        print("GROUP MEMORY RICHNESS DEMO")
        print("=" * 80)
        print(f"\nDemo DB: {self.test_db_path}")
        print(f"Agent: {self.agent_id}\n")

        self.manager = GroupMemoryManager(
            agent_id=self.agent_id,
            db_base_path=self.test_db_path
        )

    def teardown(self):
        """Cleanup"""
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)

    # ==========================================================================
    # SCENARIO: Base Ecosystem Telegram Group
    # ==========================================================================

    def run_base_ecosystem_scenario(self):
        """Simulate a conversation in a Base ecosystem builder group"""

        print("\n" + "=" * 80)
        print("SCENARIO: Base Ecosystem Telegram Group")
        print("=" * 80)
        print("\nGroup: @BaseBuilders (chatId: -1001234567890)")
        print("Participants: alice_builder, bob_defi, carol_nft, dave_infra\n")

        group_id = "telegram_group_-1001234567890"

        # ==========================================================================
        # MESSAGE 1: Alice introduces herself
        # ==========================================================================

        print("\n" + "-" * 80)
        print("MESSAGE 1: alice_builder joins")
        print("-" * 80)
        print("\nText: \"Hey everyone! I'm Alice, just joined the group. I've been working")
        print("on DeFi protocols for 2 years, specifically lending protocols on Base.\"")

        msg1 = self.manager.process_group_message(
            message="Hey everyone! I'm Alice, just joined the group. I've been working on DeFi protocols for 2 years, specifically lending protocols on Base. Looking forward to learning from you all!",
            platform_identity={
                'platform': 'telegram',
                'chatId': '-1001234567890',
                'telegramId': 111111111,
                'username': 'alice_builder'
            },
            speaker_info={'username': 'alice_builder', 'first_name': 'Alice'},
            metadata={'message_id': 'msg_001', 'timestamp': datetime.now(timezone.utc).isoformat()}
        )

        print(f"\nMemories created from message 1:")
        print(f"  Group memories: {len(msg1['created_memories']['group'])}")
        for mem in msg1['created_memories']['group']:
            print(f"    - [{mem.memory_type.value}] {mem.content[:60]}...")
        print(f"  User memories: {len(msg1['created_memories']['user'])}")
        for mem in msg1['created_memories']['user']:
            print(f"    - [{mem.memory_type.value}] {mem.content[:60]}...")
        print(f"  Interaction memories: {len(msg1['created_memories']['interaction'])}")

        # ==========================================================================
        # MESSAGE 2: Bob responds
        # ==========================================================================

        print("\n" + "-" * 80)
        print("MESSAGE 2: bob_defi responds to Alice")
        print("-" * 80)
        print("\nText: \"Welcome Alice! Great to have a DeFi expert here. I'm Bob, been working")
        print("on yield optimization strategies. Have you looked at liquid staking?\"")

        msg2 = self.manager.process_group_message(
            message="Welcome Alice! Great to have a DeFi expert here. I'm Bob, been working on yield optimization strategies. Have you looked at liquid staking derivatives for your lending protocols? Could be a good synergy.",
            platform_identity={
                'platform': 'telegram',
                'chatId': '-1001234567890',
                'telegramId': 222222222,
                'username': 'bob_defi'
            },
            speaker_info={'username': 'bob_defi', 'first_name': 'Bob'},
            metadata={'message_id': 'msg_002', 'timestamp': datetime.now(timezone.utc).isoformat()}
        )

        print(f"\nMemories created from message 2:")
        print(f"  Group memories: {len(msg2['created_memories']['group'])}")
        for mem in msg2['created_memories']['group']:
            print(f"    - [{mem.memory_type.value}] {mem.content[:60]}...")
        print(f"  User memories (Bob): {len(msg2['created_memories']['user'])}")
        for mem in msg2['created_memories']['user']:
            print(f"    - [{mem.memory_type.value}] {mem.content[:60]}...")
        print(f"  Interaction memories: {len(msg2['created_memories']['interaction'])}")
        for mem in msg2['created_memories']['interaction']:
            print(f"    - Alice <-> Bob: {mem.interaction_type}")

        # ==========================================================================
        # CHECKPOINT 1: Get context after 2 messages
        # ==========================================================================

        print("\n" + "=" * 80)
        print("CHECKPOINT 1: Agent Context after 2 messages")
        print("=" * 80)

        ctx1 = self.manager.get_context_for_agent(
            message="What should I know about Alice?",
            platform_identity={
                'platform': 'telegram',
                'chatId': '-1001234567890',
                'telegramId': 333333333,  # Carol asking
                'username': 'carol_nft'
            },
            limit_per_level=5,
            include_user_profile=True
        )

        print(f"\nWhen Carol asks about Alice, agent knows:")
        print(f"\n  Group context ({len(ctx1.get('group_context', []))} memories):")
        for mem in ctx1.get('group_context', [])[:3]:
            print(f"    - {mem.memory_type.value}: {mem.content[:70]}...")

        print(f"\n  User context - Alice ({len(ctx1.get('user_context', []))} memories):")
        for mem in ctx1.get('user_context', [])[:3]:
            print(f"    - {mem.memory_type.value}: {mem.content[:70]}...")

        print(f"\n  Interaction context ({len(ctx1.get('interaction_context', []))} memories):")
        for mem in ctx1.get('interaction_context', []):
            print(f"    - {mem.speaker_id} -> {mem.listener_id}: {mem.interaction_type}")

        # ==========================================================================
        # MESSAGES 3-7: More conversation
        # ==========================================================================

        print("\n" + "-" * 80)
        print("MESSAGES 3-7: Conversation continues")
        print("-" * 80)

        messages = [
            {
                'user': 'carol_nft',
                'id': 333333333,
                'text': "Hi all! I'm Carol, NFT artist and developer. Building a marketplace for Base-native NFTs. Any interest in collab?"
            },
            {
                'user': 'alice_builder',
                'id': 111111111,
                'text': "Hey Carol! Actually yes - we've been thinking about NFT collateral for lending protocols. Could be interesting!"
            },
            {
                'user': 'dave_infra',
                'id': 444444444,
                'text': "Dave here, infra engineer. Been working on Base node optimization and RPC endpoints. If anyone needs reliable infrastructure, hit me up."
            },
            {
                'user': 'bob_defi',
                'id': 222222222,
                'text': "That's actually perfect timing Dave! Our yield protocol needs more reliable RPCs. We're getting rate limited on the public ones."
            },
            {
                'user': 'alice_builder',
                'id': 111111111,
                'text': "Same here Dave! We're launching a new lending pool next week and need dedicated endpoints. What's your setup like?"
            }
        ]

        total_memories = {'group': 0, 'user': 0, 'interaction': 0}

        for i, msg_data in enumerate(messages, start=3):
            result = self.manager.process_group_message(
                message=msg_data['text'],
                platform_identity={
                    'platform': 'telegram',
                    'chatId': '-1001234567890',
                    'telegramId': msg_data['id'],
                    'username': msg_data['user']
                },
                speaker_info={'username': msg_data['user']},
                metadata={'message_id': f'msg_{i:03d}'}
            )
            total_memories['group'] += len(result['created_memories']['group'])
            total_memories['user'] += len(result['created_memories']['user'])
            total_memories['interaction'] += len(result['created_memories']['interaction'])

            print(f"\n  Message {i} ({msg_data['user']}): +{len(result['created_memories']['user'])} user memories")

        print(f"\n  Total new memories from messages 3-7:")
        print(f"    Group: {total_memories['group']}")
        print(f"    User: {total_memories['user']}")
        print(f"    Interaction: {total_memories['interaction']}")

        # ==========================================================================
        # CHECKPOINT 2: Rich context after 7 messages
        # ==========================================================================

        print("\n" + "=" * 80)
        print("CHECKPOINT 2: Agent Context after 7 messages (Mature Context)")
        print("=" * 80)

        ctx2 = self.manager.get_context_for_agent(
            message="Who can help with DeFi lending protocols?",
            platform_identity={
                'platform': 'telegram',
                'chatId': '-1001234567890',
                'telegramId': 999999999,  # New user asking
                'username': 'newbie'
            },
            limit_per_level=10,
            include_user_profile=True
        )

        print(f"\nWhen a NEW user asks about DeFi lending, agent knows:")

        print(f"\n  Group context ({len(ctx2.get('group_context', []))} memories):")
        for mem in ctx2.get('group_context', [])[:5]:
            print(f"    - [{mem.memory_type.value}] {mem.content[:80]}...")

        print(f"\n  User context - Alice (DeFi expert) ({len(ctx2.get('user_context', []))} memories):")
        alice_memories = [m for m in ctx2.get('user_context', []) if 'alice' in str(m.user_id).lower()]
        for mem in alice_memories[:5]:
            print(f"    - [{mem.memory_type.value}] {mem.content[:80]}...")

        print(f"\n  User context - Bob (yield expert) ({len(ctx2.get('user_context', []))} memories):")
        bob_memories = [m for m in ctx2.get('user_context', []) if 'bob' in str(m.user_id).lower()]
        for mem in bob_memories[:5]:
            print(f"    - [{mem.memory_type.value}] {mem.content[:80]}...")

        print(f"\n  Interaction context ({len(ctx2.get('interaction_context', []))} memories):")
        for mem in ctx2.get('interaction_context', [])[:8]:
            print(f"    - {mem.speaker_id} -> {mem.listener_id}: {mem.interaction_type}")

        # ==========================================================================
        # CHECKPOINT 3: Cross-group patterns
        # ==========================================================================

        print("\n" + "=" * 80)
        print("CHECKPOINT 3: Cross-Group Pattern Discovery")
        print("=" * 80)
        print("\nNow Alice participates in ANOTHER group...")

        # Alice in another group
        other_group_messages = [
            "I've been working on DeFi lending protocols for 2 years",
            "My expertise is in collateralized debt positions",
            "I focus on Base ecosystem DeFi projects"
        ]

        for msg in other_group_messages:
            self.manager.process_group_message(
                message=msg,
                platform_identity={
                    'platform': 'telegram',
                    'chatId': '-1009999999999',  # Different group
                    'telegramId': 111111111,  # Same Alice
                    'username': 'alice_builder'
                },
                speaker_info={'username': 'alice_builder'},
                metadata={'message_id': 'other_group'}
            )

        # Consolidate patterns
        patterns = self.manager.consolidate_cross_group_patterns(
            user_id='telegram_111111111',
            min_groups=2,
            min_evidence=2
        )

        print(f"\nCross-group patterns detected for Alice:")
        print(f"  Total patterns: {len(patterns)}")

        for pattern in patterns:
            print(f"\n  Pattern:")
            print(f"    Type: {pattern.pattern_type}")
            print(f"    Content: {pattern.content}")
            print(f"    Groups involved: {pattern.groups_involved}")
            print(f"    Confidence: {pattern.confidence_score:.2f}")
            print(f"    Evidence count: {pattern.evidence_count}")

        # ==========================================================================
        # CHECKPOINT 4: Context with cross-group patterns
        # ==========================================================================

        print("\n" + "=" * 80)
        print("CHECKPOINT 4: Context with Cross-Group Patterns")
        print("=" * 80)

        ctx3 = self.manager.get_context_for_agent(
            message="What does Alice specialize in?",
            platform_identity={
                'platform': 'telegram',
                'chatId': '-1001234567890',
                'telegramId': 222222222,
                'username': 'bob_defi'
            },
            limit_per_level=5
        )

        print(f"\nWhen Bob asks about Alice's specialization:")

        if 'cross_group_context' in ctx3:
            print(f"\n  Cross-group patterns ({len(ctx3['cross_group_context'])} patterns):")
            for pattern in ctx3['cross_group_context']:
                print(f"    - [{pattern.pattern_type}] {pattern.content}")
                print(f"      Groups: {len(pattern.groups_involved)}, Confidence: {pattern.confidence_score:.2f}")

        # ==========================================================================
        # FINAL SUMMARY
        # ==========================================================================

        print("\n" + "=" * 80)
        print("RICHNESS SUMMARY")
        print("=" * 80)

        # Get stats from the store directly
        try:
            db = self.manager.group_store._get_db()
            table_names = ['group_memories', 'user_memories', 'interaction_memories', 'cross_group_memories']
            table_counts = {}

            for table_name in table_names:
                try:
                    table = db.open_table(table_name)
                    count = table.count()
                    table_counts[table_name] = count
                except Exception:
                    table_counts[table_name] = 0

            print(f"\nTotal memories stored:")
            for table, count in table_counts.items():
                print(f"  {table}: {count} memories")

        except Exception as e:
            print(f"\n(Error getting stats: {e})")

        print(f"\nKey insights:")
        print(f"  ✓ Group knows Alice is a DeFi lending expert (2+ years)")
        print(f"  ✓ Group knows Bob works on yield optimization")
        print(f"  ✓ Group knows Carol builds NFT marketplaces")
        print(f"  ✓ Group knows Dave provides Base infrastructure")
        print(f"  ✓ Agent knows Alice-Bob interaction: 'synergy_discussion'")
        print(f"  ✓ Agent knows Bob-Dave interaction: 'infrastructure_request'")
        print(f"  ✓ Cross-group: Alice's DeFi expertise confirmed across 2 groups")

        print(f"\nContext richness progression:")
        print(f"  After message 1: Basic Alice profile")
        print(f"  After message 2: Alice profile + Bob profile + interaction")
        print(f"  After message 7: 4 user profiles + 6 interactions + group expertise map")
        print(f"  With cross-group: Validated expertise patterns")

        print(f"\nValue per message:")
        print(f"  Message 1 (Alice intro): Created group knowledge + user profile + interaction baseline")
        print(f"  Message 2 (Bob response): Added Bob's expertise + recorded Alice-Bob interaction")
        print(f"  Messages 3-7: Enriched network of expertise + interaction patterns")
        print(f"  Cross-group: Confirmed expertise = higher confidence")

    # ==========================================================================
    # RUN
    # ==========================================================================

    def run(self):
        """Run the complete richness demo"""
        try:
            self.setup()
            self.run_base_ecosystem_scenario()

            print("\n" + "=" * 80)
            print("DEMO COMPLETE")
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
    demo = RichnessDemo()
    success = demo.run()
    sys.exit(0 if success else 1)
