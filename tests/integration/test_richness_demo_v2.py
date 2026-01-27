"""
Richness Demo V2 - Direct memory creation to show system richness

This bypasses keyword detection and directly creates memories to demonstrate
the full richness of the group memory system.
"""
import os
import sys
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from database.group_memory_store import GroupMemoryStore
from models.group_memory import (
    GroupMemory, UserMemory, InteractionMemory, CrossGroupMemory,
    MemoryType, PrivacyScope
)
from utils.embedding import EmbeddingModel


class RichnessDemoV2:
    """Demo showing richness by directly creating memories"""

    def __init__(self):
        self.test_db_path = tempfile.mkdtemp(prefix='a0x-richness-v2-')
        self.agent_id = "jessexbt"
        self.store = None
        self.embedding_model = EmbeddingModel()

    def setup(self):
        print("=" * 80)
        print("GROUP MEMORY RICHNESS DEMO V2")
        print("=" * 80)
        print(f"\nDemo DB: {self.test_db_path}")
        print(f"Agent: {self.agent_id}\n")

        self.store = GroupMemoryStore(
            agent_id=self.agent_id,
            embedding_model=self.embedding_model,
            db_base_path=self.test_db_path
        )

    def teardown(self):
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)

    def create_sample_memories(self):
        """Create rich sample memories directly"""

        group_id = "telegram_group_-1001234567890"

        # ==========================================================================
        # 1. Alice's Expertise (User Memory)
        # ==========================================================================

        print("Creating user memories...")

        alice_expertise = UserMemory(
            agent_id=self.agent_id,
            group_id=group_id,
            user_id="telegram_111111111",
            username="alice_builder",
            platform="telegram",
            memory_type=MemoryType.EXPERTISE,
            
            content="Alice has been working on DeFi lending protocols for 2 years, specializing in collateralized debt positions on Base",
            importance_score=0.9,
            privacy_scope=PrivacyScope.PUBLIC,
            topics=["defi", "lending", "base", "cdp", "collateral"],
            keywords=["defi", "lending protocols", "2 years", "base", "collateral"],
            first_seen=datetime.now(timezone.utc).isoformat(),
            last_seen=datetime.now(timezone.utc).isoformat()
        )
        self.store.add_user_memory(alice_expertise)
        print("  + Alice: DeFi lending expert (2 years)")

        alice_project = UserMemory(
            agent_id=self.agent_id,
            group_id=group_id,
            user_id="telegram_111111111",
            username="alice_builder",
            platform="telegram",
            memory_type=MemoryType.ACTION,
            
            content="Alice is launching a new lending pool next week and needs dedicated RPC endpoints",
            importance_score=0.8,
            privacy_scope=PrivacyScope.PUBLIC,
            topics=["lending pool", "rpc", "infrastructure", "launch"],
            keywords=["lending pool", "next week", "rpc endpoints"],
            first_seen=datetime.now(timezone.utc).isoformat(),
            last_seen=datetime.now(timezone.utc).isoformat()
        )
        self.store.add_user_memory(alice_project)
        print("  + Alice: Launching lending pool, needs RPC endpoints")

        bob_expertise = UserMemory(
            agent_id=self.agent_id,
            group_id=group_id,
            user_id="telegram_222222222",
            username="bob_defi",
            platform="telegram",
            memory_type=MemoryType.EXPERTISE,
            
            content="Bob specializes in yield optimization strategies and liquid staking derivatives for DeFi protocols",
            importance_score=0.85,
            privacy_scope=PrivacyScope.PUBLIC,
            topics=["yield", "optimization", "liquid staking", "defi"],
            keywords=["yield optimization", "liquid staking", "derivatives"],
            first_seen=datetime.now(timezone.utc).isoformat(),
            last_seen=datetime.now(timezone.utc).isoformat()
        )
        self.store.add_user_memory(bob_expertise)
        print("  + Bob: Yield optimization expert")

        bob_need = UserMemory(
            agent_id=self.agent_id,
            group_id=group_id,
            user_id="telegram_222222222",
            username="bob_defi",
            platform="telegram",
            memory_type=MemoryType.PREFERENCE,
            
            content="Bob's yield protocol is getting rate limited on public RPCs and needs more reliable infrastructure",
            importance_score=0.7,
            privacy_scope=PrivacyScope.PUBLIC,
            topics=["rpc", "rate limiting", "infrastructure", "yield protocol"],
            keywords=["rate limited", "public rpcs", "reliable infrastructure"],
            first_seen=datetime.now(timezone.utc).isoformat(),
            last_seen=datetime.now(timezone.utc).isoformat()
        )
        self.store.add_user_memory(bob_need)
        print("  + Bob: Needs reliable RPCs (rate limited)")

        carol_expertise = UserMemory(
            agent_id=self.agent_id,
            group_id=group_id,
            user_id="telegram_333333333",
            username="carol_nft",
            platform="telegram",
            memory_type=MemoryType.EXPERTISE,
            
            content="Carol is an NFT artist and developer building a marketplace for Base-native NFTs with collateral integration for lending protocols",
            importance_score=0.8,
            privacy_scope=PrivacyScope.PUBLIC,
            topics=["nft", "marketplace", "base", "collateral", "lending"],
            keywords=["nft artist", "marketplace", "base-native", "nft collateral"],
            first_seen=datetime.now(timezone.utc).isoformat(),
            last_seen=datetime.now(timezone.utc).isoformat()
        )
        self.store.add_user_memory(carol_expertise)
        print("  + Carol: NFT marketplace builder")

        dave_expertise = UserMemory(
            agent_id=self.agent_id,
            group_id=group_id,
            user_id="telegram_444444444",
            username="dave_infra",
            platform="telegram",
            memory_type=MemoryType.EXPERTISE,
            
            content="Dave is an infrastructure engineer working on Base node optimization and provides dedicated RPC endpoints",
            importance_score=0.85,
            privacy_scope=PrivacyScope.PUBLIC,
            topics=["infrastructure", "base", "rpc", "node optimization"],
            keywords=["base node optimization", "rpc endpoints", "infrastructure engineer", "dedicated endpoints"],
            first_seen=datetime.now(timezone.utc).isoformat(),
            last_seen=datetime.now(timezone.utc).isoformat()
        )
        self.store.add_user_memory(dave_expertise)
        print("  + Dave: Base infrastructure, provides RPC endpoints")

        # ==========================================================================
        # 2. Group Knowledge (Group Memory)
        # ==========================================================================

        print("\nCreating group memories...")

        group_collab = GroupMemory(
            agent_id=self.agent_id,
            group_id=group_id,
            memory_type=MemoryType.DISCUSSION,
            
            content="Group members are exploring collaboration between NFT marketplaces and DeFi lending protocols, specifically using NFTs as collateral for lending",
            speaker="alice_builder",
            importance_score=0.8,
            privacy_scope=PrivacyScope.PUBLIC,
            topics=["nft", "collateral", "lending", "collaboration"],
            keywords=["nft collateral", "lending protocols", "collaboration", "synergy"],
            evidence_count=2,
            first_seen=datetime.now(timezone.utc).isoformat(),
            last_seen=datetime.now(timezone.utc).isoformat()
        )
        self.store.add_group_memory(group_collab)
        print("  + Group: NFT collateral for lending collaboration")

        group_infra = GroupMemory(
            agent_id=self.agent_id,
            group_id=group_id,
            memory_type=MemoryType.NEED,
            
            content="Multiple DeFi projects in the group need reliable RPC infrastructure - both Alice's lending protocol and Bob's yield protocol are experiencing rate limiting issues",
            speaker="bob_defi",
            importance_score=0.75,
            privacy_scope=PrivacyScope.PUBLIC,
            topics=["rpc", "infrastructure", "rate limiting"],
            keywords=["rate limiting", "rpc infrastructure", "dedicated endpoints"],
            evidence_count=3,
            first_seen=datetime.now(timezone.utc).isoformat(),
            last_seen=datetime.now(timezone.utc).isoformat()
        )
        self.store.add_group_memory(group_infra)
        print("  + Group: DeFi projects need reliable RPC infrastructure")

        # ==========================================================================
        # 3. Interactions (Interaction Memory)
        # ==========================================================================

        print("\nCreating interaction memories...")

        alice_bob = InteractionMemory(
            agent_id=self.agent_id,
            group_id=group_id,
            speaker_id="telegram_111111111",
            speaker_username="alice_builder",
            listener_id="telegram_222222222",
            listener_username="bob_defi",
            content="Alice and Bob discussed potential synergy between lending protocols and yield optimization, specifically liquid staking derivatives",
            interaction_type="synergy_discussion",
            mentioned_users=[],
            topics=["lending", "yield", "liquid staking", "synergy"],
            first_seen=datetime.now(timezone.utc).isoformat(),
            last_seen=datetime.now(timezone.utc).isoformat()
        )
        self.store.add_interaction_memory(alice_bob)
        print("  + Alice <-> Bob: Synergy discussion (lending + yield)")

        bob_dave = InteractionMemory(
            agent_id=self.agent_id,
            group_id=group_id,
            speaker_id="telegram_222222222",
            speaker_username="bob_defi",
            listener_id="telegram_444444444",
            listener_username="dave_infra",
            content="Bob requested reliable RPC infrastructure from Dave for his yield protocol that's experiencing rate limiting",
            interaction_type="infrastructure_request",
            mentioned_users=[],
            topics=["rpc", "infrastructure", "yield protocol"],
            first_seen=datetime.now(timezone.utc).isoformat(),
            last_seen=datetime.now(timezone.utc).isoformat()
        )
        self.store.add_interaction_memory(bob_dave)
        print("  + Bob -> Dave: Infrastructure request (RPCs needed)")

        alice_dave = InteractionMemory(
            agent_id=self.agent_id,
            group_id=group_id,
            speaker_id="telegram_111111111",
            speaker_username="alice_builder",
            listener_id="telegram_444444444",
            listener_username="dave_infra",
            content="Alice asked Dave about his Base node setup and RPC infrastructure for her upcoming lending pool launch",
            interaction_type="infrastructure_inquiry",
            mentioned_users=[],
            topics=["rpc", "lending pool", "infrastructure", "launch"],
            first_seen=datetime.now(timezone.utc).isoformat(),
            last_seen=datetime.now(timezone.utc).isoformat()
        )
        self.store.add_interaction_memory(alice_dave)
        print("  + Alice -> Dave: Infrastructure inquiry (lending pool)")

        alice_carol = InteractionMemory(
            agent_id=self.agent_id,
            group_id=group_id,
            speaker_id="telegram_111111111",
            speaker_username="alice_builder",
            listener_id="telegram_333333333",
            listener_username="carol_nft",
            content="Alice expressed interest in Carol's NFT marketplace idea for using NFTs as collateral in lending protocols",
            interaction_type="collaboration_interest",
            mentioned_users=[],
            topics=["nft", "collateral", "lending", "collaboration"],
            first_seen=datetime.now(timezone.utc).isoformat(),
            last_seen=datetime.now(timezone.utc).isoformat()
        )
        self.store.add_interaction_memory(alice_carol)
        print("  + Alice <-> Carol: Collaboration interest (NFT collateral)")

        # ==========================================================================
        # 4. Cross-Group Pattern (Alice in multiple groups)
        # ==========================================================================

        print("\nCreating cross-group patterns...")

        alice_cross = CrossGroupMemory(
            agent_id=self.agent_id,
            user_id="telegram_111111111",
            username="alice_builder",
            pattern_type="expertise",
            content="Alice consistently demonstrates DeFi lending expertise across multiple groups - mentions 2 years experience with lending protocols and CDPs in different contexts",
            confidence_score=0.9,
            groups_involved=[
                "telegram_group_-1001234567890",
                "telegram_group_-1009999999999"
            ],
            evidence_count=4,
            topics=["defi", "lending", "cdp"],
            keywords=["defi lending", "2 years", "collateral"],
            first_seen=datetime.now(timezone.utc).isoformat(),
            last_seen=datetime.now(timezone.utc).isoformat()
        )
        self.store.add_cross_group_memory(alice_cross)
        print("  + Cross-group: Alice's DeFi expertise confirmed across 2 groups")

        print("\n" + "=" * 80)

    def demonstrate_context_retrieval(self):
        """Show context retrieval for different queries"""

        group_id = "telegram_group_-1001234567890"

        # ==========================================================================
        # Query 1: New user asking about DeFi lending
        # ==========================================================================

        print("\nQUERY 1: New user asks 'Who can help with DeFi lending?'")
        print("-" * 80)

        context1 = self.store.get_group_context(
            group_id=group_id,
            query="Who can help with DeFi lending protocols and collateralized debt positions?",
            limit_per_level=5
        )

        print(f"\n  Relevant user memories:")
        for mem in context1['user_context'][:3]:
            print(f"    - [{mem.username}] {mem.memory_type.value}: {mem.content[:70]}...")

        print(f"\n  Relevant group memories:")
        for mem in context1['group_context'][:2]:
            print(f"    - [{mem.memory_type.value}] {mem.content[:70]}...")

        # ==========================================================================
        # Query 2: Asking about RPC infrastructure needs
        # ==========================================================================

        print("\n\nQUERY 2: Someone asks 'Who needs RPC endpoints?'")
        print("-" * 80)

        context2 = self.store.get_group_context(
            group_id=group_id,
            query="Who needs reliable RPC endpoints and infrastructure for their DeFi project?",
            limit_per_level=5
        )

        print(f"\n  Relevant user memories:")
        for mem in context2['user_context'][:3]:
            print(f"    - [{mem.username}] {mem.memory_type.value}: {mem.content[:70]}...")

        print(f"\n  Relevant interaction memories:")
        for mem in context2['interaction_context'][:3]:
            print(f"    - [{mem.speaker_username} -> {mem.listener_username}]")
            print(f"      {mem.interaction_type}: {mem.content[:60]}...")

        # ==========================================================================
        # Query 3: Asking about Alice's background
        # ==========================================================================

        print("\n\nQUERY 3: Asking 'What does Alice specialize in?'")
        print("-" * 80)

        context3 = self.store.get_group_context(
            group_id=group_id,
            user_id="telegram_111111111",  # Alice
            query="What is Alice's expertise and background?",
            limit_per_level=5
        )

        print(f"\n  Alice's memories:")
        for mem in context3['user_context'][:4]:
            print(f"    - [{mem.memory_type.value}] {mem.content[:75]}...")

        print(f"\n  Cross-group patterns for Alice:")
        for mem in context3['cross_group_context']:
            print(f"    - [{mem.pattern_type}] Confidence: {mem.confidence_score:.2f}")
            print(f"      {mem.content[:70]}...")

        # ==========================================================================
        # Query 4: NFT collateral collaboration
        # ==========================================================================

        print("\n\nQUERY 4: Asking 'Who is working on NFT collateral?'")
        print("-" * 80)

        context4 = self.store.get_group_context(
            group_id=group_id,
            query="NFT collateral for lending protocols marketplace",
            limit_per_level=5
        )

        print(f"\n  Relevant group memories:")
        for mem in context4['group_context'][:2]:
            print(f"    - [{mem.memory_type.value}] {mem.content[:70]}...")

        print(f"\n  Relevant interaction memories:")
        for mem in context4['interaction_context'][:2]:
            print(f"    - [{mem.speaker_username} <-> {mem.listener_username}]")
            print(f"      {mem.interaction_type}: {mem.content[:60]}...")

        print(f"\n  Relevant user memories:")
        for mem in context4['user_context'][:2]:
            print(f"    - [{mem.username}] {mem.memory_type.value}: {mem.content[:70]}...")

    def show_statistics(self):
        """Show database statistics"""

        print("\n" + "=" * 80)
        print("DATABASE STATISTICS")
        print("=" * 80)

        db = self.store.db
        tables = {
            'group_memories': self.store.group_memories_table,
            'user_memories': self.store.user_memories_table,
            'interaction_memories': self.store.interaction_memories_table,
            'cross_group_memories': self.store.cross_group_memories_table
        }

        total = 0
        for name, table in tables.items():
            count = table.count()
            total += count
            print(f"\n{name}: {count} memories")

        print(f"\nTotal memories: {total}")

        # Sample retrieval
        print("\n" + "-" * 80)
        print("SAMPLE: Retrieving memories about Alice")
        print("-" * 80)

        results = self.store.user_memories_table.search().where(
            "username = 'alice_builder'"
        ).to_pandas()

        if len(results) > 0:
            print(f"\nFound {len(results)} memories for Alice:")
            for _, row in results.iterrows():
                print(f"  - [{row['memory_type']}] {row['content'][:60]}...")

    def run(self):
        """Run the complete demo"""
        try:
            self.setup()
            self.create_sample_memories()
            self.demonstrate_context_retrieval()
            self.show_statistics()

            print("\n" + "=" * 80)
            print("DEMO COMPLETE")
            print("=" * 80)

            print("\nKEY INSIGHTS:")
            print("  ✓ User memories capture individual expertise and needs")
            print("  ✓ Group memories capture collective knowledge and decisions")
            print("  ✓ Interaction memories track who-talked-to-whom about what")
            print("  ✓ Cross-group patterns validate expertise across contexts")
            print("  ✓ Semantic search finds relevant memories even with different wording")

            print("\nCONTEXT RICHNESS:")
            print("  After 1 message: Basic user profile")
            print("  After 5 messages: Multi-expertise network with interactions")
            print("  After 10+ messages: Rich cross-linked knowledge graph")
            print("  Cross-group: Validated patterns with higher confidence")

            print("\n")

            return True

        except Exception as e:
            print(f"\nERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            self.teardown()


if __name__ == "__main__":
    demo = RichnessDemoV2()
    success = demo.run()
    sys.exit(0 if success else 1)
