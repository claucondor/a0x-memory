"""
Realistic Scenario Test - Quality Evaluation

Simulates real usage patterns:
1. 10 users sending DMs to the bot
2. Same 10 users in 3 different groups
3. Various interaction patterns:
   - Direct mentions to bot (@bot)
   - Replies to bot messages
   - User-to-user interactions
   - General group discussion
4. Evaluates quality of:
   - Memory extraction
   - Profile generation
   - Context retrieval

Run: USE_LOCAL_STORAGE=true python tests/test_realistic_scenario.py
"""
import sys
import os
import time
import random
from datetime import datetime, timezone, timedelta

# Use local storage for fast testing (no Firestore network calls)
os.environ["USE_LOCAL_STORAGE"] = "true"

# Add parent to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from main import SimpleMemSystem
import config

# ============================================================
# Test Configuration
# ============================================================

AGENT_ID = "test_realistic"
BOT_NAME = "jessexbt"

# 10 test users with distinct personalities
USERS = [
    {"id": 1001, "username": "alice_dev", "persona": "Senior Solidity developer, 5 years experience, works on DeFi"},
    {"id": 1002, "username": "bob_trader", "persona": "Crypto trader, interested in Base chain, uses technical analysis"},
    {"id": 1003, "username": "carol_artist", "persona": "NFT artist, creates generative art, new to smart contracts"},
    {"id": 1004, "username": "david_founder", "persona": "Startup founder, building on Base, looking for grants"},
    {"id": 1005, "username": "emma_researcher", "persona": "Academic researcher in blockchain, writes papers"},
    {"id": 1006, "username": "frank_newbie", "persona": "Complete beginner, just learned about crypto last week"},
    {"id": 1007, "username": "grace_pm", "persona": "Product manager at DeFi protocol, coordinates teams"},
    {"id": 1008, "username": "henry_auditor", "persona": "Smart contract auditor, security focused"},
    {"id": 1009, "username": "ivy_marketer", "persona": "Crypto marketing specialist, community management"},
    {"id": 1010, "username": "jack_whale", "persona": "Large holder, institutional investor background"},
]

# 3 groups with different contexts and member lists
GROUPS = [
    {
        "id": "-100001",
        "name": "Base Builders",
        "topic": "Building on Base chain",
        "members": ["alice_dev", "david_founder", "henry_auditor", "emma_researcher", "frank_newbie", "jack_whale"]
    },
    {
        "id": "-100002",
        "name": "DeFi Discussion",
        "topic": "DeFi protocols and trading",
        "members": ["bob_trader", "grace_pm", "jack_whale", "alice_dev", "henry_auditor", "david_founder"]
    },
    {
        "id": "-100003",
        "name": "NFT Creators",
        "topic": "NFT art and collections",
        "members": ["carol_artist", "ivy_marketer", "frank_newbie", "grace_pm", "bob_trader"]
    },
]

# DM conversations (user talking to bot)
DM_CONVERSATIONS = {
    "alice_dev": [
        "Hey, I've been building smart contracts for 5 years now",
        "Mostly worked on Uniswap forks and lending protocols",
        "I'm interested in the Base chain grants program",
        "Do you know what the requirements are?",
        "My specialty is gas optimization and security",
        "I've audited over 20 contracts professionally",
        "Currently working on a new AMM design",
        "The innovation is in how we handle concentrated liquidity",
        "Would love to get some feedback on the architecture",
        "My wallet is 0xAlice123...abc for reference",
    ],
    "bob_trader": [
        "What's up, I trade mostly on Base now",
        "The fees are so much better than mainnet",
        "I use a combination of TA and on-chain analysis",
        "Looking at the WETH/USDC pool liquidity",
        "Do you have any insights on Base DEX volumes?",
        "I've been profitable for 3 months straight",
        "My strategy involves mean reversion on smaller caps",
        "Risk management is key, I never risk more than 2%",
        "What tools do you recommend for on-chain analysis?",
        "I'm also exploring automated trading strategies",
    ],
    "carol_artist": [
        "Hi! I'm new to the crypto space",
        "I'm an artist and want to explore NFTs",
        "I create generative art using Processing and p5.js",
        "Never deployed a smart contract before though",
        "Is Base a good chain for NFTs?",
        "What are the gas costs like compared to Ethereum?",
        "I have a collection of 1000 unique pieces ready",
        "Would love to learn about ERC-721 basics",
        "Do you know any good tutorials for artists?",
        "My art style is abstract geometric patterns",
    ],
    "david_founder": [
        "Hey, I'm building a startup on Base",
        "We're creating a decentralized identity solution",
        "Looking into the Base grants program",
        "Our MVP is almost ready for testnet",
        "Team of 4 engineers, all remote",
        "We've raised a small pre-seed round",
        "Target launch is Q2 this year",
        "Main challenge is user onboarding complexity",
        "Any advice on grant applications?",
        "We're also considering Optimism grants",
    ],
    "emma_researcher": [
        "Hello, I'm researching L2 scaling solutions",
        "Working on a paper comparing Base, Optimism, Arbitrum",
        "Focused on transaction finality and security models",
        "Do you have data on Base's sequencer uptime?",
        "I'm affiliated with Stanford blockchain club",
        "Published 3 papers on consensus mechanisms",
        "Interested in the fraud proof system",
        "How does Base handle data availability?",
        "Planning to present findings at ETHDenver",
        "Would appreciate any technical documentation",
    ],
    "frank_newbie": [
        "Hi, I'm completely new to all this",
        "My friend told me about Base chain",
        "I don't really understand what L2 means",
        "Is it safe to use? I'm worried about scams",
        "How do I even get started?",
        "What wallet should I use?",
        "I have some ETH on Coinbase",
        "How do I move it to Base?",
        "What are gas fees exactly?",
        "Sorry for all the basic questions",
    ],
    "grace_pm": [
        "Hey, I'm a PM at a DeFi protocol",
        "We're expanding to Base next month",
        "Need to understand the ecosystem better",
        "What are the main DEXs on Base?",
        "How's the liquidity compared to mainnet?",
        "We're bringing over $10M in TVL initially",
        "Looking for integration partners",
        "Our protocol focuses on yield aggregation",
        "Do you know any good Base native protocols?",
        "Also interested in cross-chain bridges",
    ],
    "henry_auditor": [
        "Hi, I do smart contract security audits",
        "Specializing in DeFi protocol audits",
        "Certified by Code4rena and Sherlock",
        "Found critical bugs in 5 major protocols",
        "Looking to expand my Base chain expertise",
        "Are there specific patterns to watch for on Base?",
        "I use Slither, Mythril, and manual review",
        "Typical audit takes 2-4 weeks",
        "Happy to help review any community projects",
        "My audit reports are on GitHub",
    ],
    "ivy_marketer": [
        "Hey! I do marketing for crypto projects",
        "Managed communities for 3 successful launches",
        "Interested in Base ecosystem growth",
        "What's the best way to reach Base users?",
        "I've built communities of 50k+ members",
        "Focus on organic growth, no bots",
        "Twitter and Discord are my main channels",
        "Looking for projects that need marketing help",
        "I also do ambassador programs",
        "Have connections with major crypto influencers",
    ],
    "jack_whale": [
        "Hello, I manage a crypto fund",
        "We're looking at Base ecosystem investments",
        "Institutional background, ex-Goldman",
        "Typical check size is $500k-2M",
        "Interested in infrastructure plays",
        "What are the most promising Base projects?",
        "We also do OTC deals if needed",
        "Looking for serious teams with traction",
        "Happy to make introductions to other funds",
        "Our thesis is L2 infrastructure dominance",
    ],
}

# Group conversations - ONLY members of that group participate
# Extended to 30+ messages per group for better memory generation
GROUP_CONVERSATIONS = {
    "Base Builders": [
        # Members: alice_dev, david_founder, henry_auditor, emma_researcher, frank_newbie, jack_whale
        {"user": "alice_dev", "msg": "Just deployed my new contract on Base testnet!", "mentions_bot": False},
        {"user": "david_founder", "msg": "@alice_dev nice! What are you building?", "mentions_bot": False, "reply_to": "alice_dev"},
        {"user": "alice_dev", "msg": "A new AMM with concentrated liquidity, optimized for Base", "mentions_bot": False},
        {"user": "frank_newbie", "msg": f"@{BOT_NAME} what's an AMM?", "mentions_bot": True},
        {"user": "henry_auditor", "msg": "@alice_dev happy to take a look at the security when ready", "mentions_bot": False, "reply_to": "alice_dev"},
        {"user": "emma_researcher", "msg": "Interesting, concentrated liquidity on L2s has different dynamics", "mentions_bot": False},
        {"user": "david_founder", "msg": f"@{BOT_NAME} are there grants available for AMM development?", "mentions_bot": True},
        {"user": "jack_whale", "msg": "What's the expected TVL target? We might be interested in seeding", "mentions_bot": False},
        {"user": "alice_dev", "msg": "@jack_whale targeting $5M in first month, have some commitments already", "mentions_bot": False, "reply_to": "jack_whale"},
        {"user": "frank_newbie", "msg": "This is all so complex but fascinating", "mentions_bot": False},
        {"user": "emma_researcher", "msg": "I can share some research on L2 AMM performance", "mentions_bot": False},
        {"user": "henry_auditor", "msg": "Security should be priority before launch", "mentions_bot": False},
        {"user": "david_founder", "msg": "Our identity solution could help with KYC for institutional investors", "mentions_bot": False},
        {"user": "jack_whale", "msg": "@david_founder that's actually a blocker for us often", "mentions_bot": False, "reply_to": "david_founder"},
        {"user": "alice_dev", "msg": "Might need that for the seed round actually", "mentions_bot": False},
        # Extended conversations with more substantive content
        {"user": "alice_dev", "msg": "I've been coding in Solidity for 5 years, started with Uniswap V2 forks", "mentions_bot": False},
        {"user": "henry_auditor", "msg": "My background is in traditional security, moved to smart contracts 3 years ago", "mentions_bot": False},
        {"user": "emma_researcher", "msg": "I published a paper on MEV protection strategies last month", "mentions_bot": False},
        {"user": "david_founder", "msg": "Our startup raised $2M seed from a]16z scout fund", "mentions_bot": False},
        {"user": "jack_whale", "msg": "We've deployed $50M into Base ecosystem projects this year", "mentions_bot": False},
        {"user": "frank_newbie", "msg": "I'm learning Solidity on CryptoZombies right now", "mentions_bot": False},
        {"user": "alice_dev", "msg": "The AMM contract is 2000 lines, uses custom math library for precision", "mentions_bot": False},
        {"user": "henry_auditor", "msg": "Custom math is risky, I'd recommend using Solmate or PRBMath", "mentions_bot": False},
        {"user": "emma_researcher", "msg": "@alice_dev I found that fixed-point math reduces gas by 40% on L2s", "mentions_bot": False, "reply_to": "alice_dev"},
        {"user": "david_founder", "msg": "We're launching on Base mainnet next month, need beta testers", "mentions_bot": False},
        {"user": "jack_whale", "msg": "@david_founder send me the details, we can commit $500k for the pilot", "mentions_bot": False, "reply_to": "david_founder"},
        {"user": "frank_newbie", "msg": f"@{BOT_NAME} how do I become a beta tester for Base projects?", "mentions_bot": True},
        {"user": "alice_dev", "msg": "Security audit scheduled for February with Trail of Bits", "mentions_bot": False},
        {"user": "henry_auditor", "msg": "Trail of Bits is excellent, I've worked with them before", "mentions_bot": False},
        {"user": "emma_researcher", "msg": "Announcement: Base Builders meetup next Thursday at 6pm EST", "mentions_bot": False},
    ],
    "DeFi Discussion": [
        # Members: bob_trader, grace_pm, jack_whale, alice_dev, henry_auditor, david_founder
        {"user": "bob_trader", "msg": "Anyone else seeing the WETH premium on Base?", "mentions_bot": False},
        {"user": "jack_whale", "msg": "Yeah, about 0.1% right now, arb opportunity", "mentions_bot": False},
        {"user": "grace_pm", "msg": "Our protocol is showing good yields on Base, 8% APY on stables", "mentions_bot": False},
        {"user": "bob_trader", "msg": "@grace_pm is that sustainable long term?", "mentions_bot": False, "reply_to": "grace_pm"},
        {"user": "alice_dev", "msg": "Depends on the source of yield, what's the mechanism?", "mentions_bot": False},
        {"user": "henry_auditor", "msg": "Always ask where yield comes from, red flag if unclear", "mentions_bot": False},
        {"user": "bob_trader", "msg": f"@{BOT_NAME} what are typical sustainable yields on Base?", "mentions_bot": True},
        {"user": "david_founder", "msg": "We're building identity verification for DeFi, reduces risk", "mentions_bot": False},
        {"user": "jack_whale", "msg": "@david_founder interesting, institutional requirement for sure", "mentions_bot": False, "reply_to": "david_founder"},
        {"user": "alice_dev", "msg": "Would integrate well with my AMM for compliance", "mentions_bot": False},
        {"user": "grace_pm", "msg": "We need KYC for our institutional pool", "mentions_bot": False},
        {"user": "henry_auditor", "msg": f"@{BOT_NAME} any security concerns with on-chain KYC?", "mentions_bot": True},
        {"user": "bob_trader", "msg": "Privacy is a concern with on-chain identity", "mentions_bot": False},
        {"user": "david_founder", "msg": "We use ZK proofs, data stays private", "mentions_bot": False},
        {"user": "jack_whale", "msg": "That's the right approach, we'd consider that", "mentions_bot": False},
        # Extended with more DeFi-specific content
        {"user": "bob_trader", "msg": "I've been trading on Base since day 1, volume is up 300% this month", "mentions_bot": False},
        {"user": "grace_pm", "msg": "Our TVL hit $100M last week, fastest growing on Base", "mentions_bot": False},
        {"user": "jack_whale", "msg": "We're looking at protocols with at least $50M TVL for investment", "mentions_bot": False},
        {"user": "alice_dev", "msg": "Gas costs on Base are 100x cheaper than mainnet for swaps", "mentions_bot": False},
        {"user": "henry_auditor", "msg": "Found 3 critical bugs in DeFi protocols this month, always DYOR", "mentions_bot": False},
        {"user": "bob_trader", "msg": "My strategy: 60% stables, 30% ETH, 10% high-risk plays", "mentions_bot": False},
        {"user": "david_founder", "msg": "Our wallet address is 0x1234...5678 for anyone wanting to test", "mentions_bot": False},
        {"user": "grace_pm", "msg": "@bob_trader that's a solid conservative allocation", "mentions_bot": False, "reply_to": "bob_trader"},
        {"user": "jack_whale", "msg": "Institutional money prefers 80% stables minimum", "mentions_bot": False},
        {"user": "alice_dev", "msg": f"@{BOT_NAME} what's the typical slippage on Base DEXes?", "mentions_bot": True},
        {"user": "henry_auditor", "msg": "Important: always check contract verification on Basescan", "mentions_bot": False},
        {"user": "bob_trader", "msg": "Made 15% last week on the ETH/USDC pool", "mentions_bot": False},
        {"user": "grace_pm", "msg": "Reminder: our governance vote ends Friday at midnight", "mentions_bot": False},
        {"user": "david_founder", "msg": "@grace_pm I'll vote yes on proposal 7", "mentions_bot": False, "reply_to": "grace_pm"},
        {"user": "jack_whale", "msg": "Same here, the fee reduction makes sense for growth", "mentions_bot": False},
    ],
    "NFT Creators": [
        # Members: carol_artist, ivy_marketer, frank_newbie, grace_pm, bob_trader
        {"user": "carol_artist", "msg": "Just finished my new generative collection!", "mentions_bot": False},
        {"user": "ivy_marketer", "msg": "@carol_artist congrats! Can I see some previews?", "mentions_bot": False, "reply_to": "carol_artist"},
        {"user": "carol_artist", "msg": "Sure, it's geometric patterns inspired by nature", "mentions_bot": False},
        {"user": "frank_newbie", "msg": "How do you even create generative art?", "mentions_bot": False},
        {"user": "carol_artist", "msg": "@frank_newbie I use p5.js, it's JavaScript for creative coding", "mentions_bot": False, "reply_to": "frank_newbie"},
        {"user": "carol_artist", "msg": f"@{BOT_NAME} what's the best way to deploy NFTs on Base?", "mentions_bot": True},
        {"user": "bob_trader", "msg": "NFT market has been recovering, good timing", "mentions_bot": False},
        {"user": "ivy_marketer", "msg": "@carol_artist I can connect you with some collectors", "mentions_bot": False, "reply_to": "carol_artist"},
        {"user": "grace_pm", "msg": "We're exploring NFT-gated features for our protocol", "mentions_bot": False},
        {"user": "carol_artist", "msg": "@grace_pm that could be interesting, let's chat!", "mentions_bot": False, "reply_to": "grace_pm"},
        {"user": "frank_newbie", "msg": "I want to buy my first NFT, any recommendations?", "mentions_bot": False},
        {"user": "bob_trader", "msg": "@frank_newbie start with small buys, learn the market", "mentions_bot": False, "reply_to": "frank_newbie"},
        {"user": "ivy_marketer", "msg": f"@{BOT_NAME} what are the best NFT marketplaces on Base?", "mentions_bot": True},
        {"user": "grace_pm", "msg": "We might feature carol's art in our app", "mentions_bot": False},
        {"user": "carol_artist", "msg": "That would be amazing exposure!", "mentions_bot": False},
        # Extended with more NFT-specific content
        {"user": "carol_artist", "msg": "My collection has 10,000 unique pieces, mint price 0.01 ETH", "mentions_bot": False},
        {"user": "ivy_marketer", "msg": "I've managed 5 successful NFT launches, total 50k mints", "mentions_bot": False},
        {"user": "frank_newbie", "msg": "What wallet do I need to buy NFTs on Base?", "mentions_bot": False},
        {"user": "bob_trader", "msg": "@frank_newbie MetaMask works, just add Base network", "mentions_bot": False, "reply_to": "frank_newbie"},
        {"user": "carol_artist", "msg": "My OpenSea collection: opensea.io/collection/geometric-nature", "mentions_bot": False},
        {"user": "ivy_marketer", "msg": "Pro tip: build community before launch, Discord is essential", "mentions_bot": False},
        {"user": "grace_pm", "msg": "We could do an NFT drop for our top 100 users", "mentions_bot": False},
        {"user": "carol_artist", "msg": "@grace_pm I'd love to design that! My rates are 2 ETH per collection", "mentions_bot": False, "reply_to": "grace_pm"},
        {"user": "bob_trader", "msg": "Floor prices on Base NFTs are up 50% this month", "mentions_bot": False},
        {"user": "frank_newbie", "msg": "What's a floor price?", "mentions_bot": False},
        {"user": "ivy_marketer", "msg": "@frank_newbie lowest price available for an NFT in a collection", "mentions_bot": False, "reply_to": "frank_newbie"},
        {"user": "carol_artist", "msg": f"@{BOT_NAME} what royalty percentage do you recommend for creators?", "mentions_bot": True},
        {"user": "grace_pm", "msg": "Announcement: NFT Creator contest starting next week, 5 ETH prize pool", "mentions_bot": False},
        {"user": "ivy_marketer", "msg": "I can promote that to my 50k Twitter followers", "mentions_bot": False},
        {"user": "bob_trader", "msg": "I'll put up 1 ETH for the prize pool", "mentions_bot": False},
    ],
}


def get_user_by_username(username: str) -> dict:
    """Get user data by username"""
    for user in USERS:
        if user["username"] == username:
            return user
    return None


def build_event_timeline():
    """
    Build a realistic interleaved timeline of DM and group messages.

    In reality, messages don't arrive as "all DMs first, then all groups".
    Users DM the bot, then join a group discussion, then come back to DM, etc.

    Returns a list of events in realistic order:
    [{"type": "dm", "user": ..., "msg": ...}, {"type": "group", "group": ..., "user": ..., "msg": ...}]
    """
    events = []

    # Build DM events with indices
    dm_queues = {}
    for username, messages in DM_CONVERSATIONS.items():
        dm_queues[username] = list(enumerate(messages))

    # Build group events with indices
    group_queues = {}
    for group_name, messages in GROUP_CONVERSATIONS.items():
        group_queues[group_name] = list(enumerate(messages))

    # Simulate realistic arrival: round-robin with randomness
    # Some users DM first, some jump into groups first
    rng = random.Random(42)  # Fixed seed for reproducibility

    # Phase 1: A few users DM first (3-5 messages each), others jump into group
    early_dm_users = rng.sample(list(dm_queues.keys()), 4)
    for username in early_dm_users:
        # Send 3-5 early DM messages
        n = min(rng.randint(3, 5), len(dm_queues[username]))
        for _ in range(n):
            if dm_queues[username]:
                idx, msg = dm_queues[username].pop(0)
                events.append({"type": "dm", "username": username, "msg": msg, "idx": idx})

    # Phase 2: Groups start getting active, interleaved with DMs
    # Simulate rounds: each round picks a random mix of group msgs and DMs
    round_num = 0
    while any(dm_queues[u] for u in dm_queues) or any(group_queues[g] for g in group_queues):
        round_num += 1

        # Pick 1-2 active groups this round
        active_groups = [g for g in group_queues if group_queues[g]]
        if active_groups:
            groups_this_round = rng.sample(active_groups, min(rng.randint(1, 2), len(active_groups)))
            for group_name in groups_this_round:
                # Send 3-6 group messages (a burst of conversation)
                n = min(rng.randint(3, 6), len(group_queues[group_name]))
                for _ in range(n):
                    if group_queues[group_name]:
                        idx, msg_data = group_queues[group_name].pop(0)
                        events.append({
                            "type": "group",
                            "group_name": group_name,
                            "username": msg_data["user"],
                            "msg": msg_data["msg"],
                            "mentions_bot": msg_data.get("mentions_bot", False),
                            "reply_to": msg_data.get("reply_to"),
                            "idx": idx,
                        })

        # Interleave some DMs (1-3 users send 1-2 messages each)
        active_dm_users = [u for u in dm_queues if dm_queues[u]]
        if active_dm_users:
            dm_users_this_round = rng.sample(active_dm_users, min(rng.randint(1, 3), len(active_dm_users)))
            for username in dm_users_this_round:
                n = min(rng.randint(1, 2), len(dm_queues[username]))
                for _ in range(n):
                    if dm_queues[username]:
                        idx, msg = dm_queues[username].pop(0)
                        events.append({"type": "dm", "username": username, "msg": msg, "idx": idx})

    return events


# Queries to test context retrieval - each with expected evidence
CONTEXT_QUERIES = [
    {
        "query": "Who knows about smart contract development?",
        "scope": "dm",  # DM context (no group_id)
        "expect": "Should mention alice_dev (5 years Solidity, AMM) and henry_auditor (audits, Slither/Mythril)",
    },
    {
        "query": "What projects are being built on Base?",
        "scope": "dm",
        "expect": "Should mention david_founder (identity solution), alice_dev (AMM), grace_pm (yield aggregation)",
    },
    {
        "query": "Who is interested in NFTs?",
        "scope": "dm",
        "expect": "Should mention carol_artist (generative art, p5.js) and ivy_marketer (NFT launches)",
    },
    {
        "query": "Who can help with security audits?",
        "scope": "group",
        "group_name": "Base Builders",
        "expect": "Should mention henry_auditor (Code4rena, Sherlock, critical bugs)",
    },
    {
        "query": "What's the trading activity like?",
        "scope": "group",
        "group_name": "DeFi Discussion",
        "expect": "Should mention bob_trader (profitable, mean reversion) and jack_whale (fund, institutional)",
    },
    {
        "query": "Tell me about the NFT collection being launched",
        "scope": "group",
        "group_name": "NFT Creators",
        "expect": "Should mention carol_artist (geometric patterns, 10k pieces, 0.01 ETH mint)",
    },
]


class RealisticScenarioTest:
    """
    Realistic scenario test - interleaved DMs and group messages.

    Simulates real usage: users DM the bot, join groups, DM again, etc.
    Evaluates profiles and full context retrieval (window + memories).
    """

    def __init__(self):
        self.system = SimpleMemSystem(agent_id=AGENT_ID)
        self.msg_count = 0
        self.batch_count = 0
        self.results = {
            "profiles": {},
            "group_profiles": {},
            "context": {},
        }

    def run_all(self):
        """Run complete test scenario"""
        print("\n" + "=" * 70)
        print("REALISTIC SCENARIO TEST (interleaved)")
        print("=" * 70)

        # Phase 1: Feed messages in realistic order
        print("\n" + "=" * 50)
        print("PHASE 1: Message Ingestion (interleaved)")
        print("=" * 50)
        self.run_ingestion()

        # Phase 2: Evaluate profiles
        print("\n" + "=" * 50)
        print("PHASE 2: Profile Evaluation")
        print("=" * 50)
        self.evaluate_profiles()

        # Phase 3: Context retrieval (full pipeline: window + memories)
        print("\n" + "=" * 50)
        print("PHASE 3: Context Retrieval (ask)")
        print("=" * 50)
        self.evaluate_context()

        # Summary
        self.print_summary()

    def run_ingestion(self):
        """Feed messages in realistic interleaved order"""
        timeline = build_event_timeline()
        print(f"Total events: {len(timeline)}")

        group_lookup = {g["name"]: g for g in GROUPS}

        for event in timeline:
            user = get_user_by_username(event["username"])
            if not user:
                continue

            self.msg_count += 1

            if event["type"] == "dm":
                result = self.system.add_dialogue(
                    speaker=event["username"],
                    content=event["msg"],
                    platform="telegram",
                    group_id=None,
                    user_id=f"telegram:{user['id']}",
                    username=event["username"],
                    message_id=f"dm_{event['username']}_{event['idx']}",
                    add_to_firestore=True,
                    use_stateless_processing=True,
                )
                if result.get("processed"):
                    self.batch_count += 1
                    print(f"  [{self.msg_count}] DM batch from {event['username']} → {result.get('memories_created', 0)} memories")

            elif event["type"] == "group":
                group_info = group_lookup.get(event["group_name"])
                if not group_info:
                    continue
                # Validate membership
                if event["username"] not in group_info.get("members", []):
                    continue

                group_id = f"telegram_{group_info['id']}"
                result = self.system.add_dialogue(
                    speaker=event["username"],
                    content=event["msg"],
                    platform="telegram",
                    group_id=group_id,
                    user_id=f"telegram:{user['id']}",
                    username=event["username"],
                    message_id=f"group_{event['group_name']}_{event['idx']}",
                    is_reply=event.get("reply_to") is not None,
                    mentioned_users=[BOT_NAME] if event.get("mentions_bot") else [],
                    add_to_firestore=True,
                    use_stateless_processing=True,
                )
                if result.get("processed"):
                    self.batch_count += 1
                    print(f"  [{self.msg_count}] Group '{event['group_name']}' batch → {result.get('memories_created', 0)} memories")

        print(f"\nIngestion complete: {self.msg_count} messages, {self.batch_count} batches processed")

    def evaluate_profiles(self):
        """Check user and group profiles"""

        # User profiles
        print("\n--- User Profiles ---")
        for user in USERS:
            username = user["username"]
            user_id = f"telegram:{user['id']}"
            persona = user["persona"]

            try:
                profile = self.system.user_profile_store.get_profile_by_universal_id(user_id)
                if profile:
                    print(f"\n{username} (expected: {persona})")
                    print(f"  Summary: {profile.summary[:150]}")
                    print(f"  Interests: {[i.keyword for i in profile.interests[:8]]}")
                    print(f"  Messages: {profile.total_messages_processed}")
                    self.results["profiles"][username] = {
                        "has_profile": True,
                        "summary": profile.summary,
                        "interests": [i.keyword for i in profile.interests],
                        "messages": profile.total_messages_processed,
                    }
                else:
                    print(f"\n{username}: NO PROFILE (expected: {persona})")
                    self.results["profiles"][username] = {"has_profile": False}
            except Exception as e:
                print(f"\n{username}: ERROR - {e}")
                self.results["profiles"][username] = {"has_profile": False, "error": str(e)}

        # Group profiles
        print("\n--- Group Profiles ---")
        for group in GROUPS:
            group_id = f"telegram_{group['id']}"
            group_name = group["name"]

            try:
                profile = self.system.group_profile_store.get_group_profile(group_id)
                if profile:
                    print(f"\n{group_name} (topic: {group['topic']})")
                    print(f"  Summary: {profile.summary[:150]}")
                    print(f"  Topics: {profile.main_topics[:5]}")
                    print(f"  Tone: {profile.tone}")
                    self.results["group_profiles"][group_name] = {
                        "has_profile": True,
                        "summary": profile.summary,
                        "topics": profile.main_topics,
                        "tone": profile.tone,
                    }
                else:
                    print(f"\n{group_name}: NO PROFILE")
                    self.results["group_profiles"][group_name] = {"has_profile": False}
            except Exception as e:
                print(f"\n{group_name}: ERROR - {e}")
                self.results["group_profiles"][group_name] = {"has_profile": False, "error": str(e)}

    def evaluate_context(self):
        """Test context retrieval: raw search() results + ask() answer."""

        group_lookup = {g["name"]: g for g in GROUPS}

        for q in CONTEXT_QUERIES:
            query = q["query"]
            expect = q["expect"]
            print(f"\n{'─' * 60}")
            print(f"Query: {query}")
            print(f"Expected: {expect}")

            try:
                if q["scope"] == "group":
                    group_info = group_lookup[q["group_name"]]
                    group_id = f"telegram_{group_info['id']}"
                    search_results = self.system.search(
                        query=query,
                        group_id=group_id,
                    )
                    answer = self.system.ask(
                        question=query,
                        group_id=group_id,
                        include_firestore_context=True,
                    )
                else:
                    search_results = self.system.search(
                        query=query,
                    )
                    answer = self.system.ask(
                        question=query,
                        include_firestore_context=True,
                    )

                # Print raw search results
                print(f"\n  --- Raw search() results ---")
                for table_key in ["dm_memories", "group_memories", "user_memories", "interaction_memories"]:
                    items = search_results.get(table_key, [])
                    if items:
                        print(f"  [{table_key}] ({len(items)} results)")
                        for i, item in enumerate(items[:5], 1):
                            text = getattr(item, 'content', None) or getattr(item, 'lossless_restatement', str(item))
                            speaker = getattr(item, 'speaker', None) or getattr(item, 'username', None) or getattr(item, 'user_id', '')
                            print(f"    {i}. [{speaker}] {text[:120]}")

                agent_resp = search_results.get("agent_responses", [])
                if agent_resp:
                    print(f"  [agent_responses] ({len(agent_resp)} results)")

                print(f"\n  --- ask() answer ---")
                print(f"  {answer[:300]}")

                self.results["context"][query] = {
                    "answer": answer,
                    "scope": q["scope"],
                    "raw_counts": {
                        k: len(search_results.get(k, []))
                        for k in ["dm_memories", "group_memories", "user_memories", "interaction_memories"]
                    },
                }
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback; traceback.print_exc()
                self.results["context"][query] = {"error": str(e)}

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)

        # Ingestion
        print(f"\nMessages ingested: {self.msg_count}")
        print(f"Batches processed: {self.batch_count}")

        # Profiles
        profiles_created = sum(1 for v in self.results["profiles"].values() if v.get("has_profile"))
        print(f"\nUser Profiles: {profiles_created}/{len(USERS)} created")
        for username, data in self.results["profiles"].items():
            status = "OK" if data.get("has_profile") else "MISSING"
            extra = f" ({data.get('messages', 0)} msgs, {len(data.get('interests', []))} interests)" if data.get("has_profile") else ""
            print(f"  {username}: {status}{extra}")

        group_profiles = sum(1 for v in self.results["group_profiles"].values() if v.get("has_profile"))
        print(f"\nGroup Profiles: {group_profiles}/{len(GROUPS)} created")
        for name, data in self.results["group_profiles"].items():
            status = "OK" if data.get("has_profile") else "MISSING"
            extra = f" (tone={data.get('tone')}, {len(data.get('topics', []))} topics)" if data.get("has_profile") else ""
            print(f"  {name}: {status}{extra}")

        # Context
        answered = sum(1 for v in self.results["context"].values() if "answer" in v)
        print(f"\nContext Queries: {answered}/{len(CONTEXT_QUERIES)} answered")
        for query, data in self.results["context"].items():
            if "answer" in data:
                preview = data["answer"][:80].replace("\n", " ")
                counts = data.get("raw_counts", {})
                counts_str = ", ".join(f"{k.replace('_memories','')}={v}" for k, v in counts.items() if v > 0)
                print(f"  [{data['scope']}] {query[:50]}...")
                print(f"       raw: {counts_str or 'none'}")
                print(f"       → {preview}...")
            else:
                print(f"  [{data.get('scope', '?')}] {query[:50]}... → ERROR")

        # Token usage and cost
        print("\n" + "=" * 70)
        print("TOKEN USAGE & COST (OpenRouter Llama 3.1 8B)")
        print("=" * 70)

        # Main LLM client (ingestion: memory extraction, planning, retrieval)
        main_stats = self.system.llm_client.get_usage_stats()
        print(f"\n  Main LLM (ingestion + retrieval):")
        print(f"    Calls: {main_stats['total_calls']}")
        print(f"    Prompt tokens: {main_stats['prompt_tokens']:,}")
        print(f"    Completion tokens: {main_stats['completion_tokens']:,}")
        print(f"    Total tokens: {main_stats['total_tokens']:,}")
        print(f"    Cost: ${main_stats['estimated_cost_usd']:.6f}")

        # Profile LLM client (summary + interests extraction)
        profile_stats = self.system.user_profile_store.profile_llm.get_usage_stats()
        print(f"\n  Profile LLM (summary + interests):")
        print(f"    Calls: {profile_stats['total_calls']}")
        print(f"    Prompt tokens: {profile_stats['prompt_tokens']:,}")
        print(f"    Completion tokens: {profile_stats['completion_tokens']:,}")
        print(f"    Total tokens: {profile_stats['total_tokens']:,}")
        print(f"    Cost: ${profile_stats['estimated_cost_usd']:.6f}")

        # Group profile LLM (if it has its own client)
        group_profile_llm = getattr(self.system.group_profile_store, 'llm_client', None)
        if group_profile_llm and hasattr(group_profile_llm, 'get_usage_stats'):
            gp_stats = group_profile_llm.get_usage_stats()
            print(f"\n  Group Profile LLM:")
            print(f"    Calls: {gp_stats['total_calls']}")
            print(f"    Prompt tokens: {gp_stats['prompt_tokens']:,}")
            print(f"    Completion tokens: {gp_stats['completion_tokens']:,}")
            print(f"    Total tokens: {gp_stats['total_tokens']:,}")
            print(f"    Cost: ${gp_stats['estimated_cost_usd']:.6f}")

        # Total
        total_tokens = main_stats['total_tokens'] + profile_stats['total_tokens']
        total_cost = main_stats['estimated_cost_usd'] + profile_stats['estimated_cost_usd']
        total_calls = main_stats['total_calls'] + profile_stats['total_calls']
        if group_profile_llm and hasattr(group_profile_llm, 'get_usage_stats'):
            gp_stats = group_profile_llm.get_usage_stats()
            total_tokens += gp_stats['total_tokens']
            total_cost += gp_stats['estimated_cost_usd']
            total_calls += gp_stats['total_calls']

        print(f"\n  TOTAL:")
        print(f"    Total LLM calls: {total_calls}")
        print(f"    Total tokens: {total_tokens:,}")
        print(f"    Total cost: ${total_cost:.6f}")
        print(f"    Cost per message: ${total_cost / max(self.msg_count, 1):.6f}")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    test = RealisticScenarioTest()
    test.run_all()
