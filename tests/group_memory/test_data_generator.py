"""
Test Data Generator for Group Memory Architectures

Generates realistic Telegram group conversation data for testing group memory systems.
Based on research findings from GROUP_MEMORY_GENERAL_RESEARCH.md and GROUP_MEMORIES_RESEARCH.md.

Patterns included:
- Agent mentions and direct questions
- User-to-user conversations
- Group-wide announcements
- Cross-context topics
- Expertise demonstrations (Transactive Memory)
- Social identity formation
"""

import random
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class Message:
    """A single message in a Telegram group."""
    message_id: str
    timestamp: str
    sender_id: str
    sender_name: str
    group_id: str
    content: str
    message_type: str  # "agent_mention", "user_conversation", "announcement", "general"
    reply_to: str = None  # message_id if this is a reply
    mentioned_users: List[str] = None
    entities: List[str] = None  # Extracted entities
    topics: List[str] = None  # Topic tags

    def __post_init__(self):
        if self.mentioned_users is None:
            self.mentioned_users = []
        if self.entities is None:
            self.entities = []
        if self.topics is None:
            self.topics = []


class GroupMessageGenerator:
    """
    Generates realistic Telegram group messages for testing.

    Simulates:
    - 50+ users with distinct personalities and expertise areas
    - Natural conversation patterns
    - Agent interactions (mentions, questions)
    - Group decisions and announcements
    - Expertise demonstration (Transactive Memory)
    """

    # Expertise areas for users (Transactive Memory: who knows what)
    EXPERTISE_AREAS = [
        "defi", "nft", "trading", "smart_contracts", "yield_farming",
        "liquidity_mining", "dao", "governance", "layer2", "bridges",
        "wallet_security", "tokenomics", "audits", "dex", "cex"
    ]

    # Group topics (Social Identity: shared interests)
    GROUP_TOPICS = [
        "DeFi strategies", "NFT collecting", "Trading signals",
        "Smart contract security", "Yield optimization", "DAO governance"
    ]

    # Agent names for testing
    AGENT_NAMES = ["jessexbt", "a0x_agent", "crypto_helper"]

    # First names for realistic usernames
    FIRST_NAMES = [
        "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace",
        "Henry", "Ivy", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia",
        "Peter", "Quinn", "Ryan", "Sophia", "Thomas", "Uma", "Victor",
        "Wendy", "Xavier", "Yara", "Zach", "Alex", "Bella", "Chris",
        "Dana", "Eric", "Fiona", "George", "Hana", "Ivan", "Julia",
        "Kevin", "Lara", "Marcus", "Nina", "Oscar", "Paula", "Quentin",
        "Rachel", "Sam", "Tina", "Umar", "Vera", "Will", "Xena"
    ]

    # Message templates for different scenarios
    AGENT_MENTION_TEMPLATES = [
        "{agent}, what do you think about {topic}?",
        "@{agent} can you explain {concept}?",
        "Hey {agent}, is {project} a good investment?",
        "{agent} help me understand {topic}",
        "@{agent} what's your take on {concept}?",
        "{agent}, do you have info about {project}?"
    ]

    USER_CONVERSATION_TEMPLATES = [
        "{user1}, have you seen the new {project} tokenomics?",
        "Hey {user1}, what's your strategy for {topic}?",
        "@{user1} did you check out {concept}?",
        "{user1}, remember we discussed {topic} yesterday?",
        "Anyone here familiar with {concept}? @ @{user1}"
    ]

    ANNOUNCEMENT_TEMPLATES = [
        "📢 Important: {group_name} will be launching {project} soon!",
        "🚀 Big news! We're integrating with {project}",
        "⚠️ Security alert: Be careful with {concept}",
        "📊 Weekly stats: {metric} increased by {percent}%",
        "🎉 Milestone reached: {achievement}"
    ]

    EXPERTISE_DEMONSTRATION_TEMPLATES = [
        "Based on my experience with {topic}, I'd suggest {strategy}",
        "I've been working on {concept} for a while, here's my take...",
        "For those interested in {topic}, here's what I've learned:",
        "Regarding {concept}, the best approach is...",
        "I've analyzed {project} and found {finding}"
    ]

    def __init__(
        self,
        num_users: int = 50,
        num_groups: int = 3,
        messages_per_group: int = 500,
        agent_id: str = "jessexbt"
    ):
        self.num_users = num_users
        self.num_groups = num_groups
        self.messages_per_group = messages_per_group
        self.agent_id = agent_id

        # Generate users with expertise
        self.users = self._generate_users()

        # Generate groups
        self.groups = self._generate_groups()

        # Track conversation state for threading
        self.conversation_threads = {}  # group_id -> [active_threads]

    def _generate_users(self) -> List[Dict[str, Any]]:
        """Generate users with distinct personalities and expertise."""
        users = []

        for i in range(self.num_users):
            # Assign 1-3 expertise areas (Transactive Memory)
            expertise = random.sample(
                self.EXPERTISE_AREAS,
                k=random.randint(1, 3)
            )

            user = {
                "user_id": f"user_{i+1}",
                "username": f"@{self.FIRST_NAMES[i].lower()}{random.randint(10, 99)}",
                "real_name": self.FIRST_NAMES[i],
                "expertise": expertise,
                "personality": random.choice(["technical", "casual", "helpful", "skeptical"]),
                "message_frequency": random.randint(1, 10),  # Messages per day
                "join_date": (datetime.now() - timedelta(days=random.randint(30, 365))).isoformat()
            }
            users.append(user)

        return users

    def _generate_groups(self) -> List[Dict[str, Any]]:
        """Generate groups with distinct cultures (Social Identity)."""
        groups = []

        for i in range(self.num_groups):
            group = {
                "group_id": f"group_{i+1}",
                "group_name": f"{self.GROUP_TOPICS[i]} Community",
                "topic": self.GROUP_TOPICS[i],
                "member_count": random.randint(40, 60),
                "culture": random.choice(["professional", "casual", "technical", "memey"]),
                "primary_language": random.choice(["en", "es", "mixed"]),
                "created_at": (datetime.now() - timedelta(days=random.randint(60, 365))).isoformat()
            }
            groups.append(group)

        return groups

    def _select_expert_user(self, topic: str, group_id: str) -> Dict[str, Any]:
        """Select a user with expertise in the given topic (Transactive Memory)."""
        # Find users with relevant expertise
        relevant_users = []

        for user in self.users:
            # Check if user's expertise matches topic
            user_expertise = user.get("expertise", [])
            if any(exp in topic.lower() or topic.lower() in exp for exp in user_expertise):
                relevant_users.append(user)

        # If no experts found, return random user
        if not relevant_users:
            return random.choice(self.users)

        # Return random expert (weighted by frequency)
        return random.choice(relevant_users)

    def _generate_agent_mention(
        self,
        group: Dict[str, Any],
        timestamp: datetime
    ) -> Message:
        """Generate a message mentioning the agent."""
        template = random.choice(self.AGENT_MENTION_TEMPLATES)

        # Select random user
        user = random.choice(self.users)

        # Fill template
        topic = random.choice(self.EXPERTISE_AREAS)
        concept = random.choice(["APY", "impermanent loss", "slippage", "gas fees", "yield farming"])
        project = random.choice(["Uniswap", "Aave", "Compound", "Curve", "Lido"])

        content = template.format(
            agent=self.agent_id,
            topic=topic,
            concept=concept,
            project=project
        )

        return Message(
            message_id=f"msg_{group['group_id']}_{timestamp.timestamp()}",
            timestamp=timestamp.isoformat(),
            sender_id=user["user_id"],
            sender_name=user["username"],
            group_id=group["group_id"],
            content=content,
            message_type="agent_mention",
            mentioned_users=[self.agent_id],
            topics=[topic]
        )

    def _generate_user_conversation(
        self,
        group: Dict[str, Any],
        timestamp: datetime
    ) -> Message:
        """Generate a user-to-user conversation."""
        template = random.choice(self.USER_CONVERSATION_TEMPLATES)

        # Select two users
        user1 = random.choice(self.users)
        user2 = random.choice([u for u in self.users if u["user_id"] != user1["user_id"]])

        # Fill template
        topic = random.choice(self.EXPERTISE_AREAS)
        concept = random.choice(["staking", "yield", "liquidity", "collateral", "vaults"])
        project = random.choice(["Uniswap", "Aave", "Curve", "SushiSwap", "Balancer"])

        content = template.format(
            user1=user1["username"],
            topic=topic,
            concept=concept,
            project=project
        )

        # 50% chance this is a reply to previous message
        reply_to = None
        if group["group_id"] in self.conversation_threads and random.random() < 0.5:
            threads = self.conversation_threads[group["group_id"]]
            if threads:
                reply_to = random.choice(threads)

        message = Message(
            message_id=f"msg_{group['group_id']}_{timestamp.timestamp()}",
            timestamp=timestamp.isoformat(),
            sender_id=user2["user_id"],
            sender_name=user2["username"],
            group_id=group["group_id"],
            content=content,
            message_type="user_conversation",
            reply_to=reply_to,
            mentioned_users=[user1["username"]],
            topics=[topic]
        )

        # Add to conversation threads
        if group["group_id"] not in self.conversation_threads:
            self.conversation_threads[group["group_id"]] = []

        self.conversation_threads[group["group_id"]].append(message.message_id)

        # Keep only last 10 threads
        if len(self.conversation_threads[group["group_id"]]) > 10:
            self.conversation_threads[group["group_id"]] = self.conversation_threads[group["group_id"]][-10:]

        return message

    def _generate_announcement(
        self,
        group: Dict[str, Any],
        timestamp: datetime
    ) -> Message:
        """Generate a group-wide announcement."""
        template = random.choice(self.ANNOUNCEMENT_TEMPLATES)

        # Select admin user (first 5 users are admins)
        admin = random.choice(self.users[:5])

        # Fill template
        project = random.choice(["Protocol", "Platform", "Feature", "Integration", "Update"])
        concept = random.choice(["security", "audit", "launch", "migration", "upgrade"])
        metric = random.choice(["TVL", "users", "volume", "transactions"])
        achievement = random.choice(["10K users", "$1M TVL", "100K transactions", "1 year online"])

        content = template.format(
            group_name=group["group_name"],
            project=project,
            concept=concept,
            metric=metric,
            percent=random.randint(10, 50),
            achievement=achievement
        )

        return Message(
            message_id=f"msg_{group['group_id']}_{timestamp.timestamp()}",
            timestamp=timestamp.isoformat(),
            sender_id=admin["user_id"],
            sender_name=admin["username"],
            group_id=group["group_id"],
            content=content,
            message_type="announcement",
            topics=["group_announcement"]
        )

    def _generate_expertise_demonstration(
        self,
        group: Dict[str, Any],
        timestamp: datetime
    ) -> Message:
        """Generate expertise demonstration (Transactive Memory)."""
        template = random.choice(self.EXPERTISE_DEMONSTRATION_TEMPLATES)

        # Select topic and find expert
        topic = random.choice(self.EXPERTISE_AREAS)
        expert = self._select_expert_user(topic, group["group_id"])

        # Fill template
        concept = random.choice(["yield optimization", "risk management", "portfolio allocation"])
        strategy = random.choice(["diversification", "hedging", "dollar-cost averaging"])
        project = random.choice(["Aave", "Uniswap", "Curve", "Lido"])
        finding = random.choice(["good risk/reward", "high APY but risky", "solid fundamentals"])

        content = template.format(
            topic=topic,
            concept=concept,
            strategy=strategy,
            project=project,
            finding=finding
        )

        return Message(
            message_id=f"msg_{group['group_id']}_{timestamp.timestamp()}",
            timestamp=timestamp.isoformat(),
            sender_id=expert["user_id"],
            sender_name=expert["username"],
            group_id=group["group_id"],
            content=content,
            message_type="expertise_demonstration",
            topics=[topic, expert["expertise"][0]]
        )

    def generate_group_messages(self, group: Dict[str, Any]) -> List[Message]:
        """Generate all messages for a single group."""
        messages = []

        # Start time for group
        current_time = datetime.now() - timedelta(days=random.randint(1, 30))

        # Message type distribution (realistic pattern)
        # More general messages, fewer agent mentions and announcements
        type_distribution = {
            "agent_mention": 0.10,      # 10% - Agent mentions
            "user_conversation": 0.50,   # 50% - User conversations
            "announcement": 0.05,        # 5% - Announcements
            "expertise_demonstration": 0.15,  # 15% - Expertise
            "general": 0.20              # 20% - General messages
        }

        # Generate messages
        for i in range(self.messages_per_group):
            # Select message type based on distribution
            msg_type = random.choices(
                list(type_distribution.keys()),
                weights=list(type_distribution.values())
            )[0]

            # Generate message based on type
            if msg_type == "agent_mention":
                message = self._generate_agent_mention(group, current_time)
            elif msg_type == "user_conversation":
                message = self._generate_user_conversation(group, current_time)
            elif msg_type == "announcement":
                message = self._generate_announcement(group, current_time)
            elif msg_type == "expertise_demonstration":
                message = self._generate_expertise_demonstration(group, current_time)
            else:  # general
                user = random.choice(self.users)
                message = Message(
                    message_id=f"msg_{group['group_id']}_{current_time.timestamp()}",
                    timestamp=current_time.isoformat(),
                    sender_id=user["user_id"],
                    sender_name=user["username"],
                    group_id=group["group_id"],
                    content=random.choice([
                        "Hello everyone! 👋",
                        "How's it going?",
                        "What's new today?",
                        f"Anyone here interested in {random.choice(self.EXPERTISE_AREAS)}?",
                        "Good morning!",
                        "Thanks for sharing!"
                    ]),
                    message_type="general"
                )

            messages.append(message)

            # Increment time by random interval (1-60 minutes)
            current_time += timedelta(minutes=random.randint(1, 60))

        # Sort by timestamp
        messages.sort(key=lambda m: m.timestamp)

        return messages

    def generate_all(self) -> Dict[str, Any]:
        """Generate complete test dataset."""
        dataset = {
            "metadata": {
                "num_users": self.num_users,
                "num_groups": self.num_groups,
                "messages_per_group": self.messages_per_group,
                "total_messages": self.num_groups * self.messages_per_group,
                "generated_at": datetime.now().isoformat()
            },
            "users": self.users,
            "groups": self.groups,
            "messages": []
        }

        # Generate messages for each group
        for group in self.groups:
            group_messages = self.generate_group_messages(group)
            dataset["messages"].extend(group_messages)

        return dataset

    def save_to_file(self, filepath: str):
        """Save generated dataset to JSON file."""
        dataset = self.generate_all()

        # Convert messages to dicts
        dataset["messages"] = [asdict(msg) for msg in dataset["messages"]]

        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)

        print(f"Generated test data saved to {filepath}")
        print(f"  - Users: {dataset['metadata']['num_users']}")
        print(f"  - Groups: {dataset['metadata']['num_groups']}")
        print(f"  - Messages: {dataset['metadata']['total_messages']}")

    def get_query_scenarios(self) -> List[Dict[str, Any]]:
        """
        Generate test query scenarios based on the generated data.

        Scenarios:
        1. Agent mentioned directly
        2. User-to-user conversation retrieval
        3. Group-wide announcements
        4. Cross-context queries (same user, different groups)
        5. Expertise location (who knows what)
        """
        return [
            {
                "scenario": "agent_mention",
                "description": "User asks agent about specific topic",
                "query": f"{self.agent_id}, what do you know about DeFi yield farming?",
                "expected_context": ["defi", "yield_farming", "agent_mention"]
            },
            {
                "scenario": "user_conversation",
                "description": "Retrieve conversation between two users",
                "query": "What did @alice23 and @bob45 discuss about trading?",
                "expected_context": ["trading", "user_conversation"]
            },
            {
                "scenario": "group_announcement",
                "description": "Find recent announcements in group",
                "query": "What are the latest updates from this group?",
                "expected_context": ["announcement", "group_announcement"]
            },
            {
                "scenario": "cross_context",
                "description": "User's activity across multiple groups",
                "query": "What has @alice23 been asking about lately?",
                "expected_context": ["cross_group", "user_questions"]
            },
            {
                "scenario": "expertise_location",
                "description": "Find users with specific expertise",
                "query": "Who here knows about smart contract security?",
                "expected_context": ["smart_contracts", "expertise"]
            },
            {
                "scenario": "topic_aggregation",
                "description": "Aggregate information about a topic",
                "query": "What has the group discussed about NFTs?",
                "expected_context": ["nft", "group_discussion"]
            }
        ]


def main():
    """Generate test data and save to file."""
    generator = GroupMessageGenerator(
        num_users=50,
        num_groups=3,
        messages_per_group=500,
        agent_id="jessexbt"
    )

    # Save to file
    output_file = "/home/oydual3/a0x/a0x-memory/tests/group_memory/test_data.json"
    generator.save_to_file(output_file)

    # Save query scenarios
    scenarios = generator.get_query_scenarios()
    scenarios_file = "/home/oydual3/a0x/a0x-memory/tests/group_memory/query_scenarios.json"
    with open(scenarios_file, 'w') as f:
        json.dump(scenarios, f, indent=2)

    print(f"\nQuery scenarios saved to {scenarios_file}")
    print(f"  - Scenarios: {len(scenarios)}")


if __name__ == "__main__":
    main()
