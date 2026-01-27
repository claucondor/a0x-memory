"""
Quick test of the LLM-based message analyzer
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.memory_classifier import get_message_analyzer


def test_analyzer():
    """Test the LLM analyzer with sample messages"""

    print("=" * 70)
    print("Testing LLM Message Analyzer")
    print("=" * 70)

    analyzer = get_message_analyzer()

    test_messages = [
        {
            "message": "Hey everyone! I'm Alice, been working on DeFi protocols for 2 years, specifically lending protocols on Base.",
            "username": "alice_builder",
            "expected": "expertise"
        },
        {
            "message": "I'm looking for reliable RPC endpoints for my project.",
            "username": "bob_dev",
            "expected": "need"
        },
        {
            "message": "Hi all!",
            "username": "newbie",
            "expected": "skip"
        },
        {
            "message": "I prefer working with TypeScript over Python for smart contracts.",
            "username": "carol_dev",
            "expected": "preference"
        }
    ]

    for i, test in enumerate(test_messages, 1):
        print(f"\n--- Test {i}: {test['username']} ---")
        print(f"Message: {test['message'][:70]}...")
        print(f"Expected: {test['expected']}")

        result = analyzer.analyze_message(
            message=test["message"],
            username=test["username"]
        )

        print(f"\nResult:")
        print(f"  should_remember: {result.get('should_remember')}")
        print(f"  memories: {len(result.get('memories', []))} created")

        for mem in result.get('memories', []):
            print(f"    - [{mem['type']}] {mem['content'][:50]}...")
            print(f"      importance: {mem['importance']}, topics: {mem['topics']}")

        print(f"  interaction_type: {result.get('interaction_type')}")

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_analyzer()
