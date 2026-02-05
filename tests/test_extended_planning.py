"""
Test Extended Planning System for a0x-memory retrieval.

Tests the extended planning capabilities:
1. Temporal query detection (yesterday, last week, etc.)
2. Summarization intent detection (resume, summarize, catch up)
3. Entity-focused queries (@user, person names)
4. Technical query detection (code, errors, bugs)
5. Low confidence/ambiguous query handling

Usage:
    python tests/test_extended_planning.py
    python tests/test_extended_planning.py --query "your custom query"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from typing import Dict, Any, List

# Test cases for extended planning
TEST_CASES = [
    # 1. Temporal queries
    {
        "name": "temporal_yesterday",
        "query": "What happened yesterday?",
        "expected": {
            "question_type": "temporal",
            "temporal_attributes.period": "yesterday",
            "temporal_attributes.lookback_days": 1,
            "temporal_attributes.recency_bias_level": ">0.5"
        },
        "description": "Temporal query with yesterday reference"
    },
    {
        "name": "temporal_last_week",
        "query": "What did we discuss last week?",
        "expected": {
            "question_type": "temporal",
            "temporal_attributes.period": "last_week",
            "temporal_attributes.lookback_days": 7,
        },
        "description": "Temporal query with last week reference"
    },

    # 2. Summarization intent
    {
        "name": "summarization_resume",
        "query": "Resume the conversation for me",
        "expected": {
            "question_type": "summarization",
            "summarization_intent.use_summaries": True,
        },
        "description": "Summarization request with 'resume'"
    },
    {
        "name": "summarization_catch_up",
        "query": "What did I miss? Catch me up.",
        "expected": {
            "summarization_intent.use_summaries": True,
        },
        "description": "Summarization request with 'catch up'"
    },
    {
        "name": "summarization_what_happened",
        "query": "What happened while I was away?",
        "expected": {
            "summarization_intent.use_summaries": True,
        },
        "description": "Summarization request with 'what happened'"
    },

    # 3. Entity-focused queries
    {
        "name": "entity_at_username",
        "query": "What does @bob know about Python?",
        "expected": {
            "question_type": "entity_focused",
            "entity_focus.primary_entity": "bob",
            "entity_focus.entity_type": "person",
        },
        "description": "Entity-focused query with @username"
    },
    {
        "name": "entity_person_name",
        "query": "What has Alice mentioned about the project?",
        "expected": {
            "entity_focus.primary_entity": "Alice",
            "entity_focus.entity_type": "person",
        },
        "description": "Entity-focused query with person name"
    },

    # 4. Technical queries
    {
        "name": "technical_fix_error",
        "query": "How do I fix the async connection error?",
        "expected": {
            "memory_type_filter": "technical",
        },
        "description": "Technical query about fixing errors"
    },
    {
        "name": "technical_deployment",
        "query": "What's the deployment process for the API?",
        "expected": {
            "memory_type_filter": "technical",
        },
        "description": "Technical query about deployment"
    },

    # 5. Low confidence / ambiguous queries
    {
        "name": "ambiguous_vague",
        "query": "tell me about that thing",
        "expected": {
            "question_confidence": "<0.7",
            "retrieval_depth": "deep",
        },
        "description": "Ambiguous query should have low confidence"
    },

    # 6. Comparison queries
    {
        "name": "comparison_vs",
        "query": "What's the difference between Alice's approach and Bob's approach?",
        "expected": {
            "comparison.is_comparative": True,
        },
        "description": "Comparison query with two entities"
    },

    # 7. Factual query (should NOT trigger summarization)
    {
        "name": "factual_simple",
        "query": "What is Bob's email address?",
        "expected": {
            "question_type": "factual",
            "summarization_intent.use_summaries": False,
        },
        "description": "Simple factual query should not use summaries"
    },
]


def check_expected(result: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
    """
    Check if result matches expected values.
    Returns list of failures (empty if all pass).
    """
    failures = []

    for key, expected_value in expected.items():
        # Handle nested keys (e.g., "temporal_attributes.period")
        keys = key.split(".")
        actual_value = result
        for k in keys:
            if isinstance(actual_value, dict):
                actual_value = actual_value.get(k)
            else:
                actual_value = None
                break

        # Handle comparison operators
        if isinstance(expected_value, str):
            if expected_value.startswith(">"):
                threshold = float(expected_value[1:])
                if actual_value is None or actual_value <= threshold:
                    failures.append(f"{key}: expected >{threshold}, got {actual_value}")
                continue
            elif expected_value.startswith("<"):
                threshold = float(expected_value[1:])
                if actual_value is None or actual_value >= threshold:
                    failures.append(f"{key}: expected <{threshold}, got {actual_value}")
                continue

        # Direct comparison (case-insensitive for strings)
        if isinstance(expected_value, str) and isinstance(actual_value, str):
            if expected_value.lower() != actual_value.lower():
                failures.append(f"{key}: expected '{expected_value}', got '{actual_value}'")
        elif actual_value != expected_value:
            failures.append(f"{key}: expected {expected_value}, got {actual_value}")

    return failures


def run_test(test_case: Dict[str, Any], llm_client, verbose: bool = False) -> Dict[str, Any]:
    """
    Run a single test case using the LLM to generate a plan.
    """
    from core.hybrid_retriever import HybridRetriever

    # Create a minimal mock embedding model
    class MockEmbedding:
        dimension = 384
        def encode_single(self, text, is_query=False):
            import numpy as np
            return np.zeros(384)
        def encode_query(self, queries):
            import numpy as np
            return [np.zeros(384) for _ in queries]

    # Create a minimal retriever instance
    class MockStore:
        def __init__(self):
            self.embedding_model = MockEmbedding()

    retriever = HybridRetriever(
        llm_client=llm_client,
        unified_store=MockStore(),
        enable_planning=True,
        enable_reflection=False
    )

    # Run planning
    query = test_case["query"]
    print(f"\n{'='*60}")
    print(f"Test: {test_case['name']}")
    print(f"Query: {query}")
    print(f"Description: {test_case['description']}")

    try:
        information_plan, search_queries = retriever._plan_and_generate_queries(query)

        if verbose:
            print(f"\nGenerated Plan:")
            print(json.dumps(information_plan, indent=2))
            print(f"\nSearch Queries: {search_queries}")

        # Check expectations
        failures = check_expected(information_plan, test_case["expected"])

        if failures:
            print(f"\nFAILED - {len(failures)} expectation(s) not met:")
            for f in failures:
                print(f"  - {f}")
            return {"status": "FAILED", "failures": failures, "plan": information_plan}
        else:
            print(f"\nPASSED - All expectations met")
            return {"status": "PASSED", "plan": information_plan}

    except Exception as e:
        print(f"\nERROR - {e}")
        return {"status": "ERROR", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Test Extended Planning System")
    parser.add_argument("--query", type=str, help="Run a custom query instead of test cases")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full plan output")
    parser.add_argument("--test", type=str, help="Run a specific test case by name")
    args = parser.parse_args()

    # Initialize LLM client
    from utils.llm_client import LLMClient
    llm_client = LLMClient()

    if args.query:
        # Run custom query
        test_case = {
            "name": "custom",
            "query": args.query,
            "expected": {},
            "description": "Custom query"
        }
        result = run_test(test_case, llm_client, verbose=True)
        print(f"\nFull Plan:")
        print(json.dumps(result.get("plan", {}), indent=2))
        return

    # Run test cases
    if args.test:
        # Run specific test
        test_cases = [tc for tc in TEST_CASES if tc["name"] == args.test]
        if not test_cases:
            print(f"Test case '{args.test}' not found")
            print(f"Available tests: {[tc['name'] for tc in TEST_CASES]}")
            return
    else:
        test_cases = TEST_CASES

    # Run tests
    results = {"passed": 0, "failed": 0, "error": 0}

    for test_case in test_cases:
        result = run_test(test_case, llm_client, verbose=args.verbose)
        results[result["status"].lower()] = results.get(result["status"].lower(), 0) + 1

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Passed: {results['passed']}/{len(test_cases)}")
    print(f"Failed: {results['failed']}/{len(test_cases)}")
    print(f"Errors: {results['error']}/{len(test_cases)}")

    if results['failed'] > 0 or results['error'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
