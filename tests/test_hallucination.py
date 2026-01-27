"""
Test Hallucination Detection in Profile Generation

Tests whether LLM models hallucinate information that wasn't in the original
messages when generating user/group profiles.

Usage:
    python tests/test_hallucination.py

Key tests:
1. Profile should NOT contain info not in messages
2. Concepts/interests should be specific, not invented
3. Named entities should match actual mentions
4. Expertise level should match evidence
"""
import os
import sys
import json
import time

os.environ["USE_LOCAL_STORAGE"] = "true"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_client import LLMClient

# ============================================================
# Models to test (pick 2-3 best from test_profile_models.py)
# ============================================================
MODELS = {
    "meta-llama/llama-3.1-8b-instruct": "Llama 3.1 8B",
    "google/gemini-2.0-flash-001": "Gemini 2.0 Flash",
    "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
}

# ============================================================
# Test Cases - Messages with KNOWN facts to verify against
# ============================================================

TEST_CASES = [
    {
        "name": "DeFi Developer - Clear Facts",
        "messages": [
            "I'm building a DEX aggregator on Base using Solidity",
            "been coding smart contracts for 3 years now",
            "our team got an Optimism grant last month",
            "wallet is 0xABC...123, deployed on Base Sepolia",
            "we process about 500 trades daily",
        ],
        # Facts that SHOULD be extracted
        "expected_facts": [
            "builds DEX aggregator",
            "uses Solidity",
            "3 years experience",
            "Optimism grant",
            "500 trades daily",
        ],
        # Facts that should NOT appear (common hallucinations)
        "forbidden_facts": [
            "NFT",                  # never mentioned
            "staking",              # never mentioned
            "validator",            # never mentioned
            "governance",           # never mentioned
            "Ethereum mainnet",     # only Base mentioned
            "full-stack",           # only Solidity mentioned
            "React",               # never mentioned
            "frontend",            # never mentioned
            "yield farming",       # never mentioned
            "DAO",                 # never mentioned
        ],
    },
    {
        "name": "NFT Artist - Minimal Info",
        "messages": [
            "gm! just minted my first collection on Base",
            "10 pieces, all hand-drawn digital art",
            "prices start at 0.01 ETH",
            "link: opensea.io/my-collection",
        ],
        "expected_facts": [
            "NFT creator",
            "hand-drawn digital art",
            "Base chain",
            "0.01 ETH",
        ],
        "forbidden_facts": [
            "DeFi",                 # not mentioned
            "developer",            # they're an artist
            "Solidity",            # not mentioned
            "smart contract",      # not mentioned
            "trading",             # not mentioned
            "experienced",         # only first collection
            "expert",              # first collection = beginner
            "community leader",    # no evidence
            "influencer",          # no evidence
        ],
    },
    {
        "name": "Casual User - Vague Messages",
        "messages": [
            "hey whats up",
            "anyone know a good wallet for Base?",
            "thanks! will try Coinbase Wallet",
            "how do I bridge from Ethereum?",
            "ok cool, appreciate the help",
        ],
        "expected_facts": [
            "new to Base",
            "uses Coinbase Wallet",
            "needs help bridging",
        ],
        "forbidden_facts": [
            "developer",           # never mentioned
            "DeFi expert",        # clearly a beginner
            "advanced",            # clearly a beginner
            "trader",              # no trading mentioned
            "building",            # not building anything
            "project",             # no project mentioned
            "team",                # no team mentioned
            "launched",            # nothing launched
            "technical",           # asking basic questions
        ],
    },
    {
        "name": "Group Admin - Authority Without Technical Claims",
        "messages": [
            "welcome to the group everyone!",
            "rules: no spam, be respectful, no price shilling",
            "weekly call is Fridays at 3pm UTC",
            "please introduce yourselves in #introductions",
            "I'll be sharing ecosystem updates every Monday",
            "if you have questions tag me @admin_mike",
        ],
        "expected_facts": [
            "group admin",
            "manages community",
            "weekly calls Fridays",
            "shares ecosystem updates",
        ],
        "forbidden_facts": [
            "developer",           # no code discussed
            "Solidity",           # never mentioned
            "DeFi expert",        # no DeFi discussed
            "trader",             # no trading
            "built",              # didn't build anything
            "launched",           # didn't launch anything
            "wallet",             # no wallet shared
            "advanced",           # admin role, not technical
            "smart contract",     # never mentioned
        ],
    },
]

# ============================================================
# Prompts
# ============================================================

PROFILE_PROMPT = """
Analyze these messages from a single user and extract their profile.

[Messages]
{messages}

[CRITICAL RULES]
- ONLY extract information explicitly stated or clearly implied in the messages
- DO NOT infer expertise beyond what is demonstrated
- DO NOT assume technical skills unless explicitly mentioned
- If information is ambiguous, mark it as uncertain
- Be conservative: it's better to miss something than to hallucinate

Extract:
1. summary: 1-2 sentences about who this person is (ONLY based on evidence)
2. expertise_level: beginner, intermediate, advanced, or expert (based on evidence)
3. interests: Specific interests MENTIONED in messages (not inferred)
4. key_facts: Only concrete facts stated in the messages
5. confidence: How confident are you in this profile (0.0-1.0)

Return ONLY valid JSON.
"""

STRICT_PROFILE_PROMPT = """
Analyze these messages and extract ONLY verified facts about the user.

[Messages]
{messages}

[STRICT RULES]
1. NEVER add information not explicitly stated in the messages
2. NEVER assume technical skills unless the user explicitly mentions them
3. NEVER inflate expertise level - match it to the evidence
4. If the user asks basic questions -> they are likely a beginner
5. If the user only mentions one topic -> do not add related topics
6. Use DIRECT QUOTES from messages as evidence for each fact

Extract:
1. summary: Brief description (ONLY from evidence in messages)
2. expertise_level: beginner/intermediate/advanced/expert (with justification)
3. verified_facts: Facts with direct quote evidence
4. interests: ONLY topics explicitly discussed
5. things_NOT_known: What we canNOT determine from these messages

Return ONLY valid JSON.
"""


# ============================================================
# Evaluation
# ============================================================

def check_hallucinations(data: dict, test_case: dict) -> dict:
    """Check for hallucinated content in profile."""
    result = {
        "expected_found": [],
        "expected_missed": [],
        "hallucinations_found": [],
        "hallucinations_clear": [],
        "score": 0.0,
    }

    # Convert all profile data to searchable text
    profile_text = json.dumps(data).lower()

    # Check expected facts
    for fact in test_case["expected_facts"]:
        # Check if any keyword from the fact appears
        keywords = fact.lower().split()
        found = any(kw in profile_text for kw in keywords if len(kw) > 3)
        if found:
            result["expected_found"].append(fact)
        else:
            result["expected_missed"].append(fact)

    # Check forbidden facts (hallucinations)
    for forbidden in test_case["forbidden_facts"]:
        if forbidden.lower() in profile_text:
            result["hallucinations_found"].append(forbidden)
        else:
            result["hallucinations_clear"].append(forbidden)

    # Score
    total_expected = len(test_case["expected_facts"])
    total_forbidden = len(test_case["forbidden_facts"])

    recall = len(result["expected_found"]) / total_expected if total_expected else 1
    precision = len(result["hallucinations_clear"]) / total_forbidden if total_forbidden else 1

    result["recall"] = recall
    result["precision"] = precision
    result["score"] = (recall + precision * 2) / 3  # Weight precision higher (hallucinations worse than missing)

    return result


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("HALLUCINATION DETECTION TEST")
    print("Testing profile generation for hallucinated content")
    print("=" * 80)

    all_results = {}

    for model_id, model_name in MODELS.items():
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name} ({model_id})")
        print(f"{'='*70}")

        llm = LLMClient(
            base_url="https://openrouter.ai/api/v1",
            model=model_id,
        )

        model_results = {}

        for prompt_name, prompt_template in [("standard", PROFILE_PROMPT), ("strict", STRICT_PROFILE_PROMPT)]:
            print(f"\n  --- Prompt: {prompt_name} ---")

            for tc in TEST_CASES:
                print(f"\n  Test: {tc['name']}")
                try:
                    messages = [
                        {"role": "system", "content": "You are a profile analysis expert. Output valid JSON only. Be precise and factual."},
                        {"role": "user", "content": prompt_template.format(messages="\n".join(tc["messages"]))}
                    ]

                    t0 = time.time()
                    response = llm.chat_completion(messages, temperature=0.05)
                    elapsed = time.time() - t0

                    data = llm.extract_json(response)
                    if isinstance(data, list):
                        data = data[0] if data else {}

                    check = check_hallucinations(data, tc)

                    key = f"{prompt_name}_{tc['name']}"
                    model_results[key] = {
                        "check": check,
                        "time": elapsed,
                        "data": data,
                    }

                    # Print results
                    print(f"    Score: {check['score']:.2f} (recall={check['recall']:.2f}, precision={check['precision']:.2f})")
                    print(f"    Time: {elapsed:.1f}s")

                    if check["expected_missed"]:
                        print(f"    Missed: {check['expected_missed']}")
                    if check["hallucinations_found"]:
                        print(f"    HALLUCINATIONS: {check['hallucinations_found']}")
                    else:
                        print(f"    No hallucinations detected")

                    # Show summary
                    summary = data.get("summary", "N/A")
                    print(f"    Summary: {summary[:150]}")

                except Exception as e:
                    print(f"    FAILED: {e}")
                    key = f"{prompt_name}_{tc['name']}"
                    model_results[key] = {"error": str(e)}

        all_results[model_id] = model_results

    # ============================================================
    # Summary Table
    # ============================================================
    print("\n\n" + "=" * 100)
    print("HALLUCINATION SUMMARY")
    print("=" * 100)

    # Per-model summary
    for model_id, model_name in MODELS.items():
        results = all_results.get(model_id, {})
        print(f"\n{model_name}:")
        print(f"  {'Test Case':<45} {'Standard':>10} {'Strict':>10} {'Halluc.':>10}")
        print(f"  {'-'*75}")

        for tc in TEST_CASES:
            std_key = f"standard_{tc['name']}"
            strict_key = f"strict_{tc['name']}"

            std = results.get(std_key, {}).get("check", {})
            strict = results.get(strict_key, {}).get("check", {})

            std_score = std.get("score", 0) if std else 0
            strict_score = strict.get("score", 0) if strict else 0
            std_halluc = len(std.get("hallucinations_found", []))
            strict_halluc = len(strict.get("hallucinations_found", []))

            print(f"  {tc['name']:<45} {std_score:>10.2f} {strict_score:>10.2f} {std_halluc}/{strict_halluc}")

    # Best model
    print("\n" + "=" * 100)
    print("BEST MODEL PER METRIC:")
    for prompt_name in ["standard", "strict"]:
        print(f"\n  {prompt_name.upper()} prompt:")
        for model_id, model_name in MODELS.items():
            results = all_results.get(model_id, {})
            avg_score = 0
            total_halluc = 0
            count = 0
            for tc in TEST_CASES:
                key = f"{prompt_name}_{tc['name']}"
                check = results.get(key, {}).get("check", {})
                if check:
                    avg_score += check.get("score", 0)
                    total_halluc += len(check.get("hallucinations_found", []))
                    count += 1
            if count:
                avg_score /= count
            print(f"    {model_name:<30} avg={avg_score:.2f}  hallucinations={total_halluc}")

    print()


if __name__ == "__main__":
    main()
