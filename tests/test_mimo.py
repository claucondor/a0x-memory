"""
Test Xiaomi MiMo-V2-Flash for profile generation

API: https://api.xiaomimimo.com/v1 (OpenAI compatible)
Model: mimo-v2-flash
Pricing (Overseas): $0.10/1M input, $0.30/1M output
"""
import os
import sys
import json
import time

os.environ["USE_LOCAL_STORAGE"] = "true"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

# ============================================================
# MiMo Config
# ============================================================
MIMO_API_KEY = "sk-sh6sg0w98dg37htwuzfy3xbgcyowl5jaukrxx2rsukp81861"
MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2-flash"

# Pricing (overseas)
INPUT_COST = 0.10   # $/1M tokens
OUTPUT_COST = 0.30  # $/1M tokens
CACHED_INPUT_COST = 0.01  # $/1M tokens

client = OpenAI(
    api_key=MIMO_API_KEY,
    base_url=MIMO_BASE_URL,
)

# ============================================================
# Same test data as test_profile_models.py
# ============================================================
DM_MESSAGES = [
    "hey jesse! I'm building a DEX aggregator on Base, trying to optimize routing",
    "we're using 0x API but gas costs are still too high for small trades",
    "I've been a Solidity dev for 3 years, mostly working on DeFi protocols",
    "our team just got a grant from Optimism for cross-chain bridging",
    "wallet is 0xABC123...def456, we deployed on Base Sepolia already",
    "what do you think about using Uniswap v4 hooks for custom fee tiers?",
    "I was at ETHDenver last month, met some of the Base team there",
    "we're considering launching a token but want to focus on product first",
    "our GitHub is github.com/defi-routing-protocol, check the contracts folder",
    "been experimenting with account abstraction for gasless swaps",
    "my co-founder @sarah_dev handles the frontend, she's really good with React",
    "we process about 500 trades daily right now, aiming for 5000 by Q2",
    "I think the biggest challenge is MEV protection for our users",
    "looking into Flashbots Protect and private mempools on Base",
    "also interested in your grants program, we need funding for an audit",
]

GROUP_MESSAGES = [
    "alice_dev: has anyone tried the new Uniswap v4 hooks on Base?",
    "bob_trader: yeah, I deployed a custom fee hook last week. Gas was only 0.001 ETH",
    "carol_researcher: interesting, I published a paper on AMM efficiency last month",
    "alice_dev: @carol_researcher nice! can you share the link?",
    "bob_trader: we're seeing about 40% less slippage with our custom routing",
    "david_founder: gm everyone, just launched our new lending protocol on Base",
    "alice_dev: @david_founder congrats! what's the TVL looking like?",
    "david_founder: about $2M first week, growing 15% daily",
    "carol_researcher: that's a solid launch. what oracle are you using?",
    "david_founder: Chainlink for now, but exploring Pyth for faster updates",
    "eve_designer: I redesigned our DeFi dashboard UI, feedback welcome: figma.com/...",
    "bob_trader: looks clean! love the portfolio visualization",
    "alice_dev: anyone know if Base has plans for blob transactions?",
    "frank_mod: reminder: weekly DeFi call tomorrow at 3pm UTC",
    "carol_researcher: I'll present my findings on impermanent loss mitigation",
    "alice_dev: looking forward to it! I have some data on Base vs Arbitrum gas costs",
    "david_founder: we're also working on a points program for early users",
    "bob_trader: careful with tokenomics, I've seen too many projects fail there",
    "eve_designer: true, design should focus on utility first",
    "frank_mod: good discussion everyone. let's keep the energy going",
]

# ============================================================
# Prompts (same as test_profile_models.py)
# ============================================================
USER_PROFILE_PROMPT = """
Analyze these messages from a single user and extract a comprehensive profile.

[Messages]
{messages}

[Your Task]
Extract:
1. summary: 1-2 sentence description of who this person is and what they do
2. expertise_level: beginner, intermediate, advanced, or expert
3. interests: List of 5-8 specific interests (not generic like "crypto" - be specific like "DEX aggregation")
4. communication_style: formal, casual, technical, conversational
5. key_facts: 3-5 concrete facts about this person (projects, wallets, skills)
6. personality_traits: 2-3 personality traits observed

Return ONLY valid JSON.
"""

GROUP_PROFILE_PROMPT = """
Analyze this group conversation and extract group profile information.

[Conversation]
{messages}

[Your Task]
Extract:
1. summary: What this group is about (1-2 sentences)
2. group_purpose: Primary purpose of the group
3. main_topics: 3-5 main topics discussed
4. tone: formal, casual, technical, friendly, professional, educational, meme_heavy, or other
5. expertise_level: beginner, intermediate, advanced, expert, or mixed
6. activity_level: low, moderate, high, or very_high
7. communication_norms: 2-4 norms/rules observed
8. preferred_content_types: 3-5 types of content preferred

Return ONLY valid JSON.
"""

INTERESTS_PROMPT = """
Extract specific interests and expertise areas from these user messages.

[Messages]
{messages}

[Rules]
- Be SPECIFIC: "DEX aggregation on Base" not just "DeFi"
- Be CONCRETE: "Uniswap v4 hooks" not "smart contracts"
- Include technical AND non-technical interests
- Score each 0.0-1.0 based on how much they discuss it
- Return 5-10 interests maximum

Return ONLY valid JSON with format:
{{"interests": [{{"keyword": "specific interest", "score": 0.9, "evidence": "brief quote or reason"}}]}}
"""

# ============================================================
# Evaluation functions (same as test_profile_models.py)
# ============================================================
def evaluate_user_profile(data: dict) -> dict:
    scores = {}
    summary = data.get("summary", "")
    summary_score = 0.0
    if "DEX" in summary or "aggregator" in summary or "routing" in summary:
        summary_score += 0.3
    if "Base" in summary or "Solidity" in summary:
        summary_score += 0.2
    if len(summary) > 30 and len(summary) < 300:
        summary_score += 0.2
    if "3 year" in summary or "grant" in summary:
        summary_score += 0.3
    scores["summary_quality"] = min(1.0, summary_score)

    interests = data.get("interests", [])
    if isinstance(interests, list) and interests:
        generic_terms = {"crypto", "blockchain", "defi", "web3", "trading", "development"}
        specific_count = 0
        for interest in interests:
            kw = interest if isinstance(interest, str) else interest.get("keyword", "")
            if kw.lower() not in generic_terms and len(kw) > 3:
                specific_count += 1
        scores["interests_specificity"] = specific_count / len(interests)
        scores["interests_count"] = min(1.0, len(interests) / 6)
    else:
        scores["interests_specificity"] = 0.0
        scores["interests_count"] = 0.0

    key_facts = data.get("key_facts", [])
    fact_score = 0.0
    fact_checks = ["0xABC123", "Solidity", "3 year", "Optimism", "grant", "ETHDenver",
                   "500 trade", "github.com", "sarah", "account abstraction", "MEV"]
    if isinstance(key_facts, list):
        all_facts_text = " ".join(str(f) for f in key_facts).lower()
        matches = sum(1 for check in fact_checks if check.lower() in all_facts_text)
        fact_score = min(1.0, matches / 4)
    scores["facts_accuracy"] = fact_score

    scores["no_hallucination"] = 1.0
    hallucination_signals = ["NFT", "staking", "validator", "governance token"]
    all_text = json.dumps(data).lower()
    for signal in hallucination_signals:
        if signal.lower() in all_text and signal.lower() not in " ".join(DM_MESSAGES).lower():
            scores["no_hallucination"] -= 0.25

    scores["overall"] = sum(scores.values()) / len(scores)
    return scores


def evaluate_group_profile(data: dict) -> dict:
    scores = {}
    summary = data.get("summary", "")
    if "DeFi" in summary or "defi" in summary.lower():
        scores["summary_relevance"] = 0.8
        if "Base" in summary:
            scores["summary_relevance"] = 1.0
    else:
        scores["summary_relevance"] = 0.3

    topics = data.get("main_topics", [])
    if topics:
        specific = sum(1 for t in topics if len(t) > 5)
        scores["topics_quality"] = min(1.0, specific / 3)
    else:
        scores["topics_quality"] = 0.0

    tone = data.get("tone", "")
    if isinstance(tone, str):
        tone = tone.lower()
    else:
        tone = ""
    scores["tone_accuracy"] = 1.0 if tone in ["technical", "casual", "friendly"] else 0.3

    required_fields = ["summary", "main_topics", "tone", "expertise_level"]
    present = sum(1 for f in required_fields if f in data)
    scores["completeness"] = present / len(required_fields)

    scores["overall"] = sum(scores.values()) / len(scores)
    return scores


def evaluate_interests(data: dict) -> dict:
    scores = {}
    interests = data.get("interests", [])
    if not interests:
        return {"overall": 0.0}

    scores["count_score"] = 1.0 if 5 <= len(interests) <= 10 else 0.5
    generic = {"crypto", "blockchain", "defi", "web3", "trading", "technology", "finance"}
    specific_count = 0
    for item in interests:
        kw = item.get("keyword", item) if isinstance(item, dict) else str(item)
        if kw.lower().strip() not in generic and len(kw) > 4:
            specific_count += 1
    scores["specificity"] = specific_count / len(interests) if interests else 0

    has_evidence = sum(1 for i in interests if isinstance(i, dict) and i.get("evidence"))
    scores["evidence_quality"] = has_evidence / len(interests) if interests else 0

    expected = ["DEX", "aggregat", "routing", "Base", "Uniswap", "MEV", "account abstraction",
                "gas", "Solidity", "audit"]
    all_kw = " ".join(
        i.get("keyword", "") if isinstance(i, dict) else str(i) for i in interests
    ).lower()
    covered = sum(1 for e in expected if e.lower() in all_kw)
    scores["coverage"] = min(1.0, covered / 4)

    scores["overall"] = sum(scores.values()) / len(scores)
    return scores


def extract_json(text: str):
    """Extract JSON from response text."""
    text = text.strip()
    # Try direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try ```json block
    if "```json" in text.lower():
        start = text.lower().find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            try:
                return json.loads(text[start:end].strip())
            except json.JSONDecodeError:
                pass
    # Try finding { or [
    for char in ['{', '[']:
        idx = text.find(char)
        if idx != -1:
            # Find matching close
            depth = 0
            close = '}' if char == '{' else ']'
            for i in range(idx, len(text)):
                if text[i] == char:
                    depth += 1
                elif text[i] == close:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[idx:i+1])
                        except json.JSONDecodeError:
                            break
    raise ValueError(f"Failed to parse JSON: {text[:200]}")


def call_mimo(system_prompt: str, user_prompt: str) -> tuple:
    """Call MiMo API, return (response_text, usage_dict, elapsed_time)."""
    t0 = time.time()
    response = client.chat.completions.create(
        model=MIMO_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        top_p=0.95,
        stream=False,
        extra_body={"thinking": {"type": "disabled"}}
    )
    elapsed = time.time() - t0

    content = response.choices[0].message.content
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    return content, usage, elapsed


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 80)
    print("XIAOMI MiMo-V2-Flash TEST")
    print(f"Model: {MIMO_MODEL}")
    print(f"API: {MIMO_BASE_URL}")
    print(f"Pricing: ${INPUT_COST}/1M input, ${OUTPUT_COST}/1M output")
    print(f"         ${CACHED_INPUT_COST}/1M cached input")
    print("=" * 80)

    total_input_tokens = 0
    total_output_tokens = 0
    total_time = 0

    # --- Test 1: User Profile ---
    print(f"\n[1/3] User Profile Summary...")
    try:
        content, usage, elapsed = call_mimo(
            "You are a profile analysis expert. Output valid JSON only.",
            USER_PROFILE_PROMPT.format(messages="\n".join(DM_MESSAGES))
        )
        total_input_tokens += usage["input_tokens"]
        total_output_tokens += usage["output_tokens"]
        total_time += elapsed

        data = extract_json(content)
        if isinstance(data, list):
            data = data[0] if data else {}

        scores = evaluate_user_profile(data)
        print(f"  Score: {scores['overall']:.2f} | Time: {elapsed:.1f}s")
        print(f"  Tokens: {usage['input_tokens']} in / {usage['output_tokens']} out / {usage['total_tokens']} total")
        print(f"  Summary: {data.get('summary', 'N/A')[:150]}")
        interests = data.get("interests", [])
        interest_strs = [i.get("keyword", i) if isinstance(i, dict) else str(i) for i in interests[:6]]
        print(f"  Interests: {interest_strs}")
        key_facts = data.get("key_facts", [])
        print(f"  Key facts: {key_facts[:5]}")
        up_score = scores['overall']
    except Exception as e:
        print(f"  FAILED: {e}")
        up_score = 0

    # --- Test 2: Group Profile ---
    print(f"\n[2/3] Group Profile...")
    try:
        content, usage, elapsed = call_mimo(
            "You are a group analysis expert. Output valid JSON only.",
            GROUP_PROFILE_PROMPT.format(messages="\n".join(GROUP_MESSAGES))
        )
        total_input_tokens += usage["input_tokens"]
        total_output_tokens += usage["output_tokens"]
        total_time += elapsed

        data = extract_json(content)
        if isinstance(data, list):
            data = data[0] if data else {}

        scores = evaluate_group_profile(data)
        print(f"  Score: {scores['overall']:.2f} | Time: {elapsed:.1f}s")
        print(f"  Tokens: {usage['input_tokens']} in / {usage['output_tokens']} out / {usage['total_tokens']} total")
        print(f"  Summary: {data.get('summary', 'N/A')[:150]}")
        print(f"  Topics: {data.get('main_topics', [])}")
        print(f"  Tone: {data.get('tone', 'N/A')}")
        gp_score = scores['overall']
    except Exception as e:
        print(f"  FAILED: {e}")
        gp_score = 0

    # --- Test 3: Interests Extraction ---
    print(f"\n[3/3] Interests Extraction (LLM-based)...")
    try:
        content, usage, elapsed = call_mimo(
            "You are an interests extraction expert. Output valid JSON only.",
            INTERESTS_PROMPT.format(messages="\n".join(DM_MESSAGES))
        )
        total_input_tokens += usage["input_tokens"]
        total_output_tokens += usage["output_tokens"]
        total_time += elapsed

        data = extract_json(content)
        if isinstance(data, list):
            data = {"interests": data}

        scores = evaluate_interests(data)
        print(f"  Score: {scores['overall']:.2f} | Time: {elapsed:.1f}s")
        print(f"  Tokens: {usage['input_tokens']} in / {usage['output_tokens']} out / {usage['total_tokens']} total")
        for item in data.get("interests", [])[:8]:
            if isinstance(item, dict):
                ev = item.get('evidence', '')
                print(f"    - {item.get('keyword', '?')} ({item.get('score', '?')}) {f'| {ev[:60]}' if ev else ''}")
            else:
                print(f"    - {item}")
        ie_score = scores['overall']
    except Exception as e:
        print(f"  FAILED: {e}")
        ie_score = 0

    # ============================================================
    # Summary
    # ============================================================
    avg_score = (up_score + gp_score + ie_score) / 3

    # Real cost from actual token usage
    real_cost = (total_input_tokens / 1_000_000) * INPUT_COST + (total_output_tokens / 1_000_000) * OUTPUT_COST
    cached_cost = (total_input_tokens / 1_000_000) * CACHED_INPUT_COST + (total_output_tokens / 1_000_000) * OUTPUT_COST

    # Estimate per 1K profiles (1 call each, same token ratio)
    avg_input_per_call = total_input_tokens / 3
    avg_output_per_call = total_output_tokens / 3
    cost_per_call = (avg_input_per_call / 1_000_000) * INPUT_COST + (avg_output_per_call / 1_000_000) * OUTPUT_COST
    cost_per_call_cached = (avg_input_per_call / 1_000_000) * CACHED_INPUT_COST + (avg_output_per_call / 1_000_000) * OUTPUT_COST

    print(f"\n{'='*80}")
    print(f"MiMo-V2-Flash RESULTS")
    print(f"{'='*80}")
    print(f"  Profile:   {up_score:.2f}")
    print(f"  Group:     {gp_score:.2f}")
    print(f"  Interests: {ie_score:.2f}")
    print(f"  Average:   {avg_score:.2f}")
    print(f"")
    print(f"  Total time:   {total_time:.1f}s (3 calls)")
    print(f"  Avg per call: {total_time/3:.1f}s")
    print(f"")
    print(f"  Token usage (3 calls):")
    print(f"    Input:  {total_input_tokens:,} tokens")
    print(f"    Output: {total_output_tokens:,} tokens")
    print(f"    Total:  {total_input_tokens + total_output_tokens:,} tokens")
    print(f"")
    print(f"  Real cost (3 calls):     ${real_cost:.6f}")
    print(f"  With cache (3 calls):    ${cached_cost:.6f}")
    print(f"")
    print(f"  Estimated cost per 1K profiles:")
    print(f"    Without cache: ${cost_per_call * 1000:.4f}")
    print(f"    With cache:    ${cost_per_call_cached * 1000:.4f}")
    print(f"")
    print(f"  COMPARISON vs best OpenRouter models:")
    print(f"  {'Model':<30} {'Avg':>6} {'Time':>7} {'$/1K profiles':>14}")
    print(f"  {'-'*60}")
    print(f"  {'MiMo-V2-Flash':<30} {avg_score:>6.2f} {total_time:>6.1f}s ${cost_per_call * 1000:>12.4f}")
    print(f"  {'MiMo-V2 (cached)':<30} {avg_score:>6.2f} {total_time:>6.1f}s ${cost_per_call_cached * 1000:>12.4f}")
    print(f"  {'Llama 3.1 8B':<30} {'0.92':>6} {'3.9':>6}s ${'0.0310':>12}")
    print(f"  {'Qwen3 32B':<30} {'0.94':>6} {'2.6':>6}s ${'0.1360':>12}")
    print(f"  {'Qwen3 8B':<30} {'1.00':>6} {'45.9':>6}s ${'0.1150':>12}")
    print(f"  {'Llama 3.3 70B':<30} {'0.96':>6} {'2.4':>6}s ${'0.1760':>12}")
    print()


if __name__ == "__main__":
    main()
