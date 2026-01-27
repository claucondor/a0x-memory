"""
Test Profile Generation with Different OpenRouter Models

Compares quality and cost of profile summaries, group profiles, and interests
extraction across multiple LLM models via OpenRouter.

Usage:
    python tests/test_profile_models.py

Env:
    OPENAI_API_KEY - OpenRouter API key (sk-or-v1-...)
"""
import os
import sys
import json
import time

# Force local storage
os.environ["USE_LOCAL_STORAGE"] = "true"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_client import LLMClient
from utils.structured_schemas import (
    PROFILE_EVALUATION_SCHEMA, GROUP_PROFILE_EVALUATION_SCHEMA, INTERESTS_EVALUATION_SCHEMA
)

# LLM-as-Judge evaluator (non-streaming, uses structured outputs)
_judge_client = None

def _get_judge():
    """Get or create LLM-as-Judge client (Llama 3.1 8B, non-streaming)."""
    global _judge_client
    if _judge_client is None:
        _judge_client = LLMClient(
            base_url="https://openrouter.ai/api/v1",
            model="meta-llama/llama-3.1-8b-instruct",
            use_streaming=False
        )
    return _judge_client


def _llm_evaluate(prompt: str, schema: dict) -> dict:
    """Run LLM-as-judge evaluation with structured output."""
    judge = _get_judge()
    messages = [
        {"role": "system", "content": """You are a rigorous QA evaluator for AI-extracted data. Your job is to compare extracted output against source messages and score accuracy.

SCORING RULES:
- 1.0 = Perfect. Every detail correct, nothing missing, nothing invented.
- 0.8-0.9 = Good. Minor omissions but no errors.
- 0.6-0.7 = Acceptable. Some missing info OR minor inaccuracies.
- 0.4-0.5 = Poor. Significant gaps or errors.
- 0.0-0.3 = Failed. Major hallucinations or mostly wrong.

REASONING RULES (CRITICAL):
Your reasoning field MUST be a detailed analysis, NOT a generic statement. You MUST:
1. List specific things the extraction got RIGHT (quote from source messages)
2. List specific things that are MISSING (name exactly what facts/topics are absent)
3. List specific things that are WRONG or HALLUCINATED (things not in source)
4. Justify each score with evidence

BAD reasoning: "The profile is mostly accurate."
GOOD reasoning: "Summary correctly mentions '3 years Solidity' and 'DEX aggregator on Base' (from messages 1,3). Missing: ETHDenver attendance (message 7), Optimism grant (message 4). No hallucinations detected. interests_specificity: 6/8 are specific, but 'Solidity development' is too generic."

The overall score should be the average of the other scores."""},
        {"role": "user", "content": prompt}
    ]
    response = judge.chat_completion(messages, temperature=0.1, response_format=schema)
    try:
        return json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return judge.extract_json(response)

# ============================================================
# OpenRouter Models to Test
# Pricing fetched from https://openrouter.ai/api/v1/models
# All costs in $/1M tokens
# ============================================================
MODELS = {
    # --- Ultra cheap (< $0.05 input) ---
    "meta-llama/llama-3.1-8b-instruct": {
        "name": "Llama 3.1 8B",
        "input_cost": 0.02,
        "output_cost": 0.05,
    },
    "mistralai/mistral-nemo": {
        "name": "Mistral Nemo 12B",
        "input_cost": 0.02,
        "output_cost": 0.04,
    },
    "mistralai/mistral-small-3.1-24b-instruct": {
        "name": "Mistral Small 3.1 24B",
        "input_cost": 0.03,
        "output_cost": 0.11,
    },
    # --- Cheap ($0.05-$0.10 input) ---
    "qwen/qwen3-8b": {
        "name": "Qwen3 8B",
        "input_cost": 0.05,
        "output_cost": 0.25,
    },
    "qwen/qwen3-14b": {
        "name": "Qwen3 14B",
        "input_cost": 0.05,
        "output_cost": 0.22,
    },
    "openai/gpt-5-nano": {
        "name": "GPT-5 Nano",
        "input_cost": 0.05,
        "output_cost": 0.40,
    },
    "microsoft/phi-4": {
        "name": "Phi-4 14B",
        "input_cost": 0.06,
        "output_cost": 0.14,
    },
    "qwen/qwen3-30b-a3b": {
        "name": "Qwen3 30B-A3B (MoE)",
        "input_cost": 0.06,
        "output_cost": 0.22,
    },
    "google/gemini-2.0-flash-lite-001": {
        "name": "Gemini 2.0 Flash Lite",
        "input_cost": 0.07,
        "output_cost": 0.30,
    },
    "meta-llama/llama-4-scout": {
        "name": "Llama 4 Scout",
        "input_cost": 0.08,
        "output_cost": 0.30,
    },
    "qwen/qwen3-32b": {
        "name": "Qwen3 32B",
        "input_cost": 0.08,
        "output_cost": 0.24,
    },
    # --- Mid-range ($0.10 input) ---
    "meta-llama/llama-3.3-70b-instruct": {
        "name": "Llama 3.3 70B",
        "input_cost": 0.10,
        "output_cost": 0.32,
    },
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": {
        "name": "Nemotron Super 49B",
        "input_cost": 0.10,
        "output_cost": 0.40,
    },
    "openai/gpt-4.1-nano": {
        "name": "GPT-4.1 Nano",
        "input_cost": 0.10,
        "output_cost": 0.40,
    },
    "google/gemini-2.0-flash-001": {
        "name": "Gemini 2.0 Flash",
        "input_cost": 0.10,
        "output_cost": 0.40,
    },
    "google/gemini-2.5-flash-lite": {
        "name": "Gemini 2.5 Flash Lite",
        "input_cost": 0.10,
        "output_cost": 0.40,
    },
}

# ============================================================
# Test Data - Realistic Telegram Conversations
# ============================================================

# DM conversation for user profile
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

# Group conversation for group profile
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
# Prompts
# ============================================================

USER_PROFILE_PROMPT = """Analyze these messages from a single user and extract a profile.

[Messages]
{messages}

[Instructions]
- summary: 1-2 sentences. Include their ROLE (e.g. "Solidity developer"), EXPERIENCE (e.g. "3 years"), CURRENT PROJECT (e.g. "building a DEX aggregator on Base"), and any NOTABLE ACHIEVEMENTS (grants, events, milestones). Be specific.
- expertise_level: beginner, intermediate, advanced, or expert
- interests: 5-8 specific interests. ONLY include things THIS USER directly works on or expressed interest in. Do NOT attribute other people's skills to this user (e.g. if they mention a co-founder does React, that is NOT this user's interest). Be specific: "DEX aggregation routing on Base" not "DeFi".
- communication_style: formal, casual, technical, conversational
- key_facts: 5-8 concrete, verifiable facts. Include: wallet addresses, project names, GitHub links, team members mentioned, metrics (trade volumes, TVL), events attended, grants received.
- personality_traits: 2-3 traits observed from HOW they communicate.

[Example Output]
{{"summary": "Senior Solidity developer with 3 years of DeFi experience, building a DEX aggregator on Base. Received an Optimism grant for cross-chain bridging and attended ETHDenver.", "expertise_level": "advanced", "interests": [{{"keyword": "DEX aggregation routing on Base", "score": 0.9}}, {{"keyword": "Uniswap v4 hooks for fee tiers", "score": 0.8}}], "communication_style": "technical", "key_facts": ["3 years Solidity experience in DeFi protocols", "Wallet: 0xABC123...def456", "Deployed on Base Sepolia"], "personality_traits": ["product-focused", "collaborative"]}}"""

GROUP_PROFILE_PROMPT = """Analyze this group conversation and extract a profile.

[Conversation]
{messages}

[Instructions]
- summary: 1-2 sentences. Be SPECIFIC about what the group discusses - name the ecosystem/chain, the types of projects, and the community vibe. NOT generic like "DeFi enthusiasts".
- group_purpose: Why this group exists, based on what people actually do in it.
- main_topics: 3-5 topics ACTUALLY discussed in the messages. Use specific terms from the conversation (e.g. "Uniswap v4 hooks on Base", "lending protocol launch and TVL growth"). NOT generic like "DeFi protocols".
- tone: formal, casual, technical, friendly, professional, educational, meme_heavy, or other.
- expertise_level: beginner, intermediate, advanced, expert, or mixed.
- activity_level: low, moderate, high, or very_high.
- member_count_estimate: Count distinct usernames in the conversation.
- communication_norms: 2-4 norms you can ACTUALLY OBSERVE in the messages (e.g. "members congratulate each other on launches", "people share links and ask for feedback"). Do NOT invent norms you cannot see evidence of.
- preferred_content_types: 3-5 types based on what people actually share.

[Example Output]
{{"summary": "A Base L2 builder community where developers and researchers discuss DeFi protocol development, share project launches, and collaborate on technical problems.", "group_purpose": "Technical collaboration among Base DeFi builders", "main_topics": ["Uniswap v4 hooks and custom fee tiers", "Lending protocol launches and TVL tracking", "Oracle selection (Chainlink vs Pyth)"], "tone": "technical", "expertise_level": "advanced", "activity_level": "moderate", "member_count_estimate": 6, "communication_norms": ["Members congratulate each other on launches", "Technical questions get specific data-backed responses"], "preferred_content_types": ["technical discussions", "project updates", "research findings"]}}"""

INTERESTS_PROMPT = """Extract specific interests and expertise areas from these user messages.

[Messages]
{messages}

[Rules]
- ONLY extract interests of THIS USER (the speaker). If they mention someone else's skills (e.g. "my co-founder does React"), that is NOT this user's interest.
- Be SPECIFIC: "DEX aggregation routing on Base" not just "DeFi"
- Be CONCRETE: "Uniswap v4 hooks for custom fee tiers" not "smart contracts"
- Include both technical interests AND professional goals (e.g. "seeking audit funding")
- Score 0.0-1.0 based on how central it is to what they do (core work = 0.9, mentioned once = 0.5)
- Evidence must be a direct quote or paraphrase from THEIR messages
- Return 5-10 interests

[Example Output]
{{"interests": [{{"keyword": "DEX aggregation routing on Base", "score": 0.9, "evidence": "I'm building a DEX aggregator on Base, trying to optimize routing"}}, {{"keyword": "MEV protection for DEX users", "score": 0.8, "evidence": "the biggest challenge is MEV protection for our users"}}]}}"""

# ============================================================
# Evaluation Criteria
# ============================================================

def evaluate_user_profile(data: dict, model_name: str) -> dict:
    """Score profile quality using LLM-as-judge (no keyword matching)."""
    summary = data.get("summary", "")
    interests = data.get("interests", [])
    key_facts = data.get("key_facts", [])

    prompt = f"""You are evaluating an AI-extracted user profile. Read the source messages, then score the extracted profile.

[Source Messages]
{chr(10).join(DM_MESSAGES)}

[Extracted Summary]
{summary}

[Extracted Interests]
{json.dumps(interests[:8], indent=2, default=str)}

[Extracted Key Facts]
{json.dumps(key_facts[:8], indent=2, default=str)}

Score each dimension 0.0-1.0:

summary_quality: The user is a Solidity developer (3 years), building a DEX aggregator on Base, got an Optimism grant, and attended ETHDenver. Does the summary capture their role + project + experience + achievements? Score 0.9+ if all present, 0.7 if most present, 0.5 if generic.

interests_specificity: Good interests are specific like "DEX aggregation routing on Base" or "MEV protection". Bad interests are generic like "DeFi" or "crypto". Also: the user's co-founder does React - if React appears as THIS user's interest, deduct 0.2. What fraction of interests are specific?

facts_accuracy: The messages contain these facts: wallet 0xABC123, 3yr Solidity, Optimism grant, ETHDenver, 500 daily trades, GitHub repo, co-founder sarah, account abstraction, MEV research. How many were extracted? Score = fraction found.

no_hallucination: Is everything in the profile traceable to the source messages? Score 1.0 if yes, deduct 0.2 per invented claim.

overall: Average of the four scores.

In reasoning: name 2-3 things the extraction got right, and 2-3 things that are missing or wrong."""

    try:
        return _llm_evaluate(prompt, PROFILE_EVALUATION_SCHEMA)
    except Exception as e:
        print(f"    [Judge] Evaluation failed: {e}")
        return {"summary_quality": 0.5, "interests_specificity": 0.5, "facts_accuracy": 0.5, "no_hallucination": 0.5, "overall": 0.5, "reasoning": f"fallback: {e}"}


def evaluate_group_profile(data: dict, model_name: str) -> dict:
    """Score group profile quality using LLM-as-judge."""
    # Remove reasoning from data sent to judge (it's internal)
    data_for_judge = {k: v for k, v in data.items() if k != "reasoning"}

    prompt = f"""You are evaluating an AI-extracted group profile. Read the source conversation, then score the extracted profile.

[Source Conversation]
{chr(10).join(GROUP_MESSAGES)}

[Extracted Profile]
{json.dumps(data_for_judge, indent=2)}

Score each dimension 0.0-1.0:

summary_relevance: The group discusses DeFi on Base L2 - Uniswap v4 hooks, lending protocol launch, oracle selection, AMM efficiency, gas costs. Members are builders/developers/researchers who collaborate. Does the summary capture the specific ecosystem (Base), domain (DeFi), and community nature? Score 0.9+ if specific ("Base L2 DeFi builders"), 0.5 if vague ("DeFi enthusiasts"), 0.3 if generic ("crypto community").

topics_quality: The conversation covers: Uniswap v4 hooks, lending protocol + TVL, Chainlink vs Pyth oracles, AMM slippage, DeFi dashboard UI, Base vs Arbitrum gas, impermanent loss, points/tokenomics. Are the extracted topics specific and actually from the conversation? Score based on specificity. Deduct for generic topics like just "DeFi" or "crypto".

tone_accuracy: The conversation mixes technical (hooks, gas, oracles, TVL) with casual/friendly ("gm", "congrats!", "looks clean!"). Score 1.0 for technical/casual/friendly, 0.5 for professional, 0.2 for formal/meme_heavy.

completeness: Are all fields present? Is member_count reasonable? (6 distinct users: alice, bob, carol, david, eve, frank). Are communication_norms based on observable behavior, not invented?

overall: Average of the four scores.

In reasoning: name 2-3 things it got right, and 2-3 things that are wrong or could improve."""

    try:
        return _llm_evaluate(prompt, GROUP_PROFILE_EVALUATION_SCHEMA)
    except Exception as e:
        print(f"    [Judge] Evaluation failed: {e}")
        return {"summary_relevance": 0.5, "topics_quality": 0.5, "tone_accuracy": 0.5, "completeness": 0.5, "overall": 0.5, "reasoning": f"fallback: {e}"}


def evaluate_interests(data: dict, model_name: str) -> dict:
    """Score interests extraction quality using LLM-as-judge."""
    interests = data.get("interests", [])
    if not interests:
        return {"overall": 0.0, "count_score": 0.0, "specificity": 0.0, "evidence_quality": 0.0, "coverage": 0.0, "reasoning": "no interests extracted"}

    prompt = f"""You are evaluating AI-extracted interests. Read the source messages, then score the extracted interests.

[Source Messages]
{chr(10).join(DM_MESSAGES)}

[Extracted Interests ({len(interests)} items)]
{json.dumps(interests[:10], indent=2, default=str)}

Score each dimension 0.0-1.0:

count_score: There are {len(interests)} interests. Score 1.0 if 5-10, score 0.5 otherwise.

specificity: Specific interests name concrete technologies or use cases (e.g. "DEX aggregation routing on Base", "Uniswap v4 hooks for fee tiers"). Generic interests are vague (e.g. "DeFi", "crypto", "Solidity development"). Also check: the user's co-founder does React - if "React" appears as THIS user's interest, that's an attribution error (deduct 0.2). What fraction of interests are specific?

evidence_quality: Does each interest have an evidence field with a quote from the messages? Score = fraction with evidence.

coverage: The messages discuss: DEX aggregation, gas optimization, Uniswap v4, MEV protection, Flashbots, account abstraction, cross-chain bridging, audit funding. How many of these 8 topics are covered by at least one extracted interest?

overall: Average of the four scores.

In reasoning: name the best 2 interests extracted and 2 topics that are missing or poorly captured."""

    try:
        return _llm_evaluate(prompt, INTERESTS_EVALUATION_SCHEMA)
    except Exception as e:
        print(f"    [Judge] Evaluation failed: {e}")
        return {"count_score": 0.5, "specificity": 0.5, "evidence_quality": 0.5, "coverage": 0.5, "overall": 0.5, "reasoning": f"fallback: {e}"}


# ============================================================
# Main Test Runner
# ============================================================

def test_model(model_id: str, model_info: dict):
    """Test a single model across all three tasks."""
    print(f"\n{'='*70}")
    print(f"MODEL: {model_info['name']} ({model_id})")
    print(f"  Cost: ${model_info['input_cost']}/1M in, ${model_info['output_cost']}/1M out")
    print(f"{'='*70}")

    llm = LLMClient(
        base_url="https://openrouter.ai/api/v1",
        model=model_id,
    )

    results = {}

    # --- Test 1: User Profile ---
    print(f"\n  [1/3] User Profile Summary...")
    try:
        t0 = time.time()
        messages = [{"role": "system", "content": "You are a profile analysis expert. Output valid JSON only."},
                    {"role": "user", "content": USER_PROFILE_PROMPT.format(messages="\n".join(DM_MESSAGES))}]
        response = llm.chat_completion(messages, temperature=0.1)
        elapsed = time.time() - t0
        data = llm.extract_json(response)

        # Handle list response (the bug we're fixing)
        if isinstance(data, list):
            data = data[0] if data else {}

        scores = evaluate_user_profile(data, model_info['name'])
        results["user_profile"] = {
            "scores": scores,
            "time": elapsed,
            "data": data,
        }
        print(f"    Score: {scores['overall']:.2f} | Time: {elapsed:.1f}s")
        print(f"    Summary: {data.get('summary', 'N/A')[:120]}")
        interests = data.get("interests", [])
        interest_strs = [i.get("keyword", i) if isinstance(i, dict) else str(i) for i in interests[:5]]
        print(f"    Interests: {interest_strs}")
    except Exception as e:
        print(f"    FAILED: {e}")
        results["user_profile"] = {"scores": {"overall": 0}, "time": 0, "error": str(e)}

    # --- Test 2: Group Profile ---
    print(f"\n  [2/3] Group Profile...")
    try:
        t0 = time.time()
        messages = [{"role": "system", "content": "You are a group analysis expert. Output valid JSON only."},
                    {"role": "user", "content": GROUP_PROFILE_PROMPT.format(messages="\n".join(GROUP_MESSAGES))}]
        response = llm.chat_completion(messages, temperature=0.1)
        elapsed = time.time() - t0
        data = llm.extract_json(response)

        if isinstance(data, list):
            data = data[0] if data else {}

        scores = evaluate_group_profile(data, model_info['name'])
        results["group_profile"] = {
            "scores": scores,
            "time": elapsed,
            "data": data,
        }
        print(f"    Score: {scores['overall']:.2f} | Time: {elapsed:.1f}s")
        print(f"    Summary: {data.get('summary', 'N/A')[:120]}")
        print(f"    Topics: {data.get('main_topics', [])}")
    except Exception as e:
        print(f"    FAILED: {e}")
        results["group_profile"] = {"scores": {"overall": 0}, "time": 0, "error": str(e)}

    # --- Test 3: Interests Extraction ---
    print(f"\n  [3/3] Interests Extraction (LLM-based)...")
    try:
        t0 = time.time()
        messages = [{"role": "system", "content": "You are an interests extraction expert. Output valid JSON only."},
                    {"role": "user", "content": INTERESTS_PROMPT.format(messages="\n".join(DM_MESSAGES))}]
        response = llm.chat_completion(messages, temperature=0.1)
        elapsed = time.time() - t0
        data = llm.extract_json(response)

        if isinstance(data, list):
            data = {"interests": data}

        scores = evaluate_interests(data, model_info['name'])
        results["interests"] = {
            "scores": scores,
            "time": elapsed,
            "data": data,
        }
        print(f"    Score: {scores['overall']:.2f} | Time: {elapsed:.1f}s")
        for item in data.get("interests", [])[:5]:
            if isinstance(item, dict):
                print(f"      - {item.get('keyword', '?')} ({item.get('score', '?')})")
            else:
                print(f"      - {item}")
    except Exception as e:
        print(f"    FAILED: {e}")
        results["interests"] = {"scores": {"overall": 0}, "time": 0, "error": str(e)}

    return results


def main():
    print("=" * 70)
    print("PROFILE MODEL COMPARISON TEST")
    print("Testing profile generation quality across OpenRouter models")
    print("=" * 70)

    all_results = {}

    for model_id, model_info in MODELS.items():
        try:
            all_results[model_id] = test_model(model_id, model_info)
        except Exception as e:
            print(f"\n  MODEL FAILED COMPLETELY: {e}")
            all_results[model_id] = {"error": str(e)}

    # ============================================================
    # Final Comparison Table
    # ============================================================
    print("\n\n" + "=" * 100)
    print("FINAL COMPARISON")
    print("=" * 100)

    header = f"{'Model':<30} {'Profile':>8} {'Group':>8} {'Interests':>10} {'Avg':>6} {'Time':>7} {'$/M tok':>10} {'$/1K prof':>10}"
    print(header)
    print("-" * 100)

    for model_id, model_info in MODELS.items():
        results = all_results.get(model_id, {})
        if "error" in results:
            print(f"{model_info['name']:<30} {'ERROR':>8}")
            continue

        up = results.get("user_profile", {}).get("scores", {}).get("overall", 0)
        gp = results.get("group_profile", {}).get("scores", {}).get("overall", 0)
        ie = results.get("interests", {}).get("scores", {}).get("overall", 0)
        avg = (up + gp + ie) / 3

        total_time = sum(
            results.get(k, {}).get("time", 0)
            for k in ["user_profile", "group_profile", "interests"]
        )

        # Estimate cost per single profile call (~800 input + 300 output tokens)
        # and per 1000 profiles (1 LLM call each for profile generation)
        est_input_tok = 800   # ~800 tokens input (prompt + messages)
        est_output_tok = 300  # ~300 tokens output (JSON response)
        cost_per_call = (
            (est_input_tok / 1_000_000) * model_info["input_cost"] +
            (est_output_tok / 1_000_000) * model_info["output_cost"]
        )
        cost_per_1k = cost_per_call * 1000

        print(f"{model_info['name']:<30} {up:>8.2f} {gp:>8.2f} {ie:>10.2f} {avg:>6.2f} {total_time:>6.1f}s ${cost_per_call*1000000:>6.1f}/M  ${cost_per_1k:>6.4f}/1K")

    print("-" * 100)
    print("Cost/M  = cost per 1 million tokens (input_cost * 800 + output_cost * 300 per call)")
    print("Cost/1K = estimated cost to generate 1000 profiles (1 LLM call each, ~1100 tokens per call)")
    print()


if __name__ == "__main__":
    main()
