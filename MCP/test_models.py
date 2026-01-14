#!/usr/bin/env python3
"""
Model Comparison Test Script for SimpleMem Fact Extraction

Tests different LLM models to compare quality and cost of memory extraction.
Usage:
    python test_models.py                           # Interactive mode
    python test_models.py --model mistralai/mistral-nemo
    python test_models.py --model google/gemma-3-12b-it --text "Mi mensaje"
"""

import asyncio
import json
import os
import sys
import time
from typing import Optional

import httpx

# Default test messages
TEST_MESSAGES = [
    "Me llamo Carlos y estoy construyendo un DEX llamado QuickSwap en Base. Mi socio es Pedro y queremos lanzar en marzo.",
    "Yesterday I met with Alice at Starbucks. She told me about her new startup called TechFlow that uses AI for logistics.",
    "El proyecto tiene 50k usuarios activos y factura $200k mensuales. Necesitan funding de serie A para escalar.",
]

# Models to compare
MODELS = [
    ("x-ai/grok-4.1-fast", "Grok 4.1 Fast (baseline)"),
    ("mistralai/mistral-nemo", "Mistral Nemo (cheapest)"),
    ("meta-llama/llama-3.1-8b-instruct", "Llama 3.1 8B"),
    ("google/gemma-3-12b-it", "Gemma 3 12B"),
    ("qwen/qwen3-8b", "Qwen3 8B"),
    ("deepseek/deepseek-r1-distill-llama-70b", "DeepSeek R1 Distill"),
]

EXTRACTION_PROMPT = """## Dialogues to Process:
{dialogue}

---

## Extraction Requirements:

1. **Complete Coverage**: Capture ALL valuable information from the dialogues.

2. **Self-Contained Facts**: Each entry must be independently understandable.
   - BAD: "He will meet Bob tomorrow" (Who is "he"? When is "tomorrow"?)
   - GOOD: "Alice will meet Bob at Starbucks on 2025-01-15 at 14:00"

3. **Coreference Resolution**: Replace ALL pronouns with actual names.

4. **Temporal Anchoring**: Convert ALL relative times to absolute ISO 8601 format.

5. **Information Extraction**: For each entry, extract:
   - `lossless_restatement`: Complete, unambiguous fact
   - `keywords`: Core terms for search (3-7 keywords)
   - `timestamp`: ISO 8601 format if mentioned
   - `persons`: All person names involved
   - `entities`: Companies, products, organizations
   - `topic`: Brief topic phrase (2-5 words)

## Output Format (JSON only):
{{
  "entries": [
    {{
      "lossless_restatement": "...",
      "keywords": ["..."],
      "timestamp": "2025-01-15T14:00:00" or null,
      "persons": ["Alice", "Bob"],
      "entities": ["Company XYZ"],
      "topic": "Meeting arrangement"
    }}
  ]
}}

Return ONLY valid JSON."""


class ModelTester:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://a0x.co",
                "X-Title": "SimpleMem Model Test",
            },
            timeout=120.0,
        )

    async def close(self):
        await self.client.aclose()

    async def extract_facts(self, model: str, text: str) -> dict:
        """Extract facts using specified model"""
        messages = [
            {
                "role": "system",
                "content": "You are a professional information extraction assistant. Extract atomic, self-contained facts from dialogues. Always resolve pronouns and convert relative times to absolute timestamps. Output ONLY valid JSON.",
            },
            {"role": "user", "content": EXTRACTION_PROMPT.format(dialogue=text)},
        ]

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "usage": {"include": True},
        }

        start_time = time.time()
        response = await self.client.post("/chat/completions", json=payload)
        elapsed = time.time() - start_time

        if response.status_code != 200:
            return {
                "error": f"HTTP {response.status_code}: {response.text[:200]}",
                "elapsed": elapsed,
            }

        data = response.json()
        usage = data.get("usage", {})
        content = data["choices"][0]["message"]["content"]

        # Parse JSON from response
        try:
            # Try direct parse
            facts = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON block
            import re
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                try:
                    facts = json.loads(match.group())
                except:
                    facts = {"raw": content[:500]}
            else:
                facts = {"raw": content[:500]}

        return {
            "model": model,
            "facts": facts,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
                "cost": usage.get("cost", 0),
            },
            "elapsed": round(elapsed, 2),
        }


def print_result(result: dict, verbose: bool = True):
    """Pretty print extraction result"""
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return

    usage = result["usage"]
    facts = result.get("facts", {})
    entries = facts.get("entries", []) if isinstance(facts, dict) else []

    print(f"  Time: {result['elapsed']}s")
    print(f"  Tokens: {usage['prompt_tokens']} + {usage['completion_tokens']} = {usage['total_tokens']}")
    print(f"  Cost: ${usage['cost']:.6f}" if usage['cost'] else "  Cost: $0 (not reported)")
    print(f"  Facts extracted: {len(entries)}")

    if verbose and entries:
        print("  ---")
        for i, entry in enumerate(entries, 1):
            stmt = entry.get("lossless_restatement", entry.get("raw", "?"))[:80]
            topic = entry.get("topic", "?")
            print(f"  {i}. [{topic}] {stmt}")


async def compare_models(api_key: str, text: str, models: list):
    """Compare multiple models on the same text"""
    tester = ModelTester(api_key)

    print("=" * 70)
    print("INPUT TEXT:")
    print(f"  {text[:100]}..." if len(text) > 100 else f"  {text}")
    print("=" * 70)

    results = []
    for model_id, model_name in models:
        print(f"\n[{model_name}] {model_id}")
        result = await tester.extract_facts(model_id, text)
        results.append(result)
        print_result(result)

    await tester.close()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<35} {'Facts':>6} {'Tokens':>8} {'Cost':>12} {'Time':>8}")
    print("-" * 70)
    for (model_id, model_name), result in zip(models, results):
        if "error" not in result:
            facts = result.get("facts", {})
            n_facts = len(facts.get("entries", [])) if isinstance(facts, dict) else 0
            tokens = result["usage"]["total_tokens"]
            cost = result["usage"]["cost"]
            elapsed = result["elapsed"]
            print(f"{model_id:<35} {n_facts:>6} {tokens:>8} ${cost:>10.6f} {elapsed:>7.2f}s")


async def interactive_mode(api_key: str):
    """Interactive testing mode"""
    tester = ModelTester(api_key)

    print("\n" + "=" * 70)
    print("SIMPLEMEM MODEL TESTER - Interactive Mode")
    print("=" * 70)
    print("\nAvailable models:")
    for i, (model_id, name) in enumerate(MODELS, 1):
        print(f"  {i}. {name} ({model_id})")
    print("\nCommands:")
    print("  <number>  - Select model")
    print("  text      - Enter custom text")
    print("  compare   - Compare all models")
    print("  quit      - Exit")

    current_model = MODELS[0][0]
    current_text = TEST_MESSAGES[0]

    while True:
        print(f"\nCurrent: {current_model}")
        print(f"Text: {current_text[:60]}...")
        cmd = input("\n> ").strip()

        if cmd.lower() in ("quit", "exit", "q"):
            break
        elif cmd.lower() == "compare":
            await compare_models(api_key, current_text, MODELS)
        elif cmd.lower() == "text":
            new_text = input("Enter text: ").strip()
            if new_text:
                current_text = new_text
        elif cmd.isdigit() and 1 <= int(cmd) <= len(MODELS):
            current_model = MODELS[int(cmd) - 1][0]
            print(f"\nTesting {current_model}...")
            result = await tester.extract_facts(current_model, current_text)
            print_result(result, verbose=True)
        elif cmd:
            # Treat as text input
            current_text = cmd
            print(f"\nTesting {current_model}...")
            result = await tester.extract_facts(current_model, current_text)
            print_result(result, verbose=True)

    await tester.close()


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test SimpleMem fact extraction with different models")
    parser.add_argument("--model", "-m", help="Model to test (e.g., mistralai/mistral-nemo)")
    parser.add_argument("--text", "-t", help="Text to extract facts from")
    parser.add_argument("--compare", "-c", action="store_true", help="Compare all models")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        env_file = "/home/oydual3/a0x/services-backend/.env"
        if os.path.exists(env_file):
            with open(env_file) as f:
                for line in f:
                    if line.startswith("OPENROUTER_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"')
                        break

    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found")
        sys.exit(1)

    if args.interactive or (not args.model and not args.compare):
        await interactive_mode(api_key)
    elif args.compare:
        text = args.text or TEST_MESSAGES[0]
        await compare_models(api_key, text, MODELS)
    elif args.model:
        text = args.text or TEST_MESSAGES[0]
        tester = ModelTester(api_key)
        print(f"\nTesting {args.model}...")
        print(f"Text: {text}\n")
        result = await tester.extract_facts(args.model, text)
        print_result(result, verbose=True)
        await tester.close()


if __name__ == "__main__":
    asyncio.run(main())
