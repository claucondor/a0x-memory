"""
Test Planning & Reflection with different OpenRouter models.

Compares:
- Latency
- Output quality
- Cost efficiency

Usage:
    python tests/test_planning_models.py
    python tests/test_planning_models.py --query "your custom query"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import argparse
from typing import Dict, Any, List, Optional
from openai import OpenAI

# Models to test (OpenRouter)
MODELS_TO_TEST = [
    {
        "name": "gpt-4.1-mini",
        "model_id": "openai/gpt-4.1-mini",
        "cost_input": 0.40,   # per 1M tokens
        "cost_output": 1.60,  # per 1M tokens
        "description": "Current default - balanced"
    },
    {
        "name": "llama-3.1-8b",
        "model_id": "meta-llama/llama-3.1-8b-instruct",
        "cost_input": 0.06,
        "cost_output": 0.06,
        "description": "Very cheap, good for simple tasks"
    },
    {
        "name": "mistral-small-3",
        "model_id": "mistralai/mistral-small-3.1-24b-instruct",
        "cost_input": 0.03,
        "cost_output": 0.11,
        "description": "Ultra cheap, 24B params"
    },
    {
        "name": "qwen-2.5-7b",
        "model_id": "qwen/qwen-2.5-7b-instruct",
        "cost_input": 0.05,
        "cost_output": 0.05,
        "description": "Cheap, good multilingual"
    },
    {
        "name": "gemini-2.0-flash-lite",
        "model_id": "google/gemini-2.0-flash-lite-001",
        "cost_input": 0.075,
        "cost_output": 0.30,
        "description": "Google's fast cheap model"
    },
]

# Test queries - simple and complex
TEST_QUERIES = [
    {
        "type": "simple",
        "query": "What did Carlos say about the project yesterday?",
        "description": "Simple factual query"
    },
    {
        "type": "complex",
        "query": "How did the relationship between Maria and the engineering team evolve during the Q4 planning, and what were the key decisions that affected the product roadmap?",
        "description": "Multi-hop temporal query"
    },
    {
        "type": "medium",
        "query": "What are the main concerns raised in the last meeting about the deployment timeline?",
        "description": "Medium complexity"
    }
]


class PlanningTester:
    """Tests planning and reflection functions with different models."""

    def __init__(self, base_url: str = "https://openrouter.ai/api/v1"):
        self.base_url = base_url
        self.api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def _analyze_information_requirements(self, query: str, model: str) -> tuple[Dict[str, Any], float, int, int]:
        """
        Test the planning function: analyze what info is needed for a query.
        Returns: (result, latency_seconds, input_tokens, output_tokens)
        """
        prompt = f"""Analyze the following question and determine what specific information is required to answer it comprehensively.

Question: {query}

Think step by step:
1. What type of question is this? (factual, temporal, relational, explanatory, etc.)
2. What key entities, events, or concepts need to be identified?
3. What relationships or connections need to be established?
4. What minimal set of information pieces would be sufficient to answer this question?

Return your analysis in JSON format:
```json
{{
  "question_type": "type of question",
  "key_entities": ["entity1", "entity2"],
  "required_info": [
    {{
      "info_type": "what kind of information",
      "description": "specific information needed",
      "priority": "high/medium/low"
    }}
  ],
  "relationships": ["relationship1", "relationship2"],
  "minimal_queries_needed": 2
}}
```

Return ONLY the JSON, no other text."""

        messages = [
            {"role": "system", "content": "You are an intelligent information requirement analyst. Output valid JSON only."},
            {"role": "user", "content": prompt}
        ]

        start_time = time.time()

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )

        latency = time.time() - start_time

        content = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        # Parse JSON from response
        try:
            # Extract JSON from markdown code block if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            result = json.loads(content.strip())
        except:
            result = {"error": "Failed to parse JSON", "raw": content[:500]}

        return result, latency, input_tokens, output_tokens

    def _generate_targeted_queries(self, original_query: str, info_plan: Dict[str, Any], model: str) -> tuple[List[str], float, int, int]:
        """
        Test the query generation function.
        Returns: (queries, latency_seconds, input_tokens, output_tokens)
        """
        prompt = f"""Based on the information requirements analysis, generate the minimal set of targeted search queries.

Original Question: {original_query}

Information Requirements Analysis:
- Question Type: {info_plan.get('question_type', 'general')}
- Key Entities: {info_plan.get('key_entities', [])}
- Required Information: {info_plan.get('required_info', [])}
- Minimal Queries Needed: {info_plan.get('minimal_queries_needed', 1)}

Generate the minimal set of search queries (usually 1-3) that would efficiently gather all required information.

Return in JSON format:
```json
{{
  "queries": ["query1", "query2"],
  "reasoning": "brief explanation"
}}
```

Return ONLY the JSON."""

        messages = [
            {"role": "system", "content": "You are a search query optimizer. Output valid JSON only."},
            {"role": "user", "content": prompt}
        ]

        start_time = time.time()

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )

        latency = time.time() - start_time

        content = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            result = json.loads(content.strip())
            queries = result.get("queries", [original_query])
        except:
            queries = [original_query]

        return queries, latency, input_tokens, output_tokens

    def _analyze_completeness(self, query: str, results_summary: str, model: str) -> tuple[Dict[str, Any], float, int, int]:
        """
        Test the reflection function: check if retrieved info is complete.
        Returns: (result, latency_seconds, input_tokens, output_tokens)
        """
        prompt = f"""Analyze if the retrieved information is sufficient to answer the question.

Original Question: {query}

Retrieved Information Summary:
{results_summary}

Evaluate:
1. Does the retrieved info cover all aspects of the question?
2. Are there any gaps or missing information?
3. Is additional retrieval needed?

Return in JSON format:
```json
{{
  "is_complete": true/false,
  "completeness_score": 0.0-1.0,
  "missing_info": ["what's missing"],
  "recommendation": "proceed/retrieve_more"
}}
```

Return ONLY the JSON."""

        messages = [
            {"role": "system", "content": "You are an information completeness evaluator. Output valid JSON only."},
            {"role": "user", "content": prompt}
        ]

        start_time = time.time()

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )

        latency = time.time() - start_time

        content = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            result = json.loads(content.strip())
        except:
            result = {"error": "Failed to parse JSON", "raw": content[:500]}

        return result, latency, input_tokens, output_tokens

    def test_model(self, model_config: Dict, query: str) -> Dict[str, Any]:
        """Run full planning + reflection test for a model."""
        model_id = model_config["model_id"]
        results = {
            "model": model_config["name"],
            "model_id": model_id,
            "query": query,
            "steps": {},
            "totals": {
                "latency": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0
            }
        }

        # Step 1: Analyze requirements (Planning)
        print(f"  [1/3] Analyzing requirements...")
        try:
            plan, lat, inp, out = self._analyze_information_requirements(query, model_id)
            results["steps"]["analyze_requirements"] = {
                "success": "error" not in plan,
                "latency": round(lat, 2),
                "input_tokens": inp,
                "output_tokens": out,
                "output": plan
            }
            results["totals"]["latency"] += lat
            results["totals"]["input_tokens"] += inp
            results["totals"]["output_tokens"] += out
        except Exception as e:
            results["steps"]["analyze_requirements"] = {"success": False, "error": str(e)}
            plan = {}

        # Step 2: Generate queries (Planning)
        print(f"  [2/3] Generating targeted queries...")
        try:
            queries, lat, inp, out = self._generate_targeted_queries(query, plan, model_id)
            results["steps"]["generate_queries"] = {
                "success": len(queries) > 0,
                "latency": round(lat, 2),
                "input_tokens": inp,
                "output_tokens": out,
                "output": queries
            }
            results["totals"]["latency"] += lat
            results["totals"]["input_tokens"] += inp
            results["totals"]["output_tokens"] += out
        except Exception as e:
            results["steps"]["generate_queries"] = {"success": False, "error": str(e)}

        # Step 3: Analyze completeness (Reflection)
        print(f"  [3/3] Checking completeness (reflection)...")
        fake_results = "Found 3 memories about project discussions, 2 about Carlos, 1 about timeline concerns."
        try:
            completeness, lat, inp, out = self._analyze_completeness(query, fake_results, model_id)
            results["steps"]["analyze_completeness"] = {
                "success": "error" not in completeness,
                "latency": round(lat, 2),
                "input_tokens": inp,
                "output_tokens": out,
                "output": completeness
            }
            results["totals"]["latency"] += lat
            results["totals"]["input_tokens"] += inp
            results["totals"]["output_tokens"] += out
        except Exception as e:
            results["steps"]["analyze_completeness"] = {"success": False, "error": str(e)}

        # Calculate cost
        cost_input = (results["totals"]["input_tokens"] / 1_000_000) * model_config["cost_input"]
        cost_output = (results["totals"]["output_tokens"] / 1_000_000) * model_config["cost_output"]
        results["totals"]["cost"] = round(cost_input + cost_output, 6)
        results["totals"]["latency"] = round(results["totals"]["latency"], 2)

        return results


def print_comparison_table(all_results: List[Dict]):
    """Print a comparison table of all model results."""
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)

    # Header
    print(f"{'Model':<20} {'Latency':<10} {'Tokens':<12} {'Cost':<10} {'Success':<8}")
    print("-" * 80)

    for r in all_results:
        model = r["model"]
        latency = f"{r['totals']['latency']:.2f}s"
        tokens = f"{r['totals']['input_tokens']}+{r['totals']['output_tokens']}"
        cost = f"${r['totals']['cost']:.5f}"

        # Count successes
        successes = sum(1 for s in r["steps"].values() if s.get("success", False))
        success_str = f"{successes}/3"

        print(f"{model:<20} {latency:<10} {tokens:<12} {cost:<10} {success_str:<8}")

    print("=" * 80)


def print_detailed_results(results: Dict):
    """Print detailed results for a single model test."""
    print(f"\n{'─' * 60}")
    print(f"Model: {results['model']} ({results['model_id']})")
    print(f"Query: {results['query'][:60]}...")
    print(f"{'─' * 60}")

    for step_name, step_data in results["steps"].items():
        status = "✓" if step_data.get("success") else "✗"
        latency = step_data.get("latency", "N/A")
        print(f"\n  {status} {step_name}: {latency}s")

        if "output" in step_data:
            output = step_data["output"]
            if isinstance(output, dict):
                for k, v in list(output.items())[:3]:
                    print(f"      {k}: {str(v)[:50]}")
            elif isinstance(output, list):
                print(f"      queries: {output[:3]}")
        elif "error" in step_data:
            print(f"      error: {step_data['error'][:100]}")

    print(f"\n  TOTALS: {results['totals']['latency']}s, ${results['totals']['cost']:.5f}")


def main():
    parser = argparse.ArgumentParser(description="Test Planning & Reflection with different models")
    parser.add_argument("--query", help="Custom query to test", default=None)
    parser.add_argument("--model", help="Test specific model only (by name)", default=None)
    parser.add_argument("--all-queries", action="store_true", help="Test all predefined queries")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()

    print("=" * 60)
    print("Planning & Reflection Model Comparison Test")
    print("=" * 60)

    tester = PlanningTester()

    # Select models
    models = MODELS_TO_TEST
    if args.model:
        models = [m for m in MODELS_TO_TEST if args.model.lower() in m["name"].lower()]
        if not models:
            print(f"Model '{args.model}' not found. Available: {[m['name'] for m in MODELS_TO_TEST]}")
            return

    # Select queries
    if args.query:
        queries = [{"type": "custom", "query": args.query, "description": "Custom query"}]
    elif args.all_queries:
        queries = TEST_QUERIES
    else:
        # Default: test with medium complexity query
        queries = [TEST_QUERIES[2]]

    all_results = []

    for query_info in queries:
        print(f"\n{'#' * 60}")
        print(f"Query ({query_info['type']}): {query_info['query'][:50]}...")
        print(f"{'#' * 60}")

        query_results = []

        for model_config in models:
            print(f"\n[Testing: {model_config['name']}]")
            print(f"  {model_config['description']}")
            print(f"  Cost: ${model_config['cost_input']}/1M in, ${model_config['cost_output']}/1M out")

            try:
                results = tester.test_model(model_config, query_info["query"])
                query_results.append(results)

                if args.verbose:
                    print_detailed_results(results)
                else:
                    print(f"  → {results['totals']['latency']}s, ${results['totals']['cost']:.5f}")

            except Exception as e:
                print(f"  → ERROR: {e}")
                query_results.append({
                    "model": model_config["name"],
                    "error": str(e),
                    "totals": {"latency": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0},
                    "steps": {}
                })

        all_results.extend(query_results)
        print_comparison_table(query_results)

    # Final summary
    if len(queries) > 1:
        print("\n" + "=" * 60)
        print("OVERALL SUMMARY (all queries)")
        print("=" * 60)

        # Aggregate by model
        model_totals = {}
        for r in all_results:
            model = r["model"]
            if model not in model_totals:
                model_totals[model] = {"latency": 0, "cost": 0, "count": 0}
            model_totals[model]["latency"] += r["totals"]["latency"]
            model_totals[model]["cost"] += r["totals"]["cost"]
            model_totals[model]["count"] += 1

        print(f"{'Model':<20} {'Avg Latency':<12} {'Total Cost':<12}")
        print("-" * 50)
        for model, totals in model_totals.items():
            avg_lat = totals["latency"] / totals["count"]
            print(f"{model:<20} {avg_lat:.2f}s{'':<6} ${totals['cost']:.5f}")


if __name__ == "__main__":
    main()
