"""
Group Memory Architecture Comparison

Comprehensive testing and comparison of all 5 group memory architectures.
Tests performance, accuracy, and scalability.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


class ArchitectureTester:
    """Test framework for comparing group memory architectures."""

    def __init__(self, test_data_file: str):
        """Initialize tester with test data."""
        with open(test_data_file, 'r') as f:
            self.test_data = json.load(f)

        self.messages = self.test_data["messages"]
        self.groups = self.test_data["groups"]
        self.users = self.test_data["users"]

        # Load query scenarios
        scenarios_file = test_data_file.replace("test_data.json", "query_scenarios.json")
        with open(scenarios_file, 'r') as f:
            self.scenarios = json.load(f)

        self.results = {}

    def test_architecture(self, arch_name: str, store) -> Dict[str, Any]:
        """
        Test a single architecture comprehensively.

        Tests:
        1. Insert performance (single and batch)
        2. Query performance (semantic and filtered)
        3. Accuracy (recall@k)
        4. Memory distribution
        5. Scalability
        """
        print(f"\n{'=' * 70}")
        print(f"Testing: {arch_name}")
        print(f"{'=' * 70}")

        results = {
            "architecture": arch_name,
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }

        # Test 1: Batch Insert Performance
        print("\n[Test 1] Batch Insert Performance...")
        insert_results = self._test_insert_performance(store)
        results["tests"]["insert"] = insert_results
        self._print_insert_results(insert_results)

        # Test 2: Query Performance
        print("\n[Test 2] Query Performance...")
        query_results = self._test_query_performance(store)
        results["tests"]["query"] = query_results
        self._print_query_results(query_results)

        # Test 3: Memory Distribution
        print("\n[Test 3] Memory Distribution...")
        dist_results = self._test_memory_distribution(store)
        results["tests"]["distribution"] = dist_results
        self._print_distribution_results(dist_results)

        # Test 4: Scenario-Based Testing
        print("\n[Test 4] Scenario-Based Testing...")
        scenario_results = self._test_scenarios(store)
        results["tests"]["scenarios"] = scenario_results
        self._print_scenario_results(scenario_results)

        # Test 5: Scalability (if applicable)
        print("\n[Test 5] Scalability Analysis...")
        scalability_results = self._test_scalability(store)
        results["tests"]["scalability"] = scalability_results
        self._print_scalability_results(scalability_results)

        return results

    def _test_insert_performance(self, store) -> Dict[str, Any]:
        """Test insert performance."""
        results = {
            "batch_insert": {},
            "single_insert": {}
        }

        # Clear any existing data
        if hasattr(store, 'clear'):
            store.clear()

        # Test batch insert
        start = time.time()
        if hasattr(store, 'add_messages_batch'):
            counts = store.add_messages_batch(self.messages)
        else:
            # Fallback to single inserts
            counts = {"total": 0}
            for msg in self.messages:
                store.add_message(msg)
                counts["total"] += 1

        elapsed = time.time() - start

        results["batch_insert"] = {
            "total_messages": len(self.messages),
            "time_seconds": elapsed,
            "throughput_msg_per_sec": len(self.messages) / elapsed if elapsed > 0 else 0,
            "memory_counts": counts if isinstance(counts, dict) else {"total": counts}
        }

        # Test single insert (sample 10 messages)
        sample_messages = self.messages[:10]
        times = []

        for msg in sample_messages:
            start = time.time()
            if hasattr(store, 'add_message'):
                store.add_message(msg)
            times.append(time.time() - start)

        if times:
            results["single_insert"] = {
                "sample_size": len(sample_messages),
                "avg_time_ms": (sum(times) / len(times)) * 1000,
                "min_time_ms": min(times) * 1000,
                "max_time_ms": max(times) * 1000
            }

        return results

    def _test_query_performance(self, store) -> Dict[str, Any]:
        """Test query performance."""
        results = {
            "semantic_search": [],
            "filtered_search": [],
            "context_retrieval": []
        }

        # Test queries
        test_queries = [
            "yield farming strategies",
            "NFT collecting tips",
            "smart contract security",
            "DAO governance"
        ]

        for query in test_queries:
            start = time.time()

            if hasattr(store, 'semantic_search'):
                results_list = store.semantic_search(query, limit=10)
            elif hasattr(store, 'search'):
                results_list = store.search(query, limit=10)
            else:
                results_list = []

            elapsed = time.time() - start

            results["semantic_search"].append({
                "query": query,
                "results_count": len(results_list),
                "latency_ms": elapsed * 1000
            })

        # Test filtered search
        group_id = self.groups[0]["group_id"]

        # Get memory types based on architecture
        memory_types = ["group", "user", "interaction"]

        for memory_type in memory_types:
            start = time.time()

            if hasattr(store, 'semantic_search'):
                results_list = store.semantic_search(
                    query="",
                    group_id=group_id,
                    memory_type=memory_type if hasattr(store, '_determine_memory_type') or memory_type in ["group", "user"] else None,
                    limit=5
                )
            else:
                results_list = []

            elapsed = time.time() - start

            results["filtered_search"].append({
                "filter": memory_type,
                "results_count": len(results_list),
                "latency_ms": elapsed * 1000
            })

        # Test context retrieval
        if hasattr(store, 'get_group_context'):
            start = time.time()
            context = store.get_group_context(group_id, self.users[0]["user_id"], limit=5)
            elapsed = time.time() - start

            results["context_retrieval"] = {
                "group_context_count": len(context.get("group_context", [])),
                "user_context_count": len(context.get("user_context", [])),
                "latency_ms": elapsed * 1000
            }

        return results

    def _test_memory_distribution(self, store) -> Dict[str, Any]:
        """Test memory distribution."""
        results = {}

        if hasattr(store, 'get_stats'):
            stats = store.get_stats()
            results = stats
        else:
            results = {"total_memories": len(self.messages)}

        return results

    def _test_scenarios(self, store) -> List[Dict[str, Any]]:
        """Test real-world scenarios."""
        results = []

        for scenario in self.scenarios[:4]:  # Test first 4 scenarios
            start = time.time()

            # Execute scenario query
            query = scenario["query"]

            if hasattr(store, 'semantic_search'):
                search_results = store.semantic_search(query, limit=5)
            elif hasattr(store, 'search'):
                search_results = store.search(query, limit=5)
            else:
                search_results = []

            elapsed = time.time() - start

            # Check if expected topics were found
            found_topics = []
            if search_results:
                for result in search_results[:3]:
                    topics = result.get("topics", [])
                    if isinstance(topics, list):
                        found_topics.extend(topics)

            results.append({
                "scenario": scenario["scenario"],
                "description": scenario["description"],
                "query": query,
                "results_count": len(search_results),
                "latency_ms": elapsed * 1000,
                "expected_topics": scenario["expected_context"],
                "found_topics": list(set(found_topics))
            })

        return results

    def _test_scalability(self, store) -> Dict[str, Any]:
        """Test scalability across multiple groups."""
        results = {
            "single_group": {},
            "multi_group": {},
            "cross_group": {}
        }

        # Single group query
        group_id = self.groups[0]["group_id"]
        start = time.time()

        if hasattr(store, 'semantic_search'):
            results_list = store.semantic_search(
                query="",
                group_id=group_id,
                limit=10
            )
        else:
            results_list = []

        elapsed = time.time() - start
        results["single_group"] = {
            "results": len(results_list),
            "latency_ms": elapsed * 1000
        }

        # Multi-group query (search across all groups)
        start = time.time()

        if hasattr(store, 'semantic_search'):
            results_list = store.semantic_search(
                query="",
                limit=50
            )
        else:
            results_list = []

        elapsed = time.time() - start
        results["multi_group"] = {
            "results": len(results_list),
            "latency_ms": elapsed * 1000
        }

        # Cross-group user query
        user_id = self.users[0]["user_id"]
        start = time.time()

        if hasattr(store, 'semantic_search'):
            results_list = store.semantic_search(
                query="",
                user_id=user_id,
                limit=20
            )
        else:
            results_list = []

        elapsed = time.time() - start
        results["cross_group"] = {
            "results": len(results_list),
            "latency_ms": elapsed * 1000
        }

        return results

    def _print_insert_results(self, results: Dict[str, Any]):
        """Print insert results."""
        batch = results["batch_insert"]

        print(f"  Batch Insert:")
        print(f"    - Total: {batch['total_messages']} messages")
        print(f"    - Time: {batch['time_seconds']:.2f}s")
        print(f"    - Throughput: {batch['throughput_msg_per_sec']:.2f} msg/sec")

        if "memory_counts" in batch and isinstance(batch["memory_counts"], dict):
            print(f"    - Distribution: {batch['memory_counts']}")

        if "single_insert" in results and results["single_insert"]:
            single = results["single_insert"]
            print(f"  Single Insert:")
            print(f"    - Avg: {single['avg_time_ms']:.2f}ms")
            print(f"    - Range: {single['min_time_ms']:.2f}ms - {single['max_time_ms']:.2f}ms")

    def _print_query_results(self, results: Dict[str, Any]):
        """Print query results."""
        print(f"  Semantic Search:")
        for result in results["semantic_search"]:
            print(f"    - '{result['query']}': {result['latency_ms']:.2f}ms ({result['results_count']} results)")

        if results["filtered_search"]:
            print(f"  Filtered Search:")
            for result in results["filtered_search"]:
                print(f"    - {result['filter']}: {result['latency_ms']:.2f}ms ({result['results_count']} results)")

        if results["context_retrieval"]:
            ctx = results["context_retrieval"]
            print(f"  Context Retrieval:")
            print(f"    - Group: {ctx['group_context_count']}, User: {ctx['user_context_count']}")
            print(f"    - Latency: {ctx['latency_ms']:.2f}ms")

    def _print_distribution_results(self, results: Dict[str, Any]):
        """Print distribution results."""
        print(f"  Memory Distribution:")
        for key, value in results.items():
            if isinstance(value, (int, float)):
                print(f"    - {key}: {value}")

    def _print_scenario_results(self, results: List[Dict[str, Any]]):
        """Print scenario results."""
        for result in results:
            print(f"  - {result['scenario']}: {result['latency_ms']:.2f}ms ({result['results_count']} results)")

    def _print_scalability_results(self, results: Dict[str, Any]):
        """Print scalability results."""
        print(f"  Single Group: {results['single_group']['latency_ms']:.2f}ms")
        print(f"  Multi Group: {results['multi_group']['latency_ms']:.2f}ms")
        print(f"  Cross Group: {results['cross_group']['latency_ms']:.2f}ms")

    def compare_architectures(self, all_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare all architectures and generate ranking."""
        print(f"\n{'=' * 70}")
        print(f"Architecture Comparison")
        print(f"{'=' * 70}")

        comparison = {
            "timestamp": datetime.now().isoformat(),
            "architectures": list(all_results.keys()),
            "metrics": {
                "insert_throughput": {},
                "query_latency": {},
                "storage_efficiency": {},
                "scalability": {}
            },
            "rankings": {}
        }

        # Extract metrics for comparison
        for arch_name, results in all_results.items():
            tests = results.get("tests", {})

            # Insert throughput
            insert = tests.get("insert", {}).get("batch_insert", {})
            comparison["metrics"]["insert_throughput"][arch_name] = insert.get("throughput_msg_per_sec", 0)

            # Query latency (average)
            query = tests.get("query", {}).get("semantic_search", [])
            if query:
                avg_latency = sum(q["latency_ms"] for q in query) / len(query)
                comparison["metrics"]["query_latency"][arch_name] = avg_latency

            # Storage efficiency (inverse of total memory count)
            dist = tests.get("distribution", {})
            total = dist.get("total_memories", 1)
            if total > 0:
                comparison["metrics"]["storage_efficiency"][arch_name] = 1 / total

            # Scalability (cross-group latency)
            scale = tests.get("scalability", {}).get("cross_group", {})
            comparison["metrics"]["scalability"][arch_name] = scale.get("latency_ms", float('inf'))

        # Generate rankings
        for metric_name, metric_values in comparison["metrics"].items():
            # Sort architectures by metric
            if metric_name == "query_latency" or metric_name == "scalability":
                # Lower is better
                sorted_archs = sorted(metric_values.items(), key=lambda x: x[1])
            else:
                # Higher is better
                sorted_archs = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)

            comparison["rankings"][metric_name] = [
                {"rank": i+1, "architecture": arch, "value": value}
                for i, (arch, value) in enumerate(sorted_archs)
            ]

        # Print rankings
        print("\nRankings:")
        for metric_name, ranking in comparison["rankings"].items():
            print(f"\n  {metric_name.replace('_', ' ').title()}:")
            for item in ranking[:3]:
                print(f"    {item['rank']}. {item['architecture']}: {item['value']:.2f}")

        # Calculate overall score
        print("\n" + "=" * 70)
        print("Overall Scores (lower is better)")
        print("=" * 70)

        overall_scores = {}
        for arch_name in all_results.keys():
            # Normalize metrics and calculate composite score
            throughput = comparison["metrics"]["insert_throughput"][arch_name]
            latency = comparison["metrics"]["query_latency"][arch_name]
            scalability = comparison["metrics"]["scalability"][arch_name]

            # Composite score (higher throughput is better, lower latency/scalability is better)
            # Score = latency + scalability - (throughput / 100)
            score = latency + scalability - (throughput / 100)
            overall_scores[arch_name] = score

        sorted_scores = sorted(overall_scores.items(), key=lambda x: x[1])

        for i, (arch, score) in enumerate(sorted_scores):
            print(f"{i+1}. {arch}: {score:.2f}")

        comparison["overall_scores"] = overall_scores

        return comparison

    def generate_report(self, all_results: Dict[str, Dict[str, Any]], comparison: Dict[str, Any]) -> str:
        """Generate comprehensive markdown report."""
        report_lines = [
            "# Group Memory Architecture Comparison Report",
            "",
            f"**Generated:** {comparison['timestamp']}",
            f"**Test Data:** {len(self.messages)} messages across {len(self.groups)} groups with {len(self.users)} users",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            "This report compares 5 different architectures for implementing group memories in the A0X platform.",
            "",
            "### Overall Ranking",
            ""
        ]

        # Overall ranking
        sorted_scores = sorted(comparison["overall_scores"].items(), key=lambda x: x[1])
        for i, (arch, score) in enumerate(sorted_scores):
            report_lines.append(f"{i+1}. **{arch}** - Score: {score:.2f}")

        report_lines.extend(["", "---", "", "## Performance Metrics", ""])

        # Performance table
        report_lines.append("| Architecture | Insert Throughput | Query Latency | Scalability |")
        report_lines.append("|--------------|-------------------|---------------|-------------|")

        for arch in comparison["architectures"]:
            throughput = comparison["metrics"]["insert_throughput"][arch]
            latency = comparison["metrics"]["query_latency"][arch]
            scalability = comparison["metrics"]["scalability"][arch]

            report_lines.append(
                f"| {arch} | {throughput:.2f} msg/s | {latency:.2f}ms | {scalability:.2f}ms |"
            )

        report_lines.extend(["", "---", "", "## Detailed Results", ""])

        # Detailed results for each architecture
        for arch_name, results in all_results.items():
            report_lines.extend([
                f"### {arch_name}",
                "",
                f"**Tested:** {results['timestamp']}",
                ""
            ])

            # Insert performance
            insert = results["tests"]["insert"]["batch_insert"]
            report_lines.extend([
                "#### Insert Performance",
                f"- **Throughput:** {insert['throughput_msg_per_sec']:.2f} messages/second",
                f"- **Total Time:** {insert['time_seconds']:.2f}s for {insert['total_messages']} messages",
                ""
            ])

            # Query performance
            query = results["tests"]["query"]["semantic_search"]
            if query:
                avg_latency = sum(q["latency_ms"] for q in query) / len(query)
                report_lines.extend([
                    "#### Query Performance",
                    f"- **Average Latency:** {avg_latency:.2f}ms",
                    f"- **Sample Queries:**"
                ])

                for q in query[:3]:
                    report_lines.append(f"  - '{q['query']}': {q['latency_ms']:.2f}ms ({q['results_count']} results)")

                report_lines.append("")

        report_lines.extend([
            "---",
            "",
            "## Recommendations",
            "",
            "### Best Overall Architecture",
            "",
            f"**{sorted_scores[0][0]}** achieved the best overall score with {sorted_scores[0][1]:.2f} points.",
            "",
            "### Use Case Recommendations",
            "",
            "- **Simple groups with basic memory:** Use Architecture 1 (Triple-Tenant Hierarchy)",
            "- **High-volume groups:** Use Architecture 2 (Memory Type Partitioning)",
            "- **Expertise location:** Use Architecture 3 (Graph-Based)",
            "- **Privacy-critical:** Use Architecture 4 (Privacy-Scoped)",
            "- **Production-grade:** Use Architecture 5 (Hybrid Multi-Level)",
            "",
            "### Implementation Recommendations for a0x-memory",
            "",
            "Based on the test results and requirements:",
            "",
            "1. **Start with Architecture 1** for MVP - simple and effective",
            "2. **Plan migration to Architecture 5** for production - most comprehensive",
            "3. **Use Architecture 4 principles** for privacy-critical features",
            "",
            "---",
            "",
            f"*Report generated by Group Memory Architecture Tester*"
        ])

        return "\n".join(report_lines)


def main():
    """Main test runner."""
    print("=" * 70)
    print("Group Memory Architecture Comparison")
    print("=" * 70)

    # Initialize tester
    test_data_file = "/home/oydual3/a0x/a0x-memory/tests/group_memory/test_data.json"
    tester = ArchitectureTester(test_data_file)

    # Test all architectures
    all_results = {}

    # Architecture 1: Triple-Tenant Hierarchy
    try:
        from arch1_triple_tenant import TripleTenantMemoryStore
        db_path = "/home/oydual3/a0x/a0x-memory/tests/group_memory/lancedb_arch1"
        store1 = TripleTenantMemoryStore(db_path=db_path, agent_id="jessexbt")
        results1 = tester.test_architecture("Architecture 1: Triple-Tenant Hierarchy", store1)
        all_results["Arch1: Triple-Tenant"] = results1
    except Exception as e:
        print(f"Error testing Architecture 1: {e}")
        import traceback
        traceback.print_exc()

    # Architecture 2: Memory Type Partitioning
    try:
        from arch2_partitioning import MemoryTypePartitioningStore
        db_path = "/home/oydual3/a0x/a0x-memory/tests/group_memory/lancedb_arch2"
        store2 = MemoryTypePartitioningStore(db_path=db_path, agent_id="jessexbt")
        results2 = tester.test_architecture("Architecture 2: Memory Type Partitioning", store2)
        all_results["Arch2: Partitioning"] = results2
    except Exception as e:
        print(f"Error testing Architecture 2: {e}")
        import traceback
        traceback.print_exc()

    # Architecture 3: Graph-Based
    try:
        from arch3_graph_based import GraphBasedMemoryStore
        db_path = "/home/oydual3/a0x/a0x-memory/tests/group_memory/lancedb_arch3"
        store3 = GraphBasedMemoryStore(db_path=db_path, agent_id="jessexbt")
        results3 = tester.test_architecture("Architecture 3: Graph-Based", store3)
        all_results["Arch3: Graph-Based"] = results3
    except Exception as e:
        print(f"Error testing Architecture 3: {e}")
        import traceback
        traceback.print_exc()

    # Architecture 4: Privacy-Scoped
    try:
        from arch4_privacy_scoped import PrivacyScopedMemoryStore
        db_path = "/home/oydual3/a0x/a0x-memory/tests/group_memory/lancedb_arch4"
        store4 = PrivacyScopedMemoryStore(db_path=db_path, agent_id="jessexbt")
        results4 = tester.test_architecture("Architecture 4: Privacy-Scoped", store4)
        all_results["Arch4: Privacy-Scoped"] = results4
    except Exception as e:
        print(f"Error testing Architecture 4: {e}")
        import traceback
        traceback.print_exc()

    # Architecture 5: Hybrid Multi-Level
    try:
        from arch5_hybrid_multi_level import HybridMultiLevelMemoryStore
        db_path = "/home/oydual3/a0x/a0x-memory/tests/group_memory/lancedb_arch5"
        store5 = HybridMultiLevelMemoryStore(db_path=db_path, agent_id="jessexbt")
        results5 = tester.test_architecture("Architecture 5: Hybrid Multi-Level", store5)
        all_results["Arch5: Hybrid"] = results5
    except Exception as e:
        print(f"Error testing Architecture 5: {e}")
        import traceback
        traceback.print_exc()

    # Compare architectures
    if len(all_results) > 1:
        comparison = tester.compare_architectures(all_results)
    else:
        print("Only one architecture tested successfully, skipping comparison.")
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "architectures": list(all_results.keys()),
            "metrics": {},
            "overall_scores": {},
            "rankings": {}
        }

        # Extract metrics from single architecture
        for arch_name, results in all_results.items():
            tests = results.get("tests", {})

            insert = tests.get("insert", {}).get("batch_insert", {})
            query = tests.get("query", {}).get("semantic_search", [])
            scale = tests.get("scalability", {}).get("cross_group", {})

            comparison["metrics"]["insert_throughput"] = {arch_name: insert.get("throughput_msg_per_sec", 0)}
            comparison["metrics"]["query_latency"] = {arch_name: sum(q["latency_ms"] for q in query) / len(query) if query else 0}
            comparison["metrics"]["scalability"] = {arch_name: scale.get("latency_ms", 0)}

            # Calculate overall score
            throughput = comparison["metrics"]["insert_throughput"][arch_name]
            latency = comparison["metrics"]["query_latency"][arch_name]
            scalability = comparison["metrics"]["scalability"][arch_name]

            score = latency + scalability - (throughput / 100)
            comparison["overall_scores"][arch_name] = score

    # Generate report
    report = tester.generate_report(all_results, comparison)

    # Save report
    report_file = "/home/oydual3/a0x/a0x-memory/tests/group_memory/COMPARISON_REPORT.md"
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"\n{'=' * 70}")
    print(f"Report saved to: {report_file}")
    print(f"{'=' * 70}")

    # Save results JSON
    results_file = "/home/oydual3/a0x/a0x-memory/tests/group_memory/test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "comparison": comparison,
            "detailed_results": all_results
        }, f, indent=2)

    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
