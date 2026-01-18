#!/usr/bin/env python3
"""
Retrieval Quality Evaluation for Hybrid Search with RRF and MMR

This benchmark evaluates retrieval configurations using standard Information Retrieval
metrics on the LoCoMo dataset (Maharana et al., 2024).

Methodology based on:
- Clarke et al. (2008) "Novelty and diversity in information retrieval evaluation"
  ACM SIGIR. https://dl.acm.org/doi/10.1145/1390334.1390446
- Järvelin & Kekäläinen (2002) "Cumulated gain-based evaluation of IR techniques"
  ACM TOIS. https://dl.acm.org/doi/10.1145/582415.582418
- Carbonell & Goldstein (1998) "The use of MMR, diversity-based reranking"
  ACM SIGIR. https://dl.acm.org/doi/10.1145/290941.291025

Metrics implemented:
- Recall@K: Fraction of relevant documents retrieved (binary relevance)
- Precision@K: Fraction of retrieved documents that are relevant
- NDCG@K: Normalized Discounted Cumulative Gain (graded relevance, position-aware)
- α-nDCG: Extension for novelty/diversity evaluation (Clarke et al., 2008)
- MRR: Mean Reciprocal Rank (position of first relevant result)

Dataset: LoCoMo - Long Context Memory benchmark
- 10 conversations with comprehensive QA annotations
- Ground truth evidence annotations for each question
- Source: https://github.com/snap-research/locomo

Usage:
    python benchmarks/retrieval_evaluation.py
    python benchmarks/retrieval_evaluation.py --samples 5 --questions 50
    python benchmarks/retrieval_evaluation.py --full --output results.json
"""

import json
import math
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class QA:
    question: str
    answer: Optional[str]
    evidence: List[str]  # Ground truth relevant document IDs
    category: Optional[int] = None

@dataclass
class Turn:
    speaker: str
    dia_id: str
    text: str

@dataclass
class Session:
    session_id: int
    date_time: str
    turns: List[Turn]

@dataclass
class Conversation:
    speaker_a: str
    speaker_b: str
    sessions: Dict[int, Session]

@dataclass
class LoCoMoSample:
    sample_id: str
    qa: List[QA]
    conversation: Conversation


# ============================================================================
# Information Retrieval Metrics
# Based on Järvelin & Kekäläinen (2002) and Clarke et al. (2008)
# ============================================================================

class IRMetrics:
    """
    Standard Information Retrieval evaluation metrics.

    References:
    - Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation
      of IR techniques. ACM TOIS, 20(4), 422-446.
    - Clarke, C. L., et al. (2008). Novelty and diversity in information
      retrieval evaluation. ACM SIGIR, 659-666.
    """

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = None) -> float:
        """
        Recall@K: Fraction of relevant documents that were retrieved.

        R@K = |Retrieved ∩ Relevant| / |Relevant|

        Args:
            retrieved_ids: Ordered list of retrieved document IDs
            relevant_ids: Set of ground truth relevant document IDs
            k: Cutoff (if None, use all retrieved)

        Returns:
            Recall score in [0, 1]
        """
        if not relevant_ids:
            return 0.0

        if k is not None:
            retrieved_ids = retrieved_ids[:k]

        retrieved_set = set(retrieved_ids)
        hits = len(retrieved_set & relevant_ids)
        return hits / len(relevant_ids)

    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = None) -> float:
        """
        Precision@K: Fraction of retrieved documents that are relevant.

        P@K = |Retrieved@K ∩ Relevant| / K

        Args:
            retrieved_ids: Ordered list of retrieved document IDs
            relevant_ids: Set of ground truth relevant document IDs
            k: Cutoff (if None, use all retrieved)

        Returns:
            Precision score in [0, 1]
        """
        if k is not None:
            retrieved_ids = retrieved_ids[:k]

        if not retrieved_ids:
            return 0.0

        retrieved_set = set(retrieved_ids)
        hits = len(retrieved_set & relevant_ids)
        return hits / len(retrieved_ids)

    @staticmethod
    def f1_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = None) -> float:
        """
        F1@K: Harmonic mean of Precision@K and Recall@K.
        """
        p = IRMetrics.precision_at_k(retrieved_ids, relevant_ids, k)
        r = IRMetrics.recall_at_k(retrieved_ids, relevant_ids, k)

        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @staticmethod
    def dcg_at_k(relevance_scores: List[float], k: int = None) -> float:
        """
        Discounted Cumulative Gain at K.

        DCG@K = Σ (2^rel_i - 1) / log2(i + 1) for i in 1..K

        Using the formulation from Järvelin & Kekäläinen (2002).

        Args:
            relevance_scores: List of relevance scores in rank order
            k: Cutoff

        Returns:
            DCG score
        """
        if k is not None:
            relevance_scores = relevance_scores[:k]

        dcg = 0.0
        for i, rel in enumerate(relevance_scores, start=1):
            dcg += (2 ** rel - 1) / math.log2(i + 1)

        return dcg

    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = None) -> float:
        """
        Normalized Discounted Cumulative Gain at K.

        NDCG@K = DCG@K / IDCG@K

        Where IDCG is the DCG of the ideal ranking (all relevant docs first).

        References:
        - Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation.

        Args:
            retrieved_ids: Ordered list of retrieved document IDs
            relevant_ids: Set of ground truth relevant document IDs
            k: Cutoff

        Returns:
            NDCG score in [0, 1]
        """
        if k is not None:
            retrieved_ids = retrieved_ids[:k]

        # Binary relevance: 1 if relevant, 0 otherwise
        relevance_scores = [1.0 if doc_id in relevant_ids else 0.0 for doc_id in retrieved_ids]

        dcg = IRMetrics.dcg_at_k(relevance_scores, k)

        # Ideal DCG: all relevant documents ranked first
        ideal_relevance = [1.0] * min(len(relevant_ids), len(retrieved_ids))
        if k is not None:
            ideal_relevance = ideal_relevance[:k]
        idcg = IRMetrics.dcg_at_k(ideal_relevance, k)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """
        Mean Reciprocal Rank: 1 / (position of first relevant result).

        MRR = 1 / rank_first_relevant

        Args:
            retrieved_ids: Ordered list of retrieved document IDs
            relevant_ids: Set of ground truth relevant document IDs

        Returns:
            MRR score in [0, 1]
        """
        for i, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return 1.0 / i
        return 0.0

    @staticmethod
    def alpha_ndcg(
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        subtopics: Dict[str, Set[str]] = None,
        alpha: float = 0.5,
        k: int = None
    ) -> float:
        """
        α-nDCG: Novelty and diversity-aware NDCG.

        Penalizes redundant documents covering already-seen subtopics.

        References:
        - Clarke, C. L., et al. (2008). Novelty and diversity in information
          retrieval evaluation. ACM SIGIR.

        Args:
            retrieved_ids: Ordered list of retrieved document IDs
            relevant_ids: Set of ground truth relevant document IDs
            subtopics: Mapping of doc_id -> set of subtopics it covers
            alpha: Redundancy penalty (0.5 recommended by Clarke et al.)
            k: Cutoff

        Returns:
            α-nDCG score in [0, 1]
        """
        if k is not None:
            retrieved_ids = retrieved_ids[:k]

        # If no subtopics provided, treat each relevant doc as its own subtopic
        if subtopics is None:
            subtopics = {doc_id: {doc_id} for doc_id in relevant_ids}

        seen_subtopics: Set[str] = set()
        gain = []

        for doc_id in retrieved_ids:
            if doc_id in relevant_ids:
                doc_subtopics = subtopics.get(doc_id, {doc_id})
                new_subtopics = doc_subtopics - seen_subtopics

                # Gain reduced by (1-α) for each already-seen subtopic
                redundancy = len(doc_subtopics) - len(new_subtopics)
                doc_gain = len(new_subtopics) + redundancy * (1 - alpha)

                seen_subtopics.update(doc_subtopics)
                gain.append(doc_gain / len(doc_subtopics) if doc_subtopics else 0)
            else:
                gain.append(0.0)

        # Calculate α-DCG
        alpha_dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gain))

        # Ideal: all relevant docs, no redundancy
        ideal_gain = [1.0] * min(len(relevant_ids), len(retrieved_ids))
        ideal_dcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gain))

        if ideal_dcg == 0:
            return 0.0

        return alpha_dcg / ideal_dcg


# ============================================================================
# Dataset Loading
# ============================================================================

def load_locomo_dataset(file_path: Path, limit: int = None) -> List[LoCoMoSample]:
    """Load LoCoMo dataset with ground truth evidence annotations."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    samples = []
    for idx, sample in enumerate(data):
        qa_list = []
        for qa in sample.get('qa', []):
            qa_list.append(QA(
                question=qa['question'],
                answer=qa.get('answer'),
                evidence=qa.get('evidence', []),
                category=qa.get('category')
            ))

        sessions = {}
        conv_data = sample.get('conversation', {})
        for key, value in conv_data.items():
            if key.startswith('session_') and isinstance(value, list):
                session_id = int(key.split('_')[1])
                date_time = conv_data.get(f'{key}_date_time', '')
                turns = []
                for turn in value:
                    text = turn.get('text', '')
                    if 'blip_caption' in turn:
                        text = f"[Image: {turn['blip_caption']}] {text}"
                    turns.append(Turn(
                        speaker=turn['speaker'],
                        dia_id=turn['dia_id'],
                        text=text
                    ))
                if turns:
                    sessions[session_id] = Session(
                        session_id=session_id,
                        date_time=date_time,
                        turns=turns
                    )

        conversation = Conversation(
            speaker_a=conv_data.get('speaker_a', 'A'),
            speaker_b=conv_data.get('speaker_b', 'B'),
            sessions=sessions
        )

        samples.append(LoCoMoSample(
            sample_id=str(idx),
            qa=qa_list,
            conversation=conversation
        ))

    return samples


def build_turn_id_mapping(sample: LoCoMoSample) -> Dict[str, str]:
    """
    Build mapping from dia_id to turn content.
    Evidence format in LoCoMo: "D1:3" (dialogue session : turn number)
    """
    mapping = {}

    for session_id, session in sample.conversation.sessions.items():
        for turn in session.turns:
            # dia_id format is like "D1:3"
            mapping[turn.dia_id] = turn.text

    return mapping


# ============================================================================
# Retrieval Configurations
# ============================================================================

@dataclass
class RetrievalConfig:
    name: str
    use_hybrid_fusion: bool
    apply_mmr: bool = True
    description: str = ""


CONFIGURATIONS = {
    'baseline': RetrievalConfig(
        name='Baseline',
        use_hybrid_fusion=False,
        apply_mmr=False,
        description='Semantic search only'
    ),
    'rrf': RetrievalConfig(
        name='RRF',
        use_hybrid_fusion=True,
        apply_mmr=False,
        description='RRF fusion (semantic + keyword + structured)'
    ),
    'rrf_mmr': RetrievalConfig(
        name='RRF+MMR',
        use_hybrid_fusion=True,
        apply_mmr=True,
        description='RRF fusion with MMR diversity reranking'
    ),
}


# ============================================================================
# Benchmark Runner
# ============================================================================

class RetrievalBenchmark:
    """
    Benchmark for evaluating retrieval quality using standard IR metrics.
    """

    def __init__(self, dataset_path: str = 'test_ref/data/locomo10.json'):
        self.dataset_path = Path(dataset_path)
        self.metrics = IRMetrics()

    def _create_system(self):
        """Create fresh SimpleMem system."""
        import sys
        from pathlib import Path
        # Add parent directory to path for imports
        parent_dir = str(Path(__file__).parent.parent)
        sys.path.insert(0, parent_dir)

        # Override config for benchmark (use fast local model)
        import config
        config.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        config.EMBEDDING_DIMENSION = 384
        config.EMBEDDING_PROVIDER = "local"

        from main import SimpleMemSystem
        return SimpleMemSystem(clear_db=True)

    def _load_memories(self, system, sample: LoCoMoSample) -> Dict[str, str]:
        """
        Load conversation into memory system.
        Returns mapping of memory entry IDs to turn IDs.
        """
        entry_to_turn = {}

        for session_id in sorted(sample.conversation.sessions.keys()):
            session = sample.conversation.sessions[session_id]

            for turn in session.turns:
                if turn.text.strip():
                    system.add_dialogue(
                        speaker=turn.speaker,
                        content=turn.text,
                        timestamp=session.date_time
                    )
                    # Track mapping (simplified - assumes sequential IDs)
                    entry_to_turn[turn.dia_id] = turn.dia_id

        return entry_to_turn

    def _retrieve(self, system, question: str, config: RetrievalConfig, category: int = None):
        """Execute retrieval with given configuration."""
        enable_reflection = False if category == 5 else None

        # All configs use the same retrieve() path for fair comparison
        # The difference is in use_hybrid_fusion parameter
        return system.hybrid_retriever.retrieve(
            question,
            enable_reflection=enable_reflection,
            use_hybrid_fusion=config.use_hybrid_fusion
        )

    def _extract_turn_ids_from_results(self, results, turn_mapping: Dict[str, str]) -> List[str]:
        """
        Extract turn IDs from retrieval results for metric calculation.
        Uses fuzzy matching since SimpleMem transforms the original text.
        """
        retrieved_ids = []

        for entry in results:
            entry_text = entry.lossless_restatement if hasattr(entry, 'lossless_restatement') else str(entry)
            entry_text_lower = entry_text.lower()

            best_match = None
            best_score = 0

            for turn_id, turn_text in turn_mapping.items():
                turn_text_lower = turn_text.lower()

                # Check for significant overlap
                # Split into words and check intersection
                entry_words = set(entry_text_lower.split())
                turn_words = set(turn_text_lower.split())

                if not turn_words:
                    continue

                intersection = entry_words & turn_words
                score = len(intersection) / len(turn_words)

                # Require at least 50% word overlap
                if score > best_score and score >= 0.5:
                    best_score = score
                    best_match = turn_id

            if best_match:
                retrieved_ids.append(best_match)

        return retrieved_ids

    def _parse_evidence_ids(self, evidence: List[str]) -> Set[str]:
        """Parse evidence IDs. Format is 'D1:3' (dia_id)."""
        return set(evidence)

    def run(
        self,
        num_samples: int = None,
        num_questions: int = None,
        configs: List[str] = None,
        k_values: List[int] = [5, 10, 15, 20]
    ) -> Dict[str, Any]:
        """
        Run retrieval evaluation.

        Args:
            num_samples: Number of conversation samples (None = all)
            num_questions: Questions per sample (None = all)
            configs: Configuration names to test
            k_values: K values for @K metrics

        Returns:
            Dictionary with aggregated metrics and detailed results
        """
        print("\n" + "=" * 80)
        print(" Retrieval Quality Evaluation ".center(80))
        print(" LoCoMo Dataset | Standard IR Metrics ".center(80))
        print("=" * 80)

        samples = load_locomo_dataset(self.dataset_path, limit=num_samples)
        print(f"\nDataset: {len(samples)} samples loaded")

        configs = configs or list(CONFIGURATIONS.keys())
        print(f"Configurations: {', '.join(configs)}")
        print(f"K values: {k_values}")

        all_results = {cfg: [] for cfg in configs}

        for sample_idx, sample in enumerate(samples):
            print(f"\n{'─' * 60}")
            print(f" Sample {sample_idx + 1}/{len(samples)} ")
            print(f"{'─' * 60}")

            # Build turn mapping for this sample
            turn_mapping = build_turn_id_mapping(sample)

            questions = sample.qa
            if num_questions:
                questions = questions[:num_questions]

            # Filter questions with evidence
            questions = [q for q in questions if q.evidence]
            print(f"Questions with ground truth: {len(questions)}")

            for config_name in configs:
                config = CONFIGURATIONS[config_name]
                print(f"\n  [{config.name}] {config.description}")

                # Fresh system per config
                system = self._create_system()
                self._load_memories(system, sample)

                for qa in questions:
                    relevant_ids = self._parse_evidence_ids(qa.evidence)

                    # Retrieve
                    start = time.time()
                    results = self._retrieve(system, qa.question, config, qa.category)
                    retrieval_time = time.time() - start

                    # Extract IDs from results
                    retrieved_ids = self._extract_turn_ids_from_results(results, turn_mapping)

                    # Calculate metrics for each K
                    metrics = {
                        'question': qa.question,
                        'category': qa.category,
                        'num_relevant': len(relevant_ids),
                        'num_retrieved': len(results),
                        'retrieval_time': retrieval_time,
                    }

                    for k in k_values:
                        metrics[f'recall@{k}'] = self.metrics.recall_at_k(retrieved_ids, relevant_ids, k)
                        metrics[f'precision@{k}'] = self.metrics.precision_at_k(retrieved_ids, relevant_ids, k)
                        metrics[f'ndcg@{k}'] = self.metrics.ndcg_at_k(retrieved_ids, relevant_ids, k)
                        metrics[f'f1@{k}'] = self.metrics.f1_at_k(retrieved_ids, relevant_ids, k)

                    metrics['mrr'] = self.metrics.mrr(retrieved_ids, relevant_ids)

                    all_results[config_name].append(metrics)

                # Progress
                n = len(all_results[config_name])
                if n > 0:
                    avg_recall = statistics.mean([r['recall@10'] for r in all_results[config_name][-len(questions):]])
                    print(f"    Processed {len(questions)} questions | Avg Recall@10: {avg_recall:.3f}")

        # Aggregate results
        summary = self._aggregate(all_results, k_values)

        return {
            'summary': summary,
            'detailed_results': all_results,
            'k_values': k_values,
            'configurations': {k: {'name': v.name, 'description': v.description}
                             for k, v in CONFIGURATIONS.items() if k in configs}
        }

    def _aggregate(self, all_results: Dict[str, List], k_values: List[int]) -> Dict:
        """Aggregate metrics across all questions."""
        summary = {}

        print("\n" + "=" * 80)
        print(" RESULTS ".center(80))
        print("=" * 80)

        # Metrics to aggregate
        base_metrics = ['mrr', 'num_retrieved', 'retrieval_time']
        k_metrics = ['recall', 'precision', 'ndcg', 'f1']

        for config_name, results in all_results.items():
            if not results:
                continue

            config = CONFIGURATIONS[config_name]
            summary[config_name] = {
                'name': config.name,
                'description': config.description,
                'n_questions': len(results),
                'metrics': {}
            }

            print(f"\n{config.name}: {config.description}")
            print(f"  n = {len(results)} questions")
            print()

            # Aggregate base metrics
            for metric in base_metrics:
                values = [r[metric] for r in results]
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0
                summary[config_name]['metrics'][metric] = {
                    'mean': mean_val,
                    'std': std_val
                }

            # Aggregate K metrics
            print(f"  {'K':>4} {'Recall':>10} {'Precision':>10} {'NDCG':>10} {'F1':>10}")
            print(f"  {'-'*4} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

            for k in k_values:
                row = f"  {k:>4}"
                for metric in k_metrics:
                    key = f'{metric}@{k}'
                    values = [r[key] for r in results]
                    mean_val = statistics.mean(values)
                    std_val = statistics.stdev(values) if len(values) > 1 else 0
                    summary[config_name]['metrics'][key] = {
                        'mean': mean_val,
                        'std': std_val
                    }
                    row += f" {mean_val:>10.4f}"
                print(row)

            # MRR
            mrr = summary[config_name]['metrics']['mrr']['mean']
            print(f"\n  MRR: {mrr:.4f}")
            print(f"  Avg retrieved: {summary[config_name]['metrics']['num_retrieved']['mean']:.1f}")
            print(f"  Avg time: {summary[config_name]['metrics']['retrieval_time']['mean']*1000:.1f}ms")

        # Comparison table
        self._print_comparison(summary, k_values)

        return summary

    def _print_comparison(self, summary: Dict, k_values: List[int]):
        """Print comparison table across configurations."""
        print("\n" + "=" * 80)
        print(" COMPARISON ".center(80))
        print("=" * 80)

        configs = list(summary.keys())
        if len(configs) < 2:
            return

        print(f"\nRecall@K comparison:")
        header = f"  {'K':>4}"
        for cfg in configs:
            header += f" {summary[cfg]['name']:>12}"
        if len(configs) >= 2:
            header += f" {'Δ (last-first)':>14}"
        print(header)
        print(f"  {'-'*4}" + f" {'-'*12}" * len(configs) + f" {'-'*14}")

        for k in k_values:
            row = f"  {k:>4}"
            values = []
            for cfg in configs:
                val = summary[cfg]['metrics'][f'recall@{k}']['mean']
                values.append(val)
                row += f" {val:>12.4f}"
            if len(values) >= 2:
                delta = values[-1] - values[0]
                row += f" {delta:>+14.4f}"
            print(row)

        # Statistical notes
        print("\n" + "-" * 80)
        print("Notes:")
        print("  - Metrics computed using binary relevance (relevant/not relevant)")
        print("  - Ground truth from LoCoMo dataset evidence annotations")
        print("  - NDCG uses log2 discounting (Järvelin & Kekäläinen, 2002)")


def main():
    parser = argparse.ArgumentParser(
        description='Retrieval Quality Evaluation for Hybrid Search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
References:
  Clarke et al. (2008) - Novelty and diversity in IR evaluation
  Järvelin & Kekäläinen (2002) - Cumulated gain-based evaluation
  Carbonell & Goldstein (1998) - MMR diversity-based reranking
        """
    )
    parser.add_argument('--dataset', type=str, default='test_ref/data/locomo10.json')
    parser.add_argument('--samples', type=int, default=2, help='Number of samples')
    parser.add_argument('--questions', type=int, default=30, help='Questions per sample')
    parser.add_argument('--full', action='store_true', help='Run full benchmark')
    parser.add_argument('--output', type=str, default='retrieval_eval_results.json')
    parser.add_argument('--configs', nargs='+', default=['baseline', 'rrf', 'rrf_mmr'],
                       choices=['baseline', 'rrf', 'rrf_mmr'])
    parser.add_argument('--k', nargs='+', type=int, default=[5, 10, 15, 20],
                       help='K values for @K metrics')

    args = parser.parse_args()

    benchmark = RetrievalBenchmark(args.dataset)

    if args.full:
        results = benchmark.run(
            num_samples=None,
            num_questions=None,
            configs=args.configs,
            k_values=args.k
        )
    else:
        results = benchmark.run(
            num_samples=args.samples,
            num_questions=args.questions,
            configs=args.configs,
            k_values=args.k
        )

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
