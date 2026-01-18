#!/usr/bin/env python3
"""
Benchmark: RRF Fusion + MMR Reranking Impact Analysis

Compares three retrieval configurations:
1. BASELINE: Original behavior (semantic only, no fusion)
2. RRF_ONLY: RRF fusion without MMR pruning
3. RRF_MMR: Full RRF + MMR (current implementation)

Usage:
    python benchmark_rrf_mmr.py --num-samples 2 --num-questions 10
    python benchmark_rrf_mmr.py --full  # Run full benchmark
"""
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

# SimpleMem imports
from main import SimpleMemSystem
from models.memory_entry import MemoryEntry, Dialogue

# Metrics
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util as st_util

# Dataset loading
@dataclass
class QA:
    question: str
    answer: Optional[str]
    evidence: List[str]
    category: Optional[int] = None
    adversarial_answer: Optional[str] = None

    @property
    def final_answer(self) -> Optional[str]:
        if self.category == 5:
            return self.adversarial_answer
        return self.answer


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
# Benchmark Configuration
# ============================================================================

@dataclass
class RetrievalConfig:
    """Configuration for a retrieval mode"""
    name: str
    use_hybrid_fusion: bool
    apply_mmr: bool = True
    mmr_lambda: float = 0.7
    mmr_top_k: int = 15
    description: str = ""


CONFIGS = {
    'baseline': RetrievalConfig(
        name='BASELINE',
        use_hybrid_fusion=False,
        apply_mmr=False,
        description='Original: Semantic search only (no RRF/MMR)'
    ),
    'rrf_only': RetrievalConfig(
        name='RRF_ONLY',
        use_hybrid_fusion=True,
        apply_mmr=False,
        description='RRF fusion only (semantic + keyword + structured, no MMR pruning)'
    ),
    'rrf_mmr': RetrievalConfig(
        name='RRF_MMR',
        use_hybrid_fusion=True,
        apply_mmr=True,
        mmr_lambda=0.7,
        mmr_top_k=15,
        description='Full RRF + MMR (fusion + diversity pruning)'
    ),
}


# ============================================================================
# Metrics Calculation
# ============================================================================

class MetricsCalculator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Warning: Could not load SentenceTransformer: {e}")
            self.sentence_model = None

    def calculate(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calculate comprehensive metrics"""
        if not prediction or not reference:
            return self._empty_metrics()

        prediction = str(prediction).strip()
        reference = str(reference).strip()

        # Token F1
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())
        common = pred_tokens & ref_tokens

        if not pred_tokens or not ref_tokens:
            f1 = 0.0
        else:
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(ref_tokens)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # ROUGE
        rouge_scores = self.rouge_scorer.score(reference, prediction)

        # Semantic similarity
        sbert_sim = 0.0
        if self.sentence_model:
            try:
                emb1 = self.sentence_model.encode([prediction], convert_to_tensor=True)
                emb2 = self.sentence_model.encode([reference], convert_to_tensor=True)
                sbert_sim = float(st_util.cos_sim(emb1, emb2).item())
            except:
                pass

        return {
            'f1': f1,
            'rouge1_f': rouge_scores['rouge1'].fmeasure,
            'rouge2_f': rouge_scores['rouge2'].fmeasure,
            'rougeL_f': rouge_scores['rougeL'].fmeasure,
            'sbert_similarity': sbert_sim,
        }

    def _empty_metrics(self) -> Dict[str, float]:
        return {
            'f1': 0.0,
            'rouge1_f': 0.0,
            'rouge2_f': 0.0,
            'rougeL_f': 0.0,
            'sbert_similarity': 0.0,
        }


# ============================================================================
# Dataset Loading
# ============================================================================

def load_locomo_dataset(file_path: Path, limit: int = None) -> List[LoCoMoSample]:
    """Load LoCoMo dataset"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    samples = []
    for idx, sample in enumerate(data):
        # Parse QA
        qa_list = []
        for qa in sample.get('qa', []):
            qa_list.append(QA(
                question=qa['question'],
                answer=qa.get('answer'),
                evidence=qa.get('evidence', []),
                category=qa.get('category'),
                adversarial_answer=qa.get('adversarial_answer')
            ))

        # Parse conversation
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
                    sessions[session_id] = Session(session_id=session_id, date_time=date_time, turns=turns)

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


# ============================================================================
# Benchmark Runner
# ============================================================================

class RRFMMRBenchmark:
    def __init__(self, dataset_path: str = 'test_ref/data/locomo10.json'):
        self.dataset_path = Path(dataset_path)
        self.metrics_calc = MetricsCalculator()
        self.results = defaultdict(list)

    def _create_fresh_system(self) -> SimpleMemSystem:
        """Create a fresh SimpleMem system"""
        return SimpleMemSystem(clear_db=True)

    def _load_sample_memories(self, system: SimpleMemSystem, sample: LoCoMoSample):
        """Load conversation into memory system"""
        for session_id in sorted(sample.conversation.sessions.keys()):
            session = sample.conversation.sessions[session_id]

            for turn in session.turns:
                if turn.text.strip():
                    system.add_dialogue(
                        speaker=turn.speaker,
                        content=turn.text,
                        timestamp=session.date_time
                    )

    def _retrieve_with_config(
        self,
        system: SimpleMemSystem,
        question: str,
        config: RetrievalConfig,
        category: int = None
    ) -> List[MemoryEntry]:
        """Execute retrieval with specific configuration"""
        # For category 5 (adversarial), disable reflection
        enable_reflection = False if category == 5 else None

        if not config.use_hybrid_fusion:
            # Baseline: just semantic search
            return system.hybrid_retriever.retrieve(
                question,
                enable_reflection=enable_reflection,
                use_hybrid_fusion=False
            )
        else:
            # RRF with or without MMR
            # We need to call the hybrid search directly to control apply_mmr
            if config.apply_mmr:
                return system.hybrid_retriever.retrieve(
                    question,
                    enable_reflection=enable_reflection,
                    use_hybrid_fusion=True
                )
            else:
                # RRF only - need to bypass MMR
                # Call _hybrid_search_with_fusion directly with apply_mmr=False
                return system.hybrid_retriever._hybrid_search_with_fusion(
                    question,
                    apply_mmr=False
                )

    def run_benchmark(
        self,
        num_samples: int = None,
        num_questions_per_sample: int = None,
        configs_to_test: List[str] = None
    ) -> Dict[str, Any]:
        """Run the complete benchmark"""

        print("\n" + "="*80)
        print(" RRF/MMR Impact Benchmark ".center(80))
        print("="*80)

        # Load dataset
        samples = load_locomo_dataset(self.dataset_path, limit=num_samples)
        print(f"\nLoaded {len(samples)} samples")

        configs_to_test = configs_to_test or list(CONFIGS.keys())

        all_results = {cfg: [] for cfg in configs_to_test}

        for sample_idx, sample in enumerate(samples):
            print(f"\n{'='*60}")
            print(f" Sample {sample_idx + 1}/{len(samples)}")
            print(f"{'='*60}")

            # Get questions to test
            questions = sample.qa
            if num_questions_per_sample:
                questions = questions[:num_questions_per_sample]

            print(f"Testing {len(questions)} questions across {len(configs_to_test)} configurations")

            # Test each configuration
            for config_name in configs_to_test:
                config = CONFIGS[config_name]
                print(f"\n--- {config.name}: {config.description} ---")

                # Fresh system for each config
                system = self._create_fresh_system()
                self._load_sample_memories(system, sample)

                config_results = []

                for qa_idx, qa in enumerate(questions):
                    question = qa.question
                    reference = qa.final_answer
                    category = qa.category

                    # Skip if no reference answer
                    if not reference:
                        continue

                    # Retrieve
                    start_time = time.time()
                    contexts = self._retrieve_with_config(system, question, config, category)
                    retrieval_time = time.time() - start_time

                    # Generate answer
                    answer_start = time.time()
                    answer = system.answer_generator.generate_answer(question, contexts)
                    answer_time = time.time() - answer_start

                    # Calculate metrics
                    metrics = self.metrics_calc.calculate(answer, reference)

                    result = {
                        'sample_idx': sample_idx,
                        'question_idx': qa_idx,
                        'category': category,
                        'question': question,
                        'answer': answer,
                        'reference': reference,
                        'num_retrieved': len(contexts),
                        'retrieval_time': retrieval_time,
                        'answer_time': answer_time,
                        **metrics
                    }

                    config_results.append(result)

                    if (qa_idx + 1) % 10 == 0:
                        print(f"  Processed {qa_idx + 1}/{len(questions)} questions...")

                all_results[config_name].extend(config_results)

                # Print config summary for this sample
                if config_results:
                    avg_f1 = statistics.mean([r['f1'] for r in config_results])
                    avg_rougeL = statistics.mean([r['rougeL_f'] for r in config_results])
                    avg_retrieved = statistics.mean([r['num_retrieved'] for r in config_results])
                    print(f"  Results: F1={avg_f1:.4f}, ROUGE-L={avg_rougeL:.4f}, Avg Retrieved={avg_retrieved:.1f}")

        # Aggregate results
        summary = self._aggregate_results(all_results)

        return {
            'summary': summary,
            'detailed_results': all_results
        }

    def _aggregate_results(self, all_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Aggregate results across all samples"""
        summary = {}

        print("\n" + "="*80)
        print(" BENCHMARK SUMMARY ".center(80))
        print("="*80)

        metrics_to_report = ['f1', 'rouge1_f', 'rouge2_f', 'rougeL_f', 'sbert_similarity', 'num_retrieved', 'retrieval_time']

        for config_name, results in all_results.items():
            if not results:
                continue

            config = CONFIGS[config_name]

            summary[config_name] = {
                'description': config.description,
                'num_questions': len(results),
                'metrics': {}
            }

            print(f"\n{config.name}:")
            print(f"  {config.description}")
            print(f"  Questions tested: {len(results)}")
            print()

            for metric in metrics_to_report:
                values = [r[metric] for r in results if metric in r]
                if values:
                    mean_val = statistics.mean(values)
                    std_val = statistics.stdev(values) if len(values) > 1 else 0.0
                    summary[config_name]['metrics'][metric] = {
                        'mean': mean_val,
                        'std': std_val,
                        'min': min(values),
                        'max': max(values)
                    }
                    print(f"  {metric:20s}: {mean_val:.4f} (+/- {std_val:.4f})")

            # Per-category breakdown
            categories = set(r['category'] for r in results if r.get('category'))
            if categories:
                print(f"\n  Per-category F1:")
                for cat in sorted(categories):
                    cat_results = [r for r in results if r.get('category') == cat]
                    if cat_results:
                        cat_f1 = statistics.mean([r['f1'] for r in cat_results])
                        print(f"    Category {cat}: {cat_f1:.4f} (n={len(cat_results)})")

        # Comparison table
        print("\n" + "-"*80)
        print(" COMPARISON TABLE ".center(80))
        print("-"*80)
        print(f"\n{'Config':<15} {'F1':>10} {'ROUGE-L':>10} {'SBERT':>10} {'#Retrieved':>12} {'Time(s)':>10}")
        print("-"*80)

        for config_name in all_results.keys():
            if config_name in summary:
                m = summary[config_name]['metrics']
                f1 = m.get('f1', {}).get('mean', 0)
                rougeL = m.get('rougeL_f', {}).get('mean', 0)
                sbert = m.get('sbert_similarity', {}).get('mean', 0)
                num_ret = m.get('num_retrieved', {}).get('mean', 0)
                ret_time = m.get('retrieval_time', {}).get('mean', 0)

                print(f"{config_name:<15} {f1:>10.4f} {rougeL:>10.4f} {sbert:>10.4f} {num_ret:>12.1f} {ret_time:>10.3f}")

        print("-"*80)

        # Key insights
        print("\n" + "="*80)
        print(" KEY INSIGHTS ".center(80))
        print("="*80)

        if 'baseline' in summary and 'rrf_mmr' in summary:
            baseline_f1 = summary['baseline']['metrics'].get('f1', {}).get('mean', 0)
            rrf_mmr_f1 = summary['rrf_mmr']['metrics'].get('f1', {}).get('mean', 0)
            delta = rrf_mmr_f1 - baseline_f1
            pct_change = (delta / baseline_f1 * 100) if baseline_f1 > 0 else 0

            print(f"\n1. F1 Score Impact:")
            print(f"   Baseline: {baseline_f1:.4f}")
            print(f"   RRF+MMR:  {rrf_mmr_f1:.4f}")
            print(f"   Delta:    {delta:+.4f} ({pct_change:+.1f}%)")

            if 'rrf_only' in summary:
                rrf_only_f1 = summary['rrf_only']['metrics'].get('f1', {}).get('mean', 0)
                mmr_impact = rrf_mmr_f1 - rrf_only_f1
                print(f"\n2. MMR Pruning Impact (RRF+MMR vs RRF_ONLY):")
                print(f"   RRF only: {rrf_only_f1:.4f}")
                print(f"   RRF+MMR:  {rrf_mmr_f1:.4f}")
                print(f"   Delta:    {mmr_impact:+.4f}")

                if mmr_impact < -0.01:
                    print(f"   >> MMR pruning is hurting recall!")
                elif mmr_impact > 0.01:
                    print(f"   >> MMR pruning is improving quality!")
                else:
                    print(f"   >> MMR pruning has minimal impact")

        return summary


def main():
    parser = argparse.ArgumentParser(description='Benchmark RRF/MMR impact on retrieval quality')
    parser.add_argument('--dataset', type=str, default='test_ref/data/locomo10.json',
                       help='Path to LoCoMo dataset')
    parser.add_argument('--num-samples', type=int, default=2,
                       help='Number of samples to test (default: 2)')
    parser.add_argument('--num-questions', type=int, default=20,
                       help='Number of questions per sample (default: 20)')
    parser.add_argument('--full', action='store_true',
                       help='Run full benchmark (all samples, all questions)')
    parser.add_argument('--output', type=str, default='benchmark_rrf_mmr_results.json',
                       help='Output file for detailed results')
    parser.add_argument('--configs', type=str, nargs='+',
                       default=['baseline', 'rrf_only', 'rrf_mmr'],
                       choices=['baseline', 'rrf_only', 'rrf_mmr'],
                       help='Configurations to test')

    args = parser.parse_args()

    benchmark = RRFMMRBenchmark(args.dataset)

    if args.full:
        results = benchmark.run_benchmark(
            num_samples=None,
            num_questions_per_sample=None,
            configs_to_test=args.configs
        )
    else:
        results = benchmark.run_benchmark(
            num_samples=args.num_samples,
            num_questions_per_sample=args.num_questions,
            configs_to_test=args.configs
        )

    # Save detailed results
    with open(args.output, 'w') as f:
        # Convert to serializable format
        output = {
            'summary': results['summary'],
            'detailed_results': {
                k: v for k, v in results['detailed_results'].items()
            }
        }
        json.dump(output, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {args.output}")


if __name__ == '__main__':
    main()
