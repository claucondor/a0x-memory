"""
TechQA Dataset Test for SimpleMem System
Tests adaptive keyword boost on technical documentation queries
"""
from pathlib import Path
import time
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm
import statistics
from collections import defaultdict
import argparse

# Metrics
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim

from main import SimpleMemSystem

# Initialize SentenceTransformer model for semantic similarity
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Warning: Could not load SentenceTransformer model: {e}")
    sentence_model = None


@dataclass
class TechQAExample:
    id: str
    document: str
    question: str
    answer: str


class TechQATester:
    def __init__(self, system: SimpleMemSystem, use_adaptive_keyword: bool = True):
        self.system = system
        self.use_adaptive_keyword = use_adaptive_keyword
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def load_dataset(self, split: str = 'train', max_examples: int = None) -> List[TechQAExample]:
        """Load TechQA dataset from HuggingFace"""
        from datasets import load_dataset

        print(f"Loading TechQA dataset (split={split})...")
        ds = load_dataset('rojagtap/tech-qa', split=split)

        examples = []
        for i, item in enumerate(ds):
            if max_examples and i >= max_examples:
                break
            examples.append(TechQAExample(
                id=item['id'],
                document=item['document'],
                question=item['question'],
                answer=item['answer']
            ))

        print(f"Loaded {len(examples)} examples")
        return examples

    def ingest_documents(self, examples: List[TechQAExample]):
        """Ingest all documents into the memory system"""
        print(f"Ingesting {len(examples)} documents...")

        for i, ex in enumerate(tqdm(examples, desc="Ingesting")):
            # Use add_dialogue method
            self.system.add_dialogue(
                speaker="TechDoc",
                content=ex.document[:4000],  # Truncate if too long
                timestamp=None
            )

        # Finalize to flush buffer and process all dialogues
        self.system.finalize()
        print(f"Ingestion complete. Total entries in DB: {self.system.vector_store.table.count_rows()}")

    def test_single_question(self, example: TechQAExample) -> Dict:
        """Test a single question and return metrics"""
        start_time = time.time()

        # Retrieve relevant context
        results = self.system.hybrid_retriever.retrieve(
            example.question,
            use_hybrid_fusion=self.use_adaptive_keyword
        )

        retrieval_time = time.time() - start_time

        # Generate answer using retrieved context
        if results:
            context = "\n\n".join([
                f"Context {i+1}: {r.lossless_restatement[:500]}"
                for i, r in enumerate(results[:5])
            ])
        else:
            context = "No relevant information found."

        answer_start = time.time()
        generated_answer = self.system.llm_client.chat_completion([
            {"role": "system", "content": "Answer the technical question based on the provided context. Be concise and specific."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {example.question}\n\nAnswer:"}
        ], temperature=0.1)
        answer_time = time.time() - answer_start

        # Calculate metrics
        metrics = self.calculate_metrics(generated_answer, example.answer)
        metrics['retrieval_time'] = retrieval_time
        metrics['answer_time'] = answer_time
        metrics['total_time'] = retrieval_time + answer_time
        metrics['num_results'] = len(results)
        metrics['generated_answer'] = generated_answer
        metrics['reference_answer'] = example.answer

        return metrics

    def calculate_metrics(self, generated: str, reference: str) -> Dict:
        """Calculate evaluation metrics"""
        # F1 (token overlap)
        gen_tokens = set(generated.lower().split())
        ref_tokens = set(reference.lower().split())

        if len(gen_tokens) == 0 or len(ref_tokens) == 0:
            f1 = 0.0
        else:
            precision = len(gen_tokens & ref_tokens) / len(gen_tokens)
            recall = len(gen_tokens & ref_tokens) / len(ref_tokens)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # ROUGE-L
        rouge_scores = self.rouge_scorer.score(reference, generated)
        rouge_l = rouge_scores['rougeL'].fmeasure

        # SBERT similarity
        if sentence_model:
            gen_emb = sentence_model.encode([generated])
            ref_emb = sentence_model.encode([reference])
            sbert_sim = float(pytorch_cos_sim(gen_emb, ref_emb)[0][0])
        else:
            sbert_sim = 0.0

        return {
            'f1': f1,
            'rouge_l': rouge_l,
            'sbert_similarity': sbert_sim
        }

    def run_test(self, num_examples: int = 50, split: str = 'train') -> Dict:
        """Run full test on TechQA"""
        examples = self.load_dataset(split=split, max_examples=num_examples)

        # Ingest documents
        self.ingest_documents(examples)

        # Test each question
        all_metrics = []
        keyword_boost_used = 0
        keyword_boost_skipped = 0

        print(f"\nTesting {len(examples)} questions...")
        for i, ex in enumerate(tqdm(examples, desc="Testing")):
            print(f"\n[Q{i+1}] {ex.question[:80]}...")

            metrics = self.test_single_question(ex)
            all_metrics.append(metrics)

            print(f"  F1: {metrics['f1']:.3f}, ROUGE-L: {metrics['rouge_l']:.3f}, SBERT: {metrics['sbert_similarity']:.3f}")
            print(f"  Time: {metrics['total_time']:.2f}s, Results: {metrics['num_results']}")

        # Aggregate results
        summary = {
            'num_examples': len(examples),
            'use_adaptive_keyword': self.use_adaptive_keyword,
            'metrics': {
                'f1': {
                    'mean': statistics.mean([m['f1'] for m in all_metrics]),
                    'std': statistics.stdev([m['f1'] for m in all_metrics]) if len(all_metrics) > 1 else 0
                },
                'rouge_l': {
                    'mean': statistics.mean([m['rouge_l'] for m in all_metrics]),
                    'std': statistics.stdev([m['rouge_l'] for m in all_metrics]) if len(all_metrics) > 1 else 0
                },
                'sbert_similarity': {
                    'mean': statistics.mean([m['sbert_similarity'] for m in all_metrics]),
                    'std': statistics.stdev([m['sbert_similarity'] for m in all_metrics]) if len(all_metrics) > 1 else 0
                },
                'retrieval_time': {
                    'mean': statistics.mean([m['retrieval_time'] for m in all_metrics]),
                    'std': statistics.stdev([m['retrieval_time'] for m in all_metrics]) if len(all_metrics) > 1 else 0
                }
            },
            'all_results': all_metrics
        }

        # Print summary
        print("\n" + "=" * 80)
        print("TechQA Test Summary")
        print("=" * 80)
        print(f"Mode: {'Adaptive Keyword Boost' if self.use_adaptive_keyword else 'Semantic Only (Baseline)'}")
        print(f"Examples: {summary['num_examples']}")
        print(f"\nOverall Performance:")
        print(f"  F1:              {summary['metrics']['f1']['mean']:.4f} (±{summary['metrics']['f1']['std']:.4f})")
        print(f"  ROUGE-L:         {summary['metrics']['rouge_l']['mean']:.4f} (±{summary['metrics']['rouge_l']['std']:.4f})")
        print(f"  SBERT:           {summary['metrics']['sbert_similarity']['mean']:.4f} (±{summary['metrics']['sbert_similarity']['std']:.4f})")
        print(f"  Retrieval Time:  {summary['metrics']['retrieval_time']['mean']:.3f}s (±{summary['metrics']['retrieval_time']['std']:.3f})")
        print("=" * 80)

        return summary


def main():
    parser = argparse.ArgumentParser(description='Test SimpleMem on TechQA dataset')
    parser.add_argument('--num-examples', type=int, default=30,
                       help='Number of examples to test (default: 30)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'validation', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--no-adaptive', action='store_true',
                       help='Disable adaptive keyword boost (baseline mode)')
    parser.add_argument('--local-embeddings', action='store_true',
                       help='Use local small embedding model (all-MiniLM-L6-v2)')
    parser.add_argument('--result-file', type=str, default=None,
                       help='File to save results JSON')
    parser.add_argument('--fusion-method', type=str, default='cc', choices=['cc', 'rrf', 'none'],
                       help='Fusion method: cc (Convex Combination), rrf (RRF), none (semantic only)')
    parser.add_argument('--cc-alpha', type=float, default=0.7,
                       help='Convex Combination alpha (0-1, higher=more semantic)')

    args = parser.parse_args()

    # Override embedding model if requested
    if args.local_embeddings:
        import config
        config.EMBEDDING_PROVIDER = "local"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.EMBEDDING_DIMENSION = 384
        print(f"Using local embeddings: {config.EMBEDDING_MODEL} ({config.EMBEDDING_DIMENSION}D)")

    # Initialize system
    print("Initializing SimpleMem system...")
    system = SimpleMemSystem(clear_db=True)

    # Configure fusion method
    if args.no_adaptive:
        system.hybrid_retriever.fusion_method = "none"
        use_adaptive = False
    else:
        system.hybrid_retriever.fusion_method = args.fusion_method
        system.hybrid_retriever.cc_alpha = args.cc_alpha
        use_adaptive = True

    print(f"Fusion method: {system.hybrid_retriever.fusion_method}")
    if system.hybrid_retriever.fusion_method == 'cc':
        print(f"CC Alpha: {args.cc_alpha}")

    # Create tester and run
    tester = TechQATester(system, use_adaptive_keyword=use_adaptive)
    results = tester.run_test(num_examples=args.num_examples, split=args.split)

    # Save results
    if args.result_file:
        # Remove non-serializable data
        save_results = {k: v for k, v in results.items() if k != 'all_results'}
        save_results['all_results'] = [
            {k: v for k, v in r.items() if k not in ['generated_answer', 'reference_answer']}
            for r in results['all_results']
        ]
        with open(args.result_file, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to {args.result_file}")


if __name__ == '__main__':
    main()
