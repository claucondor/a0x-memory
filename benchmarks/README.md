# Retrieval Quality Evaluation

Benchmark suite for evaluating hybrid search configurations using standard Information Retrieval metrics.

## Methodology

This evaluation follows established IR evaluation frameworks:

### Metrics

| Metric | Description | Reference |
|--------|-------------|-----------|
| **Recall@K** | Fraction of relevant documents retrieved within top-K | Standard IR |
| **Precision@K** | Fraction of top-K results that are relevant | Standard IR |
| **NDCG@K** | Normalized Discounted Cumulative Gain (position-aware) | Järvelin & Kekäläinen (2002) |
| **α-nDCG** | NDCG with novelty/diversity penalty | Clarke et al. (2008) |
| **MRR** | Mean Reciprocal Rank | Standard IR |
| **F1@K** | Harmonic mean of Precision@K and Recall@K | Standard IR |

### Dataset

**LoCoMo** (Long Context Memory) benchmark:
- 10 multi-session conversations
- ~200 QA pairs per conversation with ground truth evidence annotations
- Source: [snap-research/locomo](https://github.com/snap-research/locomo)

### Configurations Tested

1. **Baseline**: Semantic search only (dense retrieval)
2. **RRF**: Reciprocal Rank Fusion combining semantic + keyword + structured search
3. **RRF+MMR**: RRF with Maximal Marginal Relevance diversity reranking

## Usage

```bash
# Quick evaluation (2 samples, 30 questions each)
python benchmarks/retrieval_evaluation.py

# Full benchmark
python benchmarks/retrieval_evaluation.py --full

# Custom parameters
python benchmarks/retrieval_evaluation.py --samples 5 --questions 50 --k 5 10 20
```

## References

1. **Järvelin, K., & Kekäläinen, J.** (2002). Cumulated gain-based evaluation of IR techniques. *ACM Transactions on Information Systems*, 20(4), 422-446. https://dl.acm.org/doi/10.1145/582415.582418

2. **Clarke, C. L., Kolla, M., Cormack, G. V., Vechtomova, O., Ashkan, A., Büttcher, S., & MacKinnon, I.** (2008). Novelty and diversity in information retrieval evaluation. *Proceedings of the 31st annual international ACM SIGIR conference*, 659-666. https://dl.acm.org/doi/10.1145/1390334.1390446

3. **Carbonell, J., & Goldstein, J.** (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries. *Proceedings of the 21st annual international ACM SIGIR conference*, 335-336. https://dl.acm.org/doi/10.1145/290941.291025

4. **Maharana, A., et al.** (2024). LoCoMo: Long Context Memory benchmark. https://github.com/snap-research/locomo
