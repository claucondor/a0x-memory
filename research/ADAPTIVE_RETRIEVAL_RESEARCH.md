# Adaptive Retrieval Research for Conversational Memory Systems

**Date**: January 2025
**Context**: Investigating optimal retrieval strategies for a0x-memory system

---

## Executive Summary

Our benchmarks on LoCoMo dataset showed that hybrid search (RRF, Convex Combination) does not improve over pure semantic search for conversational memory. This document explores why and proposes an adaptive approach.

---

## 1. Benchmark Results

### LoCoMo Dataset (5 samples, 250 questions)

| Method | F1 | ROUGE-L | Notes |
|--------|-----|---------|-------|
| Baseline (semantic only) | **0.3059** | **0.3021** | Best overall |
| RRF Fusion | 0.2818 | 0.2759 | -7.9% worse |
| Convex Combination (α=0.7) | 0.2998 | 0.2987 | -2.0% worse |

### Alpha Tuning for Convex Combination (2 samples, 60 questions)

| Alpha | F1 | Interpretation |
|-------|-----|----------------|
| 0.7 | 0.3298 | Best alpha |
| 0.9 | 0.2999 | Too little keyword |
| 0.5 | 0.2822 | Too much keyword |
| 0.8 | 0.2539 | Inconsistent |

---

## 2. Why Hybrid Search Didn't Help

### LoCoMo Dataset Characteristics
- Long conversational memory (300 turns, 9K tokens average)
- 35 sessions per conversation
- Question types: single-hop, multi-hop, temporal, adversarial
- Questions are **semantic by nature**: "When did X happen?", "What did Y do?"

Reference: [Evaluating Very Long-Term Conversational Memory of LLM Agents](https://arxiv.org/abs/2402.17753)

### When Keyword Search Helps (Literature)
- Abbreviations: GAN, LLaMA, API
- Error codes: ECONNREFUSED, 404, 500
- Product codes: SKU-12345, 1099-MISC
- Technical terms: parseJWT, async/await
- Exact names in technical docs

Reference: [Optimizing RAG with Hybrid Search](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking)

### Conclusion
LoCoMo (conversational memory) ≠ Technical documentation RAG. Hybrid search helps with **technical precision**, not **conversational understanding**.

---

## 3. Retrieval Methods Taxonomy

### 3.1 Vector/Semantic Search
- **How**: Dense embeddings, cosine similarity
- **Good for**: Conceptual queries, paraphrasing, synonyms
- **Bad for**: Exact matches, codes, abbreviations

### 3.2 Keyword/Lexical Search (BM25)
- **How**: Term frequency, sparse vectors
- **Good for**: Exact terms, codes, names
- **Bad for**: Semantic understanding, paraphrasing

### 3.3 Graph Search (GraphRAG)
- **How**: Knowledge graphs, entity relationships, traversal
- **Good for**: Multi-hop reasoning, relationships, aggregations
- **Bad for**: Simple lookups

Reference: [GraphRAG by Microsoft Research](https://medium.com/data-science/how-to-implement-graph-rag-using-knowledge-graphs-and-vector-databases-60bb69a22759)

### 3.4 SQL/Structured Query
- **How**: Text-to-SQL, structured filters
- **Good for**: Aggregations, filters, exact metadata
- **Bad for**: Unstructured content

### 3.5 Multi-Modal
- **How**: Vision encoders, table understanding
- **Good for**: Images, diagrams, tables
- **Bad for**: Text-only systems

---

## 4. Query Routing Approaches

### 4.1 LLM-Based Routing
The LLM analyzes the query and decides which retriever to use.

**Pros**: Flexible, understands context
**Cons**: Latency, inconsistent, may not understand domain-specific terms

Reference: [Routing in RAG-Driven Applications](https://towardsdatascience.com/routing-in-rag-driven-applications-a685460a7220/)

### 4.2 Semantic Router
Uses embeddings to classify query type before retrieval.

**Pros**: Fast, consistent
**Cons**: Requires training examples

### 4.3 RouterRetriever (2024)
Mixture of domain-specific expert embedding models with routing mechanism.

**Performance**: +2.1 nDCG@10 over MSMARCO baseline

Reference: [RouterRetriever Paper](https://arxiv.org/abs/2409.02685)

### 4.4 RAGRouter (2025)
Contrastive learning framework that models knowledge shifts in RAG.

Reference: [RAGRouter Paper](https://arxiv.org/html/2505.23052v1)

---

## 5. Proposed Approach: Adaptive Keyword Detection

### Concept
Integrate keyword detection into the existing Planning phase. The LLM already analyzes queries - extend it to detect terms requiring exact match.

### Modified Planning Output
```json
{
  "required_info": ["function behavior", "error handling"],
  "queries": ["parseJWT function", "JWT error handling"],
  "exact_match_terms": ["parseJWT", "ECONNREFUSED"],
  "use_keyword_boost": true
}
```

### Logic
```
if use_keyword_boost and exact_match_terms:
    keyword_results = keyword_search(exact_match_terms)
    final_results = boost_matches(semantic_results, keyword_results)
else:
    final_results = semantic_results
```

### Benefits
1. No extra latency for conversational queries (majority)
2. Keyword boost only when LLM detects technical terms
3. Uses existing planning infrastructure
4. Configurable per-query, not global

---

## 6. Detection Heuristics

The planning LLM should flag `use_keyword_boost: true` when detecting:

| Pattern | Example | Reason |
|---------|---------|--------|
| camelCase | `parseJWT`, `getUserById` | Function names |
| snake_case | `user_id`, `api_key` | Variables, configs |
| SCREAMING_CASE | `ECONNREFUSED`, `API_KEY` | Constants, errors |
| Codes with numbers | `HTTP 404`, `error-1234` | Error codes |
| File extensions | `.env`, `config.yaml` | File references |
| Version numbers | `v2.1.0`, `node@18` | Versions |
| URLs/paths | `/api/users`, `https://` | Endpoints |

---

## 7. Implementation Plan

### Phase 1: Modify Planning Prompt
Add instruction to detect exact match terms.

### Phase 2: Conditional Keyword Search
Only execute keyword search when `use_keyword_boost: true`.

### Phase 3: Smart Boosting
Use targeted CC fusion only for documents matching exact terms.

### Phase 4: Benchmark
Test on LoCoMo (should maintain baseline) + synthetic technical queries (should improve).

---

## 8. References

### Papers
1. [Evaluating Very Long-Term Conversational Memory of LLM Agents](https://arxiv.org/abs/2402.17753) - LoCoMo dataset
2. [Blended RAG: Improving RAG Accuracy with Semantic Search and Hybrid Query-Based Retrievers](https://arxiv.org/html/2404.07220v1)
3. [RouterRetriever: Routing over a Mixture of Expert Embedding Models](https://arxiv.org/abs/2409.02685)
4. [RAGRouter: Learning to Route Queries to Multiple Retrieval-Augmented Language Models](https://arxiv.org/html/2505.23052v1)
5. [HybridRAG: Integrating Knowledge Graphs and Vector Retrieval](https://arxiv.org/html/2408.04948v1)

### Articles
1. [Optimizing RAG with Hybrid Search & Reranking](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking)
2. [Routing in RAG-Driven Applications](https://towardsdatascience.com/routing-in-rag-driven-applications-a685460a7220/)
3. [How to Build Helpful RAGs with Query Routing](https://towardsdatascience.com/rags-with-query-routing-5552e4e41c54/)
4. [GraphRAG Explained](https://medium.com/@zilliz_learn/graphrag-explained-enhancing-rag-with-knowledge-graphs-3312065f99e1)
5. [Advanced RAG Techniques - Neo4j](https://neo4j.com/blog/genai/advanced-rag-techniques/)

### Project Pages
1. [LoCoMo Benchmark](https://snap-research.github.io/locomo/)
2. [RAG-Anything Framework](https://github.com/HKUDS/RAG-Anything)

---

## 9. TechQA Benchmark Results (Technical Documentation)

### Dataset: IBM TechQA
- Source: IBM Developer forums + Technotes
- Size: 450 train, 160 validation examples
- Query types: Technical support, error codes, configuration, version-specific issues

### Results (20 examples)

| Method | F1 | ROUGE-L | SBERT |
|--------|-----|---------|-------|
| Baseline (semantic only) | 0.2339 | 0.1834 | 0.5549 |
| **Adaptive Keyword Boost** | **0.2448** | **0.1936** | 0.5520 |
| Improvement | **+4.7%** | **+5.6%** | -0.5% |

### Keyword Boost Activation
- **Enabled**: 14/20 queries (70%) - detected technical terms
- **Skipped**: 6/20 queries (30%) - conversational style

### Examples of Detected Terms
```
['ulimit', 'file descriptor']
['WPS 6.1.0.6', 'Oracle 12c', 'Oracle 11g']
['SPSS 24', 'Authorization failed.', 'End Of Transaction.']
['RDZ 9.0', 'HATS 9.0.0.0', 'IBM Rational Application Developer']
['CVE-2017-3156', 'Apache CXF']
```

---

## 10. Cross-Dataset Comparison

| Dataset | Type | Keyword Boost Rate | F1 Improvement |
|---------|------|-------------------|----------------|
| LoCoMo | Conversational | 0% (0/60) | N/A (not triggered) |
| TechQA | Technical | 70% (14/20) | **+4.7%** |

**Key Insight**: The adaptive approach correctly identifies when to use keyword boost:
- Conversational queries → semantic only (no penalty)
- Technical queries → keyword boost (measurable improvement)

---

## 11. LanceDB Native FTS Implementation (COMPLETED)

### Problem (Solved)
Our original keyword search did a **full table scan** (O(n)):
```python
for row in all_entries:
    if keyword in row["lossless_restatement"]:
        results.append(row)
```

### Solution Implemented
Upgraded to LanceDB native **BM25 Full-Text Search** with Tantivy index:
```python
# Auto-creates FTS index on first data insertion
self.table.create_fts_index("lossless_restatement", use_tantivy=True)

# BM25 search with scores
results = self.table.search(query).limit(top_k).to_list()
score = result.get("_score", 0.0)  # BM25 score
```

### FTS Benchmark Results (TechQA, 20 examples)

| Configuration | F1 | vs Old Baseline | Notes |
|---------------|-----|-----------------|-------|
| Old Baseline (manual scan) | 0.2339 | - | O(n) full scan |
| Old Adaptive (manual scan) | 0.2448 | +4.7% | Adaptive works |
| **FTS Baseline** | 0.2395 | +2.4% | BM25 ranking helps |
| **FTS Adaptive** | **0.2539** | **+8.5%** | Best combination |

### Key Findings
1. **FTS alone** improves baseline by +2.4% (better BM25 ranking vs manual scoring)
2. **Adaptive + FTS** gives additional +6.0% improvement
3. **Total improvement**: +8.5% over original baseline
4. Keyword boost triggered: 75% on TechQA (15/20 queries)

Reference: [LanceDB Full-Text Search](https://lancedb.com/docs/search/full-text-search/)

---

## 12. Final Conclusion

### Validated Hypothesis
**Adaptive keyword detection with native FTS works.** The system correctly:
1. Skips keyword boost for conversational queries (LoCoMo: 0% trigger)
2. Enables keyword boost for technical queries (TechQA: 75% trigger)
3. Improves F1 by **+8.5%** on technical datasets (with FTS)

### Recommended Configuration
```python
# Default for conversational memory
FUSION_METHOD = "cc"  # Convex Combination
CC_ALPHA = 0.7        # 70% semantic, 30% keyword boost
# Keyword boost is adaptive - only triggered when LLM detects exact match terms
# Uses LanceDB native BM25 FTS for keyword search
```

### Option 2 Results (Adaptive Alpha)

Tested categorical alpha based on keyword_importance:
- none → skip, low → 0.85, medium → 0.7, high → 0.5

| Query | Option 1 (α=0.7) | Option 2 (α=0.5 for high) | Diff |
|-------|------------------|---------------------------|------|
| Q2 (Exit function) | 0.444 | 0.408 | -0.036 |
| Q5 (CVE-2017-3156) | 0.168 | 0.143 | -0.025 |
| Q6 (BPM exception) | 0.126 | 0.107 | -0.019 |
| Q15 (open files) | 0.206 | 0.157 | -0.049 |

**Conclusion**: Adaptive alpha (Option 2) performed worse. Even for highly technical queries, α=0.5 gives too much weight to keywords and introduces noise. The semantic search with planning already captures technical terms well. **Fixed α=0.7 is recommended.**

---

## 13. Limitations and Recommendations

### Hardware Constraints
Due to hardware limitations (consumer laptop), our benchmarks used small sample sizes:
- **TechQA**: 20 examples (of 450 available)
- **LoCoMo**: 2-5 samples, 30-50 questions per run

### Statistical Significance
Results are **indicative but not conclusive**:
- Small sample sizes increase variance
- Differences of <5% may be within noise margin
- Option 2 vs Option 1 difference (-7.7%) is likely significant

### Recommendations for Future Testing
1. **Larger benchmarks**: Run full TechQA (450 examples) and LoCoMo (10 samples, 50 questions)
2. **Multiple runs**: Average results over 3-5 runs to reduce variance
3. **Diverse datasets**: Test on additional datasets (MS MARCO, BEIR subsets)
4. **Statistical tests**: Apply significance tests (t-test, bootstrap) to compare methods

### Future Improvements
1. **Option 3 Testing**: Heuristic keyword detection without LLM (faster, no API cost)
2. **GraphRAG**: For complex multi-hop relationship queries
3. **Reranker**: Cross-encoder reranking for final results

---

## 14. References

### Papers
1. [Evaluating Very Long-Term Conversational Memory of LLM Agents](https://arxiv.org/abs/2402.17753) - LoCoMo
2. [The TechQA Dataset](https://arxiv.org/abs/1911.02984) - IBM Technical QA
3. [RAGBench: Explainable Benchmark for RAG Systems](https://arxiv.org/abs/2407.11005)
4. [RouterRetriever: Routing over Expert Embedding Models](https://arxiv.org/abs/2409.02685)
5. [RAGRouter: Learning to Route Queries](https://arxiv.org/html/2505.23052v1)

### Technical Resources
1. [LanceDB Full-Text Search](https://lancedb.com/docs/search/full-text-search/)
2. [LanceDB Hybrid Search with BM25](https://lancedb.com/blog/hybrid-search-combining-bm25-and-semantic-search-for-better-results-with-lan-1358038fe7e6/)
3. [TechQA on HuggingFace](https://huggingface.co/datasets/rojagtap/tech-qa)

### Benchmarks Used
- LoCoMo: Conversational memory (SNAP Research)
- TechQA: Technical documentation (IBM Research)
