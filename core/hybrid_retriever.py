"""
Hybrid Retriever - Stage 3: Adaptive Query-Aware Retrieval with Pruning (Section 3.3)

Paper Reference: Section 3.3 - Adaptive Query-Aware Retrieval with Pruning
Implements:
- Hybrid scoring function S(q, m_k) aggregating semantic, lexical, and symbolic signals
- Query Complexity estimation C_q for adaptive retrieval depth
- Dynamic retrieval depth k_dyn = k_base * (1 + delta * C_q)
- Complexity-Aware Pruning to minimize token usage while maximizing accuracy

Extended for Unified Memory:
- Multi-table search for groups (group_memories, user_memories, interaction_memories)
- Cross-group search support
- Result fusion across multiple tables
"""
from typing import List, Optional, Dict, Any, Union
from models.memory_entry import MemoryEntry
from models.group_memory import GroupMemory, UserMemory, InteractionMemory, CrossGroupMemory
from utils.llm_client import LLMClient
from utils.structured_schemas import (
    QUERY_ANALYSIS_SCHEMA, SEARCH_QUERIES_SCHEMA,
    ANSWER_ADEQUACY_SCHEMA, ADDITIONAL_QUERIES_SCHEMA,
    INFORMATION_REQUIREMENTS_SCHEMA, TARGETED_QUERIES_SCHEMA,
    INFORMATION_COMPLETENESS_SCHEMA, MISSING_INFO_QUERIES_SCHEMA
)
# Note: unified_store (UnifiedMemoryStore or VectorStore) is injected via __init__
import config
import json
import re
from datetime import datetime, timedelta, timezone
import dateparser
import concurrent.futures
import numpy as np
from lancedb.rerankers import RRFReranker
from core.rerankers import get_cross_encoder_reranker
import pyarrow as pa


class HybridRetriever:
    """
    Hybrid Retriever - Stage 3: Adaptive Query-Aware Retrieval with Pruning

    Paper Reference: Section 3.3 - Adaptive Query-Aware Retrieval with Pruning

    Core Components:
    1. Query-aware retrieval across three structured layers:
       - Semantic Layer: Dense vector similarity
       - Lexical Layer: Sparse keyword matching (BM25)
       - Symbolic Layer: Metadata filtering
    2. Hybrid Scoring Function S(q, m_k): aggregates multi-layer signals
    3. Complexity-Aware Pruning: dynamic depth based on C_q
    4. Planning-based multi-query decomposition for comprehensive retrieval

    Extended for Unified Memory:
    5. Multi-table search for groups (search_all)
    6. Result fusion across multiple memory types
    """

    def __init__(
        self,
        llm_client: LLMClient,
        unified_store,  # UnifiedMemoryStore or VectorStore (backward compatible)
        semantic_top_k: int = None,
        keyword_top_k: int = None,
        structured_top_k: int = None,
        enable_planning: bool = True,
        enable_reflection: bool = True,
        max_reflection_rounds: int = 2,
        enable_parallel_retrieval: bool = True,
        max_retrieval_workers: int = 3,
        cc_alpha: float = None,
        user_profile_store=None,  # UserProfileStore for entity lookups
        group_profile_store=None,  # GroupProfileStore for group context
        group_summary_store=None  # GroupSummaryStore for hierarchical summaries
    ):
        self.llm_client = llm_client
        self.unified_store = unified_store

        # Backward compatibility
        self.vector_store = unified_store

        # Profile stores for lightweight entity system
        self.user_profile_store = user_profile_store
        self.group_profile_store = group_profile_store
        self.group_summary_store = group_summary_store

        self.semantic_top_k = semantic_top_k or config.SEMANTIC_TOP_K
        self.keyword_top_k = keyword_top_k or config.KEYWORD_TOP_K
        self.structured_top_k = structured_top_k or config.STRUCTURED_TOP_K

        # Use config values as default if not explicitly provided
        self.enable_planning = enable_planning if enable_planning is not None else getattr(config, 'ENABLE_PLANNING', True)
        self.enable_reflection = enable_reflection if enable_reflection is not None else getattr(config, 'ENABLE_REFLECTION', True)
        self.max_reflection_rounds = max_reflection_rounds if max_reflection_rounds is not None else getattr(config, 'MAX_REFLECTION_ROUNDS', 2)
        self.enable_parallel_retrieval = enable_parallel_retrieval if enable_parallel_retrieval is not None else getattr(config, 'ENABLE_PARALLEL_RETRIEVAL', True)
        self.max_retrieval_workers = max_retrieval_workers if max_retrieval_workers is not None else getattr(config, 'MAX_RETRIEVAL_WORKERS', 3)

        # Convex Combination alpha for adaptive keyword boost
        self.cc_alpha = cc_alpha if cc_alpha is not None else getattr(config, 'CC_ALPHA', 0.7)

        # Check if unified_store supports multi-table search
        self._supports_multi_table = hasattr(unified_store, 'search_all')

    def retrieve(self, query: str, enable_reflection: Optional[bool] = None) -> List[MemoryEntry]:
        """
        Execute retrieval with planning and optional reflection

        Args:
        - query: Search query
        - enable_reflection: Override the global reflection setting for this query
                           (useful for adversarial questions that shouldn't use reflection)

        Returns: List of relevant MemoryEntry
        """
        if self.enable_planning:
            return self._retrieve_with_planning(query, enable_reflection)
        else:
            # Fallback to simple semantic search
            return self._semantic_search(query)
    
    def _retrieve_with_planning(self, query: str, enable_reflection: Optional[bool] = None) -> List[MemoryEntry]:
        """
        Execute retrieval with intelligent planning process

        Args:
        - query: Search query
        - enable_reflection: Override reflection setting for this query
        """
        print(f"\n[Planning] Analyzing information requirements for: {query}")

        # Step 1+2: Combined analysis + query generation (single LLM call)
        information_plan, search_queries = self._plan_and_generate_queries(query)
        print(f"[Planning] Identified {len(information_plan.get('required_info', []))} information requirements")
        print(f"[Planning] Generated {len(search_queries)} targeted queries")
        
        # Step 3: Batch-encode queries, then execute searches (parallel or sequential)
        query_vectors = self.unified_store.embedding_model.encode_query(search_queries)
        if self.enable_parallel_retrieval and len(search_queries) > 1:
            all_results = self._execute_parallel_searches_with_vectors(search_queries, query_vectors)
        else:
            all_results = []
            for i, search_query in enumerate(search_queries, 1):
                print(f"[Search {i}] {search_query}")
                results = self.unified_store.search_memories(search_query, top_k=self.semantic_top_k, query_vector=query_vectors[i-1])
                all_results.extend(results)
        
        # Step 4: Merge and deduplicate results
        merged_results = self._merge_and_deduplicate_entries(all_results)
        print(f"[Planning] Found {len(merged_results)} unique results")

        # Step 5: Adaptive keyword boost (only when LLM detects exact match terms)
        use_keyword_boost = information_plan.get("use_keyword_boost", False)
        exact_match_terms = information_plan.get("exact_match_terms", [])

        if merged_results and use_keyword_boost and exact_match_terms:
            print(f"[Planning] Keyword boost enabled for terms: {exact_match_terms}")

            # Convert planning results to scored format
            total = len(merged_results)
            semantic_with_scores = [
                (entry, 1.0 - (i / max(total, 1)))
                for i, entry in enumerate(merged_results)
            ]

            # Get keyword results with BM25 FTS scores
            keyword_with_scores = self.vector_store.keyword_search_with_scores(
                exact_match_terms, top_k=self.keyword_top_k
            )

            if keyword_with_scores:
                merged_results = self._convex_combination_fusion(
                    semantic_with_scores,
                    keyword_with_scores,
                    alpha=self.cc_alpha
                )
                print(f"[Planning] CC fusion (α={self.cc_alpha}): {len(merged_results)} results")

        # Step 6: Optional reflection-based additional retrieval
        # Use override parameter if provided, otherwise use global setting
        should_use_reflection = enable_reflection if enable_reflection is not None else self.enable_reflection

        if should_use_reflection:
            merged_results = self._retrieve_with_intelligent_reflection(query, merged_results, information_plan)

        # Step 6.5: Apply temporal scoring before final rerank
        if merged_results:
            merged_results = self._apply_temporal_scoring(merged_results)

        # Final step: Rerank with cross-encoder (always run to ensure semantic ordering)
        if merged_results:
            merged_results = self._rerank_with_cross_encoder(query, merged_results, top_k=10)

        return merged_results
    
    def _retrieve_with_reflection(self, query: str, initial_results: List[MemoryEntry]) -> List[MemoryEntry]:
        """
        Execute reflection-based additional retrieval
        """
        current_results = initial_results
        
        for round_num in range(self.max_reflection_rounds):
            print(f"\n[Reflection Round {round_num + 1}] Checking if results are sufficient...")
            
            # Quick answer attempt with current results
            if not current_results:
                answer_status = "no_results"
            else:
                answer_status = self._check_answer_adequacy(query, current_results)
            
            if answer_status == "sufficient":
                print(f"[Reflection Round {round_num + 1}] Information is sufficient")
                break
            elif answer_status == "insufficient":
                print(f"[Reflection Round {round_num + 1}] Information is insufficient, generating additional queries...")
                
                # Generate additional targeted queries based on what's missing
                additional_queries = self._generate_additional_queries(query, current_results)
                print(f"[Reflection Round {round_num + 1}] Generated {len(additional_queries)} additional queries")
                
                # Execute additional searches (parallel or sequential)
                if self.enable_parallel_retrieval and len(additional_queries) > 1:
                    print(f"[Reflection Round {round_num + 1}] Executing {len(additional_queries)} additional queries in parallel")
                    additional_results = self._execute_parallel_additional_searches(additional_queries, round_num + 1)
                else:
                    additional_results = []
                    for i, add_query in enumerate(additional_queries, 1):
                        print(f"[Additional Search {i}] {add_query}")
                        results = self._semantic_search(add_query)
                        additional_results.extend(results)
                
                # Merge with existing results
                all_results = current_results + additional_results
                current_results = self._merge_and_deduplicate_entries(all_results)
                print(f"[Reflection Round {round_num + 1}] Total results: {len(current_results)}")
                
            else:  # "no_results"
                print(f"[Reflection Round {round_num + 1}] No results found, cannot continue reflection")
                break
        
        return current_results

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Use LLM to analyze query intent and extract structured information
        """
        prompt = f"""
Analyze the following query and extract key information:

Query: {query}

Please extract:
1. keywords: List of keywords (names, places, topic words, etc.)
2. persons: Person names mentioned
3. time_expression: Time expression (if any)
4. location: Location (if any)
5. entities: Entities (companies, products, etc.)

Return in JSON format:
```json
{{
  "keywords": ["keyword1", "keyword2", ...],
  "persons": ["name1", "name2", ...],
  "time_expression": "time expression or null",
  "location": "location or null",
  "entities": ["entity1", ...]
}}
```

Return ONLY JSON, no other content.
"""

        messages = [
            {"role": "system", "content": "You are a query analysis assistant. You must output valid JSON format."},
            {"role": "user", "content": prompt}
        ]

        # Retry up to 3 times
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm_client.chat_completion(
                    messages,
                    temperature=0.1,
                    response_format=QUERY_ANALYSIS_SCHEMA
                )
                try:
                    analysis = json.loads(response)
                except (json.JSONDecodeError, TypeError):
                    analysis = self.llm_client.extract_json(response)
                return analysis
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Query analysis attempt {attempt + 1}/{max_retries} failed: {e}. Retrying...")
                else:
                    print(f"Query analysis failed after {max_retries} attempts: {e}")
                    # Return default values
                    return {
                        "keywords": [query],
                        "persons": [],
                        "time_expression": None,
                        "location": None,
                        "entities": []
                    }

    def _semantic_search(self, query: str) -> List[MemoryEntry]:
        """
        Semantic Layer Retrieval

        Paper Reference: Section 3.3 - Part of hybrid scoring function S(q, m_k)
        Retrieves based on dense vector similarity: λ₁ · cos(e_q, v_k)
        """
        return self.vector_store.semantic_search(query, top_k=self.semantic_top_k)

    def _keyword_search(
        self,
        query: str,
        query_analysis: Dict[str, Any]
    ) -> List[MemoryEntry]:
        """
        Lexical Layer Retrieval

        Paper Reference: Section 3.3 - Part of hybrid scoring function S(q, m_k)
        Retrieves based on sparse keyword matching: λ₂ · BM25(q_lex, S_k)
        """
        keywords = query_analysis.get("keywords", [])
        if not keywords:
            # If no keywords extracted, use query itself
            keywords = [query]

        return self.vector_store.keyword_search(keywords, top_k=self.keyword_top_k)

    def _structured_search(self, query_analysis: Dict[str, Any]) -> List[MemoryEntry]:
        """
        Symbolic Layer Retrieval

        Paper Reference: Section 3.3 - Part of hybrid scoring function S(q, m_k)
        Hard filter based on symbolic constraints: γ · 𝕀(R_k ⊨ C_meta)
        """
        persons = query_analysis.get("persons", [])
        location = query_analysis.get("location")
        entities = query_analysis.get("entities", [])
        time_expression = query_analysis.get("time_expression")

        # Parse time range
        timestamp_range = None
        if time_expression:
            timestamp_range = self._parse_time_range(time_expression)

        # Return empty if no structured conditions
        if not any([persons, location, entities, timestamp_range]):
            return []

        # Execute structured search
        return self.vector_store.structured_search(
            persons=persons if persons else None,
            location=location,
            entities=entities if entities else None,
            timestamp_range=timestamp_range,
            top_k=self.structured_top_k
        )

    def _parse_time_range(self, time_expression: str) -> Optional[tuple]:
        """
        Parse time expression to time range

        Examples:
        - "last week" -> (last Monday 00:00, last Sunday 23:59)
        - "November 15" -> (2025-11-15 00:00, 2025-11-15 23:59)
        """
        try:
            # Use dateparser to parse
            parsed_date = dateparser.parse(
                time_expression,
                settings={'PREFER_DATES_FROM': 'past'}
            )

            if parsed_date:
                # Generate time range (for the day)
                start_time = parsed_date.replace(hour=0, minute=0, second=0)
                end_time = parsed_date.replace(hour=23, minute=59, second=59)

                # Expand range for weekly expressions
                if "week" in time_expression.lower() or "周" in time_expression:
                    start_time = start_time - timedelta(days=7)
                    end_time = end_time + timedelta(days=7)

                return (
                    start_time.isoformat(),
                    end_time.isoformat()
                )
        except Exception as e:
            print(f"Time parsing failed: {e}")

        return None

    def _merge_and_deduplicate(
        self,
        results: Dict[str, List[MemoryEntry]]
    ) -> List[MemoryEntry]:
        """
        Merge multi-path retrieval results and deduplicate
        """
        seen_ids = set()
        merged = []

        # Merge by priority (structured > semantic > keyword)
        for source in ['structured', 'semantic', 'keyword']:
            for entry in results.get(source, []):
                if entry.entry_id not in seen_ids:
                    seen_ids.add(entry.entry_id)
                    merged.append(entry)

        return merged
    
    def _generate_search_queries(self, query: str) -> List[str]:
        """
        Generate multiple search queries for comprehensive retrieval
        """
        prompt = f"""
You are helping with information retrieval. Given a user question, generate multiple search queries that would help find comprehensive information to answer the question.

Original Question: {query}

Please generate 3-5 different search queries that cover various aspects and angles of this question. Each query should be focused and specific.

Guidelines:
1. Include the original question as one query
2. Break down complex questions into component parts
3. Consider synonyms and alternative phrasings
4. Think about related concepts that might be relevant
5. Consider temporal, spatial, or contextual variations

Return your response in JSON format:
```json
{{
  "queries": [
    "search query 1",
    "search query 2", 
    "search query 3",
    ...
  ]
}}
```

Return ONLY the JSON, no other text.
"""
        
        messages = [
            {"role": "system", "content": "You are a search query generation assistant. You must output valid JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.llm_client.chat_completion(
                messages,
                temperature=0.3,
                response_format=SEARCH_QUERIES_SCHEMA
            )
            try:
                result = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                result = self.llm_client.extract_json(response)
            queries = result.get("queries", [query])
            
            # Ensure original query is included
            if query not in queries:
                queries.insert(0, query)
                
            return queries
            
        except Exception as e:
            print(f"Failed to generate search queries: {e}")
            # Fallback to original query
            return [query]
    
    def _merge_and_deduplicate_entries(self, entries: List[MemoryEntry]) -> List[MemoryEntry]:
        """
        Merge and deduplicate memory entries by entry_id
        """
        seen_ids = set()
        merged = []
        
        for entry in entries:
            if entry.entry_id not in seen_ids:
                seen_ids.add(entry.entry_id)
                merged.append(entry)
        
        return merged

    def _convex_combination_fusion(
        self,
        semantic_results: List[tuple],
        keyword_results: List[tuple],
        alpha: float = 0.7
    ) -> List[MemoryEntry]:
        """
        Convex Combination (CC) fusion for hybrid retrieval.

        Formula: S_final = α·S_sem + (1-α)·S_kw

        Args:
            semantic_results: List of (MemoryEntry, score) from semantic search
            keyword_results: List of (MemoryEntry, score) from keyword search
            alpha: Weight for semantic scores (default 0.7)

        Returns:
            List of MemoryEntry sorted by fused score
        """
        semantic_scores: Dict[str, float] = {}
        keyword_scores: Dict[str, float] = {}
        entry_map: Dict[str, MemoryEntry] = {}

        for entry, score in semantic_results:
            semantic_scores[entry.entry_id] = score
            entry_map[entry.entry_id] = entry

        for entry, score in keyword_results:
            keyword_scores[entry.entry_id] = score
            if entry.entry_id not in entry_map:
                entry_map[entry.entry_id] = entry

        all_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())

        fused_scores: Dict[str, float] = {}
        for entry_id in all_ids:
            sem_score = semantic_scores.get(entry_id, 0.0)
            kw_score = keyword_scores.get(entry_id, 0.0)

            if entry_id in semantic_scores and entry_id in keyword_scores:
                fused_scores[entry_id] = alpha * sem_score + (1 - alpha) * kw_score
            elif entry_id in semantic_scores:
                fused_scores[entry_id] = alpha * sem_score
            else:
                fused_scores[entry_id] = (1 - alpha) * kw_score

        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        return [entry_map[entry_id] for entry_id in sorted_ids]

    def _check_answer_adequacy(self, query: str, contexts: List[MemoryEntry]) -> str:
        """
        Check if current contexts are sufficient to answer the query
        Returns: "sufficient", "insufficient", or "no_results"
        """
        if not contexts:
            return "no_results"
        
        # Format contexts
        context_str = self._format_contexts_for_check(contexts)
        
        prompt = f"""
You are evaluating whether the provided context contains sufficient information to answer a user question.

Question: {query}

Context:
{context_str}

Please evaluate whether the context contains enough information to provide a meaningful, accurate answer to the question.

Consider these criteria:
1. Does the context directly address the question being asked?
2. Are there key details necessary to answer the question?
3. Is the information specific enough to avoid vague responses?

Return your evaluation in JSON format:
```json
{{
  "assessment": "sufficient" OR "insufficient",
  "reasoning": "Brief explanation of why the context is or isn't sufficient",
  "missing_info": ["list", "of", "missing", "information"] (only if insufficient)
}}
```

Return ONLY the JSON, no other text.
"""
        
        messages = [
            {"role": "system", "content": "You are an information adequacy evaluator. You must output valid JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.llm_client.chat_completion(
                messages,
                temperature=0.1,
                response_format=ANSWER_ADEQUACY_SCHEMA
            )
            try:
                result = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                result = self.llm_client.extract_json(response)
            return result.get("assessment", "insufficient")

        except Exception as e:
            print(f"Failed to check answer adequacy: {e}")
            return "insufficient"
    
    def _generate_additional_queries(self, original_query: str, current_contexts: List[MemoryEntry]) -> List[str]:
        """
        Generate additional targeted queries based on what's missing
        """
        context_str = self._format_contexts_for_check(current_contexts)
        
        prompt = f"""
Based on the original question and current available information, generate additional specific search queries that would help find the missing information needed to answer the question completely.

Original Question: {original_query}

Current Available Information:
{context_str}

Analyze what specific information is still missing and generate 2-4 targeted search queries that would help find this missing information.

The queries should be:
1. Specific and focused on the missing information
2. Different from the original question
3. Likely to find complementary information

Return your response in JSON format:
```json
{{
  "missing_analysis": "Brief analysis of what's missing",
  "additional_queries": [
    "specific search query 1",
    "specific search query 2",
    ...
  ]
}}
```

Return ONLY the JSON, no other text.
"""
        
        messages = [
            {"role": "system", "content": "You are a search strategy assistant. You must output valid JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.llm_client.chat_completion(
                messages,
                temperature=0.3,
                response_format=ADDITIONAL_QUERIES_SCHEMA
            )
            try:
                result = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                result = self.llm_client.extract_json(response)
            return result.get("additional_queries", [])

        except Exception as e:
            print(f"Failed to generate additional queries: {e}")
            return []
    
    def _format_contexts_for_check(self, contexts: List[MemoryEntry]) -> str:
        """
        Format contexts for adequacy checking (more concise than full format)
        """
        formatted = []
        for i, entry in enumerate(contexts, 1):
            parts = [f"[Info {i}] {entry.lossless_restatement}"]
            if entry.timestamp:
                parts.append(f"Time: {entry.timestamp}")
            formatted.append(" | ".join(parts))
        
        return "\n".join(formatted)
    
    def _execute_parallel_searches_with_vectors(self, search_queries: List[str], query_vectors) -> List[MemoryEntry]:
        """Execute multiple search queries in parallel using pre-computed vectors."""
        print(f"[Parallel Search] Executing {len(search_queries)} queries in parallel with pre-computed vectors")
        all_results = []

        def _search_worker(args):
            i, query, vector = args
            print(f"[Search {i}] {query}")
            return self.unified_store.search_memories(query, top_k=self.semantic_top_k, query_vector=vector)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_retrieval_workers) as executor:
                tasks = [(i+1, q, query_vectors[i]) for i, q in enumerate(search_queries)]
                future_to_idx = {executor.submit(_search_worker, t): t[0] for t in tasks}
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results = future.result()
                        all_results.extend(results)
                        print(f"[Parallel Search] Query {idx} completed: {len(results)} results")
                    except Exception as e:
                        print(f"[Parallel Search] Query {idx} failed: {e}")
        except Exception as e:
            print(f"[Parallel Search] Failed: {e}. Falling back to sequential.")
            for i, query in enumerate(search_queries):
                results = self.unified_store.search_memories(query, top_k=self.semantic_top_k, query_vector=query_vectors[i])
                all_results.extend(results)

        return all_results

    def _execute_parallel_searches(self, search_queries: List[str]) -> List[MemoryEntry]:
        """
        Execute multiple search queries in parallel using ThreadPoolExecutor
        """
        print(f"[Parallel Search] Executing {len(search_queries)} queries in parallel with {self.max_retrieval_workers} workers")
        all_results = []
        
        try:
            # Use ThreadPoolExecutor for parallel retrieval
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_retrieval_workers) as executor:
                # Submit all search tasks
                future_to_query = {}
                for i, query in enumerate(search_queries, 1):
                    future = executor.submit(self._semantic_search_worker, query, i)
                    future_to_query[future] = (query, i)
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_query):
                    query, query_num = future_to_query[future]
                    try:
                        results = future.result()
                        all_results.extend(results)
                        print(f"[Parallel Search] Query {query_num} completed: {len(results)} results")
                    except Exception as e:
                        print(f"[Parallel Search] Query {query_num} failed: {e}")
                        
        except Exception as e:
            print(f"[Parallel Search] Parallel execution failed: {e}. Falling back to sequential search...")
            # Fallback to sequential processing
            for i, query in enumerate(search_queries, 1):
                try:
                    print(f"[Sequential Search {i}] {query}")
                    results = self._semantic_search(query)
                    all_results.extend(results)
                except Exception as search_e:
                    print(f"[Sequential Search {i}] Failed: {search_e}")
        
        return all_results
    
    def _semantic_search_worker(self, query: str, query_num: int) -> List[MemoryEntry]:
        """
        Worker function for parallel semantic search
        """
        print(f"[Search {query_num}] {query}")
        return self._semantic_search(query)
    
    def _execute_parallel_additional_searches(self, additional_queries: List[str], round_num: int) -> List[MemoryEntry]:
        """
        Execute additional reflection queries in parallel
        """
        all_results = []
        
        try:
            # Use ThreadPoolExecutor for parallel retrieval
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_retrieval_workers) as executor:
                # Submit all search tasks
                future_to_query = {}
                for i, query in enumerate(additional_queries, 1):
                    future = executor.submit(self._additional_search_worker, query, i, round_num)
                    future_to_query[future] = (query, i)
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_query):
                    query, query_num = future_to_query[future]
                    try:
                        results = future.result()
                        all_results.extend(results)
                        print(f"[Reflection Round {round_num}] Additional query {query_num} completed: {len(results)} results")
                    except Exception as e:
                        print(f"[Reflection Round {round_num}] Additional query {query_num} failed: {e}")
                        
        except Exception as e:
            print(f"[Reflection Round {round_num}] Parallel execution failed: {e}. Falling back to sequential search...")
            # Fallback to sequential processing
            for i, query in enumerate(additional_queries, 1):
                try:
                    print(f"[Additional Search {i}] {query}")
                    results = self._semantic_search(query)
                    all_results.extend(results)
                except Exception as search_e:
                    print(f"[Additional Search {i}] Failed: {search_e}")
        
        return all_results
    
    def _additional_search_worker(self, query: str, query_num: int, round_num: int) -> List[MemoryEntry]:
        """
        Worker function for parallel additional search in reflection
        """
        print(f"[Additional Search {query_num}] {query}")
        return self._semantic_search(query)
    
    def _analyze_information_requirements(self, query: str) -> Dict[str, Any]:
        """
        Query Complexity Estimation C_q

        Paper Reference: Section 3.3 - Eq. (8)
        Analyzes query complexity to determine minimal information requirements
        and optimal retrieval depth k_dyn
        """
        prompt = f"""
Analyze the following question and determine what specific information is required to answer it comprehensively.

Question: {query}

Think step by step:
1. What type of question is this? (factual, temporal, relational, explanatory, etc.)
2. What key entities, events, or concepts need to be identified?
3. What relationships or connections need to be established?
4. What minimal set of information pieces would be sufficient to answer this question?
5. Are there technical terms requiring exact lexical matching?

Return your analysis in JSON format:
```json
{{
  "question_type": "type of question",
  "key_entities": ["entity1", "entity2", ...],
  "required_info": [
    {{
      "info_type": "what kind of information",
      "description": "specific information needed",
      "priority": "high/medium/low"
    }}
  ],
  "relationships": ["relationship1", "relationship2", ...],
  "minimal_queries_needed": 2,
  "exact_match_terms": [],
  "use_keyword_boost": false
}}
```

For exact_match_terms, include ONLY terms requiring exact lexical matching:
- Function/method names: parseJWT, get_user_id
- Error codes: ECONNREFUSED, CVE-2017-3156
- Version numbers: v2.1.0, Oracle 12c
- File names: config.yaml, .env

Set use_keyword_boost=true ONLY if exact_match_terms is non-empty.
For conversational queries about people/events, leave both fields as defaults.

Return ONLY the JSON, no other text.
"""
        
        messages = [
            {"role": "system", "content": "You are an intelligent information requirement analyst. You must output valid JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.llm_client.chat_completion(
                messages,
                temperature=0.2,
                response_format=INFORMATION_REQUIREMENTS_SCHEMA
            )
            try:
                result = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                result = self.llm_client.extract_json(response)
            return result

        except Exception as e:
            print(f"Failed to analyze information requirements: {e}")
            # Fallback to simple analysis
            return {
                "question_type": "general",
                "key_entities": [query],
                "required_info": [{"info_type": "general", "description": "relevant information", "priority": "high"}],
                "relationships": [],
                "minimal_queries_needed": 1,
                "exact_match_terms": [],
                "use_keyword_boost": False
            }
    
    def _generate_targeted_queries(self, original_query: str, information_plan: Dict[str, Any]) -> List[str]:
        """
        Generate minimal targeted queries based on information requirements analysis
        """
        prompt = f"""
Based on the information requirements analysis, generate the minimal set of targeted search queries needed to gather the required information.

Original Question: {original_query}

Information Requirements Analysis:
- Question Type: {information_plan.get('question_type', 'general')}
- Key Entities: {information_plan.get('key_entities', [])}
- Required Information: {information_plan.get('required_info', [])}
- Relationships: {information_plan.get('relationships', [])}
- Minimal Queries Needed: {information_plan.get('minimal_queries_needed', 1)}

Generate the minimal set of search queries that would efficiently gather all the required information. Each query should be focused and specific to retrieve distinct types of information.

Guidelines:
1. Always include the original query as one option
2. Generate only the minimal necessary queries (usually 1-3)
3. Each query should target a specific information requirement
4. Avoid redundant or overlapping queries
5. Focus on efficiency - fewer, more targeted queries are better

Return your response in JSON format:
```json
{{
  "reasoning": "Brief explanation of the query strategy",
  "queries": [
    "targeted query 1",
    "targeted query 2",
    ...
  ]
}}
```

Return ONLY the JSON, no other text.
"""
        
        messages = [
            {"role": "system", "content": "You are a query generation specialist. You must output valid JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.llm_client.chat_completion(
                messages,
                temperature=0.3,
                response_format=TARGETED_QUERIES_SCHEMA
            )
            try:
                result = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                result = self.llm_client.extract_json(response)
            queries = result.get("queries", [original_query])
            
            # Ensure original query is included and limit to reasonable number
            if original_query not in queries:
                queries.insert(0, original_query)
            
            # Limit to max 4 queries for efficiency
            queries = queries[:4]
            
            print(f"[Planning] Strategy: {result.get('reasoning', 'Generate targeted queries')}")
            return queries
            
        except Exception as e:
            print(f"Failed to generate targeted queries: {e}")
            # Fallback to original query
            return [original_query]
    
    def _plan_and_generate_queries(self, query: str) -> tuple:
        """
        Combined planning: analyze information requirements AND generate targeted queries
        in a single LLM call instead of two sequential calls.

        Returns:
            Tuple of (information_plan dict, list of search queries)
        """
        prompt = f"""
Analyze the following question and generate targeted search queries to find the answer in a PERSONAL MEMORY STORE (conversations, facts shared by users, user profiles). This is NOT a web search.

Question: {query}

Think step by step:
1. What type of question is this? (factual, temporal, relational, etc.)
2. What key entities need to be identified?
3. What minimal set of search queries would find the relevant memories?
4. Are there technical terms requiring exact lexical matching?

Return in JSON format:
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
  "exact_match_terms": [],
  "use_keyword_boost": false,
  "queries": [
    "targeted search query 1",
    "targeted search query 2"
  ]
}}
```

For exact_match_terms, include ONLY terms requiring exact lexical matching (function names, error codes, version numbers, file names).
Set use_keyword_boost=true ONLY if exact_match_terms is non-empty.

For queries: generate 1-3 focused queries. Always include the original question. Each query should target a distinct information need. Keep queries natural and conversational (these search a memory store, NOT the web).

Return ONLY the JSON, no other text.
"""

        messages = [
            {"role": "system", "content": "You are a memory retrieval planner. You must output valid JSON format."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.llm_client.chat_completion(
                messages,
                temperature=0.2,
                response_format=None  # Let it return raw JSON
            )
            try:
                result = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                result = self.llm_client.extract_json(response)

            queries = result.get("queries", [query])
            if query not in queries:
                queries.insert(0, query)
            queries = queries[:4]

            # Build information_plan from the combined response
            information_plan = {
                "question_type": result.get("question_type", "general"),
                "key_entities": result.get("key_entities", []),
                "required_info": result.get("required_info", []),
                "exact_match_terms": result.get("exact_match_terms", []),
                "use_keyword_boost": result.get("use_keyword_boost", False),
            }

            return information_plan, queries

        except Exception as e:
            print(f"Failed combined planning: {e}")
            return {
                "question_type": "general",
                "key_entities": [],
                "required_info": [{"info_type": "general", "description": "relevant information", "priority": "high"}],
                "exact_match_terms": [],
                "use_keyword_boost": False,
            }, [query]

    def _retrieve_with_intelligent_reflection(self, query: str, initial_results: List[MemoryEntry], information_plan: Dict[str, Any]) -> List[MemoryEntry]:
        """
        Execute intelligent reflection-based additional retrieval.
        Limited to 1 round max and skips if initial results already have good coverage.
        """
        current_results = initial_results

        # Skip reflection entirely if we already have a decent number of results
        if len(current_results) >= 5:
            print(f"\n[Intelligent Reflection] Skipping - already have {len(current_results)} results")
            return current_results

        # Only do 1 round of reflection (the second round almost never helps)
        max_rounds = min(self.max_reflection_rounds, 1)

        for round_num in range(max_rounds):
            print(f"\n[Intelligent Reflection Round {round_num + 1}] Analyzing information completeness...")

            if not current_results:
                completeness_status = "no_results"
                coverage = 0
            else:
                completeness_status, coverage = self._analyze_information_completeness_fast(query, current_results, information_plan)

            if completeness_status == "complete":
                print(f"[Intelligent Reflection Round {round_num + 1}] Information is complete")
                break
            elif completeness_status == "incomplete":
                print(f"[Intelligent Reflection Round {round_num + 1}] Information is incomplete, generating targeted additional queries...")

                additional_queries = self._generate_missing_info_queries(query, current_results, information_plan)
                print(f"[Intelligent Reflection Round {round_num + 1}] Generated {len(additional_queries)} targeted queries")

                if self.enable_parallel_retrieval and len(additional_queries) > 1:
                    print(f"[Intelligent Reflection Round {round_num + 1}] Executing {len(additional_queries)} queries in parallel")
                    additional_results = self._execute_parallel_additional_searches(additional_queries, round_num + 1)
                else:
                    additional_results = []
                    for i, add_query in enumerate(additional_queries, 1):
                        print(f"[Additional Search {i}] {add_query}")
                        results = self._semantic_search(add_query)
                        additional_results.extend(results)

                all_results = current_results + additional_results
                current_results = self._merge_and_deduplicate_entries(all_results)
                print(f"[Intelligent Reflection Round {round_num + 1}] Total results: {len(current_results)}")

            else:  # "no_results"
                print(f"[Intelligent Reflection Round {round_num + 1}] No results found, cannot continue reflection")
                break

        return current_results
    
    def _analyze_information_completeness_fast(self, query: str, current_results: List[MemoryEntry], information_plan: Dict[str, Any]) -> tuple:
        """
        Fast completeness check. Returns (status, coverage_percentage).
        If coverage >= 70%, considers it complete.
        """
        if not current_results:
            return ("no_results", 0)

        context_str = self._format_contexts_for_check(current_results)
        required_info = information_plan.get('required_info', [])

        prompt = f"""
You are checking if a PERSONAL MEMORY STORE has enough information to answer a question.
This is NOT a web search - we only have memories from personal conversations.

Question: {query}

Required Information Types: {required_info}

Current Available Information:
{context_str}

Does the available information contain enough to provide a reasonable answer?
If the memories mention relevant people, facts, or events - that's sufficient.
Don't mark as incomplete just because the information isn't encyclopedic.

Return in JSON:
```json
{{
  "assessment": "complete" OR "incomplete",
  "coverage_percentage": 85,
  "reasoning": "Brief explanation"
}}
```

Return ONLY JSON.
"""

        messages = [
            {"role": "system", "content": "You are evaluating memory store completeness. Output valid JSON."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.llm_client.chat_completion(
                messages, temperature=0.1,
                response_format=INFORMATION_COMPLETENESS_SCHEMA
            )
            try:
                result = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                result = self.llm_client.extract_json(response)

            coverage = result.get("coverage_percentage", 0)
            assessment = result.get("assessment", "incomplete")

            # Override: if coverage >= 70%, consider complete
            if coverage >= 70:
                assessment = "complete"

            print(f"[Intelligent Reflection] Coverage: {coverage}% - {result.get('reasoning', '')}")
            return (assessment, coverage)

        except Exception as e:
            print(f"Failed to analyze completeness: {e}")
            return ("incomplete", 0)

    def _analyze_information_completeness(self, query: str, current_results: List[MemoryEntry], information_plan: Dict[str, Any]) -> str:
        """
        Analyze if current results provide complete information to answer the query
        """
        if not current_results:
            return "no_results"
        
        context_str = self._format_contexts_for_check(current_results)
        required_info = information_plan.get('required_info', [])
        
        prompt = f"""
Analyze whether the provided information is sufficient to completely answer the original question, based on the identified information requirements.

Original Question: {query}

Required Information Types: {required_info}

Current Available Information:
{context_str}

Evaluate whether:
1. All required information types are addressed
2. The information is complete enough to provide a comprehensive answer
3. Any critical gaps remain that would prevent a satisfactory answer

Return your evaluation in JSON format:
```json
{{
  "assessment": "complete" OR "incomplete",
  "reasoning": "Brief explanation of completeness assessment",
  "missing_info_types": ["list", "of", "missing", "information", "types"],
  "coverage_percentage": 85
}}
```

Return ONLY the JSON, no other text.
"""
        
        messages = [
            {"role": "system", "content": "You are an information completeness evaluator. You must output valid JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.llm_client.chat_completion(
                messages,
                temperature=0.1,
                response_format=INFORMATION_COMPLETENESS_SCHEMA
            )
            try:
                result = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                result = self.llm_client.extract_json(response)
            assessment = result.get("assessment", "incomplete")
            coverage = result.get("coverage_percentage", 0)
            
            print(f"[Intelligent Reflection] Coverage: {coverage}% - {result.get('reasoning', '')}")
            return assessment
            
        except Exception as e:
            print(f"Failed to analyze information completeness: {e}")
            return "incomplete"
    
    def _generate_missing_info_queries(self, original_query: str, current_results: List[MemoryEntry], information_plan: Dict[str, Any]) -> List[str]:
        """
        Generate targeted queries to find missing information
        """
        context_str = self._format_contexts_for_check(current_results)
        required_info = information_plan.get('required_info', [])
        
        prompt = f"""
You are searching a PERSONAL MEMORY STORE containing conversations and facts shared by users. This is NOT a web search engine.

Based on the original question and currently available memories, generate 1-3 additional search queries to find missing information.

Original Question: {original_query}

Required Information Types: {required_info}

Currently Available Memories:
{context_str}

IMPORTANT: Generate queries that would match memories from conversations. Use natural language like:
- "Who mentioned working on X?"
- "What did users say about Y?"
- "experiences with Z"

Do NOT generate web-style queries like "site:linkedin.com" or "demographics of X".

Return in JSON:
```json
{{
  "missing_analysis": "Brief analysis of what's missing",
  "targeted_queries": [
    "natural conversational query 1",
    "natural conversational query 2"
  ]
}}
```

Return ONLY JSON.
"""
        
        messages = [
            {"role": "system", "content": "You are a missing information query generator. You must output valid JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.llm_client.chat_completion(
                messages,
                temperature=0.3,
                response_format=MISSING_INFO_QUERIES_SCHEMA
            )
            try:
                result = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                result = self.llm_client.extract_json(response)
            queries = result.get("targeted_queries", [])

            print(f"[Intelligent Reflection] Missing info: {result.get('missing_analysis', 'Unknown')}")
            return queries

        except Exception as e:
            print(f"Failed to generate missing info queries: {e}")
            return []

    # ============================================================
    # Multi-Table Retrieval (Extended for Unified Memory)
    # ============================================================

    def retrieve_for_context(
        self,
        query: str,
        context: Dict[str, Any],
        limit_per_table: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve from all relevant tables using context-aware access control.

        This is the main entry point for group/multi-table retrieval.
        Uses planning + multi-query fan-out + optional BM25 keyword boost.

        Context-Aware Access:
        - DM: Returns DM memories + shareable group memories + cross-group + user facts
        - Group: Returns ONLY that group's memories + speaker's facts

        Args:
            query: Search query
            context: Dict with group_id, user_id, platform, etc.
            limit_per_table: Max results per table

        Returns:
            Dict with results from each table type and formatted context string
        """
        group_id = context.get('group_id')
        user_id = context.get('user_id')
        is_dm_with_group_reference = context.get('is_dm_with_group_reference', False)

        # Determine if this is a group context or DM context
        # Important: if is_dm_with_group_reference is True, this is a DM asking about a group,
        # so we should take the DM path (which will search in the reference group)
        is_group = group_id is not None and not group_id.startswith('dm_') and not is_dm_with_group_reference

        # Fetch conversation summary
        conversation_summary = self._fetch_conversation_summary(group_id, user_id)

        # DM path: use context-aware retrieval
        if not is_group or not self._supports_multi_table:
            results = self._retrieve_for_dm(query, user_id, context, limit_per_table)
            return {
                **results,
                "conversation_summary": conversation_summary,
            }

        # Group path: intelligent pipeline with multi-table fan-out
        print(f"\n[retrieve_for_context] Intelligent group retrieval for: {query}")
        involved_users = context.get('involved_users', [])
        return self._retrieve_for_group(query, group_id, user_id, context, limit_per_table, conversation_summary, involved_users)

    def _retrieve_for_dm(
        self,
        query: str,
        user_id: str,
        context: Dict[str, Any],
        limit: int
    ) -> Dict[str, Any]:
        """
        Context-aware retrieval for DM conversations.

        Access Rules for DM with User X:
        - SEES: DM history with X, shareable group memories from X, cross-group memories, user facts
        - NOT: DMs of other users, non-shareable group memories

        Args:
            query: Search query
            user_id: User ID (format: platform:user_id)
            context: Full context dict
            limit: Max results per source

        Returns:
            Dict with dm_memories, shareable_context, cross_group_memories, user_facts, formatted_context
        """
        print(f"\n[retrieve_for_context] DM retrieval for user {user_id}: {query}")

        # Step 1: Combined planning + query generation
        information_plan, search_queries = self._plan_and_generate_queries(query)
        print(f"[DM Retrieval] Generated {len(search_queries)} targeted queries")

        top_k = self.semantic_top_k

        # Step 2: Batch-encode queries
        query_vectors = self.unified_store.embedding_model.encode_query(search_queries)
        query_vector_map = {sq: query_vectors[i] for i, sq in enumerate(search_queries)}

        # Step 3: Search DM memories (from dm_memories table)
        dm_memories = []
        try:
            for sq in search_queries:
                qv = query_vector_map[sq]
                results = self.unified_store.search_memories(sq, user_id=user_id, top_k=top_k, query_vector=qv)
                dm_memories.extend(results)
            dm_memories = self._deduplicate_by_id(dm_memories, "entry_id")
            # Apply content-based deduplication to remove near-duplicates
            # Higher threshold (0.95) because e5-small has high similarity for structurally similar texts
            dm_memories = self._deduplicate_by_content(dm_memories, "dm_memories", similarity_threshold=0.95)
            print(f"[DM Retrieval] Found {len(dm_memories)} DM memories")
        except Exception as e:
            print(f"[DM Retrieval] DM memories search failed: {e}")

        # Step 4: Search user memories from groups for this user (context from group interactions)
        user_memories_from_groups = []
        try:
            if hasattr(self.unified_store, 'user_memories'):
                for sq in search_queries:
                    qv = query_vector_map[sq]
                    # Search user memories where this user participated
                    results = self.unified_store.user_memories.search_by_user(user_id, qv, top_k)
                    user_memories_from_groups.extend(results)
                user_memories_from_groups = self._deduplicate_by_id(user_memories_from_groups, "memory_id")
                print(f"[DM Retrieval] Found {len(user_memories_from_groups)} user memories from groups")
        except Exception as e:
            print(f"[DM Retrieval] User memories search failed: {e}")

        # Step 5: Search cross-group memories for this user
        cross_group_memories = []
        if user_id:
            # Use user_id directly if it already has platform prefix, otherwise assume format
            universal_id = user_id if ':' in user_id else f"telegram:{user_id}"
            for sq in search_queries:
                qv = query_vector_map[sq]
                try:
                    cg = self.unified_store.search_cross_group(universal_id, sq, limit=top_k, query_vector=qv)
                    cross_group_memories.extend(cg)
                except Exception as e:
                    print(f"[DM Retrieval] Cross-group search failed: {e}")

        cross_group_memories = self._deduplicate_by_id(cross_group_memories, "memory_id")

        # Step 5b: If reference_group_id provided, search in that specific group
        group_memories = []
        reference_group_id = context.get('group_id')  # This is the reference group for cross-context
        if reference_group_id and context.get('is_dm_with_group_reference'):
            print(f"[DM Retrieval] Cross-context: searching in reference group {reference_group_id}")
            try:
                for sq in search_queries:
                    qv = query_vector_map[sq]
                    # Search group memories
                    gm = self.unified_store.search_group_memories(reference_group_id, sq, limit=top_k, query_vector=qv)
                    group_memories.extend(gm)
                    # Search user memories in that group
                    um = self.unified_store.search_user_memories_in_group(reference_group_id, sq, limit=top_k, query_vector=qv)
                    user_memories_from_groups.extend(um)
                group_memories = self._deduplicate_by_id(group_memories, "memory_id")
                user_memories_from_groups = self._deduplicate_by_id(user_memories_from_groups, "memory_id")
                print(f"[DM Retrieval] Found {len(group_memories)} group memories from reference group")
            except Exception as e:
                print(f"[DM Retrieval] Reference group search failed: {e}")

        # Step 5c: Search memories about involved/mentioned users
        involved_user_memories = []
        involved_users = context.get('involved_users', [])
        if involved_users:
            print(f"[DM Retrieval] Searching for involved users: {involved_users}")
            fact_store = getattr(self, 'fact_store', None) or getattr(self.unified_store, 'fact_store', None)
            for involved_user_id in involved_users:
                try:
                    # Get facts about the mentioned user
                    if fact_store:
                        facts = fact_store.get_user_facts(involved_user_id, min_confidence=0.3)
                        for fact in facts[:5]:  # Limit per user
                            involved_user_memories.append({
                                "type": "user_fact",
                                "user_id": involved_user_id,
                                "content": fact.content,
                                "fact_type": fact.fact_type.value if hasattr(fact.fact_type, 'value') else fact.fact_type
                            })
                    # Get user memories from groups
                    if hasattr(self.unified_store, 'user_memories'):
                        for sq in search_queries[:2]:  # Limit queries
                            qv = query_vector_map[sq]
                            um = self.unified_store.user_memories.search_by_user(involved_user_id, qv, 3)
                            for m in um:
                                involved_user_memories.append({
                                    "type": "user_memory",
                                    "user_id": involved_user_id,
                                    "content": m.content,
                                    "group_id": m.group_id
                                })
                except Exception as e:
                    print(f"[DM Retrieval] Search for involved user {involved_user_id} failed: {e}")
            print(f"[DM Retrieval] Found {len(involved_user_memories)} memories about involved users")

        # Step 6: Apply temporal scoring to all memory types
        if dm_memories:
            dm_memories = self._apply_temporal_scoring(dm_memories)
        if user_memories_from_groups:
            user_memories_from_groups = self._apply_temporal_scoring(user_memories_from_groups)
        if cross_group_memories:
            cross_group_memories = self._apply_temporal_scoring(cross_group_memories)

        # Step 7: Rerank and limit
        if dm_memories:
            dm_memories = self._rerank_with_cross_encoder(query, dm_memories, limit)
        if user_memories_from_groups:
            user_memories_from_groups = self._rerank_with_cross_encoder(query, user_memories_from_groups, limit)
        if cross_group_memories:
            cross_group_memories = self._rerank_with_cross_encoder(query, cross_group_memories, limit)

        print(f"[DM Retrieval] Results: dm={len(dm_memories)}, user_from_groups={len(user_memories_from_groups)}, cross_group={len(cross_group_memories)}")

        # Step 8: Get user facts (if fact store is available)
        user_facts = []
        fact_store = getattr(self, 'fact_store', None) or getattr(self.unified_store, 'fact_store', None)
        if fact_store:
            try:
                user_facts = fact_store.get_user_facts(user_id, min_confidence=0.3)
                print(f"[DM Retrieval] Found {len(user_facts)} user facts")
            except Exception as e:
                print(f"[DM Retrieval] Fact store lookup failed: {e}")

        # Step 9: Format results
        raw_results = {
            "dm_memories": dm_memories,
            "group_memories": group_memories,
            "user_memories": user_memories_from_groups,
            "interaction_memories": [],
            "cross_group_memories": cross_group_memories,
            "involved_user_memories": involved_user_memories
        }

        formatted = self._format_dm_context_with_shareable(raw_results, context, user_facts)

        return {
            **raw_results,
            "formatted_context": formatted,
            "user_facts": user_facts,
        }

    def _search_shareable_user_memories(
        self,
        query: str,
        user_id: str,
        query_vector,
        limit: int
    ) -> List[Any]:
        """Search for shareable user memories across ALL groups for a specific user."""
        if not hasattr(self.unified_store, 'user_memories_table'):
            return []

        try:
            # Search all user memories for this user, filtered by is_shareable=true
            results = (
                self.unified_store.user_memories_table.search(query_vector.tolist())
                .where(f"user_id = '{user_id}' AND is_shareable = true", prefilter=True)
                .limit(limit)
                .to_list()
            )

            return [
                self.unified_store._row_to_user_memory(r)
                for r in results
            ]
        except Exception as e:
            print(f"[_search_shareable_user_memories] Error: {e}")
            return []

    def _search_shareable_dm_memories(
        self,
        query: str,
        user_id: str,
        query_vector,
        limit: int
    ) -> List[Any]:
        """Search for shareable DM memories for a specific user (is_shareable=true).

        DM memories are stored in the dm_memories table (MemoryStore.dm_memories).
        We filter by user_id and is_shareable=true to get cross-context memories.
        """
        # DM memories are in the MemoryStore.dm_memories table
        if not hasattr(self.unified_store, 'dm_memories'):
            return []

        try:
            dm_table = self.unified_store.dm_memories
            agent_id = self.unified_store.agent_id

            # Build filter: DM memories with is_shareable=true for this user
            filter_parts = [
                f"agent_id = '{agent_id}'",
                f"user_id = '{user_id}'",
                "is_shareable = true"
            ]
            filter_clause = " AND ".join(filter_parts)

            vec = query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector
            results = (
                dm_table.table.search(vec)
                .where(filter_clause, prefilter=True)
                .limit(limit)
                .to_list()
            )

            shareable_count = len(results)
            print(f"[retrieve_for_context] Found {shareable_count} shareable DM memories for speaker")

            return [dm_table._row_to_memory_entry(r) for r in results]
        except Exception as e:
            print(f"[_search_shareable_dm_memories] Error: {e}")
            return []

    def _retrieve_for_group(
        self,
        query: str,
        group_id: str,
        user_id: str,
        context: Dict[str, Any],
        limit_per_table: int,
        conversation_summary: str,
        involved_users: List[str] = None
    ) -> Dict[str, Any]:
        """
        Context-aware retrieval for group conversations.

        Access Rules for Group A when SPEAKER talks:
        - SEES: Group A's memories, Group A's user memories, Group A's interactions
        - SEES: SPEAKER's shareable DM memories (is_shareable=true)
        - SEES: Profiles of involved_users
        - NOT: DMs of other users, memories from other groups

        Args:
            query: Search query
            group_id: Group ID
            user_id: Current user ID (SPEAKER)
            context: Full context dict
            limit_per_table: Max results per table
            conversation_summary: Pre-fetched conversation summary
            involved_users: List of user IDs mentioned/involved in the query

        Returns:
            Dict with results from each table type and formatted context string
        """
        print(f"\n[retrieve_for_context] Intelligent group retrieval for: {query}")

        # Step 1: Combined planning + query generation (single LLM call)
        information_plan, search_queries = self._plan_and_generate_queries(query)
        print(f"[retrieve_for_context] Identified {len(information_plan.get('required_info', []))} info requirements")
        print(f"[retrieve_for_context] Generated {len(search_queries)} targeted queries")

        use_keyword_boost = information_plan.get("use_keyword_boost", False)
        exact_match_terms = information_plan.get("exact_match_terms", [])

        top_k = self.semantic_top_k

        # Step 2: Batch-encode ALL queries at once (1 API call instead of N*4)
        query_vectors = self.unified_store.embedding_model.encode_query(search_queries)
        query_vector_map = {sq: query_vectors[i] for i, sq in enumerate(search_queries)}
        print(f"[retrieve_for_context] Batch-encoded {len(search_queries)} queries (1 API call)")

        # Step 3: Fan-out queries to all group tables (parallel) with pre-computed vectors
        all_group_memories = []
        all_user_memories = []
        all_interaction_memories = []
        all_cross_group_memories = []
        all_speaker_dm_memories = []

        def _search_group_for_query(sq: str):
            """Search all group tables + speaker DM for a single query using pre-computed vector."""
            qv = query_vector_map[sq]
            gm = self.unified_store.search_group_memories(group_id, sq, limit=top_k, query_vector=qv)
            um = self.unified_store.search_user_memories_in_group(group_id, sq, limit=top_k, exclude_user_id=user_id, query_vector=qv)
            im = self.unified_store.search_interactions(group_id, speaker_id=user_id, query=sq, limit=top_k, query_vector=qv)
            cg = []
            dm = []
            if user_id:
                cg = self.unified_store.search_cross_group(user_id, sq, limit=top_k, query_vector=qv)
                # Search speaker's shareable DM memories in parallel
                dm = self._search_shareable_dm_memories(sq, user_id, qv, top_k)
            return gm, um, im, cg, dm

        if self.enable_parallel_retrieval and len(search_queries) > 1:
            print(f"[retrieve_for_context] Parallel fan-out: {len(search_queries)} queries x 4 tables (vectors pre-computed)")
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_retrieval_workers) as executor:
                    futures = {executor.submit(_search_group_for_query, sq): sq for sq in search_queries}
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            gm, um, im, cg, dm = future.result()
                            all_group_memories.extend(gm)
                            all_user_memories.extend(um)
                            all_interaction_memories.extend(im)
                            all_cross_group_memories.extend(cg)
                            all_speaker_dm_memories.extend(dm)
                        except Exception as e:
                            print(f"[retrieve_for_context] Query failed: {e}")
            except Exception as e:
                print(f"[retrieve_for_context] Parallel execution failed, falling back to sequential: {e}")
                for sq in search_queries:
                    gm, um, im, cg, dm = _search_group_for_query(sq)
                    all_group_memories.extend(gm)
                    all_user_memories.extend(um)
                    all_interaction_memories.extend(im)
                    all_cross_group_memories.extend(cg)
                    all_speaker_dm_memories.extend(dm)
        else:
            for sq in search_queries:
                print(f"[retrieve_for_context] Searching: {sq}")
                gm, um, im, cg, dm = _search_group_for_query(sq)
                all_group_memories.extend(gm)
                all_user_memories.extend(um)
                all_interaction_memories.extend(im)
                all_cross_group_memories.extend(cg)
                all_speaker_dm_memories.extend(dm)

        # Step 3: BM25 keyword boost for group tables
        if use_keyword_boost and exact_match_terms:
            print(f"[retrieve_for_context] Keyword boost for terms: {exact_match_terms}")
            kw_top_k = self.keyword_top_k

            all_group_memories = self._apply_group_keyword_boost(
                all_group_memories,
                lambda: self.unified_store.keyword_search_group_memories(group_id, exact_match_terms, top_k=kw_top_k),
                id_attr="memory_id"
            )
            all_user_memories = self._apply_group_keyword_boost(
                all_user_memories,
                lambda: self.unified_store.keyword_search_user_memories(group_id, exact_match_terms, top_k=kw_top_k),
                id_attr="memory_id"
            )
            all_interaction_memories = self._apply_group_keyword_boost(
                all_interaction_memories,
                lambda: self.unified_store.keyword_search_interactions(group_id, exact_match_terms, top_k=kw_top_k),
                id_attr="memory_id"
            )

        # Step 4: Deduplicate within each table (speaker DM search was done in parallel above)
        all_group_memories = self._deduplicate_by_id(all_group_memories, "memory_id")
        all_user_memories = self._deduplicate_by_id(all_user_memories, "memory_id")
        all_interaction_memories = self._deduplicate_by_id(all_interaction_memories, "memory_id")
        all_cross_group_memories = self._deduplicate_by_id(all_cross_group_memories, "memory_id")
        all_speaker_dm_memories = self._deduplicate_by_id(all_speaker_dm_memories, "entry_id")

        # Step 4.5: Apply content-based deduplication to remove near-duplicates
        # Higher threshold (0.95) because e5-small has high similarity for structurally similar texts
        all_group_memories = self._deduplicate_by_content(all_group_memories, "group_memories", similarity_threshold=0.95)
        all_user_memories = self._deduplicate_by_content(all_user_memories, "user_memories", similarity_threshold=0.95)
        all_interaction_memories = self._deduplicate_by_content(all_interaction_memories, "interaction_memories", similarity_threshold=0.95)
        if all_cross_group_memories:
            all_cross_group_memories = self._deduplicate_by_content(all_cross_group_memories, "cross_group_memories", similarity_threshold=0.95)
        if all_speaker_dm_memories:
            all_speaker_dm_memories = self._deduplicate_by_content(all_speaker_dm_memories, "speaker_dm_memories", similarity_threshold=0.95)

        print(f"[retrieve_for_context] Found {len(all_speaker_dm_memories)} shareable DM memories for speaker")

        print(f"[retrieve_for_context] Results (pre-temporal): group={len(all_group_memories)}, user={len(all_user_memories)}, "
              f"interaction={len(all_interaction_memories)}, cross_group={len(all_cross_group_memories)}, speaker_dm={len(all_speaker_dm_memories)}")

        # Step 4.6: Apply temporal scoring (recency boost + decay)
        # This is done BEFORE reranking so the cross-encoder sees temporally-adjusted scores
        if all_group_memories:
            all_group_memories = self._apply_temporal_scoring(all_group_memories)
        if all_user_memories:
            all_user_memories = self._apply_temporal_scoring(all_user_memories)
        if all_interaction_memories:
            all_interaction_memories = self._apply_temporal_scoring(all_interaction_memories)
        if all_cross_group_memories:
            all_cross_group_memories = self._apply_temporal_scoring(all_cross_group_memories)
        if all_speaker_dm_memories:
            all_speaker_dm_memories = self._apply_temporal_scoring(all_speaker_dm_memories)

        print(f"[retrieve_for_context] Results (pre-rerank): group={len(all_group_memories)}, user={len(all_user_memories)}, "
              f"interaction={len(all_interaction_memories)}, cross_group={len(all_cross_group_memories)}, speaker_dm={len(all_speaker_dm_memories)}")

        # Step 4.7: Rerank by importance score (local, no HTTP)
        rerank_top_k = limit_per_table
        all_group_memories = self._rerank_with_cross_encoder(query, all_group_memories, rerank_top_k)
        all_user_memories = self._rerank_with_cross_encoder(query, all_user_memories, rerank_top_k)
        all_interaction_memories = self._rerank_with_cross_encoder(query, all_interaction_memories, rerank_top_k)
        if all_cross_group_memories:
            all_cross_group_memories = self._rerank_with_cross_encoder(query, all_cross_group_memories, rerank_top_k)
        if all_speaker_dm_memories:
            all_speaker_dm_memories = self._rerank_with_cross_encoder(query, all_speaker_dm_memories, rerank_top_k)

        print(f"[retrieve_for_context] Results (post-rerank): group={len(all_group_memories)}, user={len(all_user_memories)}, "
              f"interaction={len(all_interaction_memories)}, cross_group={len(all_cross_group_memories)}, speaker_dm={len(all_speaker_dm_memories)}")

        # Step 5: Format using existing method
        raw_results = {
            "dm_memories": [],
            "group_memories": all_group_memories,
            "user_memories": all_user_memories,
            "interaction_memories": all_interaction_memories,
            "cross_group_memories": all_cross_group_memories,
            "speaker_dm_memories": all_speaker_dm_memories  # Speaker's shareable DM memories
        }

        # Step 6: Fetch relevant profiles and group context (lightweight entity system)
        relevant_profiles = []
        group_context = None

        # Step 6.5: Search agent responses (dual-vector search)
        agent_context = {
            "similar_questions": [],
            "said_to_user": [],
            "said_in_group": []
        }

        if hasattr(self.unified_store, 'search_agent_responses_by_trigger'):
            try:
                # Search by trigger (similar questions asked before)
                similar_qs = self.unified_store.search_agent_responses_by_trigger(
                    query,
                    group_id=group_id,
                    limit=3
                )
                agent_context["similar_questions"] = similar_qs
                if similar_qs:
                    print(f"[retrieve_for_context] Found {len(similar_qs)} similar questions asked before")
            except Exception as e:
                print(f"[retrieve_for_context] search_agent_responses_by_trigger failed: {e}")

        if hasattr(self.unified_store, 'search_agent_responses_by_response'):
            # Search by response (what I said to this user)
            if user_id:
                try:
                    said_to_user = self.unified_store.search_agent_responses_by_response(
                        query,
                        user_id=user_id,
                        limit=3
                    )
                    agent_context["said_to_user"] = said_to_user
                    if said_to_user:
                        print(f"[retrieve_for_context] Found {len(said_to_user)} responses said to this user")
                except Exception as e:
                    print(f"[retrieve_for_context] search_agent_responses_by_response (user) failed: {e}")

            # Search by response (what I said in this group)
            if group_id:
                try:
                    said_in_group = self.unified_store.search_agent_responses_by_response(
                        query,
                        group_id=group_id,
                        limit=3
                    )
                    agent_context["said_in_group"] = said_in_group
                    if said_in_group:
                        print(f"[retrieve_for_context] Found {len(said_in_group)} responses said in this group")
                except Exception as e:
                    print(f"[retrieve_for_context] search_agent_responses_by_response (group) failed: {e}")

        if self.user_profile_store:
            try:
                # Use involved_users if provided, otherwise fall back to query-based lookup
                if involved_users:
                    # Lookup profiles for specific users (no embedding needed)
                    for uid in involved_users:
                        try:
                            profile = self.user_profile_store.get_profile_by_universal_id(uid)
                            if profile:
                                relevant_profiles.append(profile)
                        except Exception as e:
                            print(f"[retrieve_for_context] Profile lookup failed for {uid}: {e}")
                    print(f"[retrieve_for_context] Loaded {len(relevant_profiles)} profiles for involved_users")
                else:
                    # Fall back to semantic profile search
                    relevant_profiles = self.user_profile_store.get_relevant_profiles(
                        query=query,
                        group_id=group_id,
                        top_k=5
                    )
                    if relevant_profiles:
                        print(f"[retrieve_for_context] Found {len(relevant_profiles)} relevant user profiles")
            except Exception as e:
                print(f"[retrieve_for_context] Profile lookup failed: {e}")

        if self.group_profile_store and group_id:
            try:
                group_context = self.group_profile_store.get_group_context(group_id)
                print(f"[retrieve_for_context] Retrieved group context for {group_id}")
            except Exception as e:
                print(f"[retrieve_for_context] Group context lookup failed: {e}")

        # Get hierarchical summaries for context
        historical_context = ""
        if self.group_summary_store and group_id:
            try:
                summaries = self.group_summary_store.get_context_summaries(
                    group_id=group_id,
                    limit_daily=3,   # Last 3 days
                    limit_weekly=2,  # Last 2 weeks
                    limit_monthly=1  # Last month
                )
                historical_context = self._build_summary_context(summaries)
                if historical_context:
                    print(f"[retrieve_for_context] Retrieved hierarchical summaries for {group_id}")
            except Exception as e:
                print(f"[retrieve_for_context] Summary lookup failed: {e}")

        formatted = self._format_multi_table_context(raw_results, context)

        return {
            **raw_results,
            "formatted_context": formatted,
            "conversation_summary": conversation_summary,
            "relevant_profiles": relevant_profiles,
            "group_context": group_context,
            "historical_context": historical_context,
            "agent_responses": agent_context
        }

    def _apply_group_keyword_boost(
        self,
        semantic_items: list,
        keyword_search_fn,
        id_attr: str = "memory_id"
    ) -> list:
        """
        Apply BM25 keyword boost to a list of group memory items using CC fusion.

        Args:
            semantic_items: Items from semantic search (no scores)
            keyword_search_fn: Callable that returns List[Tuple[item, float]]
            id_attr: Attribute name for the unique ID

        Returns:
            Re-ranked list of items
        """
        if not semantic_items:
            return semantic_items

        try:
            keyword_with_scores = keyword_search_fn()
            if not keyword_with_scores:
                return semantic_items

            # Convert semantic items to scored format (rank-based scores)
            total = len(semantic_items)
            semantic_scores = {}
            item_map = {}
            for i, item in enumerate(semantic_items):
                item_id = getattr(item, id_attr)
                semantic_scores[item_id] = 1.0 - (i / max(total, 1))
                item_map[item_id] = item

            keyword_scores = {}
            for item, score in keyword_with_scores:
                item_id = getattr(item, id_attr)
                keyword_scores[item_id] = score
                if item_id not in item_map:
                    item_map[item_id] = item

            # CC fusion
            alpha = self.cc_alpha
            all_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
            fused = {}
            for item_id in all_ids:
                sem = semantic_scores.get(item_id, 0.0)
                kw = keyword_scores.get(item_id, 0.0)
                if item_id in semantic_scores and item_id in keyword_scores:
                    fused[item_id] = alpha * sem + (1 - alpha) * kw
                elif item_id in semantic_scores:
                    fused[item_id] = alpha * sem
                else:
                    fused[item_id] = (1 - alpha) * kw

            sorted_ids = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)
            return [item_map[item_id] for item_id in sorted_ids]

        except Exception as e:
            print(f"[retrieve_for_context] Keyword boost failed: {e}")
            return semantic_items

    def _deduplicate_by_id(self, items: list, id_attr: str = "memory_id") -> list:
        """Deduplicate a list of items by their ID attribute."""
        seen = set()
        deduped = []
        for item in items:
            item_id = getattr(item, id_attr)
            if item_id not in seen:
                seen.add(item_id)
                deduped.append(item)
        return deduped

    def _deduplicate_by_content(
        self,
        items: list,
        table_name: str,
        similarity_threshold: float = 0.85
    ) -> list:
        """
        Deduplicate items by content similarity using cosine similarity.

        Computes embeddings for all items and removes items with high similarity,
        keeping only the one with higher importance_score (or first seen if tied).

        Args:
            items: List of memory objects with 'content' and 'importance_score' attributes
            table_name: Table name for logging
            similarity_threshold: Cosine similarity threshold (default 0.85)

        Returns:
            Deduplicated list of items
        """
        if not items or len(items) <= 1:
            return items

        # Check if items have required attributes (content or lossless_restatement)
        first_item = items[0]
        if hasattr(first_item, 'content'):
            contents = [item.content for item in items]
        elif hasattr(first_item, 'lossless_restatement'):
            contents = [item.lossless_restatement for item in items]
        else:
            return items

        try:
            # Get embeddings using the unified_store's embedding model
            embeddings = self.unified_store.embedding_model.encode_documents(contents)

            # Compute cosine similarity matrix
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / np.where(norms > 0, norms, 1)

            # Compute similarity matrix (all pairs)
            similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

            # Find duplicates (excluding self-similarity on diagonal)
            to_remove = set()
            n = len(items)

            for i in range(n):
                if i in to_remove:
                    continue
                for j in range(i + 1, n):
                    if j in to_remove:
                        continue

                    # Check if similarity exceeds threshold
                    if similarity_matrix[i, j] > similarity_threshold:
                        # Keep the one with higher importance_score
                        score_i = getattr(items[i], 'importance_score', 0.5)
                        score_j = getattr(items[j], 'importance_score', 0.5)

                        if score_i >= score_j:
                            to_remove.add(j)
                        else:
                            to_remove.add(i)
                            # If we remove i, we should continue comparing with j as the reference
                            # So we break the inner loop to let j be compared to others
                            break

            # Create deduplicated list
            removed_count = len(to_remove)
            deduped = [item for idx, item in enumerate(items) if idx not in to_remove]

            if removed_count > 0:
                print(f"[content-dedup] Removed {removed_count} content duplicates from {table_name}")

            return deduped

        except Exception as e:
            print(f"[content-dedup] Failed to deduplicate {table_name}: {e}")
            return items

    def _rerank_with_cross_encoder(
        self,
        query: str,
        items: list,
        top_k: int = 10,
        text_field: str = "lossless_restatement",
        force_rerank: bool = True
    ) -> list:
        """
        Rerank items using LanceDB's native CrossEncoderReranker.

        Implements "Solution B: Reranking post-retrieval" from the plan.

        Args:
            force_rerank: If True, always run reranker even if items <= top_k
                         This ensures semantic relevance ordering even for small result sets
        Converts Python objects to PyArrow table, reranks, and converts back.

        Falls back to importance_score sorting if reranking fails.

        Args:
            query: Search query for reranking
            items: List of memory objects to rerank
            top_k: Number of top results to return
            text_field: Field name to use as text content (default: lossless_restatement)

        Returns:
            Reranked list of items
        """
        if not items or len(items) <= 1:
            return items

        # If items count is already <= top_k and force_rerank is False, return as-is
        # With force_rerank=True, we still run reranking to ensure semantic ordering
        if len(items) <= top_k and not force_rerank:
            return items

        try:
            reranker = get_cross_encoder_reranker()

            # Convert items to PyArrow table
            data = []
            for idx, item in enumerate(items):
                # Get text content - try multiple possible fields
                content = getattr(item, text_field, None)
                if content is None:
                    content = getattr(item, 'content', None)
                if content is None:
                    content = getattr(item, 'lossless_restatement', '')
                if not isinstance(content, str):
                    content = str(content) if content else ''

                data.append({
                    "text": content,
                    "_original_index": idx,
                    "_distance": float(idx) / max(len(items), 1),  # Dummy distance for reranker
                    "importance_score": getattr(item, 'importance_score', 0.5)
                })

            table = pa.Table.from_pylist(data)

            # Use rerank_vector (requires _distance column for LanceDB rerankers)
            reranked_table = reranker.rerank_vector(query, table)
            reranked = reranked_table.to_pylist()

            # Retrieve original items in new order (sorted by relevance_score descending)
            reranked_items = []
            for row in reranked[:top_k]:
                original_idx = int(row["_original_index"])
                reranked_items.append(items[original_idx])

            print(f"[rerank] Cross-encoder reranked {len(items)} → {len(reranked_items)} items (query: {query[:50]}...)")
            return reranked_items

        except Exception as e:
            print(f"[rerank] CrossEncoderReranker failed: {e}, fallback to importance_score sorting")
            # Fallback: Sort by importance_score descending
            def _get_score(item):
                if hasattr(item, 'importance_score'):
                    return item.importance_score
                return 0.5

            sorted_items = sorted(items, key=_get_score, reverse=True)
            return sorted_items[:top_k]

    def _apply_temporal_scoring(
        self,
        items: list,
        recency_boost: float = 0.2,
        decay_days: int = 30
    ) -> list:
        """
        Apply temporal scoring (recency boost + decay) to memory items.

        Adjusts importance_score based on memory age:
        - < 7 days old: boost importance_score by +recency_boost (like Zep's +0.2)
        - 7-30 days: no change
        - 30-90 days: multiply importance_score by 0.8
        - > 90 days: multiply importance_score by 0.5

        Uses last_seen or first_seen field (available on all memory models).
        This is READ-side only - doesn't modify stored importance_score.

        Args:
            items: List of memory objects with importance_score and timestamps
            recency_boost: Boost amount for recent memories (default 0.2)
            decay_days: Threshold for decay (default 30 days)

        Returns:
            List of items with adjusted importance_score (temporary, in-memory only)
        """
        if not items:
            return items

        now = datetime.now(timezone.utc)
        scored_items = []

        for item in items:
            # Get the base importance_score
            base_score = getattr(item, 'importance_score', 0.5)

            # Get timestamp - try last_seen first, then first_seen, then timestamp
            memory_time = None
            for field in ['last_seen', 'first_seen', 'timestamp']:
                if hasattr(item, field):
                    ts = getattr(item, field)
                    if ts:
                        try:
                            # Parse ISO timestamp
                            memory_time = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                            # Ensure it's timezone-aware
                            if memory_time.tzinfo is None:
                                memory_time = memory_time.replace(tzinfo=timezone.utc)
                            break
                        except (ValueError, AttributeError):
                            continue

            if memory_time is None:
                # No valid timestamp found, keep original score as dict
                scored_items.append({
                    'item': item,
                    'temporal_score': getattr(item, 'importance_score', 0.5),
                    'base_score': getattr(item, 'importance_score', 0.5),
                    'age_days': 0
                })
                continue

            # Calculate age in days
            age_days = (now - memory_time).days

            # Apply temporal adjustment
            if age_days < 7:
                # Boost recent memories
                adjusted_score = min(1.0, base_score + recency_boost)
            elif age_days < 30:
                # No change for memories 7-30 days old
                adjusted_score = base_score
            elif age_days < 90:
                # Decay memories 30-90 days old
                adjusted_score = base_score * 0.8
            else:
                # Heavy decay for memories > 90 days old
                adjusted_score = base_score * 0.5

            # Create a wrapper with adjusted score (temporary)
            # Use a dict to avoid modifying the original object
            scored_items.append({
                'item': item,
                'temporal_score': adjusted_score,
                'base_score': base_score,
                'age_days': age_days
            })

        # Sort by temporally-adjusted score
        scored_items.sort(key=lambda x: x['temporal_score'], reverse=True)

        # Return original items in new order
        return [x['item'] for x in scored_items]

    def _apply_decay_to_results(self, items: list) -> list:
        """
        Re-sort items by temporally-adjusted importance_score.

        This is a convenience method that applies temporal scoring and returns
        the sorted list. Use this when you want the final sorted results.

        Args:
            items: List of memory objects

        Returns:
            List sorted by temporally-adjusted importance_score
        """
        return self._apply_temporal_scoring(items)

    def _format_dm_context(self, memories: List[MemoryEntry]) -> str:
        """Format DM memories as context string."""
        if not memories:
            return "[No relevant memories found]"

        formatted = []
        for i, entry in enumerate(memories, 1):
            parts = [f"[Memory {i}] {entry.lossless_restatement}"]
            if entry.timestamp:
                parts.append(f"(Time: {entry.timestamp})")
            if entry.topic:
                parts.append(f"(Topic: {entry.topic})")
            formatted.append(" ".join(parts))

        return "\n".join(formatted)

    def _format_dm_context_with_shareable(
        self,
        results: Dict[str, List[Any]],
        context: Dict[str, Any],
        user_facts: List[Any] = None
    ) -> str:
        """
        Format DM context with DM memories, shareable group memories and user facts.

        Access-aware formatting for DM conversations:
        - Shows: DM history, shareable user memories from groups, cross-group patterns, user facts
        - Hides: non-shareable memories, other users' private data
        """
        sections = []
        user_id = context.get('user_id')

        # DM memories (conversation history with this user)
        dm_mems = results.get("dm_memories", [])
        if dm_mems:
            dm_lines = ["## Previous Conversations"]
            for i, mem in enumerate(dm_mems[:10], 1):  # Limit to 10 most relevant
                # MemoryEntry uses lossless_restatement
                content = getattr(mem, 'lossless_restatement', str(mem))
                topic = getattr(mem, 'topic', None)
                line = f"{i}. {content}"
                if topic:
                    line += f" (topic: {topic})"
                dm_lines.append(line)
            if len(dm_mems) > 10:
                dm_lines.append(f"... and {len(dm_mems) - 10} more memories")
            sections.append("\n".join(dm_lines))

        # User memories from groups (context from group interactions)
        user_mems = results.get("user_memories", [])
        if user_mems:
            user_lines = ["## About You (from Groups)"]
            for i, mem in enumerate(user_mems, 1):
                username = mem.username or mem.user_id
                source_info = f"from group context" if mem.group_id else ""
                line = f"{i}. {mem.content}"
                if username:
                    line = f"{i}. [{username}] {mem.content}"
                if source_info:
                    line += f" ({source_info})"
                user_lines.append(line)
            sections.append("\n".join(user_lines))

        # Group memories (from reference group in cross-context query)
        group_mems = results.get("group_memories", [])
        if group_mems:
            ref_group = context.get('group_id', 'referenced group')
            group_lines = [f"## From Group ({ref_group})"]
            for i, mem in enumerate(group_mems[:10], 1):
                content = getattr(mem, 'content', str(mem))
                speaker = getattr(mem, 'speaker', None)
                line = f"{i}. {content}"
                if speaker:
                    line = f"{i}. [{speaker}] {content}"
                group_lines.append(line)
            sections.append("\n".join(group_lines))

        # Involved/mentioned users (facts and memories about them)
        involved_mems = results.get("involved_user_memories", [])
        if involved_mems:
            involved_lines = ["## About Mentioned Users"]
            # Group by user
            by_user = {}
            for mem in involved_mems:
                uid = mem.get("user_id", "unknown")
                if uid not in by_user:
                    by_user[uid] = []
                by_user[uid].append(mem)

            for uid, mems in by_user.items():
                involved_lines.append(f"### {uid}")
                for mem in mems[:5]:
                    content = mem.get("content", "")
                    mem_type = mem.get("type", "")
                    fact_type = mem.get("fact_type", "")
                    if fact_type:
                        involved_lines.append(f"- [{fact_type}] {content}")
                    else:
                        involved_lines.append(f"- {content}")
            sections.append("\n".join(involved_lines))

        # Cross-group memories
        cross_mems = results.get("cross_group_memories", [])
        if cross_mems:
            cross_lines = ["## Your Profile (Across All Groups)"]
            for i, mem in enumerate(cross_mems, 1):
                groups = ", ".join(mem.groups_involved[:3])
                if len(mem.groups_involved) > 3:
                    groups += f" (+{len(mem.groups_involved) - 3} more)"
                line = f"{i}. {mem.content}"
                line += f" (seen in: {groups})"
                if mem.confidence_score:
                    line += f" [confidence: {mem.confidence_score:.0%}]"
                cross_lines.append(line)
            sections.append("\n".join(cross_lines))

        # User facts
        if user_facts:
            fact_lines = ["## Facts About You"]
            # Group by fact type
            facts_by_type = {}
            for fact in user_facts:
                ft = fact.fact_type.value if hasattr(fact.fact_type, 'value') else fact.fact_type
                if ft not in facts_by_type:
                    facts_by_type[ft] = []
                facts_by_type[ft].append(fact)

            for fact_type, facts in facts_by_type.items():
                type_header = f"### {fact_type.title()}"
                fact_lines.append(type_header)
                for fact in facts[:5]:  # Limit to 5 per type
                    line = f"- {fact.content}"
                    if fact.confidence:
                        line += f" [{fact.confidence:.0%}]"
                    fact_lines.append(line)
                if len(facts) > 5:
                    fact_lines.append(f"- ... and {len(facts) - 5} more")
            sections.append("\n".join(fact_lines))

        if not sections:
            return "[No relevant memories found. Start a conversation and I'll remember things about you!]"

        return "\n\n".join(sections)

    def _format_multi_table_context(
        self,
        results: Dict[str, List[Any]],
        context: Dict[str, Any]
    ) -> str:
        """Format multi-table results as context string."""
        sections = []
        group_id = context.get('group_id')
        user_id = context.get('user_id')

        # DM memories
        dm_mems = results.get("dm_memories", [])
        if dm_mems:
            dm_lines = ["## Personal Memories"]
            for i, entry in enumerate(dm_mems, 1):
                line = f"{i}. {entry.lossless_restatement}"
                if entry.timestamp:
                    line += f" ({entry.timestamp})"
                dm_lines.append(line)
            sections.append("\n".join(dm_lines))

        # Group memories
        group_mems = results.get("group_memories", [])
        if group_mems:
            group_lines = ["## Group Knowledge"]
            for i, mem in enumerate(group_mems, 1):
                line = f"{i}. {mem.content}"
                if mem.speaker:
                    line += f" (shared by {mem.speaker})"
                group_lines.append(line)
            sections.append("\n".join(group_lines))

        # User memories (about other users in the group)
        user_mems = results.get("user_memories", [])
        if user_mems:
            user_lines = ["## About Group Members"]
            for i, mem in enumerate(user_mems, 1):
                username = mem.username or mem.user_id
                line = f"{i}. [{username}] {mem.content}"
                user_lines.append(line)
            sections.append("\n".join(user_lines))

        # Interaction memories
        interaction_mems = results.get("interaction_memories", [])
        if interaction_mems:
            interaction_lines = ["## Notable Interactions"]
            for i, mem in enumerate(interaction_mems, 1):
                line = f"{i}. {mem.content}"
                if mem.interaction_type:
                    line += f" ({mem.interaction_type})"
                interaction_lines.append(line)
            sections.append("\n".join(interaction_lines))

        # Cross-group memories
        cross_mems = results.get("cross_group_memories", [])
        if cross_mems:
            cross_lines = ["## User Profile (Cross-Group)"]
            for i, mem in enumerate(cross_mems, 1):
                line = f"{i}. {mem.content}"
                if mem.confidence_score:
                    line += f" (confidence: {mem.confidence_score:.0%})"
                cross_lines.append(line)
            sections.append("\n".join(cross_lines))

        # Speaker's shareable DM memories (what the current speaker shared privately)
        speaker_dm_mems = results.get("speaker_dm_memories", [])
        if speaker_dm_mems:
            speaker_dm_lines = ["## Speaker's Personal Context (shareable)"]
            for i, entry in enumerate(speaker_dm_mems, 1):
                content = getattr(entry, 'lossless_restatement', None) or getattr(entry, 'content', str(entry))
                line = f"{i}. {content}"
                timestamp = getattr(entry, 'timestamp', None)
                if timestamp:
                    line += f" ({timestamp})"
                speaker_dm_lines.append(line)
            sections.append("\n".join(speaker_dm_lines))

        if not sections:
            return "[No relevant memories found]"

        return "\n\n".join(sections)

    def _build_summary_context(self, summaries: Dict[str, List]) -> str:
        """Build context string from hierarchical summaries (micro/chunk/block)."""
        parts = []

        # Block summaries (highest level - ~1250 messages each)
        if summaries.get("block"):
            for s in summaries["block"][:2]:
                topics_str = ", ".join(s.topics[:3]) if s.topics else "general discussion"
                parts.append(f"[Historical] Messages {s.message_start}-{s.message_end}: {s.summary} (Topics: {topics_str})")

        # Chunk summaries (mid level - ~250 messages each)
        if summaries.get("chunk"):
            for s in summaries["chunk"][:3]:
                topics_str = ", ".join(s.topics[:3]) if s.topics else "general"
                parts.append(f"[Recent period] Messages {s.message_start}-{s.message_end}: {s.summary}")

        # Micro summaries (lowest level - ~50 messages each)
        if summaries.get("micro"):
            micro_summaries = summaries["micro"][:5]
            if micro_summaries:
                micro_parts = []
                for s in micro_summaries:
                    topic = s.topics[0] if s.topics else "discussion"
                    micro_parts.append(f"msgs {s.message_start}-{s.message_end}: {topic}")
                parts.append(f"[Latest activity] {' | '.join(micro_parts)}")

        return "\n".join(parts) if parts else ""

    def retrieve_user_profile(
        self,
        user_id: str,
        group_id: str = None
    ) -> Dict[str, Any]:
        """
        Retrieve comprehensive user profile across tables.

        Args:
            user_id: User identifier
            group_id: Optional group to focus on

        Returns:
            Dict with user information from various tables
        """
        if not self._supports_multi_table:
            return {"error": "Multi-table search not supported"}

        profile = {
            "user_id": user_id,
            "group_id": group_id,
            "memories": [],
            "interactions": [],
            "cross_group": []
        }

        # Get user memories from this group
        if group_id:
            user_mems = self.unified_store.user_memories_table.search().where(
                f"user_id = '{user_id}' AND group_id = '{group_id}'",
                prefilter=True
            ).limit(20).to_list()

            profile["memories"] = [
                self.unified_store._row_to_user_memory(r)
                for r in user_mems
            ]

            # Get interactions involving this user
            interactions = self.unified_store.search_interactions(
                group_id=group_id,
                speaker_id=user_id,
                limit=10
            )
            profile["interactions"] = interactions

        # Get cross-group profile
        cross_group = self.unified_store.cross_group_memories_table.search().where(
            f"universal_user_id = '{user_id}'",
            prefilter=True
        ).limit(10).to_list()

        profile["cross_group"] = [
            self.unified_store._row_to_cross_group_memory(r)
            for r in cross_group
        ]

        return profile

    def _fetch_conversation_summary(
        self,
        group_id: Optional[str],
        user_id: Optional[str]
    ) -> str:
        """
        Fetch conversation summary for a thread.

        Args:
            group_id: Group ID (None for DMs)
            user_id: User ID

        Returns:
            Summary text (empty if not exists)
        """
        if not self._supports_multi_table or not hasattr(self.unified_store, 'get_or_create_summary'):
            return ""

        try:
            # Construct thread_id format matches unified_store
            # For DMs: "{agent_id}_dm_{user_id}"
            # For groups: "{agent_id}_group_{group_id}"
            agent_id = getattr(self.unified_store, 'agent_id', 'default')

            if group_id is None or group_id.startswith('dm_'):
                # DM - extract user_id if needed
                effective_user_id = user_id
                if group_id and group_id.startswith('dm_'):
                    effective_user_id = group_id.replace('dm_', '', 1)
                if not effective_user_id:
                    return ""
                thread_id = f"{agent_id}_dm_{effective_user_id}"
            else:
                # Group
                thread_id = f"{agent_id}_group_{group_id}"

            return self.unified_store.get_or_create_summary(thread_id)
        except Exception as e:
            print(f"[_fetch_conversation_summary] Error: {e}")
            return ""
