"""
LanceDB Native Rerankers for Memory Retrieval

Provides singleton access to LanceDB's native rerankers:
- CrossEncoderReranker: Local cross-encoder reranking (free, fast)
- DedupCrossEncoderReranker: Custom reranker with deduplication

Based on: https://docs.lancedb.com/reranking
"""
from lancedb.rerankers import CrossEncoderReranker
import pyarrow as pa
import numpy as np
from typing import Optional
from utils.embedding import EmbeddingModel


# Singleton instance for cross-encoder reranker
_cross_encoder_reranker = None


def get_cross_encoder_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
) -> CrossEncoderReranker:
    """
    Get or create a singleton CrossEncoderReranker instance.

    The cross-encoder model is loaded once and reused to avoid loading overhead.

    Args:
        model_name: HuggingFace model name for cross-encoder
            - "cross-encoder/ms-marco-TinyBERT-L-6" (faster, lower quality)
            - "cross-encoder/ms-marco-MiniLM-L-6-v2" (balanced - DEFAULT)
            - "cross-encoder/ms-marco-MiniLM-L-12-v2" (higher quality, slower)

    Returns:
        CrossEncoderReranker instance
    """
    global _cross_encoder_reranker
    if _cross_encoder_reranker is None:
        print(f"[rerankers] Initializing CrossEncoderReranker with {model_name}")
        _cross_encoder_reranker = CrossEncoderReranker(model_name=model_name)
    return _cross_encoder_reranker


class DedupCrossEncoderReranker(CrossEncoderReranker):
    """
    CrossEncoderReranker with deduplication by content similarity.

    Combines deduplication and cross-encoder reranking:
    1. First deduplicates results with high similarity (> threshold)
    2. Then applies cross-encoder reranking for query relevance

    This is useful when combining results from multiple sources where
    duplicates may exist.

    Example:
        reranker = DedupCrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            dedup_threshold=0.85,
            embedding_model=your_embedding_model
        )

        # Use in LanceDB queries
        results = table.search(query).rerank(reranker=reranker).to_list()
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        dedup_threshold: float = 0.85,
        embedding_model = None,
        **kwargs
    ):
        """
        Initialize DedupCrossEncoderReranker.

        Args:
            model_name: Cross-encoder model name
            dedup_threshold: Cosine similarity threshold for deduplication (0-1)
            embedding_model: EmbeddingModel instance for computing similarities
            **kwargs: Additional arguments passed to CrossEncoderReranker
        """
        super().__init__(model_name=model_name, **kwargs)
        self.dedup_threshold = dedup_threshold
        self.embedding_model = embedding_model

    def _deduplicate(self, results: pa.Table, text_column: str = "text") -> pa.Table:
        """
        Remove duplicates based on content similarity.

        Computes embeddings for all texts and removes items with high similarity,
        keeping only the one with higher relevance score (if available).

        Args:
            results: PyArrow table with search results
            text_column: Name of column containing text content

        Returns:
            Deduplicated PyArrow table
        """
        df = results.to_pandas()

        if len(df) <= 1 or self.embedding_model is None:
            return results

        try:
            # Get embeddings for all texts
            texts = df[text_column].tolist()
            embeddings = self.embedding_model.encode_documents(texts)

            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / np.where(norms > 0, norms, 1)

            # Compute similarity matrix
            similarity_matrix = np.dot(normalized, normalized.T)

            # Find duplicates
            to_remove = set()
            n = len(df)

            for i in range(n):
                if i in to_remove:
                    continue
                for j in range(i + 1, n):
                    if j in to_remove:
                        continue

                    if similarity_matrix[i, j] > self.dedup_threshold:
                        # Keep item with higher relevance score
                        score_i = df.iloc[i].get('_relevance_score', 0)
                        score_j = df.iloc[j].get('_relevance_score', 0)

                        to_remove.add(j if score_i >= score_j else i)

            if to_remove:
                print(f"[DedupReranker] Removing {len(to_remove)} duplicates (threshold={self.dedup_threshold})")

            # Remove duplicates
            df_deduped = df.drop(index=list(to_remove)).reset_index(drop=True)
            return pa.Table.from_pandas(df_deduped)

        except Exception as e:
            print(f"[DedupReranker] Deduplication failed: {e}")
            return results

    def rerank_vector(self, query: str, vector_results: pa.Table) -> pa.Table:
        """
        Deduplicate then rerank vector results.

        Args:
            query: Search query
            vector_results: Vector search results as PyArrow table

        Returns:
            Reranked and deduplicated results
        """
        deduped = self._deduplicate(vector_results)
        return super().rerank_vector(query, deduped)

    def rerank_fts(self, query: str, fts_results: pa.Table) -> pa.Table:
        """
        Deduplicate then rerank FTS results.

        Args:
            query: Search query
            fts_results: Full-text search results as PyArrow table

        Returns:
            Reranked and deduplicated results
        """
        deduped = self._deduplicate(fts_results)
        return super().rerank_fts(query, deduped)

    def rerank_hybrid(
        self,
        query: str,
        vector_results: pa.Table,
        fts_results: pa.Table
    ) -> pa.Table:
        """
        Deduplicate then rerank hybrid results.

        Args:
            query: Search query
            vector_results: Vector search results as PyArrow table
            fts_results: Full-text search results as PyArrow table

        Returns:
            Reranked and deduplicated results
        """
        # Merge first (using parent's merge method)
        combined = self.merge_results(vector_results, fts_results)

        # Deduplicate
        deduped = self._deduplicate(combined)

        # Cross-encoder rerank
        return super().rerank_vector(query, deduped)
