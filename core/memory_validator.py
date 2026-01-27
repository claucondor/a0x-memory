"""
Memory Validator - Validates LLM-extracted memories for quality and coherence

Three validation levels:
1. Schema validation - required fields, types, ranges
2. Content validation - length, relevance, no empty values
3. Semantic validation - embedding similarity with original dialogue
"""
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from models.memory_entry import MemoryEntry
from utils.embedding import EmbeddingModel


@dataclass
class ValidationResult:
    """Result of memory validation"""
    is_valid: bool
    score: float  # 0.0 to 1.0
    issues: List[str]
    memory: Optional[MemoryEntry] = None


class MemoryValidator:
    """
    Validates extracted memories for quality and coherence.

    Catches common LLM extraction issues:
    - Empty or too short restatements
    - Hallucinated information
    - Missing required fields
    - Irrelevant keywords
    - Unreasonable importance scores
    """

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        min_restatement_length: int = 20,
        max_restatement_length: int = 500,
        min_semantic_similarity: float = 0.3,
        min_keyword_overlap: float = 0.2
    ):
        self.embedding_model = embedding_model
        self.min_restatement_length = min_restatement_length
        self.max_restatement_length = max_restatement_length
        self.min_semantic_similarity = min_semantic_similarity
        self.min_keyword_overlap = min_keyword_overlap

    def validate_memory(
        self,
        memory: MemoryEntry,
        original_dialogue: str
    ) -> ValidationResult:
        """
        Validate a single memory entry against original dialogue.

        Args:
            memory: The extracted MemoryEntry
            original_dialogue: The original dialogue text

        Returns:
            ValidationResult with score and issues
        """
        issues = []
        scores = []

        # ============================================================
        # Level 1: Schema Validation
        # ============================================================
        schema_score, schema_issues = self._validate_schema(memory)
        scores.append(schema_score)
        issues.extend(schema_issues)

        # ============================================================
        # Level 2: Content Validation
        # ============================================================
        content_score, content_issues = self._validate_content(memory, original_dialogue)
        scores.append(content_score)
        issues.extend(content_issues)

        # ============================================================
        # Level 3: Semantic Validation (if embedding model available)
        # ============================================================
        if self.embedding_model:
            semantic_score, semantic_issues = self._validate_semantic(memory, original_dialogue)
            scores.append(semantic_score)
            issues.extend(semantic_issues)

        # Calculate final score (weighted average)
        weights = [0.2, 0.4, 0.4] if self.embedding_model else [0.3, 0.7]
        final_score = sum(s * w for s, w in zip(scores, weights))

        # Determine if valid (score >= 0.5 and no critical issues)
        critical_issues = [i for i in issues if i.startswith("[CRITICAL]")]
        is_valid = final_score >= 0.5 and len(critical_issues) == 0

        return ValidationResult(
            is_valid=is_valid,
            score=final_score,
            issues=issues,
            memory=memory if is_valid else None
        )

    def validate_batch(
        self,
        memories: List[MemoryEntry],
        original_dialogue: str
    ) -> Tuple[List[MemoryEntry], List[ValidationResult]]:
        """
        Validate a batch of memories, returning only valid ones.

        Returns:
            Tuple of (valid_memories, all_results)
        """
        results = []
        valid_memories = []

        for memory in memories:
            result = self.validate_memory(memory, original_dialogue)
            results.append(result)

            if result.is_valid:
                valid_memories.append(memory)
            else:
                print(f"[Validator] Rejected memory (score={result.score:.2f}): {result.issues}")

        acceptance_rate = len(valid_memories) / len(memories) if memories else 0
        print(f"[Validator] Accepted {len(valid_memories)}/{len(memories)} memories ({acceptance_rate:.0%})")

        return valid_memories, results

    # ============================================================
    # Level 1: Schema Validation
    # ============================================================

    def _validate_schema(self, memory: MemoryEntry) -> Tuple[float, List[str]]:
        """Check required fields and types."""
        issues = []
        score = 1.0

        # Check lossless_restatement exists
        if not memory.lossless_restatement:
            issues.append("[CRITICAL] Missing lossless_restatement")
            score -= 0.5

        # Check keywords is a list
        if not isinstance(memory.keywords, list):
            issues.append("[CRITICAL] Keywords must be a list")
            score -= 0.3

        # Check importance_score range (if present)
        if hasattr(memory, 'importance_score') and memory.importance_score is not None:
            if not 0.0 <= memory.importance_score <= 1.0:
                issues.append(f"[WARNING] Importance score out of range: {memory.importance_score}")
                score -= 0.1

        # Check memory_type is valid (if present)
        valid_types = {"expertise", "preference", "fact", "announcement", "conversation", None}
        if hasattr(memory, 'memory_type') and memory.memory_type not in valid_types:
            issues.append(f"[WARNING] Invalid memory_type: {memory.memory_type}")
            score -= 0.1

        return max(0.0, score), issues

    # ============================================================
    # Level 2: Content Validation
    # ============================================================

    def _validate_content(
        self,
        memory: MemoryEntry,
        original_dialogue: str
    ) -> Tuple[float, List[str]]:
        """Check content quality and relevance."""
        issues = []
        score = 1.0

        restatement = memory.lossless_restatement or ""

        # Check length
        if len(restatement) < self.min_restatement_length:
            issues.append(f"[WARNING] Restatement too short ({len(restatement)} chars)")
            score -= 0.2

        if len(restatement) > self.max_restatement_length:
            issues.append(f"[WARNING] Restatement too long ({len(restatement)} chars)")
            score -= 0.1

        # Check for generic/useless restatements
        generic_patterns = [
            r"^the user said",
            r"^someone mentioned",
            r"^a message was sent",
            r"^there was a conversation",
            r"^the group discussed"
        ]
        for pattern in generic_patterns:
            if re.match(pattern, restatement.lower()):
                issues.append("[WARNING] Restatement is too generic")
                score -= 0.2
                break

        # Check keyword relevance (keywords should appear in dialogue or restatement)
        if memory.keywords:
            dialogue_lower = original_dialogue.lower()
            restatement_lower = restatement.lower()

            relevant_keywords = 0
            for kw in memory.keywords:
                kw_lower = kw.lower()
                if kw_lower in dialogue_lower or kw_lower in restatement_lower:
                    relevant_keywords += 1

            keyword_relevance = relevant_keywords / len(memory.keywords) if memory.keywords else 0

            if keyword_relevance < self.min_keyword_overlap:
                issues.append(f"[WARNING] Low keyword relevance ({keyword_relevance:.0%})")
                score -= 0.2

        # Check for hallucination indicators
        # (entities/names in restatement that don't appear in original)
        if memory.persons:
            dialogue_lower = original_dialogue.lower()
            for person in memory.persons:
                # Normalize person name for comparison
                # Handle variations: "David Founder" vs "david_founder" vs "davidfounder"
                person_lower = person.lower()
                person_normalized = person_lower.replace(" ", "_")  # "david founder" -> "david_founder"
                person_no_sep = person_lower.replace(" ", "").replace("_", "")  # "davidfounder"

                # Check all variations
                found = (
                    person_lower in dialogue_lower or           # exact match
                    person_normalized in dialogue_lower or      # with underscores
                    person_no_sep in dialogue_lower.replace("_", "")  # no separators
                )

                if not found:
                    issues.append(f"[CRITICAL] Possible hallucination: person '{person}' not in dialogue")
                    score -= 0.3

        return max(0.0, score), issues

    # ============================================================
    # Level 3: Semantic Validation
    # ============================================================

    def _validate_semantic(
        self,
        memory: MemoryEntry,
        original_dialogue: str
    ) -> Tuple[float, List[str]]:
        """Check semantic similarity between restatement and original."""
        issues = []
        score = 1.0

        if not self.embedding_model:
            return score, issues

        try:
            # Get embeddings
            dialogue_emb = self.embedding_model.encode_single(original_dialogue, is_query=False)
            restatement_emb = self.embedding_model.encode_single(
                memory.lossless_restatement,
                is_query=False
            )

            # Calculate cosine similarity
            similarity = np.dot(dialogue_emb, restatement_emb) / (
                np.linalg.norm(dialogue_emb) * np.linalg.norm(restatement_emb)
            )

            if similarity < self.min_semantic_similarity:
                issues.append(f"[CRITICAL] Low semantic similarity ({similarity:.2f})")
                score = similarity / self.min_semantic_similarity  # Proportional penalty
            else:
                # Bonus for high similarity
                score = min(1.0, 0.8 + (similarity - self.min_semantic_similarity) * 0.5)

        except Exception as e:
            issues.append(f"[WARNING] Semantic validation failed: {e}")
            score = 0.7  # Neutral score on failure

        return max(0.0, score), issues


# ============================================================
# Convenience function for integration
# ============================================================

def validate_and_filter_memories(
    memories: List[MemoryEntry],
    original_dialogue: str,
    embedding_model: Optional[EmbeddingModel] = None
) -> List[MemoryEntry]:
    """
    Convenience function to validate and filter memories.

    Usage in MemoryBuilder:
        entries = self._parse_llm_response(response, dialogues)
        entries = validate_and_filter_memories(entries, dialogue_text, self.embedding_model)
    """
    validator = MemoryValidator(embedding_model=embedding_model)
    valid_memories, _ = validator.validate_batch(memories, original_dialogue)
    return valid_memories
