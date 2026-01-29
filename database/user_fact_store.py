"""
UserFactStore - LanceDB storage for evidence-based user facts.

Facts are extracted from both groups and DMs.
No decay - facts remain valid until contradicted.
Confidence increases with more evidence from diverse sources.
"""
import lancedb
import pyarrow as pa
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from models.group_memory import UserFact, FactType
from utils.embedding import EmbeddingModel
import config


CONFIDENCE_CONFIG = {
    "base_confidence": 0.5,
    "evidence_boost": 0.05,      # +0.05 per additional evidence
    "source_diversity_boost": 0.1,  # +0.1 per unique source type
    "max_confidence": 0.95,
    "consolidation_threshold": 10,  # Merge after 10 similar facts
    "similarity_threshold": 0.85,   # For deduplication
}


class UserFactStore:
    """LanceDB storage for user facts with evidence tracking."""

    def __init__(
        self,
        db_path: str = None,
        embedding_model: EmbeddingModel = None,
        agent_id: Optional[str] = None
    ):
        self.db_path = db_path or config.LANCEDB_PATH
        self.embedding_model = embedding_model or EmbeddingModel()
        self.agent_id = agent_id

        # Connect to database
        self.db = lancedb.connect(self.db_path)
        self._init_table()

    def _init_table(self):
        """Initialize user_facts table."""
        schema = pa.schema([
            pa.field("fact_id", pa.string()),
            pa.field("agent_id", pa.string()),
            pa.field("user_id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("fact_type", pa.string()),
            pa.field("keywords", pa.list_(pa.string())),
            pa.field("evidence_count", pa.int32()),
            pa.field("confidence", pa.float32()),
            pa.field("sources", pa.list_(pa.string())),
            pa.field("source_types", pa.list_(pa.string())),
            pa.field("first_seen", pa.string()),
            pa.field("last_confirmed", pa.string()),
            pa.field("is_consolidated", pa.bool_()),
            pa.field("consolidated_from", pa.list_(pa.string())),
            pa.field("contradicted_by", pa.list_(pa.string())),
            pa.field("fact_vector", pa.list_(pa.float32(), self.embedding_model.dimension)),
        ])

        table_name = "user_facts"
        if table_name not in self.db.table_names():
            self.table = self.db.create_table(table_name, schema=schema)
            print(f"[UserFactStore] Created {table_name} table")
        else:
            self.table = self.db.open_table(table_name)
            print(f"[UserFactStore] Opened {table_name} ({self.table.count_rows()} rows)")

        # Create scalar indexes for faster filtering
        self._create_indexes()

    def _create_indexes(self):
        """Create scalar indexes for common query patterns."""
        try:
            self.table.create_scalar_index("user_id", replace=True)
            self.table.create_scalar_index("fact_type", replace=True)
            self.table.create_scalar_index("is_consolidated", replace=True)
        except Exception as e:
            print(f"[UserFactStore] Index creation note: {e}")

    def add_fact(self, fact: UserFact) -> str:
        """Add a new fact or update existing if similar."""
        # Check for similar existing fact
        similar = self._find_similar_fact(fact.user_id, fact.content)
        if similar:
            return self._merge_fact(similar, fact)

        # Add new fact
        vector = self.embedding_model.encode_single(fact.content, is_query=False)
        fact_type = fact.fact_type.value if hasattr(fact.fact_type, 'value') else fact.fact_type

        data = {
            "fact_id": fact.fact_id,
            "agent_id": fact.agent_id,
            "user_id": fact.user_id,
            "content": fact.content,
            "fact_type": fact_type,
            "keywords": fact.keywords,
            "evidence_count": fact.evidence_count,
            "confidence": fact.confidence,
            "sources": fact.sources,
            "source_types": fact.source_types,
            "first_seen": fact.first_seen,
            "last_confirmed": fact.last_confirmed,
            "is_consolidated": fact.is_consolidated,
            "consolidated_from": fact.consolidated_from,
            "contradicted_by": fact.contradicted_by,
            "fact_vector": vector.tolist()
        }

        self.table.add([data])
        print(f"[UserFactStore] Added fact {fact.fact_id} for user {fact.user_id}")
        return fact.fact_id

    def _find_similar_fact(self, user_id: str, content: str) -> Optional[UserFact]:
        """Find existing fact with similar content using vector search."""
        if self.table.count_rows() == 0:
            return None

        query_vector = self.embedding_model.encode_single(content, is_query=True)

        # Search for similar facts for this user
        results = (
            self.table.search(query_vector.tolist())
            .where(f"user_id = '{user_id}' AND is_consolidated = false", prefilter=True)
            .limit(5)
            .to_list()
        )

        if not results:
            return None

        # Check if any result is above similarity threshold
        for r in results:
            distance = r.get("_distance", 1.0)
            similarity = 1.0 - distance
            if similarity >= CONFIDENCE_CONFIG["similarity_threshold"]:
                return self._row_to_fact(r)

        return None

    def _merge_fact(self, existing: UserFact, new: UserFact) -> str:
        """Merge new evidence into existing fact."""
        # Delete existing fact
        self.table.delete(f"fact_id = '{existing.fact_id}'")

        # Update fields
        existing.evidence_count += 1
        existing.last_confirmed = new.last_confirmed

        # Add source if new
        if new.sources and new.sources[0] not in existing.sources:
            existing.sources.append(new.sources[0])
        if new.source_types and new.source_types[0] not in existing.source_types:
            existing.source_types.append(new.source_types[0])

        # Recalculate confidence
        existing.confidence = self._calculate_confidence(existing)

        # Re-add with updated values
        return self.add_fact(existing)

    def _calculate_confidence(self, fact: UserFact) -> float:
        """Calculate confidence based on evidence and source diversity."""
        config = CONFIDENCE_CONFIG

        confidence = config["base_confidence"]

        # Evidence boost
        evidence_boost = (fact.evidence_count - 1) * config["evidence_boost"]
        confidence += evidence_boost

        # Source diversity boost
        unique_source_types = len(set(fact.source_types))
        diversity_boost = (unique_source_types - 1) * config["source_diversity_boost"]
        confidence += diversity_boost

        return min(confidence, config["max_confidence"])

    def get_user_facts(
        self,
        user_id: str,
        min_confidence: float = 0.3,
        fact_type: Optional[FactType] = None
    ) -> List[UserFact]:
        """Get all facts for a user above confidence threshold."""
        conditions = [f"user_id = '{user_id}'", f"confidence >= {min_confidence}"]

        if fact_type:
            fact_type_str = fact_type.value if hasattr(fact_type, 'value') else fact_type
            conditions.append(f"fact_type = '{fact_type_str}'")

        where_clause = " AND ".join(conditions)

        results = self.table.search().where(
            where_clause,
            prefilter=True
        ).limit(100).to_list()

        return [self._row_to_fact(r) for r in results]

    def search_facts(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        min_confidence: float = 0.3
    ) -> List[UserFact]:
        """Semantic search for relevant facts."""
        if self.table.count_rows() == 0:
            return []

        query_vector = self.embedding_model.encode_single(query, is_query=True)

        results = (
            self.table.search(query_vector.tolist())
            .where(f"user_id = '{user_id}' AND confidence >= {min_confidence}", prefilter=True)
            .limit(limit)
            .to_list()
        )

        return [self._row_to_fact(r) for r in results]

    def consolidate_facts(self, user_id: str):
        """
        Consolidate similar facts into general statements.

        Called when a user has many facts (>consolidation_threshold).
        Groups similar facts and creates consolidated fact.
        """
        facts = self.get_user_facts(user_id)
        if len(facts) < CONFIDENCE_CONFIG["consolidation_threshold"]:
            print(f"[UserFactStore] User {user_id} has {len(facts)} facts, below consolidation threshold")
            return

        print(f"[UserFactStore] Consolidating {len(facts)} facts for user {user_id}")

        # Group by fact_type
        by_type: Dict[str, List[UserFact]] = {}
        for fact in facts:
            if not fact.is_consolidated:
                fact_type = fact.fact_type.value if hasattr(fact.fact_type, 'value') else fact.fact_type
                if fact_type not in by_type:
                    by_type[fact_type] = []
                by_type[fact_type].append(fact)

        # For each type, find clusters and consolidate
        for fact_type, type_facts in by_type.items():
            if len(type_facts) < 3:
                continue  # Need at least 3 facts to consolidate

            # Simple consolidation: take the fact with highest confidence as base
            type_facts.sort(key=lambda f: f.confidence, reverse=True)
            base_fact = type_facts[0]

            # Create consolidated fact
            consolidated = UserFact(
                agent_id=base_fact.agent_id,
                user_id=user_id,
                content=f"[Consolidated] {base_fact.content}",
                fact_type=base_fact.fact_type,
                keywords=list(set([k for f in type_facts for k in f.keywords]))[:10],
                evidence_count=sum(f.evidence_count for f in type_facts),
                confidence=max(f.confidence for f in type_facts),
                sources=list(set([s for f in type_facts for s in f.sources]))[:20],
                source_types=list(set([st for f in type_facts for st in f.source_types])),
                first_seen=min(f.first_seen for f in type_facts),
                last_confirmed=max(f.last_confirmed for f in type_facts),
                is_consolidated=True,
                consolidated_from=[f.fact_id for f in type_facts],
                contradicted_by=[]
            )

            # Add consolidated fact
            self.add_fact(consolidated)

            # Mark original facts as consolidated
            for fact in type_facts:
                self.table.delete(f"fact_id = '{fact.fact_id}'")
                fact.is_consolidated = True
                fact.consolidated_from = []
                # Re-add as consolidated
                self.add_fact(fact)

            print(f"[UserFactStore] Consolidated {len(type_facts)} {fact_type} facts")

    def add_contradiction(self, fact_id: str, contradicting_fact_id: str):
        """Mark two facts as contradicting each other."""
        # Get both facts
        fact1_results = self.table.search().where(
            f"fact_id = '{fact_id}'",
            prefilter=True
        ).limit(1).to_list()

        fact2_results = self.table.search().where(
            f"fact_id = '{contradicting_fact_id}'",
            prefilter=True
        ).limit(1).to_list()

        if not fact1_results or not fact2_results:
            print(f"[UserFactStore] Could not find facts for contradiction")
            return

        # Update contradiction lists
        fact1 = self._row_to_fact(fact1_results[0])
        fact2 = self._row_to_fact(fact2_results[0])

        if contradicting_fact_id not in fact1.contradicted_by:
            fact1.contradicted_by.append(contradicting_fact_id)
        if fact_id not in fact2.contradicted_by:
            fact2.contradicted_by.append(fact_id)

        # Delete and re-add with updated contradictions
        self.table.delete(f"fact_id = '{fact_id}'")
        self.table.delete(f"fact_id = '{contradicting_fact_id}'")

        self.add_fact(fact1)
        self.add_fact(fact2)

        print(f"[UserFactStore] Marked contradiction between {fact_id} and {contradicting_fact_id}")

    def _row_to_fact(self, row: dict) -> UserFact:
        """Convert LanceDB row to UserFact."""
        fact_type = row["fact_type"]
        if isinstance(fact_type, str):
            fact_type = FactType(fact_type)

        return UserFact(
            fact_id=row["fact_id"],
            agent_id=row["agent_id"],
            user_id=row["user_id"],
            content=row["content"],
            fact_type=fact_type,
            keywords=row.get("keywords", []),
            evidence_count=row.get("evidence_count", 1),
            confidence=row.get("confidence", 0.5),
            sources=row.get("sources", []),
            source_types=row.get("source_types", []),
            first_seen=row["first_seen"],
            last_confirmed=row["last_confirmed"],
            is_consolidated=row.get("is_consolidated", False),
            consolidated_from=row.get("consolidated_from", []),
            contradicted_by=row.get("contradicted_by", [])
        )

    def count_facts(self, user_id: str = None) -> int:
        """Count facts with optional user filter."""
        if not user_id:
            return self.table.count_rows()

        results = self.table.search().where(
            f"user_id = '{user_id}'",
            prefilter=True
        ).limit(10000).to_list()
        return len(results)

    def get_fact_by_id(self, fact_id: str) -> Optional[UserFact]:
        """Get a specific fact by ID."""
        results = self.table.search().where(
            f"fact_id = '{fact_id}'",
            prefilter=True
        ).limit(1).to_list()

        if results:
            return self._row_to_fact(results[0])
        return None
