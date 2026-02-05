"""
Adaptive Threshold Manager - Stateless version for Cloud Run

Uses Firestore to persist activity metrics across requests.
Each request reads metrics, calculates threshold, and updates metrics.

For local testing, set USE_LOCAL_STORAGE=true to use in-memory storage.

Problems with fixed thresholds:
- Active groups wait too long (10 msgs = seconds)
- Inactive groups process too often (10 msgs = days)
- No urgency consideration (important messages wait)

Adaptive approach:
- Store activity metrics in Firestore per group
- Calculate threshold on each request
- Time-based fallback (max wait time)
- Urgency detection (important messages trigger faster processing)
"""
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import defaultdict

import config

# Conditional Firestore import
try:
    from google.cloud import firestore
    FIRESTORE_AVAILABLE = True
except ImportError:
    FIRESTORE_AVAILABLE = False
    firestore = None


@dataclass
class ThresholdConfig:
    """Configuration for adaptive thresholds"""
    # Base thresholds
    min_batch_size: int = 5
    max_batch_size: int = 30
    default_batch_size: int = 10

    # Activity-based adjustments
    high_activity_threshold: float = 20.0  # msgs/hour = high activity
    low_activity_threshold: float = 2.0    # msgs/hour = low activity

    # Time-based triggers
    max_wait_time_seconds: int = 3600      # 1 hour max wait
    min_wait_time_seconds: int = 60        # 1 minute min wait

    # Urgency settings
    urgency_trigger_count: int = 3         # N high-importance msgs = urgent
    urgency_batch_size: int = 5            # Smaller batch for urgency

    # Dynamic window size settings
    min_window_size: int = 10              # Minimum messages in window
    max_window_size: int = 50              # Maximum messages in window
    default_window_size: int = 15          # Default window size

    # Spam detection settings
    # Note: e5-small (384 dims) produces high similarity for topically similar messages
    # 0.97 threshold catches near-duplicates while allowing normal conversation
    spam_similarity_threshold: float = 0.97  # Similarity > this = likely spam
    spam_check_window: int = 5               # Compare with last N messages from same user
    spam_score_decay: float = 0.9            # Decay factor for spam score over time
    spam_block_threshold: float = 5.0        # Block user if spam_score exceeds this (was 3.0)


# ============================================================
# In-Memory Metrics Store (for local testing)
# ============================================================

# Simple Increment class for in-memory mode
class InMemoryIncrement:
    def __init__(self, value: int):
        self._value = value


class InMemoryMetricsStore:
    """
    In-memory implementation of metrics storage for local testing.
    Singleton pattern to persist across AdaptiveThresholdManager instances.
    """
    _instance = None
    _metrics: Dict[str, Dict[str, Any]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._metrics = {}
        return cls._instance

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self._metrics.get(doc_id)

    def set(self, doc_id: str, data: Dict[str, Any], merge: bool = True):
        if merge and doc_id in self._metrics:
            existing = self._metrics[doc_id]
            for key, value in data.items():
                if isinstance(value, InMemoryIncrement):
                    # Handle increment - add to existing value
                    existing[key] = existing.get(key, 0) + value._value
                else:
                    existing[key] = value
        else:
            # New document - convert InMemoryIncrement to plain values
            processed = {}
            for key, value in data.items():
                if isinstance(value, InMemoryIncrement):
                    processed[key] = value._value
                else:
                    processed[key] = value
            self._metrics[doc_id] = processed

    def clear(self):
        """Clear all metrics (for tests)."""
        self._metrics = {}


class AdaptiveThresholdManager:
    """
    Stateless adaptive threshold manager using Firestore or in-memory storage.

    All state is persisted in Firestore (or memory for tests), making it Cloud Run compatible.

    For local testing, set USE_LOCAL_STORAGE=true to use fast in-memory storage.

    Firestore structure:
        activity_metrics/{agent_id}_{group_id}:
            - message_count: int
            - first_message_time: timestamp
            - last_message_time: timestamp
            - last_process_time: timestamp
            - total_processed: int
            - high_importance_pending: int
            - messages_per_hour_cached: float (calculated periodically)
    """

    COLLECTION = "activity_metrics"

    def __init__(
        self,
        firestore_client=None,
        threshold_config: Optional[ThresholdConfig] = None
    ):
        self.threshold_config = threshold_config or ThresholdConfig()
        self._use_memory = config.USE_LOCAL_STORAGE
        self._memory_store = InMemoryMetricsStore() if self._use_memory else None
        self.db = firestore_client if not self._use_memory else None

    def _get_db(self):
        """Lazy initialization of Firestore client."""
        if self._use_memory:
            return None
        if self.db is None and FIRESTORE_AVAILABLE:
            self.db = firestore.Client()
        return self.db

    def _get_doc_id(self, agent_id: str, group_id: str) -> str:
        """Generate document ID for metrics."""
        # Sanitize for Firestore (no slashes)
        safe_group = group_id.replace("/", "_").replace(":", "_")
        return f"{agent_id}_{safe_group}"

    def _get_metrics(self, agent_id: str, group_id: str) -> Dict[str, Any]:
        """Get metrics from Firestore or in-memory store."""
        doc_id = self._get_doc_id(agent_id, group_id)

        if self._use_memory:
            data = self._memory_store.get(doc_id)
            return data if data else self._default_metrics()

        try:
            doc = self._get_db().collection(self.COLLECTION).document(doc_id).get()
            if doc.exists:
                return doc.to_dict()
            else:
                return self._default_metrics()
        except Exception as e:
            print(f"[AdaptiveThreshold] Error reading metrics: {e}")
            return self._default_metrics()

    def _update_metrics(self, agent_id: str, group_id: str, updates: Dict[str, Any]):
        """Update metrics in Firestore or in-memory store."""
        doc_id = self._get_doc_id(agent_id, group_id)

        if self._use_memory:
            self._memory_store.set(doc_id, updates, merge=True)
            return

        try:
            self._get_db().collection(self.COLLECTION).document(doc_id).set(
                updates,
                merge=True
            )
        except Exception as e:
            print(f"[AdaptiveThreshold] Error updating metrics: {e}")

    def _default_metrics(self) -> Dict[str, Any]:
        """Default metrics for new group."""
        return {
            "message_count": 0,
            "last_message_time": None,
            "last_process_time": None,
            "total_processed": 0,
            "high_importance_pending": 0
        }

    def _calculate_messages_per_hour(self, metrics: Dict[str, Any]) -> float:
        """
        Estimate activity rate based on messages since last processing.

        Simple heuristic: pending_messages / time_since_last_process
        """
        last_process = metrics.get("last_process_time")
        total_count = metrics.get("message_count", 0)
        processed = metrics.get("total_processed", 0)
        pending = total_count - processed

        if pending <= 0:
            return 0.0

        if not last_process:
            # No processing yet - use default rate
            return 5.0  # Assume moderate activity

        # Convert Firestore timestamp if needed
        if hasattr(last_process, 'timestamp'):
            last_process_ts = last_process.timestamp()
        else:
            last_process_ts = last_process

        hours_since_process = (time.time() - last_process_ts) / 3600
        if hours_since_process < 0.05:  # Less than 3 minutes
            return 20.0  # Assume high activity if very recent

        return pending / hours_since_process

    def _calculate_batch_size(self, metrics: Dict[str, Any]) -> int:
        """
        Calculate optimal batch size based on activity.

        High activity → smaller batches (process more frequently)
        Low activity → larger batches (wait for more context)
        """
        cfg = self.threshold_config
        msgs_per_hour = self._calculate_messages_per_hour(metrics)

        if msgs_per_hour >= cfg.high_activity_threshold:
            # High activity: smaller batches
            activity_ratio = min(msgs_per_hour / cfg.high_activity_threshold, 3.0)
            batch_size = cfg.default_batch_size - int(
                (cfg.default_batch_size - cfg.min_batch_size) * (activity_ratio - 1) / 2
            )
            return max(cfg.min_batch_size, batch_size)

        elif msgs_per_hour <= cfg.low_activity_threshold and msgs_per_hour > 0:
            # Low activity: larger batches
            activity_ratio = cfg.low_activity_threshold / msgs_per_hour
            batch_size = cfg.default_batch_size + int(
                (cfg.max_batch_size - cfg.default_batch_size) * min(activity_ratio - 1, 2) / 2
            )
            return min(cfg.max_batch_size, batch_size)

        else:
            # Normal activity or no data: default
            return cfg.default_batch_size

    def get_dynamic_window_size(self, agent_id: str, group_id: str) -> int:
        """
        Calculate dynamic window size based on activity.

        High activity → larger window (more context needed)
        Low activity → smaller window (less storage, sufficient context)

        Returns:
            int: Window size (number of messages to keep)
        """
        metrics = self._get_metrics(agent_id, group_id)
        cfg = self.threshold_config
        msgs_per_hour = self._calculate_messages_per_hour(metrics)

        if msgs_per_hour >= cfg.high_activity_threshold:
            # High activity: larger window for more context
            activity_ratio = min(msgs_per_hour / cfg.high_activity_threshold, 3.0)
            window_size = cfg.default_window_size + int(
                (cfg.max_window_size - cfg.default_window_size) * (activity_ratio - 1) / 2
            )
            return min(cfg.max_window_size, window_size)

        elif msgs_per_hour <= cfg.low_activity_threshold and msgs_per_hour > 0:
            # Low activity: smaller window
            activity_ratio = cfg.low_activity_threshold / msgs_per_hour
            window_size = cfg.default_window_size - int(
                (cfg.default_window_size - cfg.min_window_size) * min(activity_ratio - 1, 2) / 2
            )
            return max(cfg.min_window_size, window_size)

        else:
            # Normal activity or no data: default
            return cfg.default_window_size

    def check_spam(
        self,
        agent_id: str,
        group_id: str,
        user_id: str,
        message_embedding: list,
        recent_embeddings: list
    ) -> Tuple[bool, float, str]:
        """
        Check if a message is likely spam based on similarity to recent messages.

        Args:
            agent_id: Agent ID
            group_id: Group/conversation ID
            user_id: User who sent the message
            message_embedding: Embedding vector of the new message
            recent_embeddings: List of (embedding, user_id, timestamp) tuples from recent messages

        Returns:
            Tuple of (is_spam: bool, spam_score: float, reason: str)
        """
        import numpy as np

        cfg = self.threshold_config

        # Filter to only this user's recent messages
        user_recent = [
            (emb, ts) for emb, uid, ts in recent_embeddings
            if uid == user_id and emb is not None
        ][-cfg.spam_check_window:]

        if not user_recent:
            return False, 0.0, "no_history"

        # Calculate similarities with recent messages
        msg_vec = np.array(message_embedding)
        similarities = []

        for emb, ts in user_recent:
            recent_vec = np.array(emb)
            # Cosine similarity
            similarity = np.dot(msg_vec, recent_vec) / (
                np.linalg.norm(msg_vec) * np.linalg.norm(recent_vec) + 1e-8
            )
            similarities.append(similarity)

        max_similarity = max(similarities) if similarities else 0.0
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        # Spam detection logic
        is_spam = False
        reason = "ok"
        spam_score = 0.0

        # High similarity with any recent message = likely spam
        if max_similarity >= cfg.spam_similarity_threshold:
            is_spam = True
            spam_score = max_similarity
            reason = f"high_similarity:{max_similarity:.3f}"

        # Multiple similar messages in a row = spam pattern
        elif avg_similarity >= cfg.spam_similarity_threshold - 0.05:
            high_sim_count = sum(1 for s in similarities if s >= cfg.spam_similarity_threshold - 0.05)
            if high_sim_count >= 2:
                is_spam = True
                spam_score = avg_similarity
                reason = f"repeated_pattern:{high_sim_count}_similar"

        return is_spam, spam_score, reason

    def update_spam_score(
        self,
        agent_id: str,
        group_id: str,
        user_id: str,
        spam_detected: bool
    ):
        """
        Update spam score for a user in a group.

        Spam score increases when spam is detected, decays over time.
        Used to track repeat offenders.
        """
        doc_id = f"{self._get_doc_id(agent_id, group_id)}_spam_{user_id.replace(':', '_')}"
        cfg = self.threshold_config

        if self._use_memory:
            current = self._memory_store.get(doc_id) or {"spam_score": 0.0, "last_update": 0}
            # Apply time decay
            time_since = time.time() - current.get("last_update", 0)
            hours_passed = time_since / 3600
            decayed_score = current.get("spam_score", 0.0) * (cfg.spam_score_decay ** hours_passed)

            # Update score
            new_score = decayed_score + (1.0 if spam_detected else -0.1)
            new_score = max(0.0, new_score)

            self._memory_store.set(doc_id, {
                "spam_score": new_score,
                "last_update": time.time(),
                "total_spam_count": current.get("total_spam_count", 0) + (1 if spam_detected else 0)
            })
            return new_score

        try:
            doc_ref = self._get_db().collection("spam_scores").document(doc_id)
            doc = doc_ref.get()
            current = doc.to_dict() if doc.exists else {"spam_score": 0.0, "last_update": time.time()}

            # Apply time decay
            last_update = current.get("last_update")
            if hasattr(last_update, 'timestamp'):
                last_update = last_update.timestamp()
            time_since = time.time() - (last_update or time.time())
            hours_passed = time_since / 3600
            decayed_score = current.get("spam_score", 0.0) * (cfg.spam_score_decay ** hours_passed)

            # Update score
            new_score = decayed_score + (1.0 if spam_detected else -0.1)
            new_score = max(0.0, new_score)

            doc_ref.set({
                "spam_score": new_score,
                "last_update": firestore.SERVER_TIMESTAMP if firestore else time.time(),
                "total_spam_count": firestore.Increment(1) if spam_detected and firestore else current.get("total_spam_count", 0) + (1 if spam_detected else 0),
                "user_id": user_id,
                "group_id": group_id,
                "agent_id": agent_id
            }, merge=True)

            return new_score
        except Exception as e:
            print(f"[SpamScore] Error updating: {e}")
            return 0.0

    def is_user_blocked(self, agent_id: str, group_id: str, user_id: str) -> bool:
        """Check if a user should be blocked due to high spam score."""
        doc_id = f"{self._get_doc_id(agent_id, group_id)}_spam_{user_id.replace(':', '_')}"
        cfg = self.threshold_config

        if self._use_memory:
            data = self._memory_store.get(doc_id)
            return data.get("spam_score", 0.0) >= cfg.spam_block_threshold if data else False

        try:
            doc = self._get_db().collection("spam_scores").document(doc_id).get()
            if doc.exists:
                return doc.to_dict().get("spam_score", 0.0) >= cfg.spam_block_threshold
            return False
        except:
            return False

    def get_user_spam_info(self, agent_id: str, group_id: str, user_id: str) -> dict:
        """Get spam information for a user."""
        doc_id = f"{self._get_doc_id(agent_id, group_id)}_spam_{user_id.replace(':', '_')}"
        cfg = self.threshold_config

        default_info = {
            "user_id": user_id,
            "agent_id": agent_id,
            "group_id": group_id,
            "spam_score": 0.0,
            "is_blocked": False,
            "block_threshold": cfg.spam_block_threshold,
            "total_spam_count": 0,
            "last_update": None
        }

        if self._use_memory:
            data = self._memory_store.get(doc_id)
            if data:
                spam_score = data.get("spam_score", 0.0)
                return {
                    **default_info,
                    "spam_score": spam_score,
                    "is_blocked": spam_score >= cfg.spam_block_threshold,
                    "total_spam_count": data.get("total_spam_count", 0),
                    "last_update": data.get("last_update")
                }
            return default_info

        try:
            doc = self._get_db().collection("spam_scores").document(doc_id).get()
            if doc.exists:
                data = doc.to_dict()
                spam_score = data.get("spam_score", 0.0)
                last_update = data.get("last_update")
                if hasattr(last_update, 'isoformat'):
                    last_update = last_update.isoformat()
                return {
                    **default_info,
                    "spam_score": spam_score,
                    "is_blocked": spam_score >= cfg.spam_block_threshold,
                    "total_spam_count": data.get("total_spam_count", 0),
                    "last_update": last_update
                }
            return default_info
        except Exception as e:
            print(f"[SpamInfo] Error: {e}")
            return default_info

    def unblock_user(self, agent_id: str, group_id: str, user_id: str) -> dict:
        """Manually unblock a user by resetting their spam score."""
        doc_id = f"{self._get_doc_id(agent_id, group_id)}_spam_{user_id.replace(':', '_')}"

        if self._use_memory:
            self._memory_store.set(doc_id, {
                "spam_score": 0.0,
                "last_update": time.time(),
                "manually_unblocked": True,
                "unblocked_at": time.time()
            }, merge=True)
            return {"success": True, "new_spam_score": 0.0}

        try:
            from google.cloud import firestore
            doc_ref = self._get_db().collection("spam_scores").document(doc_id)
            doc_ref.set({
                "spam_score": 0.0,
                "last_update": firestore.SERVER_TIMESTAMP,
                "manually_unblocked": True,
                "unblocked_at": firestore.SERVER_TIMESTAMP
            }, merge=True)
            return {"success": True, "new_spam_score": 0.0}
        except Exception as e:
            print(f"[Unblock] Error: {e}")
            return {"success": False, "error": str(e)}

    def get_blocked_users(self, agent_id: str, group_id: str = None) -> list:
        """Get all blocked users for an agent (optionally filtered by group)."""
        cfg = self.threshold_config
        blocked = []

        if self._use_memory:
            # In-memory: scan all spam docs
            prefix = f"{agent_id}_"
            for key, data in self._memory_store._store.items():
                if key.startswith(prefix) and "_spam_" in key:
                    if data.get("spam_score", 0.0) >= cfg.spam_block_threshold:
                        blocked.append({
                            "doc_id": key,
                            "spam_score": data.get("spam_score", 0.0),
                            "total_spam_count": data.get("total_spam_count", 0)
                        })
            return blocked

        try:
            query = self._get_db().collection("spam_scores").where(
                "agent_id", "==", agent_id
            ).where(
                "spam_score", ">=", cfg.spam_block_threshold
            )

            if group_id:
                query = query.where("group_id", "==", group_id)

            docs = query.get()
            for doc in docs:
                data = doc.to_dict()
                blocked.append({
                    "doc_id": doc.id,
                    "user_id": data.get("user_id"),
                    "group_id": data.get("group_id"),
                    "spam_score": data.get("spam_score", 0.0),
                    "total_spam_count": data.get("total_spam_count", 0)
                })
            return blocked
        except Exception as e:
            print(f"[BlockedUsers] Error: {e}")
            return []

    def record_message(
        self,
        agent_id: str,
        group_id: str,
        importance_score: float = 0.5
    ):
        """
        Record a new message - fire and forget, single write.
        """
        doc_id = self._get_doc_id(agent_id, group_id)

        if self._use_memory:
            # In-memory mode
            updates = {
                "message_count": InMemoryIncrement(1),
                "last_message_time": time.time(),
            }
            if importance_score >= 0.8:
                updates["high_importance_pending"] = InMemoryIncrement(1)
            self._memory_store.set(doc_id, updates, merge=True)
            return

        # Firestore mode
        try:
            updates = {
                "message_count": firestore.Increment(1),
                "last_message_time": firestore.SERVER_TIMESTAMP,
            }

            if importance_score >= 0.8:
                updates["high_importance_pending"] = firestore.Increment(1)

            self._get_db().collection(self.COLLECTION).document(doc_id).set(
                updates, merge=True
            )
        except Exception as e:
            # Non-blocking - just log and continue
            print(f"[AdaptiveThreshold] Warning: {e}")

    def record_processing(self, agent_id: str, group_id: str, processed_count: int):
        """Record that batch processing occurred."""
        now = datetime.now(timezone.utc)

        if self._use_memory:
            self._update_metrics(agent_id, group_id, {
                "last_process_time": time.time(),
                "total_processed": InMemoryIncrement(processed_count),
                "high_importance_pending": 0
            })
        else:
            self._update_metrics(agent_id, group_id, {
                "last_process_time": now,
                "total_processed": firestore.Increment(processed_count),
                "high_importance_pending": 0
            })

    def should_process(
        self,
        agent_id: str,
        group_id: str,
        pending_count: int
    ) -> Tuple[bool, int]:
        """
        Determine if batch processing should occur.

        Args:
            agent_id: Agent identifier
            group_id: Group/conversation identifier
            pending_count: Current count of unprocessed messages

        Returns:
            Tuple of (should_process: bool, recommended_batch_size: int)
        """
        metrics = self._get_metrics(agent_id, group_id)
        cfg = self.threshold_config

        # Calculate adaptive batch size
        batch_size = self._calculate_batch_size(metrics)

        # ============================================================
        # Trigger Conditions
        # ============================================================

        # 1. Enough messages accumulated
        if pending_count >= batch_size:
            print(f"[AdaptiveThreshold] Trigger: batch size ({pending_count} >= {batch_size})")
            return True, batch_size

        # 2. Time-based trigger
        last_process = metrics.get("last_process_time")
        if last_process and pending_count > 0:
            if hasattr(last_process, 'timestamp'):
                last_process_ts = last_process.timestamp()
            else:
                last_process_ts = last_process

            time_since = time.time() - last_process_ts

            if time_since >= cfg.max_wait_time_seconds:
                print(f"[AdaptiveThreshold] Trigger: max wait time ({time_since:.0f}s)")
                return True, pending_count

        # 3. Urgency trigger
        high_importance = metrics.get("high_importance_pending", 0)
        if high_importance >= cfg.urgency_trigger_count:
            print(f"[AdaptiveThreshold] Trigger: urgency ({high_importance} high-importance)")
            return True, min(pending_count, cfg.urgency_batch_size)

        return False, batch_size

    def get_threshold(self, agent_id: str, group_id: str) -> int:
        """Get current threshold for a group."""
        metrics = self._get_metrics(agent_id, group_id)
        return self._calculate_batch_size(metrics)

    def get_metrics_debug(self, agent_id: str, group_id: str) -> Dict[str, Any]:
        """Get metrics for debugging."""
        metrics = self._get_metrics(agent_id, group_id)
        return {
            "group_id": group_id,
            "message_count": metrics.get("message_count", 0),
            "messages_per_hour": self._calculate_messages_per_hour(metrics),
            "high_importance_pending": metrics.get("high_importance_pending", 0),
            "current_threshold": self._calculate_batch_size(metrics),
            "total_processed": metrics.get("total_processed", 0)
        }


# ============================================================
# Convenience functions for stateless usage
# ============================================================

def should_process_batch(
    agent_id: str,
    group_id: str,
    pending_count: int,
    importance_score: float = 0.5,
    firestore_client=None
) -> Tuple[bool, int]:
    """
    Stateless convenience function.

    Usage in main.py:
        from core.adaptive_threshold import should_process_batch, record_batch_processed

        should_process, batch_size = should_process_batch(
            agent_id=self.agent_id,
            group_id=effective_group_id,
            pending_count=len(unprocessed),
            importance_score=0.5
        )

        if should_process:
            # Process batch...
            record_batch_processed(agent_id, group_id, len(processed))
    """
    manager = AdaptiveThresholdManager(firestore_client=firestore_client)
    manager.record_message(agent_id, group_id, importance_score)
    return manager.should_process(agent_id, group_id, pending_count)


def record_batch_processed(
    agent_id: str,
    group_id: str,
    count: int,
    firestore_client=None
):
    """Record that a batch was processed."""
    manager = AdaptiveThresholdManager(firestore_client=firestore_client)
    manager.record_processing(agent_id, group_id, count)


def get_adaptive_threshold(
    agent_id: str,
    group_id: str,
    firestore_client=None
) -> int:
    """Get current adaptive threshold for a group."""
    manager = AdaptiveThresholdManager(firestore_client=firestore_client)
    return manager.get_threshold(agent_id, group_id)
