"""
Firestore Window Store - Recent messages window for immediate context

This provides a sliding window of the most recent messages in a group.
Used for immediate context without needing to wait for LLM processing.

Architecture:
- Messages added to Firestore immediately (low latency)
- Sliding window keeps only last N messages
- When threshold reached, trigger batch processing
- Local LanceDB used for persistent memory with embeddings

For local testing, set USE_LOCAL_STORAGE=true to use in-memory storage.
"""
import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from collections import defaultdict

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIRESTORE_AVAILABLE = True
except ImportError:
    FIRESTORE_AVAILABLE = False
    print("[FirestoreWindow] firebase-admin not installed - Firestore features disabled")

import config


class InMemoryWindowStore:
    """
    In-memory implementation of window store for local testing.

    Mimics FirestoreWindowStore interface but stores everything in memory.
    Fast and doesn't require network calls - perfect for tests.
    """

    def __init__(self):
        # Structure: {agent_id: {group_id: [messages]}}
        self._messages: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        self._enabled = True
        self.db = None  # Compatibility with FirestoreWindowStore
        print("[InMemoryWindow] Using in-memory storage for testing")

    def is_enabled(self) -> bool:
        return self._enabled

    def _get_messages(self, agent_id: str, group_id: str) -> List[Dict[str, Any]]:
        return self._messages[agent_id][group_id]

    def add_message(
        self,
        agent_id: str,
        group_id: str,
        message: str,
        username: str,
        platform_identity: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Add a message to the in-memory window with spam detection."""
        import numpy as np

        result = {
            'doc_id': None,
            'is_spam': False,
            'is_blocked': False,
            'spam_score': 0.0,
            'spam_reason': 'ok'
        }

        doc_id = str(uuid.uuid4())[:8]
        user_id = platform_identity.get('user_id', '') if platform_identity else ''

        # Simple spam detection for in-memory mode
        if embedding and user_id:
            messages = self._get_messages(agent_id, group_id)
            user_recent = [
                m for m in messages[-10:]
                if m.get('platform_identity', {}).get('user_id') == user_id
                and m.get('embedding') is not None
            ]

            if user_recent:
                msg_vec = np.array(embedding[:384] if len(embedding) > 384 else embedding)
                for m in user_recent[-5:]:
                    recent_vec = np.array(m['embedding'])
                    similarity = np.dot(msg_vec, recent_vec) / (
                        np.linalg.norm(msg_vec) * np.linalg.norm(recent_vec) + 1e-8
                    )
                    if similarity >= 0.92:
                        result['is_spam'] = True
                        result['spam_score'] = similarity
                        result['spam_reason'] = f"high_similarity:{similarity:.3f}"
                        print(f"[InMemoryWindow] Spam detected: {result['spam_reason']}")
                        break

        doc_data = {
            'doc_id': doc_id,
            'content': message,
            'username': username,
            'platform_identity': platform_identity or {},
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'processed': False,
            'is_spam': result['is_spam'],  # Track spam status
            'spam_score': result['spam_score'],
            'metadata': metadata or {},
            'embedding': embedding[:384] if embedding and len(embedding) > 384 else embedding
        }

        messages = self._get_messages(agent_id, group_id)
        messages.append(doc_data)

        # Smart window maintenance: prioritize keeping unprocessed non-spam
        window_size = getattr(config, 'RECENT_WINDOW_SIZE', 15)
        while len(messages) > window_size:
            # Priority 1: Remove oldest processed messages
            processed_idx = next(
                (i for i, m in enumerate(messages) if m.get('processed', False)),
                None
            )
            if processed_idx is not None:
                messages.pop(processed_idx)
                continue

            # Priority 2: Remove oldest spam messages (keep for context but expendable)
            spam_idx = next(
                (i for i, m in enumerate(messages) if m.get('is_spam', False)),
                None
            )
            if spam_idx is not None:
                messages.pop(spam_idx)
                continue

            # Priority 3: Remove oldest message (last resort)
            messages.pop(0)

        result['doc_id'] = doc_id
        return result

    def get_recent(
        self,
        agent_id: str,
        group_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent messages with full metadata."""
        messages = self._get_messages(agent_id, group_id)
        # Return oldest first (for context)
        return [
            {
                'content': m['content'],
                'username': m['username'],
                'timestamp': m['timestamp'],
                'message_id': m['doc_id'],
                'metadata': m.get('metadata', {}),
                'platform_identity': m.get('platform_identity', {})
            }
            for m in messages[-limit:]
        ]

    def get_unprocessed(
        self,
        agent_id: str,
        group_id: str,
        min_count: int = 10
    ) -> List[Dict[str, Any]]:
        """Get unprocessed non-spam messages for batch processing."""
        messages = self._get_messages(agent_id, group_id)
        # Filter: unprocessed AND not spam
        unprocessed = [
            m for m in messages
            if not m.get('processed', False) and not m.get('is_spam', False)
        ]

        spam_count = len([m for m in messages if m.get('is_spam', False) and not m.get('processed', False)])
        if spam_count > 0:
            print(f"[InMemoryWindow] Filtered out {spam_count} spam messages from batch")

        if len(unprocessed) >= min_count:
            return [
                {
                    'doc_id': m['doc_id'],
                    'content': m['content'],
                    'username': m['username'],
                    'platform_identity': m.get('platform_identity', {}),
                    'metadata': m.get('metadata', {}),
                    'timestamp': m['timestamp']
                }
                for m in unprocessed
            ]
        return []

    def mark_processed(self, agent_id: str, group_id: str, doc_ids: List[str]):
        """Mark messages as processed."""
        messages = self._get_messages(agent_id, group_id)
        doc_id_set = set(doc_ids)
        for m in messages:
            if m['doc_id'] in doc_id_set:
                m['processed'] = True

    def clear_recent(self, agent_id: str, group_id: str):
        """Clear all messages for a group."""
        self._messages[agent_id][group_id] = []

    def get_stats(self, agent_id: str, group_id: str) -> Dict[str, Any]:
        """Get statistics about the window."""
        messages = self._get_messages(agent_id, group_id)
        processed = sum(1 for m in messages if m.get('processed', False))
        return {
            'enabled': True,
            'total': len(messages),
            'processed': processed,
            'unprocessed': len(messages) - processed,
            'window_size': config.RECENT_WINDOW_SIZE
        }


class FirestoreWindowStore:
    """
    Stores recent messages in Firestore for immediate context access.

    Uses sliding window to maintain only last N messages per group.
    """

    def __init__(self, threshold_manager=None):
        self._threshold_manager = threshold_manager

        if not FIRESTORE_AVAILABLE:
            print("[FirestoreWindow] Firestore not available - running in local-only mode")
            self._enabled = False
            self.db = None
            return

        try:
            # Initialize Firebase Admin
            # In Cloud Run with Workload Identity, ADC is automatic
            # Locally, uses gcloud auth application-default login
            if not firebase_admin._apps:
                firebase_admin.initialize_app()

            self.db = firestore.client()
            self._enabled = True
            print(f"[FirestoreWindow] Connected to project: {config.FIRESTORE_PROJECT}")

            # Initialize threshold manager for dynamic window + spam detection
            if not self._threshold_manager:
                from core.adaptive_threshold import AdaptiveThresholdManager
                self._threshold_manager = AdaptiveThresholdManager(firestore_client=self.db)
        except Exception as e:
            print(f"[FirestoreWindow] Failed to connect: {e}")
            print("[FirestoreWindow] Running in local-only mode")
            self._enabled = False
            self.db = None

    def is_enabled(self) -> bool:
        """Check if Firestore is enabled and connected"""
        return self._enabled

    def _get_collection(self, agent_id: str, group_id: str):
        """Get Firestore collection path for a group"""
        if not self.db:
            return None
        # Path: agents/{agent_id}/groups/{group_id}/recent_messages
        prefix = config.FIRESTORE_COLLECTION_PREFIX
        return self.db.collection(f'{prefix}/{agent_id}/groups/{group_id}/recent_messages')

    def add_message(
        self,
        agent_id: str,
        group_id: str,
        message: str,
        username: str,
        platform_identity: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Add a message to the recent window with spam detection and dynamic window.

        Args:
            agent_id: Agent ID
            group_id: Group ID
            message: Message content
            username: Sender username
            platform_identity: Platform identity dict
            metadata: Additional metadata (message_id, timestamp, etc.)
            embedding: Optional message embedding for spam detection

        Returns:
            Dict with 'doc_id', 'is_spam', 'spam_score', 'spam_reason'
        """
        result = {
            'doc_id': None,
            'is_spam': False,
            'is_blocked': False,
            'spam_score': 0.0,
            'spam_reason': 'ok'
        }

        if not self._enabled:
            return result

        try:
            collection = self._get_collection(agent_id, group_id)
            if not collection:
                return result

            user_id = platform_identity.get('user_id', '') if platform_identity else ''

            # Spam detection if embedding provided
            if embedding and self._threshold_manager and user_id:
                # Get recent embeddings for spam check
                recent_embeddings = self._get_recent_embeddings(agent_id, group_id, limit=10)

                is_spam, spam_score, reason = self._threshold_manager.check_spam(
                    agent_id=agent_id,
                    group_id=group_id,
                    user_id=user_id,
                    message_embedding=embedding,
                    recent_embeddings=recent_embeddings
                )

                result['is_spam'] = is_spam
                result['spam_score'] = spam_score
                result['spam_reason'] = reason

                if is_spam:
                    # Update spam score for user
                    self._threshold_manager.update_spam_score(agent_id, group_id, user_id, True)
                    print(f"[FirestoreWindow] Spam detected from {username}: {reason}")

                    # Check if user should be blocked
                    if self._threshold_manager.is_user_blocked(agent_id, group_id, user_id):
                        print(f"[FirestoreWindow] User {username} blocked due to spam")
                        result['is_blocked'] = True
                        return result

            # Prepare document data
            doc_data = {
                'content': message,
                'username': username,
                'platform_identity': platform_identity or {},
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'processed': False,  # Mark for batch processing
                'is_spam': result['is_spam'],  # Track spam status
                'spam_score': result['spam_score'],
                'metadata': metadata or {}
            }

            # Store embedding if provided (truncated for Firestore size limits)
            if embedding:
                # Store first 384 dims (enough for similarity) to save space
                doc_data['embedding'] = embedding[:384] if len(embedding) > 384 else embedding

            # Add new message
            doc_ref = collection.add(doc_data)
            doc_id = doc_ref[1].id
            result['doc_id'] = doc_id

            # Get dynamic window size
            window_size = config.RECENT_WINDOW_SIZE
            if self._threshold_manager:
                window_size = self._threshold_manager.get_dynamic_window_size(agent_id, group_id)

            # Maintain sliding window with dynamic size
            self._maintain_window_size(collection, window_size)

            return result

        except Exception as e:
            print(f"[FirestoreWindow] Error adding message: {e}")
            return result

    def _get_recent_embeddings(
        self,
        agent_id: str,
        group_id: str,
        limit: int = 10
    ) -> List[tuple]:
        """
        Get recent message embeddings for spam detection.

        Returns:
            List of (embedding, user_id, timestamp) tuples
        """
        if not self._enabled:
            return []

        try:
            collection = self._get_collection(agent_id, group_id)
            if not collection:
                return []

            # Get recent messages with embeddings
            msgs = collection.order_by(
                'timestamp',
                direction=firestore.Query.DESCENDING
            ).limit(limit).get()

            results = []
            for msg in msgs:
                data = msg.to_dict()
                embedding = data.get('embedding')
                user_id = data.get('platform_identity', {}).get('user_id', '')
                timestamp = data.get('timestamp', '')
                if embedding:
                    results.append((embedding, user_id, timestamp))

            return results

        except Exception as e:
            print(f"[FirestoreWindow] Error getting embeddings: {e}")
            return []

    def _maintain_window_size(self, collection, max_size: int):
        """
        Smart window maintenance: prioritize keeping unprocessed non-spam messages.
        Delete order: 1) processed, 2) spam, 3) oldest
        """
        try:
            # Get all messages
            all_msgs = list(collection.get())

            if len(all_msgs) <= max_size:
                return

            # Categorize messages
            to_delete = []
            processed = []
            spam = []
            unprocessed_clean = []

            for msg in all_msgs:
                data = msg.to_dict()
                if data.get('processed', False):
                    processed.append((msg.id, data.get('timestamp', '')))
                elif data.get('is_spam', False):
                    spam.append((msg.id, data.get('timestamp', '')))
                else:
                    unprocessed_clean.append((msg.id, data.get('timestamp', '')))

            # Sort each category by timestamp (oldest first)
            processed.sort(key=lambda x: x[1])
            spam.sort(key=lambda x: x[1])
            unprocessed_clean.sort(key=lambda x: x[1])

            # Calculate how many to delete
            excess = len(all_msgs) - max_size

            # Priority 1: Delete oldest processed
            while excess > 0 and processed:
                to_delete.append(processed.pop(0)[0])
                excess -= 1

            # Priority 2: Delete oldest spam
            while excess > 0 and spam:
                to_delete.append(spam.pop(0)[0])
                excess -= 1

            # Priority 3: Delete oldest unprocessed (last resort)
            while excess > 0 and unprocessed_clean:
                to_delete.append(unprocessed_clean.pop(0)[0])
                excess -= 1

            # Batch delete
            if to_delete:
                for doc_id in to_delete:
                    collection.document(doc_id).delete()
                print(f"[FirestoreWindow] Smart window cleanup: deleted {len(to_delete)} messages")

        except Exception as e:
            print(f"[FirestoreWindow] Error maintaining window: {e}")

    def get_recent(
        self,
        agent_id: str,
        group_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent messages for immediate context.

        Args:
            agent_id: Agent ID
            group_id: Group ID
            limit: Max messages to return (default: 10)

        Returns:
            List of message dicts with content, username, timestamp
        """
        if not self._enabled:
            return []

        try:
            collection = self._get_collection(agent_id, group_id)
            if not collection:
                return []

            # Get messages ordered by timestamp ASC (oldest first for context)
            msgs = collection.order_by(
                'timestamp',
                direction=firestore.Query.ASCENDING
            ).limit(limit).get()

            results = []
            for m in msgs:
                data = m.to_dict()
                results.append({
                    'content': data.get('content', ''),
                    'username': data.get('username', ''),
                    'timestamp': data.get('timestamp', ''),
                    'message_id': m.id,
                    'metadata': data.get('metadata', {}),
                    'platform_identity': data.get('platform_identity', {})
                })
            return results

        except Exception as e:
            print(f"[FirestoreWindow] Error getting recent: {e}")
            return []

    def get_unprocessed(
        self,
        agent_id: str,
        group_id: str,
        min_count: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get unprocessed messages for batch processing.

        Args:
            agent_id: Agent ID
            group_id: Group ID
            min_count: Minimum count to trigger batch processing

        Returns:
            List of unprocessed message documents
        """
        if not self._enabled:
            return []

        try:
            collection = self._get_collection(agent_id, group_id)
            if not collection:
                return []

            # Get all unprocessed messages
            msgs = collection.where(
                filter=firestore.FieldFilter('processed', '==', False)
            ).get()
            messages = list(msgs)

            # Filter out spam messages - don't process spam into memories
            non_spam_messages = []
            spam_count = 0
            for m in messages:
                data = m.to_dict()
                if not data.get('is_spam', False):
                    non_spam_messages.append(m)
                else:
                    spam_count += 1

            if spam_count > 0:
                print(f"[FirestoreWindow] Filtered out {spam_count} spam messages from batch")

            # Only return if we have enough non-spam messages for a batch
            if len(non_spam_messages) >= min_count:
                results = []
                for m in non_spam_messages:
                    data = m.to_dict()
                    results.append({
                        'doc_id': m.id,
                        'content': data.get('content', ''),
                        'username': data.get('username', ''),
                        'platform_identity': data.get('platform_identity', {}),
                        'metadata': data.get('metadata', {}),
                        'timestamp': data.get('timestamp', '')
                    })
                return results

            return []

        except Exception as e:
            print(f"[FirestoreWindow] Error getting unprocessed: {e}")
            return []

    def mark_processed(self, agent_id: str, group_id: str, doc_ids: List[str]):
        """
        Mark messages as processed.

        Args:
            agent_id: Agent ID
            group_id: Group ID
            doc_ids: List of document IDs to mark as processed
        """
        if not self._enabled:
            return

        try:
            collection = self._get_collection(agent_id, group_id)
            if not collection:
                return

            # Batch update
            batch = self.db.batch()
            for doc_id in doc_ids:
                doc_ref = collection.document(doc_id)
                batch.update(doc_ref, {'processed': True})

            batch.commit()
            print(f"[FirestoreWindow] Marked {len(doc_ids)} messages as processed")

        except Exception as e:
            print(f"[FirestoreWindow] Error marking processed: {e}")

    def mark_processed_to_lancedb(self, agent_id: str, group_id: str, doc_id: str):
        """
        Mark a single agent response message as processed to LanceDB.

        Args:
            agent_id: Agent ID
            group_id: Group ID
            doc_id: Document ID to mark as processed to LanceDB
        """
        if not self._enabled:
            return

        try:
            collection = self._get_collection(agent_id, group_id)
            if not collection:
                return

            doc_ref = collection.document(doc_id)
            doc_ref.update({'metadata.processed_to_lancedb': True})

        except Exception as e:
            print(f"[FirestoreWindow] Error marking processed_to_lancedb: {e}")

    def clear_recent(self, agent_id: str, group_id: str):
        """
        Clear all recent messages for a group (useful for testing).

        Args:
            agent_id: Agent ID
            group_id: Group ID
        """
        if not self._enabled:
            return

        try:
            collection = self._get_collection(agent_id, group_id)
            if not collection:
                return

            # Get all documents and delete
            msgs = collection.get()
            batch = self.db.batch()
            for msg in msgs:
                batch.delete(msg.reference)

            batch.commit()
            print(f"[FirestoreWindow] Cleared recent messages for {group_id}")

        except Exception as e:
            print(f"[FirestoreWindow] Error clearing: {e}")

    def get_stats(self, agent_id: str, group_id: str) -> Dict[str, Any]:
        """
        Get statistics about the recent window.

        Args:
            agent_id: Agent ID
            group_id: Group ID

        Returns:
            Dict with total, processed, unprocessed counts
        """
        if not self._enabled:
            return {'enabled': False}

        try:
            collection = self._get_collection(agent_id, group_id)
            if not collection:
                return {'enabled': False}

            # Get all messages
            all_msgs = list(collection.get())

            # Count processed/unprocessed
            processed = sum(1 for m in all_msgs if m.get('processed', False))
            unprocessed = len(all_msgs) - processed

            return {
                'enabled': True,
                'total': len(all_msgs),
                'processed': processed,
                'unprocessed': unprocessed,
                'window_size': config.RECENT_WINDOW_SIZE
            }

        except Exception as e:
            print(f"[FirestoreWindow] Error getting stats: {e}")
            return {'enabled': False, 'error': str(e)}


# Singleton instance
_window_store_instance = None


def get_firestore_store():
    """
    Get or create the singleton window store instance.

    Returns InMemoryWindowStore if USE_LOCAL_STORAGE=true,
    otherwise returns FirestoreWindowStore.
    """
    global _window_store_instance
    if _window_store_instance is None:
        if config.USE_LOCAL_STORAGE:
            _window_store_instance = InMemoryWindowStore()
        else:
            _window_store_instance = FirestoreWindowStore()
    return _window_store_instance


def reset_store():
    """Reset the singleton instance (useful for tests)."""
    global _window_store_instance
    _window_store_instance = None
