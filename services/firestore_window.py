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
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Add a message to the in-memory window."""
        doc_id = str(uuid.uuid4())[:8]

        doc_data = {
            'doc_id': doc_id,
            'content': message,
            'username': username,
            'platform_identity': platform_identity or {},
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'processed': False,
            'metadata': metadata or {}
        }

        messages = self._get_messages(agent_id, group_id)
        messages.append(doc_data)

        # Maintain sliding window
        if len(messages) > config.RECENT_WINDOW_SIZE:
            messages.pop(0)  # Remove oldest

        return doc_id

    def get_recent(
        self,
        agent_id: str,
        group_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent messages."""
        messages = self._get_messages(agent_id, group_id)
        # Return oldest first (for context)
        return [
            {
                'content': m['content'],
                'username': m['username'],
                'timestamp': m['timestamp'],
                'message_id': m['doc_id']
            }
            for m in messages[-limit:]
        ]

    def get_unprocessed(
        self,
        agent_id: str,
        group_id: str,
        min_count: int = 10
    ) -> List[Dict[str, Any]]:
        """Get unprocessed messages for batch processing."""
        messages = self._get_messages(agent_id, group_id)
        unprocessed = [m for m in messages if not m.get('processed', False)]

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

    def __init__(self):
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
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Add a message to the recent window and maintain sliding window.

        Args:
            agent_id: Agent ID
            group_id: Group ID
            message: Message content
            username: Sender username
            platform_identity: Platform identity dict
            metadata: Additional metadata (message_id, timestamp, etc.)

        Returns:
            Document ID if successful, None otherwise
        """
        if not self._enabled:
            return None

        try:
            collection = self._get_collection(agent_id, group_id)
            if not collection:
                return None

            # Prepare document data
            doc_data = {
                'content': message,
                'username': username,
                'platform_identity': platform_identity or {},
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'processed': False,  # Mark for batch processing
                'metadata': metadata or {}
            }

            # Add new message
            doc_ref = collection.add(doc_data)
            doc_id = doc_ref[1].id

            # Maintain sliding window - keep only last N messages
            self._maintain_window_size(collection, config.RECENT_WINDOW_SIZE)

            return doc_id

        except Exception as e:
            print(f"[FirestoreWindow] Error adding message: {e}")
            return None

    def _maintain_window_size(self, collection, max_size: int):
        """Maintain sliding window by deleting oldest messages beyond max_size"""
        try:
            # Get all messages ordered by timestamp DESC (newest first)
            all_msgs = collection.order_by(
                'timestamp',
                direction=firestore.Query.DESCENDING
            ).limit(max_size + 1).get()

            messages = list(all_msgs)

            # If we have more than max_size, delete the oldest
            if len(messages) > max_size:
                # The last one in DESC order is the oldest
                oldest = messages[-1]
                collection.document(oldest.id).delete()
                print(f"[FirestoreWindow] Slid window - deleted oldest message")

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
                    'message_id': m.id
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

            # Only return if we have enough for a batch
            if len(messages) >= min_count:
                results = []
                for m in messages:
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
