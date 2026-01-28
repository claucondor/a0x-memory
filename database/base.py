"""
Base LanceDB connection manager.
Single connection shared across all tables.
"""
import os
from typing import Dict, Optional, Any

import lancedb

import config


class LanceDBConnection:
    """Singleton connection to LanceDB."""

    _instances: Dict[str, lancedb.DBConnection] = {}  # agent_id -> connection
    _global_instance: Optional[lancedb.DBConnection] = None
    _storage_options: Optional[Dict[str, Any]] = None

    @classmethod
    def configure(cls, storage_options: Optional[Dict[str, Any]] = None):
        """Configure storage options for cloud storage."""
        cls._storage_options = storage_options

    @classmethod
    def get_agent_db(cls, agent_id: str, db_base_path: str = None) -> lancedb.DBConnection:
        """Get or create agent-specific database connection."""
        if agent_id in cls._instances:
            return cls._instances[agent_id]

        base_path = db_base_path or config.LANCEDB_PATH
        agent_db_path = f"{base_path}/agents/{agent_id}"

        # Detect cloud storage
        is_cloud_storage = base_path.startswith(("gs://", "s3://", "az://"))

        if is_cloud_storage:
            db = lancedb.connect(agent_db_path, storage_options=cls._storage_options)
        else:
            os.makedirs(agent_db_path, exist_ok=True)
            db = lancedb.connect(agent_db_path)

        cls._instances[agent_id] = db
        return db

    @classmethod
    def get_global_db(cls, db_base_path: str = None) -> lancedb.DBConnection:
        """Get global database connection (cross-agent data)."""
        if cls._global_instance is not None:
            return cls._global_instance

        base_path = db_base_path or config.LANCEDB_PATH
        global_db_path = f"{base_path}/global"

        # Detect cloud storage
        is_cloud_storage = base_path.startswith(("gs://", "s3://", "az://"))

        if is_cloud_storage:
            db = lancedb.connect(global_db_path, storage_options=cls._storage_options)
        else:
            os.makedirs(global_db_path, exist_ok=True)
            db = lancedb.connect(global_db_path)

        cls._global_instance = db
        return db

    @classmethod
    def reset(cls):
        """Reset all connections (mainly for testing)."""
        cls._instances.clear()
        cls._global_instance = None
        cls._storage_options = None
