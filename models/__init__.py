"""
Models package
"""
from .memory_entry import MemoryEntry, Dialogue
from .user_profile import UserProfile, TraitScore, Interest, EntityInfo
from .group_memory import (
    GroupMemory, UserMemory, InteractionMemory,
    CrossGroupMemory, CrossAgentLink,
    MemoryLevel, MemoryType, PrivacyScope
)

__all__ = [
    'MemoryEntry', 'Dialogue',
    'UserProfile', 'TraitScore', 'Interest', 'EntityInfo',
    'GroupMemory', 'UserMemory', 'InteractionMemory',
    'CrossGroupMemory', 'CrossAgentLink',
    'MemoryLevel', 'MemoryType', 'PrivacyScope'
]
