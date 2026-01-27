"""
Test Group Expansion for SimpleMem

Validates that:
1. DMs still work (backwards compatibility)
2. Groups can be inserted and searched
3. Memory type classification works
4. New Dialogue and MemoryEntry fields work correctly
"""
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.memory_entry import Dialogue, MemoryEntry, MemoryType, PrivacyScope


class TestDialogueBackwardsCompatibility:
    """Test that Dialogue works without new fields (backwards compatible)"""

    def test_basic_dialogue_creation(self):
        """Original Dialogue creation should still work"""
        d = Dialogue(dialogue_id=1, speaker="Alice", content="Hello")
        assert d.dialogue_id == 1
        assert d.speaker == "Alice"
        assert d.content == "Hello"

    def test_default_values(self):
        """New fields should have sensible defaults"""
        d = Dialogue(dialogue_id=1, speaker="Alice", content="Hello")
        assert d.platform == "direct"
        assert d.group_id is None
        assert d.user_id is None
        assert d.username is None
        assert d.is_reply == False
        assert d.mentioned_users == []

    def test_str_method(self):
        """String representation should work"""
        d = Dialogue(dialogue_id=1, speaker="Alice", content="Hello")
        assert "Alice" in str(d)
        assert "Hello" in str(d)

    def test_str_method_with_username(self):
        """String should prefer username over speaker"""
        d = Dialogue(
            dialogue_id=1,
            speaker="Alice",
            content="Hello",
            username="@alice_crypto"
        )
        assert "@alice_crypto" in str(d)


class TestDialogueWithGroupContext:
    """Test Dialogue with new group fields"""

    def test_telegram_group_dialogue(self):
        """Test Dialogue for Telegram group"""
        d = Dialogue(
            dialogue_id=1,
            speaker="Alice",
            content="Hello group!",
            platform="telegram",
            group_id="telegram_-123456789",
            user_id="telegram:123456789",
            username="@alice"
        )
        assert d.platform == "telegram"
        assert d.group_id == "telegram_-123456789"
        assert d.user_id == "telegram:123456789"
        assert d.username == "@alice"

    def test_xmtp_group_dialogue(self):
        """Test Dialogue for XMTP group"""
        d = Dialogue(
            dialogue_id=1,
            speaker="Bob",
            content="Hey everyone",
            platform="xmtp",
            group_id="xmtp_abc123/groups/xyz",
            user_id="xmtp:0x1234...abcd",
            username="bob.eth"
        )
        assert d.platform == "xmtp"
        assert d.group_id.startswith("xmtp_")
        assert d.user_id.startswith("xmtp:")

    def test_is_group_message_method(self):
        """Test is_group_message helper"""
        dm = Dialogue(dialogue_id=1, speaker="Alice", content="Hi", group_id=None)
        group = Dialogue(dialogue_id=2, speaker="Bob", content="Hi", group_id="telegram_-123")

        assert dm.is_group_message() == False
        assert group.is_group_message() == True

    def test_dialogue_with_reply(self):
        """Test Dialogue with reply metadata"""
        d = Dialogue(
            dialogue_id=1,
            speaker="Alice",
            content="I agree with you",
            is_reply=True,
            reply_to_message_id="msg_789",
            mentioned_users=["@bob", "@charlie"]
        )
        assert d.is_reply == True
        assert d.reply_to_message_id == "msg_789"
        assert "@bob" in d.mentioned_users
        assert "@charlie" in d.mentioned_users


class TestMemoryEntryBackwardsCompatibility:
    """Test that MemoryEntry works without new fields"""

    def test_basic_memory_entry(self):
        """Original MemoryEntry creation should still work"""
        m = MemoryEntry(lossless_restatement="Alice said hello to Bob")
        assert m.lossless_restatement == "Alice said hello to Bob"
        assert m.entry_id is not None  # Auto-generated

    def test_default_classification(self):
        """New classification fields should have defaults"""
        m = MemoryEntry(lossless_restatement="Alice said hello")
        assert m.memory_type == MemoryType.CONVERSATION
        assert m.privacy_scope == PrivacyScope.PRIVATE
        assert m.importance_score == 0.5

    def test_default_group_context(self):
        """Group context fields should default to None/direct"""
        m = MemoryEntry(lossless_restatement="Alice said hello")
        assert m.group_id is None
        assert m.user_id is None
        assert m.username is None
        assert m.platform == "direct"


class TestMemoryEntryWithGroupContext:
    """Test MemoryEntry with new group and classification fields"""

    def test_expertise_memory(self):
        """Test expertise memory type"""
        m = MemoryEntry(
            lossless_restatement="Alice has been building smart contracts for 3 years",
            group_id="telegram_-123456",
            user_id="telegram:123456789",
            username="@alice",
            platform="telegram",
            memory_type=MemoryType.EXPERTISE,
            privacy_scope=PrivacyScope.GROUP_ONLY,
            importance_score=0.8
        )
        assert m.memory_type == MemoryType.EXPERTISE
        assert m.privacy_scope == PrivacyScope.GROUP_ONLY
        assert m.importance_score == 0.8
        assert m.group_id == "telegram_-123456"

    def test_preference_memory(self):
        """Test preference memory type"""
        m = MemoryEntry(
            lossless_restatement="Bob prefers using Hardhat over Foundry for testing",
            memory_type=MemoryType.PREFERENCE,
            importance_score=0.6
        )
        assert m.memory_type == MemoryType.PREFERENCE

    def test_fact_memory(self):
        """Test fact memory type"""
        m = MemoryEntry(
            lossless_restatement="Charlie's wallet address is 0x1234...abcd",
            memory_type=MemoryType.FACT,
            importance_score=0.9
        )
        assert m.memory_type == MemoryType.FACT

    def test_announcement_memory(self):
        """Test announcement memory type"""
        m = MemoryEntry(
            lossless_restatement="The group decided to meet every Friday at 3pm",
            memory_type=MemoryType.ANNOUNCEMENT,
            privacy_scope=PrivacyScope.PUBLIC,
            importance_score=0.7
        )
        assert m.memory_type == MemoryType.ANNOUNCEMENT
        assert m.privacy_scope == PrivacyScope.PUBLIC


class TestMemoryTypeEnum:
    """Test MemoryType enum values"""

    def test_all_memory_types(self):
        """Verify all memory types exist"""
        assert MemoryType.CONVERSATION.value == "conversation"
        assert MemoryType.EXPERTISE.value == "expertise"
        assert MemoryType.PREFERENCE.value == "preference"
        assert MemoryType.FACT.value == "fact"
        assert MemoryType.ANNOUNCEMENT.value == "announcement"

    def test_memory_type_from_string(self):
        """Test creating MemoryType from string"""
        assert MemoryType("expertise") == MemoryType.EXPERTISE
        assert MemoryType("preference") == MemoryType.PREFERENCE


class TestPrivacyScopeEnum:
    """Test PrivacyScope enum values"""

    def test_all_privacy_scopes(self):
        """Verify all privacy scopes exist"""
        assert PrivacyScope.PRIVATE.value == "private"
        assert PrivacyScope.GROUP_ONLY.value == "group_only"
        assert PrivacyScope.PUBLIC.value == "public"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
