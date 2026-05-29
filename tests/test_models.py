"""Tests for core/memory/models.py — EpisodicMessage, MemoryEntry, UserProfile."""

import pytest
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

from core.memory.models import EpisodicMessage, MemoryEntry, UserProfile


_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class TestEpisodicMessage:
    def test_fields(self):
        msg = EpisodicMessage(id=1, user_id="u1", role="user", content="hello", created_at=_NOW)
        assert msg.id == 1
        assert msg.user_id == "u1"
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.created_at == _NOW

    def test_frozen(self):
        msg = EpisodicMessage(id=1, user_id="u1", role="user", content="hello", created_at=_NOW)
        with pytest.raises(FrozenInstanceError):
            msg.content = "changed"

    def test_assistant_role(self):
        msg = EpisodicMessage(id=2, user_id="u1", role="assistant", content="reply", created_at=_NOW)
        assert msg.role == "assistant"


class TestMemoryEntry:
    def test_fields(self):
        entry = MemoryEntry(id="m1", content="likes pizza", short_term=False, created_at=_NOW)
        assert entry.id == "m1"
        assert entry.content == "likes pizza"
        assert entry.short_term is False
        assert entry.created_at == _NOW

    def test_short_term_true(self):
        entry = MemoryEntry(id="m2", content="watching NBA playoffs", short_term=True, created_at=_NOW)
        assert entry.short_term is True

    def test_frozen(self):
        entry = MemoryEntry(id="m1", content="likes pizza", short_term=False, created_at=_NOW)
        with pytest.raises(FrozenInstanceError):
            entry.content = "changed"


class TestUserProfile:
    def _entry(self, content: str, short_term: bool = False) -> MemoryEntry:
        return MemoryEntry(id="x", content=content, short_term=short_term, created_at=_NOW)

    def test_fields(self):
        entry = self._entry("is vegetarian")
        profile = UserProfile(user_id="u1", facts=(entry,))
        assert profile.user_id == "u1"
        assert len(profile.facts) == 1

    def test_frozen(self):
        profile = UserProfile(user_id="u1", facts=())
        with pytest.raises(FrozenInstanceError):
            profile.user_id = "changed"

    def test_summary_contains_fact_content(self):
        entry = self._entry("lives in Toronto")
        profile = UserProfile(user_id="u1", facts=(entry,))
        assert "lives in Toronto" in profile.summary()

    def test_summary_multiple_facts(self):
        facts = (self._entry("is vegetarian"), self._entry("loves spicy food"))
        profile = UserProfile(user_id="u1", facts=facts)
        summary = profile.summary()
        assert "is vegetarian" in summary
        assert "loves spicy food" in summary

    def test_summary_empty_facts(self):
        profile = UserProfile(user_id="u1", facts=())
        summary = profile.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_summary_returns_string(self):
        profile = UserProfile(user_id="u1", facts=(self._entry("test fact"),))
        assert isinstance(profile.summary(), str)
