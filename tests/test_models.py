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

    def test_summary_groups_dimensioned_facts_under_bold_header(self):
        facts = (
            self._entry("hobby: loves hiking"),
            self._entry("hobby: plays piano"),
            self._entry("work: software developer"),
        )
        summary = UserProfile(user_id="u1", facts=facts).summary()
        assert "**Hobby**" in summary
        assert "**Work**" in summary
        hobby_pos = summary.index("**Hobby**")
        assert summary.index("loves hiking") > hobby_pos
        assert summary.index("plays piano") > hobby_pos

    def test_summary_capitalizes_dimension_header(self):
        profile = UserProfile(user_id="u1", facts=(self._entry("diet: vegetarian"),))
        assert "**Diet**" in profile.summary()

    def test_summary_plain_facts_still_appear(self):
        facts = (
            self._entry("diet: vegetarian"),
            self._entry("no dimension here"),
        )
        summary = UserProfile(user_id="u1", facts=facts).summary()
        assert "vegetarian" in summary
        assert "no dimension here" in summary

    def test_summary_groups_separated_by_blank_line(self):
        facts = (
            self._entry("diet: vegetarian"),
            self._entry("work: engineer"),
        )
        summary = UserProfile(user_id="u1", facts=facts).summary()
        assert "\n\n" in summary

    def test_summary_groups_by_dimension_attribute(self):
        def _dim_entry(content: str, dimension: str) -> MemoryEntry:
            return MemoryEntry(id="x", content=content, short_term=False,
                               created_at=_NOW, dimension=dimension)

        facts = (
            _dim_entry("loves hiking", "hobby"),
            _dim_entry("plays piano", "hobby"),
            _dim_entry("software developer", "work"),
        )
        summary = UserProfile(user_id="u1", facts=facts).summary()
        assert "**Hobby**" in summary
        assert "**Work**" in summary
        assert "- loves hiking" in summary
        assert "- plays piano" in summary
        assert "- software developer" in summary

    def test_summary_dimension_attribute_takes_precedence_over_content_split(self):
        entry = MemoryEntry(id="x", content="raw mem0 sentence", short_term=False,
                            created_at=_NOW, dimension="goal")
        summary = UserProfile(user_id="u1", facts=(entry,)).summary()
        assert "**Goal**" in summary
        assert "- raw mem0 sentence" in summary

    def test_summary_ungrouped_facts_go_to_other(self):
        entry = self._entry("some random fact with no dimension")
        summary = UserProfile(user_id="u1", facts=(entry,)).summary()
        assert "**Other**" in summary
        assert "some random fact with no dimension" in summary

