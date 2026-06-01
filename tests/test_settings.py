"""Tests for core/settings.py — Settings validation."""

import pytest
from pydantic import ValidationError

from core.settings import Settings


def _base_env(**overrides) -> dict:
    """Minimum env vars needed to construct a valid Settings object."""
    base = {
        "discord_bot_token": "tok",
        "discord_guild_id": "123",
        "discord_user_id": "456",
        "openai_api_key": "sk-test",
        "llm_provider": "openai",
    }
    base.update(overrides)
    return base


class TestDigestTimezone:
    def test_valid_iana_timezone_passes(self):
        s = Settings(**_base_env(digest_timezone="America/New_York"))
        assert s.digest_timezone == "America/New_York"

    def test_default_timezone_is_valid(self):
        s = Settings(**_base_env())
        assert s.digest_timezone == "America/Toronto"

    def test_utc_is_valid(self):
        s = Settings(**_base_env(digest_timezone="UTC"))
        assert s.digest_timezone == "UTC"

    def test_invalid_timezone_raises(self):
        with pytest.raises(ValidationError, match="IANA timezone"):
            Settings(**_base_env(digest_timezone="Not/AReal/Zone"))

    def test_empty_timezone_raises(self):
        with pytest.raises(ValidationError):
            Settings(**_base_env(digest_timezone=""))


class TestDigestTimes:
    def test_default_digest_time(self):
        s = Settings(**_base_env())
        assert s.digest_time == "09:00"

    def test_custom_digest_time(self):
        s = Settings(**_base_env(digest_time="07:30"))
        assert s.digest_time == "07:30"


class TestTaskTokenBudget:
    def test_default_task_token_budget(self):
        s = Settings(**_base_env())
        assert s.task_token_budget == 8000

    def test_custom_task_token_budget(self):
        s = Settings(**_base_env(task_token_budget="5000"))
        assert s.task_token_budget == 5000


class TestLlmKeyValidation:
    def test_openai_requires_key(self):
        with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
            Settings(**_base_env(openai_api_key="", llm_provider="openai"))

    def test_openrouter_requires_key(self):
        with pytest.raises(ValidationError, match="OPENROUTER_API_KEY"):
            Settings(**_base_env(
                openai_api_key="",
                openrouter_api_key="",
                llm_provider="openrouter",
            ))

    def test_openrouter_with_key_passes(self):
        s = Settings(**_base_env(
            openai_api_key="",
            openrouter_api_key="or-key",
            llm_provider="openrouter",
        ))
        assert s.llm_provider == "openrouter"
