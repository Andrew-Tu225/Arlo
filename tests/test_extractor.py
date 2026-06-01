"""Tests for core/memory/extractor.py — passive profile extraction."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import core.memory.extractor as extractor
from core.memory.models import EpisodicMessage


def _make_message(role: str = "user", content: str = "I am vegetarian") -> EpisodicMessage:
    return EpisodicMessage(
        id=1,
        user_id="u1",
        role=role,
        content=content,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _make_llm_response(facts: list[dict]) -> MagicMock:
    msg = MagicMock()
    msg.content = json.dumps({"facts": facts})
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_client(facts: list[dict] | None = None) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_make_llm_response(facts or [])
    )
    return client


def _make_settings(interval: int = 10) -> MagicMock:
    settings = MagicMock()
    settings.profile_extraction_interval = interval
    return settings


class TestIntervalGate:
    @pytest.mark.asyncio
    async def test_skips_when_count_not_divisible(self):
        pool = MagicMock()
        with (
            patch("core.memory.extractor.get_settings", return_value=_make_settings(10)),
            patch("core.memory.extractor.db.count_user_messages", new_callable=AsyncMock, return_value=7),
            patch(
                "core.memory.extractor.db.get_recent_messages", new_callable=AsyncMock
            ) as mock_db,
        ):
            await extractor.maybe_extract(pool, "u1")
        mock_db.assert_not_called()

    @pytest.mark.asyncio
    async def test_runs_when_count_divisible(self):
        pool = MagicMock()
        with (
            patch("core.memory.extractor.get_settings", return_value=_make_settings(10)),
            patch("core.memory.extractor.db.count_user_messages", new_callable=AsyncMock, return_value=10),
            patch(
                "core.memory.extractor.db.get_recent_messages",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("core.memory.extractor.llm.get_client", return_value=_make_client()),
            patch("core.memory.extractor.store.add", new_callable=AsyncMock),
        ):
            await extractor.maybe_extract(pool, "u1")

    @pytest.mark.asyncio
    async def test_skips_when_pool_is_none(self):
        with patch(
            "core.memory.extractor.db.get_recent_messages", new_callable=AsyncMock
        ) as mock_db:
            await extractor.maybe_extract(None, "u1")
        mock_db.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_count_is_zero(self):
        pool = MagicMock()
        with (
            patch("core.memory.extractor.get_settings", return_value=_make_settings(10)),
            patch("core.memory.extractor.db.count_user_messages", new_callable=AsyncMock, return_value=0),
            patch(
                "core.memory.extractor.db.get_recent_messages", new_callable=AsyncMock
            ) as mock_db,
        ):
            await extractor.maybe_extract(pool, "u1")
        mock_db.assert_not_called()


class TestMessageFetching:
    @pytest.mark.asyncio
    async def test_fetches_n_recent_messages(self):
        pool = MagicMock()
        settings = _make_settings(interval=10)
        with (
            patch("core.memory.extractor.get_settings", return_value=settings),
            patch("core.memory.extractor.db.count_user_messages", new_callable=AsyncMock, return_value=10),
            patch(
                "core.memory.extractor.db.get_recent_messages",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_db,
            patch("core.memory.extractor.llm.get_client", return_value=_make_client()),
            patch("core.memory.extractor.store.add", new_callable=AsyncMock),
        ):
            await extractor.maybe_extract(pool, "u1")
        mock_db.assert_called_once_with(pool, user_id="u1", n=10)

    @pytest.mark.asyncio
    async def test_no_llm_call_when_no_messages(self):
        pool = MagicMock()
        client = _make_client()
        with (
            patch("core.memory.extractor.get_settings", return_value=_make_settings(10)),
            patch("core.memory.extractor.db.count_user_messages", new_callable=AsyncMock, return_value=10),
            patch(
                "core.memory.extractor.db.get_recent_messages",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("core.memory.extractor.llm.get_client", return_value=client),
            patch("core.memory.extractor.store.add", new_callable=AsyncMock),
        ):
            await extractor.maybe_extract(pool, "u1")
        client.chat.completions.create.assert_not_called()


class TestFactExtraction:
    @pytest.mark.asyncio
    async def test_calls_store_add_per_fact(self):
        pool = MagicMock()
        messages = [_make_message("user", "I am vegetarian"), _make_message("assistant", "Got it!")]
        facts = [
            {"dimension": "diet", "value": "vegetarian", "is_short_term": False},
            {"dimension": "hobby", "value": "cooking", "is_short_term": False},
        ]
        with (
            patch("core.memory.extractor.get_settings", return_value=_make_settings(10)),
            patch("core.memory.extractor.db.count_user_messages", new_callable=AsyncMock, return_value=10),
            patch(
                "core.memory.extractor.db.get_recent_messages",
                new_callable=AsyncMock,
                return_value=messages,
            ),
            patch("core.memory.extractor.llm.get_client", return_value=_make_client(facts)),
            patch("core.memory.extractor.store.add", new_callable=AsyncMock) as mock_add,
        ):
            await extractor.maybe_extract(pool, "u1")
        assert mock_add.call_count == 2

    @pytest.mark.asyncio
    async def test_passes_value_as_fact_text(self):
        pool = MagicMock()
        messages = [_make_message()]
        facts = [{"dimension": "diet", "value": "vegetarian", "is_short_term": False}]
        with (
            patch("core.memory.extractor.get_settings", return_value=_make_settings(10)),
            patch("core.memory.extractor.db.count_user_messages", new_callable=AsyncMock, return_value=10),
            patch(
                "core.memory.extractor.db.get_recent_messages",
                new_callable=AsyncMock,
                return_value=messages,
            ),
            patch("core.memory.extractor.llm.get_client", return_value=_make_client(facts)),
            patch("core.memory.extractor.store.add", new_callable=AsyncMock) as mock_add,
        ):
            await extractor.maybe_extract(pool, "u1")
        assert mock_add.call_args.args[0] == "vegetarian"

    @pytest.mark.asyncio
    async def test_passes_dimension_to_store(self):
        pool = MagicMock()
        messages = [_make_message()]
        facts = [{"dimension": "diet", "value": "vegetarian", "is_short_term": False}]
        with (
            patch("core.memory.extractor.get_settings", return_value=_make_settings(10)),
            patch("core.memory.extractor.db.count_user_messages", new_callable=AsyncMock, return_value=10),
            patch(
                "core.memory.extractor.db.get_recent_messages",
                new_callable=AsyncMock,
                return_value=messages,
            ),
            patch("core.memory.extractor.llm.get_client", return_value=_make_client(facts)),
            patch("core.memory.extractor.store.add", new_callable=AsyncMock) as mock_add,
        ):
            await extractor.maybe_extract(pool, "u1")
        assert mock_add.call_args.kwargs.get("dimension") == "diet"

    @pytest.mark.asyncio
    async def test_passes_short_term_true(self):
        pool = MagicMock()
        messages = [_make_message("user", "I'm in Tokyo this week")]
        facts = [{"dimension": "location", "value": "in Tokyo this week", "is_short_term": True}]
        with (
            patch("core.memory.extractor.get_settings", return_value=_make_settings(10)),
            patch("core.memory.extractor.db.count_user_messages", new_callable=AsyncMock, return_value=10),
            patch(
                "core.memory.extractor.db.get_recent_messages",
                new_callable=AsyncMock,
                return_value=messages,
            ),
            patch("core.memory.extractor.llm.get_client", return_value=_make_client(facts)),
            patch("core.memory.extractor.store.add", new_callable=AsyncMock) as mock_add,
        ):
            await extractor.maybe_extract(pool, "u1")
        assert mock_add.call_args.kwargs["short_term"] is True

    @pytest.mark.asyncio
    async def test_passes_short_term_false(self):
        pool = MagicMock()
        messages = [_make_message()]
        facts = [{"dimension": "diet", "value": "vegetarian", "is_short_term": False}]
        with (
            patch("core.memory.extractor.get_settings", return_value=_make_settings(10)),
            patch("core.memory.extractor.db.count_user_messages", new_callable=AsyncMock, return_value=10),
            patch(
                "core.memory.extractor.db.get_recent_messages",
                new_callable=AsyncMock,
                return_value=messages,
            ),
            patch("core.memory.extractor.llm.get_client", return_value=_make_client(facts)),
            patch("core.memory.extractor.store.add", new_callable=AsyncMock) as mock_add,
        ):
            await extractor.maybe_extract(pool, "u1")
        assert mock_add.call_args.kwargs["short_term"] is False

    @pytest.mark.asyncio
    async def test_no_store_add_when_facts_empty(self):
        pool = MagicMock()
        messages = [_make_message("user", "Hello there")]
        with (
            patch("core.memory.extractor.get_settings", return_value=_make_settings(10)),
            patch("core.memory.extractor.db.count_user_messages", new_callable=AsyncMock, return_value=10),
            patch(
                "core.memory.extractor.db.get_recent_messages",
                new_callable=AsyncMock,
                return_value=messages,
            ),
            patch("core.memory.extractor.llm.get_client", return_value=_make_client([])),
            patch("core.memory.extractor.store.add", new_callable=AsyncMock) as mock_add,
        ):
            await extractor.maybe_extract(pool, "u1")
        mock_add.assert_not_called()


class TestErrorSwallowing:
    @pytest.mark.asyncio
    async def test_swallows_db_exception(self):
        pool = MagicMock()
        with (
            patch("core.memory.extractor.get_settings", return_value=_make_settings(10)),
            patch("core.memory.extractor.db.count_user_messages", new_callable=AsyncMock, return_value=10),
            patch(
                "core.memory.extractor.db.get_recent_messages",
                new_callable=AsyncMock,
                side_effect=RuntimeError("db down"),
            ),
            patch("core.memory.extractor.store.add", new_callable=AsyncMock),
        ):
            await extractor.maybe_extract(pool, "u1")

    @pytest.mark.asyncio
    async def test_swallows_llm_exception(self):
        pool = MagicMock()
        messages = [_make_message()]
        bad_client = MagicMock()
        bad_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("LLM down"))
        with (
            patch("core.memory.extractor.get_settings", return_value=_make_settings(10)),
            patch("core.memory.extractor.db.count_user_messages", new_callable=AsyncMock, return_value=10),
            patch(
                "core.memory.extractor.db.get_recent_messages",
                new_callable=AsyncMock,
                return_value=messages,
            ),
            patch("core.memory.extractor.llm.get_client", return_value=bad_client),
            patch("core.memory.extractor.store.add", new_callable=AsyncMock),
        ):
            await extractor.maybe_extract(pool, "u1")

    @pytest.mark.asyncio
    async def test_swallows_store_exception(self):
        pool = MagicMock()
        messages = [_make_message()]
        facts = [{"dimension": "diet", "value": "vegetarian", "is_short_term": False}]
        with (
            patch("core.memory.extractor.get_settings", return_value=_make_settings(10)),
            patch("core.memory.extractor.db.count_user_messages", new_callable=AsyncMock, return_value=10),
            patch(
                "core.memory.extractor.db.get_recent_messages",
                new_callable=AsyncMock,
                return_value=messages,
            ),
            patch("core.memory.extractor.llm.get_client", return_value=_make_client(facts)),
            patch(
                "core.memory.extractor.store.add",
                new_callable=AsyncMock,
                side_effect=RuntimeError("mem0 down"),
            ),
        ):
            await extractor.maybe_extract(pool, "u1")
