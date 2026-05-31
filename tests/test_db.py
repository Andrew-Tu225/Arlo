"""Tests for core/db.py — asyncpg pool and database operations."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.db import (
    get_digest_config,
    get_recent_messages,
    init_tables,
    insert_episodic_message,
    prune_old_messages,
    upsert_digest_config,
)
from core.memory.models import EpisodicMessage


def _make_pool(fetchval=None, fetch=None, execute=None):
    """Build a minimal asyncpg pool mock with a context-manager acquire()."""
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=fetchval)
    conn.fetch = AsyncMock(return_value=fetch or [])
    conn.execute = AsyncMock(return_value=execute)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=_AsyncCtx(conn))
    return pool, conn


class _AsyncCtx:
    """Async context manager that returns a fixed value."""

    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *_):
        pass


_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class TestInitTables:
    @pytest.mark.asyncio
    async def test_executes_create_statements(self):
        pool, conn = _make_pool()
        await init_tables(pool)
        assert conn.execute.called

    @pytest.mark.asyncio
    async def test_creates_episodic_messages_table(self):
        pool, conn = _make_pool()
        await init_tables(pool)
        combined = " ".join(str(c) for c in conn.execute.call_args_list)
        assert "episodic_messages" in combined

    @pytest.mark.asyncio
    async def test_creates_digest_config_table(self):
        pool, conn = _make_pool()
        await init_tables(pool)
        combined = " ".join(str(c) for c in conn.execute.call_args_list)
        assert "digest_config" in combined


class TestInsertEpisodicMessage:
    @pytest.mark.asyncio
    async def test_inserts_row(self):
        pool, conn = _make_pool()
        await insert_episodic_message(pool, user_id="u1", role="user", content="hello")
        assert conn.execute.called

    @pytest.mark.asyncio
    async def test_passes_correct_values(self):
        pool, conn = _make_pool()
        await insert_episodic_message(pool, user_id="u1", role="assistant", content="reply")
        call_args = conn.execute.call_args
        args = call_args.args
        assert "u1" in args
        assert "assistant" in args
        assert "reply" in args


class TestGetRecentMessages:
    def _make_record(self, id_, role, content):
        rec = MagicMock()
        rec.__getitem__ = lambda self, k: {
            "id": id_,
            "user_id": "u1",
            "role": role,
            "content": content,
            "created_at": _NOW,
        }[k]
        return rec

    @pytest.mark.asyncio
    async def test_returns_list_of_episodic_messages(self):
        record = self._make_record(1, "user", "hello")
        pool, conn = _make_pool(fetch=[record])
        result = await get_recent_messages(pool, user_id="u1", n=5)
        assert len(result) == 1
        assert isinstance(result[0], EpisodicMessage)

    @pytest.mark.asyncio
    async def test_maps_fields_correctly(self):
        record = self._make_record(42, "assistant", "world")
        pool, conn = _make_pool(fetch=[record])
        result = await get_recent_messages(pool, user_id="u1", n=5)
        msg = result[0]
        assert msg.id == 42
        assert msg.role == "assistant"
        assert msg.content == "world"

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_rows(self):
        pool, conn = _make_pool(fetch=[])
        result = await get_recent_messages(pool, user_id="u1", n=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_passes_limit_n(self):
        pool, conn = _make_pool(fetch=[])
        await get_recent_messages(pool, user_id="u1", n=7)
        call_args = conn.fetch.call_args
        args = tuple(call_args.args) + tuple(call_args.kwargs.values())
        assert 7 in args


class TestPruneOldMessages:
    @pytest.mark.asyncio
    async def test_executes_delete(self):
        pool, conn = _make_pool()
        await prune_old_messages(pool, days=30)
        assert conn.execute.called

    @pytest.mark.asyncio
    async def test_passes_days_value(self):
        pool, conn = _make_pool()
        await prune_old_messages(pool, days=14)
        call_args = conn.execute.call_args
        args = tuple(call_args.args) + tuple(call_args.kwargs.values())
        assert 14 in args


class TestGetDigestConfig:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_row(self):
        pool, conn = _make_pool(fetch=[])
        result = await get_digest_config(pool, user_id="u1")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_dict_when_row_exists(self):
        rec = MagicMock()
        rec.__getitem__ = lambda self, k: {"user_id": "u1", "channel_id": "123", "enabled": True}[k]
        rec.keys = MagicMock(return_value=["user_id", "channel_id", "enabled"])
        pool, conn = _make_pool(fetch=[rec])
        result = await get_digest_config(pool, user_id="u1")
        assert result is not None


class TestUpsertDigestConfig:
    @pytest.mark.asyncio
    async def test_executes_upsert(self):
        pool, conn = _make_pool()
        await upsert_digest_config(pool, user_id="u1", channel_id="ch1", enabled=True)
        assert conn.execute.called

    @pytest.mark.asyncio
    async def test_passes_correct_values(self):
        pool, conn = _make_pool()
        await upsert_digest_config(pool, user_id="u1", channel_id="ch99", enabled=False)
        call_args = conn.execute.call_args
        args = tuple(call_args.args) + tuple(call_args.kwargs.values())
        assert "u1" in args
        assert "ch99" in args
        assert False in args
