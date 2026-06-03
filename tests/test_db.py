"""Tests for core/db.py — asyncpg pool and database operations."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.db import (
    count_user_messages,
    delete_schedule,
    get_channel_by_discord_id,
    get_enabled_channels,
    get_enabled_schedules,
    get_recent_messages,
    get_schedule,
    get_schedule_by_name,
    list_schedules_for_user,
    init_tables,
    insert_channel,
    insert_episodic_message,
    insert_schedule,
    prune_old_messages,
    set_channels_enabled,
    set_schedules_enabled,
    update_schedule,
    update_schedule_last_sent,
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
    async def test_creates_schedules_table(self):
        pool, conn = _make_pool()
        await init_tables(pool)
        combined = " ".join(str(c) for c in conn.execute.call_args_list)
        assert "schedules" in combined

    @pytest.mark.asyncio
    async def test_creates_arlo_channels_table(self):
        pool, conn = _make_pool()
        await init_tables(pool)
        combined = " ".join(str(c) for c in conn.execute.call_args_list)
        assert "arlo_channels" in combined


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
        args = conn.execute.call_args.args
        assert "u1" in args
        assert "assistant" in args
        assert "reply" in args


class TestGetRecentMessages:
    def _make_record(self, id_, role, content):
        rec = MagicMock()
        rec.__getitem__ = lambda self, k: {
            "id": id_, "user_id": "u1", "role": role,
            "content": content, "created_at": _NOW,
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
        assert result[0].id == 42
        assert result[0].role == "assistant"

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_rows(self):
        pool, conn = _make_pool(fetch=[])
        result = await get_recent_messages(pool, user_id="u1", n=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_passes_limit_n(self):
        pool, conn = _make_pool(fetch=[])
        await get_recent_messages(pool, user_id="u1", n=7)
        args = tuple(conn.fetch.call_args.args) + tuple(conn.fetch.call_args.kwargs.values())
        assert 7 in args


class TestCountUserMessages:
    @pytest.mark.asyncio
    async def test_returns_count(self):
        pool, conn = _make_pool(fetchval=5)
        result = await count_user_messages(pool, user_id="u1")
        assert result == 5

    @pytest.mark.asyncio
    async def test_passes_user_id(self):
        pool, conn = _make_pool(fetchval=0)
        await count_user_messages(pool, user_id="u42")
        assert "u42" in conn.fetchval.call_args.args

    @pytest.mark.asyncio
    async def test_counts_only_user_role(self):
        pool, conn = _make_pool(fetchval=3)
        await count_user_messages(pool, user_id="u1")
        query = conn.fetchval.call_args.args[0]
        assert "role" in query.lower() and "user" in query.lower()


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
        args = conn.execute.call_args.args
        assert 14 in args


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def _make_schedule_record(
    id_=1, user_id="u1", name="morning-proactive",
    task="Morning task", discord_channel_id=None,
    channel_topic=None, cron_schedule="0 9 * * *",
    enabled=True,
):
    rec = MagicMock()
    rec.__getitem__ = lambda self, k: {
        "id": id_, "user_id": user_id, "name": name, "task": task,
        "discord_channel_id": discord_channel_id, "channel_topic": channel_topic,
        "cron_schedule": cron_schedule, "poll_interval_secs": None,
        "last_sent_at": None, "enabled": enabled, "created_at": _NOW,
    }[k]
    return rec


class TestInsertSchedule:
    @pytest.mark.asyncio
    async def test_returns_id(self):
        rec = MagicMock()
        rec.__getitem__ = lambda self, k: {"id": 5}[k]
        pool, conn = _make_pool()
        conn.fetchrow = AsyncMock(return_value=rec)
        result = await insert_schedule(pool, user_id="u1", name="morning-proactive", task="task")
        assert result == 5

    @pytest.mark.asyncio
    async def test_passes_user_id_name_task(self):
        rec = MagicMock()
        rec.__getitem__ = lambda self, k: {"id": 1}[k]
        pool, conn = _make_pool()
        conn.fetchrow = AsyncMock(return_value=rec)
        await insert_schedule(pool, user_id="u1", name="morning-proactive", task="Do something")
        args = conn.fetchrow.call_args.args
        assert "u1" in args
        assert "morning-proactive" in args
        assert "Do something" in args

    @pytest.mark.asyncio
    async def test_passes_none_discord_channel_id_for_dm(self):
        rec = MagicMock()
        rec.__getitem__ = lambda self, k: {"id": 1}[k]
        pool, conn = _make_pool()
        conn.fetchrow = AsyncMock(return_value=rec)
        await insert_schedule(pool, user_id="u1", name="dm-schedule", task="task",
                               discord_channel_id=None)
        args = conn.fetchrow.call_args.args
        assert None in args

    @pytest.mark.asyncio
    async def test_passes_discord_channel_id_for_channel_schedule(self):
        rec = MagicMock()
        rec.__getitem__ = lambda self, k: {"id": 1}[k]
        pool, conn = _make_pool()
        conn.fetchrow = AsyncMock(return_value=rec)
        await insert_schedule(pool, user_id="u1", name="ch-schedule", task="task",
                               discord_channel_id="999")
        args = conn.fetchrow.call_args.args
        assert "999" in args


class TestGetSchedule:
    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        pool, conn = _make_pool(fetch=[])
        result = await get_schedule(pool, schedule_id=99)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_dict_when_found(self):
        rec = _make_schedule_record()
        pool, conn = _make_pool(fetch=[rec])
        result = await get_schedule(pool, schedule_id=1)
        assert result is not None
        assert result["name"] == "morning-proactive"

    @pytest.mark.asyncio
    async def test_dm_schedule_has_none_channel_id(self):
        rec = _make_schedule_record(discord_channel_id=None)
        pool, conn = _make_pool(fetch=[rec])
        result = await get_schedule(pool, schedule_id=1)
        assert result["discord_channel_id"] is None

    @pytest.mark.asyncio
    async def test_channel_schedule_has_channel_id(self):
        rec = _make_schedule_record(discord_channel_id="111")
        pool, conn = _make_pool(fetch=[rec])
        result = await get_schedule(pool, schedule_id=1)
        assert result["discord_channel_id"] == "111"


class TestGetScheduleByName:
    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        pool, conn = _make_pool(fetch=[])
        result = await get_schedule_by_name(pool, user_id="u1", name="missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_row_when_found(self):
        rec = _make_schedule_record()
        pool, conn = _make_pool(fetch=[rec])
        result = await get_schedule_by_name(pool, user_id="u1", name="morning-proactive")
        assert result is not None
        assert result["name"] == "morning-proactive"


class TestListSchedulesForUser:
    @pytest.mark.asyncio
    async def test_returns_all_schedules_including_disabled(self):
        rec = _make_schedule_record(enabled=False)
        pool, conn = _make_pool(fetch=[rec])
        result = await list_schedules_for_user(pool, user_id="u1")
        assert len(result) == 1
        assert result[0]["enabled"] is False

    @pytest.mark.asyncio
    async def test_passes_user_id(self):
        pool, conn = _make_pool(fetch=[])
        await list_schedules_for_user(pool, user_id="u42")
        assert "u42" in conn.fetch.call_args.args


class TestGetEnabledSchedules:
    @pytest.mark.asyncio
    async def test_returns_list_of_dicts(self):
        rec = _make_schedule_record()
        pool, conn = _make_pool(fetch=[rec])
        result = await get_enabled_schedules(pool, user_id="u1")
        assert len(result) == 1
        assert isinstance(result[0], dict)

    @pytest.mark.asyncio
    async def test_returns_empty_when_none(self):
        pool, conn = _make_pool(fetch=[])
        result = await get_enabled_schedules(pool, user_id="u1")
        assert result == []

    @pytest.mark.asyncio
    async def test_passes_user_id(self):
        pool, conn = _make_pool(fetch=[])
        await get_enabled_schedules(pool, user_id="u42")
        assert "u42" in conn.fetch.call_args.args


class TestUpdateSchedule:
    @pytest.mark.asyncio
    async def test_executes_update_when_fields_provided(self):
        pool, conn = _make_pool()
        await update_schedule(pool, schedule_id=1, task="New task")
        assert conn.execute.called

    @pytest.mark.asyncio
    async def test_passes_new_task_and_id(self):
        pool, conn = _make_pool()
        await update_schedule(pool, schedule_id=3, task="Updated task")
        args = conn.execute.call_args.args
        assert "Updated task" in args
        assert 3 in args

    @pytest.mark.asyncio
    async def test_no_op_when_no_fields(self):
        pool, conn = _make_pool()
        await update_schedule(pool, schedule_id=1)
        conn.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_passes_enabled_false(self):
        pool, conn = _make_pool()
        await update_schedule(pool, schedule_id=2, enabled=False)
        args = conn.execute.call_args.args
        assert False in args
        assert 2 in args


class TestDeleteSchedule:
    @pytest.mark.asyncio
    async def test_executes_delete(self):
        pool, conn = _make_pool()
        await delete_schedule(pool, schedule_id=5)
        assert conn.execute.called

    @pytest.mark.asyncio
    async def test_passes_schedule_id(self):
        pool, conn = _make_pool()
        await delete_schedule(pool, schedule_id=7)
        assert 7 in conn.execute.call_args.args


class TestSetSchedulesEnabled:
    @pytest.mark.asyncio
    async def test_executes_update(self):
        pool, conn = _make_pool()
        await set_schedules_enabled(pool, user_id="u1", enabled=False)
        assert conn.execute.called

    @pytest.mark.asyncio
    async def test_passes_user_id_and_enabled(self):
        pool, conn = _make_pool()
        await set_schedules_enabled(pool, user_id="u1", enabled=False)
        args = conn.execute.call_args.args
        assert "u1" in args
        assert False in args


class TestUpdateScheduleLastSent:
    @pytest.mark.asyncio
    async def test_executes_update(self):
        pool, conn = _make_pool()
        await update_schedule_last_sent(pool, schedule_id=3)
        assert conn.execute.called

    @pytest.mark.asyncio
    async def test_passes_schedule_id(self):
        pool, conn = _make_pool()
        await update_schedule_last_sent(pool, schedule_id=7)
        assert 7 in conn.execute.call_args.args


# ---------------------------------------------------------------------------
# Arlo channels (Phase 4 channel registry)
# ---------------------------------------------------------------------------

def _make_channel_record(id_=1, user_id="u1", discord_channel_id="111",
                          name="general", topic="General chat"):
    rec = MagicMock()
    rec.__getitem__ = lambda self, k: {
        "id": id_, "user_id": user_id, "discord_channel_id": discord_channel_id,
        "name": name, "topic": topic, "enabled": True, "created_at": _NOW,
    }[k]
    return rec


class TestInsertChannel:
    @pytest.mark.asyncio
    async def test_returns_id(self):
        rec = MagicMock()
        rec.__getitem__ = lambda self, k: {"id": 7}[k]
        pool, conn = _make_pool()
        conn.fetchrow = AsyncMock(return_value=rec)
        result = await insert_channel(pool, user_id="u1", discord_channel_id="111",
                                       name="general", topic="General chat")
        assert result == 7

    @pytest.mark.asyncio
    async def test_passes_user_id_and_channel_id(self):
        rec = MagicMock()
        rec.__getitem__ = lambda self, k: {"id": 1}[k]
        pool, conn = _make_pool()
        conn.fetchrow = AsyncMock(return_value=rec)
        await insert_channel(pool, user_id="u1", discord_channel_id="999",
                              name="general", topic="topic")
        args = conn.fetchrow.call_args.args
        assert "u1" in args
        assert "999" in args


class TestGetChannelByDiscordId:
    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        pool, conn = _make_pool(fetch=[])
        result = await get_channel_by_discord_id(pool, user_id="u1", discord_channel_id="111")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_dict_when_found(self):
        rec = _make_channel_record()
        pool, conn = _make_pool(fetch=[rec])
        result = await get_channel_by_discord_id(pool, user_id="u1", discord_channel_id="111")
        assert result is not None
        assert result["name"] == "general"


class TestGetEnabledChannels:
    @pytest.mark.asyncio
    async def test_returns_list_of_dicts(self):
        rec = _make_channel_record()
        pool, conn = _make_pool(fetch=[rec])
        result = await get_enabled_channels(pool, user_id="u1")
        assert len(result) == 1
        assert isinstance(result[0], dict)

    @pytest.mark.asyncio
    async def test_returns_empty_when_none(self):
        pool, conn = _make_pool(fetch=[])
        result = await get_enabled_channels(pool, user_id="u1")
        assert result == []


class TestSetChannelsEnabled:
    @pytest.mark.asyncio
    async def test_executes_update(self):
        pool, conn = _make_pool()
        await set_channels_enabled(pool, user_id="u1", enabled=False)
        assert conn.execute.called

    @pytest.mark.asyncio
    async def test_passes_enabled_and_user_id(self):
        pool, conn = _make_pool()
        await set_channels_enabled(pool, user_id="u1", enabled=True)
        args = conn.execute.call_args.args
        assert "u1" in args
        assert True in args
