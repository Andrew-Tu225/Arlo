"""Tests for core/db.py — asyncpg pool and database operations."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.db import (
    get_digest_config,
    get_due_reminders,
    get_note,
    get_open_reminders,
    get_open_tasks,
    get_recent_messages,
    init_tables,
    insert_episodic_message,
    insert_reminder,
    insert_task,
    prune_old_messages,
    update_reminder_status,
    update_task_status,
    upsert_digest_config,
    upsert_note,
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

    @pytest.mark.asyncio
    async def test_creates_reminders_table(self):
        pool, conn = _make_pool()
        await init_tables(pool)
        combined = " ".join(str(c) for c in conn.execute.call_args_list)
        assert "reminders" in combined

    @pytest.mark.asyncio
    async def test_creates_tasks_table(self):
        pool, conn = _make_pool()
        await init_tables(pool)
        combined = " ".join(str(c) for c in conn.execute.call_args_list)
        assert "tasks" in combined

    @pytest.mark.asyncio
    async def test_creates_notes_table(self):
        pool, conn = _make_pool()
        await init_tables(pool)
        combined = " ".join(str(c) for c in conn.execute.call_args_list)
        assert "notes" in combined


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


# ---------------------------------------------------------------------------
# Reminders
# ---------------------------------------------------------------------------

def _make_reminder_record(
    id_=1,
    user_id="u1",
    text="Buy milk",
    due_at=_NOW,
    recurrence=None,
    status="pending",
):
    rec = MagicMock()
    rec.__getitem__ = lambda self, k: {
        "id": id_,
        "user_id": user_id,
        "text": text,
        "due_at": due_at,
        "recurrence": recurrence,
        "status": status,
        "created_at": _NOW,
    }[k]
    return rec


class TestInsertReminder:
    @pytest.mark.asyncio
    async def test_returns_id(self):
        pool, conn = _make_pool(fetchval=42)
        result = await insert_reminder(pool, user_id="u1", text="Buy milk", due_at=_NOW, recurrence=None)
        assert result == 42

    @pytest.mark.asyncio
    async def test_passes_user_id_and_text(self):
        pool, conn = _make_pool(fetchval=1)
        await insert_reminder(pool, user_id="u1", text="Buy milk", due_at=_NOW, recurrence=None)
        args = conn.fetchval.call_args.args
        assert "u1" in args
        assert "Buy milk" in args

    @pytest.mark.asyncio
    async def test_passes_recurrence(self):
        pool, conn = _make_pool(fetchval=1)
        await insert_reminder(pool, user_id="u1", text="Stand-up", due_at=_NOW, recurrence="daily")
        args = conn.fetchval.call_args.args
        assert "daily" in args

    @pytest.mark.asyncio
    async def test_passes_none_due_at(self):
        pool, conn = _make_pool(fetchval=1)
        await insert_reminder(pool, user_id="u1", text="Someday", due_at=None, recurrence=None)
        args = conn.fetchval.call_args.args
        assert None in args


class TestGetOpenReminders:
    @pytest.mark.asyncio
    async def test_returns_list_of_dicts(self):
        rec = _make_reminder_record()
        pool, conn = _make_pool(fetch=[rec])
        result = await get_open_reminders(pool, user_id="u1")
        assert len(result) == 1
        assert isinstance(result[0], dict)

    @pytest.mark.asyncio
    async def test_maps_fields(self):
        rec = _make_reminder_record(id_=7, text="Stand-up", status="pending")
        pool, conn = _make_pool(fetch=[rec])
        result = await get_open_reminders(pool, user_id="u1")
        assert result[0]["id"] == 7
        assert result[0]["text"] == "Stand-up"
        assert result[0]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_rows(self):
        pool, conn = _make_pool(fetch=[])
        result = await get_open_reminders(pool, user_id="u1")
        assert result == []

    @pytest.mark.asyncio
    async def test_passes_user_id(self):
        pool, conn = _make_pool(fetch=[])
        await get_open_reminders(pool, user_id="u99")
        args = conn.fetch.call_args.args
        assert "u99" in args


class TestGetDueReminders:
    @pytest.mark.asyncio
    async def test_returns_list_of_dicts(self):
        rec = _make_reminder_record()
        pool, conn = _make_pool(fetch=[rec])
        result = await get_due_reminders(pool, user_id="u1", before=_NOW)
        assert len(result) == 1
        assert isinstance(result[0], dict)

    @pytest.mark.asyncio
    async def test_passes_before_datetime(self):
        pool, conn = _make_pool(fetch=[])
        await get_due_reminders(pool, user_id="u1", before=_NOW)
        args = conn.fetch.call_args.args
        assert _NOW in args

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_due(self):
        pool, conn = _make_pool(fetch=[])
        result = await get_due_reminders(pool, user_id="u1", before=_NOW)
        assert result == []


class TestUpdateReminderStatus:
    @pytest.mark.asyncio
    async def test_executes_update(self):
        pool, conn = _make_pool()
        await update_reminder_status(pool, reminder_id=5, status="done")
        assert conn.execute.called

    @pytest.mark.asyncio
    async def test_passes_status_and_id(self):
        pool, conn = _make_pool()
        await update_reminder_status(pool, reminder_id=5, status="snoozed")
        args = conn.execute.call_args.args
        assert "snoozed" in args
        assert 5 in args


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def _make_task_record(
    id_=1,
    user_id="u1",
    text="Write tests",
    due_date=None,
    priority="normal",
    project=None,
    status="open",
):
    rec = MagicMock()
    rec.__getitem__ = lambda self, k: {
        "id": id_,
        "user_id": user_id,
        "text": text,
        "due_date": due_date,
        "priority": priority,
        "project": project,
        "status": status,
        "created_at": _NOW,
    }[k]
    return rec


class TestInsertTask:
    @pytest.mark.asyncio
    async def test_returns_id(self):
        pool, conn = _make_pool(fetchval=10)
        result = await insert_task(pool, user_id="u1", text="Write tests", due_date=None, project=None)
        assert result == 10

    @pytest.mark.asyncio
    async def test_passes_user_id_and_text(self):
        pool, conn = _make_pool(fetchval=1)
        await insert_task(pool, user_id="u1", text="Write tests", due_date=None, project=None)
        args = conn.fetchval.call_args.args
        assert "u1" in args
        assert "Write tests" in args

    @pytest.mark.asyncio
    async def test_passes_priority(self):
        pool, conn = _make_pool(fetchval=1)
        await insert_task(pool, user_id="u1", text="Urgent", due_date=None, priority="high", project=None)
        args = conn.fetchval.call_args.args
        assert "high" in args

    @pytest.mark.asyncio
    async def test_default_priority_is_normal(self):
        pool, conn = _make_pool(fetchval=1)
        await insert_task(pool, user_id="u1", text="Task", due_date=None, project=None)
        args = conn.fetchval.call_args.args
        assert "normal" in args


class TestGetOpenTasks:
    @pytest.mark.asyncio
    async def test_returns_list_of_dicts(self):
        rec = _make_task_record()
        pool, conn = _make_pool(fetch=[rec])
        result = await get_open_tasks(pool, user_id="u1")
        assert len(result) == 1
        assert isinstance(result[0], dict)

    @pytest.mark.asyncio
    async def test_maps_fields(self):
        rec = _make_task_record(id_=3, text="Write tests", priority="high")
        pool, conn = _make_pool(fetch=[rec])
        result = await get_open_tasks(pool, user_id="u1")
        assert result[0]["id"] == 3
        assert result[0]["text"] == "Write tests"
        assert result[0]["priority"] == "high"

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_rows(self):
        pool, conn = _make_pool(fetch=[])
        result = await get_open_tasks(pool, user_id="u1")
        assert result == []

    @pytest.mark.asyncio
    async def test_passes_user_id(self):
        pool, conn = _make_pool(fetch=[])
        await get_open_tasks(pool, user_id="u77")
        args = conn.fetch.call_args.args
        assert "u77" in args


class TestUpdateTaskStatus:
    @pytest.mark.asyncio
    async def test_executes_update(self):
        pool, conn = _make_pool()
        await update_task_status(pool, task_id=3, status="done")
        assert conn.execute.called

    @pytest.mark.asyncio
    async def test_passes_status_and_id(self):
        pool, conn = _make_pool()
        await update_task_status(pool, task_id=3, status="done")
        args = conn.execute.call_args.args
        assert "done" in args
        assert 3 in args


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

def _make_note_record(id_=1, user_id="u1", topic="workout", content="3x10 squats"):
    rec = MagicMock()
    rec.__getitem__ = lambda self, k: {
        "id": id_,
        "user_id": user_id,
        "topic": topic,
        "content": content,
        "created_at": _NOW,
        "updated_at": _NOW,
    }[k]
    return rec


class TestUpsertNote:
    @pytest.mark.asyncio
    async def test_executes_upsert(self):
        pool, conn = _make_pool()
        await upsert_note(pool, user_id="u1", topic="workout", content="3x10 squats")
        assert conn.execute.called

    @pytest.mark.asyncio
    async def test_passes_user_id_topic_content(self):
        pool, conn = _make_pool()
        await upsert_note(pool, user_id="u1", topic="workout", content="3x10 squats")
        args = conn.execute.call_args.args
        assert "u1" in args
        assert "workout" in args
        assert "3x10 squats" in args


class TestGetNote:
    @pytest.mark.asyncio
    async def test_returns_dict_when_found(self):
        rec = _make_note_record()
        pool, conn = _make_pool(fetch=[rec])
        result = await get_note(pool, user_id="u1", topic="workout")
        assert result is not None
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_maps_fields(self):
        rec = _make_note_record(id_=9, topic="workout", content="3x10 squats")
        pool, conn = _make_pool(fetch=[rec])
        result = await get_note(pool, user_id="u1", topic="workout")
        assert result["id"] == 9
        assert result["topic"] == "workout"
        assert result["content"] == "3x10 squats"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        pool, conn = _make_pool(fetch=[])
        result = await get_note(pool, user_id="u1", topic="nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_passes_user_id_and_topic(self):
        pool, conn = _make_pool(fetch=[])
        await get_note(pool, user_id="u1", topic="workout")
        args = conn.fetch.call_args.args
        assert "u1" in args
        assert "workout" in args
