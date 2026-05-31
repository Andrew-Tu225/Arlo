"""asyncpg connection pool and database operations.

Owns all direct PostgreSQL access. The rest of the codebase calls functions
here rather than managing connections themselves.

Tables managed here:
  episodic_messages — raw interaction log; source of truth for the context
                      window (handlers.py reads) and the extraction job
                      (extractor.py reads). 30-day retention; pruned weekly.
  digest_config     — APScheduler job state (channel_id, enabled, schedule).
                      Re-read at startup to re-register the job after restarts.
  reminders         — user reminders with optional recurrence; digest reads
                      due reminders for the morning brief.
  tasks             — user tasks with priority and project grouping; digest
                      surfaces open tasks in the morning brief and evening wrap-up.
  notes             — key-value topic notes; upserted by topic so each topic
                      has exactly one current version.
"""

from __future__ import annotations

from datetime import date, datetime

import asyncpg

from core.memory.models import EpisodicMessage


async def get_pool(database_url: str) -> asyncpg.Pool:
    """Create and return a shared asyncpg connection pool."""
    return await asyncpg.create_pool(database_url)


async def init_tables(pool: asyncpg.Pool) -> None:
    """Create all application tables if they don't exist."""
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS episodic_messages (
                id         BIGSERIAL PRIMARY KEY,
                user_id    TEXT NOT NULL,
                role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content    TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS episodic_messages_user_time_idx
                ON episodic_messages (user_id, created_at DESC)
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS digest_config (
                user_id    TEXT PRIMARY KEY,
                channel_id TEXT NOT NULL,
                enabled    BOOLEAN NOT NULL DEFAULT true,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id         SERIAL PRIMARY KEY,
                user_id    TEXT NOT NULL,
                text       TEXT NOT NULL,
                due_at     TIMESTAMPTZ,
                recurrence TEXT,
                status     TEXT NOT NULL DEFAULT 'pending',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS reminders_user_status_idx
                ON reminders (user_id, status, due_at)
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id         SERIAL PRIMARY KEY,
                user_id    TEXT NOT NULL,
                text       TEXT NOT NULL,
                due_date   DATE,
                priority   TEXT NOT NULL DEFAULT 'normal',
                project    TEXT,
                status     TEXT NOT NULL DEFAULT 'open',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS tasks_user_status_idx
                ON tasks (user_id, status)
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id         SERIAL PRIMARY KEY,
                user_id    TEXT NOT NULL,
                topic      TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (user_id, topic)
            )
        """)


async def insert_episodic_message(
    pool: asyncpg.Pool,
    *,
    user_id: str,
    role: str,
    content: str,
) -> None:
    """Insert a single message row into episodic_messages."""
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO episodic_messages (user_id, role, content) VALUES ($1, $2, $3)",
            user_id,
            role,
            content,
        )


async def get_recent_messages(
    pool: asyncpg.Pool,
    *,
    user_id: str,
    n: int,
) -> list[EpisodicMessage]:
    """Return the n most recent messages for user_id, oldest-first."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, user_id, role, content, created_at
            FROM (
                SELECT id, user_id, role, content, created_at
                FROM episodic_messages
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            ) sub
            ORDER BY created_at ASC
            """,
            user_id,
            n,
        )
    return [
        EpisodicMessage(
            id=row["id"],
            user_id=row["user_id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
        )
        for row in rows
    ]


async def prune_old_messages(pool: asyncpg.Pool, *, days: int = 30) -> None:
    """Delete episodic_messages older than `days` days."""
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM episodic_messages WHERE created_at < now() - ($1 || ' days')::interval",
            days,
        )


async def get_digest_config(pool: asyncpg.Pool, *, user_id: str) -> dict | None:
    """Return the digest config row for user_id, or None if not set."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT user_id, channel_id, enabled FROM digest_config WHERE user_id = $1",
            user_id,
        )
    if not rows:
        return None
    row = rows[0]
    return {k: row[k] for k in ("user_id", "channel_id", "enabled")}


async def upsert_digest_config(
    pool: asyncpg.Pool,
    *,
    user_id: str,
    channel_id: str,
    enabled: bool,
) -> None:
    """Insert or update the digest config row for user_id."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO digest_config (user_id, channel_id, enabled, updated_at)
            VALUES ($1, $2, $3, now())
            ON CONFLICT (user_id) DO UPDATE
                SET channel_id = EXCLUDED.channel_id,
                    enabled    = EXCLUDED.enabled,
                    updated_at = now()
            """,
            user_id,
            channel_id,
            enabled,
        )


# ---------------------------------------------------------------------------
# Reminders
# ---------------------------------------------------------------------------

_REMINDER_COLS = ("id", "user_id", "text", "due_at", "recurrence", "status", "created_at")


async def insert_reminder(
    pool: asyncpg.Pool,
    *,
    user_id: str,
    text: str,
    due_at: datetime | None,
    recurrence: str | None,
) -> int:
    """Insert a reminder and return its new id."""
    async with pool.acquire() as conn:
        return await conn.fetchval(
            """
            INSERT INTO reminders (user_id, text, due_at, recurrence)
            VALUES ($1, $2, $3, $4)
            RETURNING id
            """,
            user_id,
            text,
            due_at,
            recurrence,
        )


async def get_open_reminders(pool: asyncpg.Pool, *, user_id: str) -> list[dict]:
    """Return all pending reminders for user_id, ordered by due_at ascending."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, user_id, text, due_at, recurrence, status, created_at
            FROM reminders
            WHERE user_id = $1 AND status = 'pending'
            ORDER BY due_at ASC NULLS LAST
            """,
            user_id,
        )
    return [{k: row[k] for k in _REMINDER_COLS} for row in rows]


async def get_due_reminders(
    pool: asyncpg.Pool,
    *,
    user_id: str,
    before: datetime,
) -> list[dict]:
    """Return pending reminders with due_at <= before."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, user_id, text, due_at, recurrence, status, created_at
            FROM reminders
            WHERE user_id = $1 AND due_at <= $2 AND status = 'pending'
            ORDER BY due_at ASC
            """,
            user_id,
            before,
        )
    return [{k: row[k] for k in _REMINDER_COLS} for row in rows]


async def update_reminder_status(
    pool: asyncpg.Pool,
    *,
    reminder_id: int,
    status: str,
) -> None:
    """Update the status of a single reminder."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE reminders SET status = $1 WHERE id = $2",
            status,
            reminder_id,
        )


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

_TASK_COLS = ("id", "user_id", "text", "due_date", "priority", "project", "status", "created_at")


async def insert_task(
    pool: asyncpg.Pool,
    *,
    user_id: str,
    text: str,
    due_date: date | None,
    priority: str = "normal",
    project: str | None,
) -> int:
    """Insert a task and return its new id."""
    async with pool.acquire() as conn:
        return await conn.fetchval(
            """
            INSERT INTO tasks (user_id, text, due_date, priority, project)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
            """,
            user_id,
            text,
            due_date,
            priority,
            project,
        )


async def get_open_tasks(pool: asyncpg.Pool, *, user_id: str) -> list[dict]:
    """Return open tasks ordered high → normal → low priority, then by created_at."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, user_id, text, due_date, priority, project, status, created_at
            FROM tasks
            WHERE user_id = $1 AND status = 'open'
            ORDER BY
                CASE priority WHEN 'high' THEN 0 WHEN 'normal' THEN 1 ELSE 2 END,
                created_at ASC
            """,
            user_id,
        )
    return [{k: row[k] for k in _TASK_COLS} for row in rows]


async def update_task_status(
    pool: asyncpg.Pool,
    *,
    task_id: int,
    status: str,
) -> None:
    """Update the status of a single task."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE tasks SET status = $1 WHERE id = $2",
            status,
            task_id,
        )


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

_NOTE_COLS = ("id", "user_id", "topic", "content", "created_at", "updated_at")


async def upsert_note(
    pool: asyncpg.Pool,
    *,
    user_id: str,
    topic: str,
    content: str,
) -> None:
    """Insert a note or overwrite content if the topic already exists."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO notes (user_id, topic, content)
            VALUES ($1, $2, $3)
            ON CONFLICT (user_id, topic) DO UPDATE
                SET content    = EXCLUDED.content,
                    updated_at = now()
            """,
            user_id,
            topic,
            content,
        )


async def get_note(pool: asyncpg.Pool, *, user_id: str, topic: str) -> dict | None:
    """Return the note for (user_id, topic), or None if not found."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, user_id, topic, content, created_at, updated_at
            FROM notes
            WHERE user_id = $1 AND topic = $2
            """,
            user_id,
            topic,
        )
    if not rows:
        return None
    return {k: rows[0][k] for k in _NOTE_COLS}
