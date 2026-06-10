"""asyncpg connection pool and database operations.

Owns all direct PostgreSQL access. The rest of the codebase calls functions
here rather than managing connections themselves.

Tables managed here:
  episodic_messages — raw interaction log; source of truth for the context
                      window (handlers.py reads) and the extraction job
                      (extractor.py reads). 30-day retention; pruned weekly.
  schedules         — all proactive jobs Arlo runs for the user: DM-based or
                      channel-based, cron or event-driven. The morning proactive
                      DM is seeded here at startup so the orchestrator can
                      search, edit, or delete it via conversation in Phase 4.
  arlo_channels     — Discord channels Arlo manages, each with a topic/purpose.
                      Phase 4: handlers.py injects channel topic into the
                      orchestrator prompt for channel-aware conversation.
"""

from __future__ import annotations

import json
import logging

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
            CREATE TABLE IF NOT EXISTS schedules (
                id                  SERIAL PRIMARY KEY,
                user_id             TEXT NOT NULL,
                name                TEXT NOT NULL,
                task                TEXT NOT NULL,
                discord_channel_id  TEXT,
                channel_topic       TEXT,
                cron_schedule       TEXT,
                poll_interval_secs  INTEGER,
                last_sent_at        TIMESTAMPTZ,
                enabled             BOOLEAN NOT NULL DEFAULT true,
                created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (user_id, name)
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS arlo_channels (
                id                  SERIAL PRIMARY KEY,
                user_id             TEXT NOT NULL,
                discord_channel_id  TEXT NOT NULL,
                name                TEXT NOT NULL,
                topic               TEXT NOT NULL,
                enabled             BOOLEAN NOT NULL DEFAULT true,
                created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (user_id, discord_channel_id)
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_actions (
                id              SERIAL PRIMARY KEY,
                user_id         TEXT NOT NULL,
                tool_name       TEXT NOT NULL,
                tool_args       JSONB NOT NULL,
                status          TEXT NOT NULL DEFAULT 'pending',
                discord_msg_id  TEXT,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                resolved_at     TIMESTAMPTZ,
                agent_state     JSONB
            )
        """)
        await conn.execute("ALTER TABLE pending_actions ADD COLUMN IF NOT EXISTS agent_state JSONB")


# ---------------------------------------------------------------------------
# Episodic messages
# ---------------------------------------------------------------------------

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


async def count_user_messages(
    pool: asyncpg.Pool,
    *,
    user_id: str,
) -> int:
    """Return the total number of user-role messages stored for user_id."""
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT COUNT(*) FROM episodic_messages WHERE user_id = $1 AND role = 'user'",
            user_id,
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


# ---------------------------------------------------------------------------
# Schedules
#
# All proactive Arlo jobs live here — DM-based and channel-based alike.
# discord_channel_id = null means DM the user directly.
#
# Phase 4 orchestrator tools (list_schedules, edit_schedule, delete_schedule)
# read and write this table conversationally.
# ---------------------------------------------------------------------------

_SCHEDULE_COLS = (
    "id", "user_id", "name", "task", "discord_channel_id", "channel_topic",
    "cron_schedule", "poll_interval_secs", "last_sent_at", "enabled", "created_at",
)


async def insert_schedule(
    pool: asyncpg.Pool,
    *,
    user_id: str,
    name: str,
    task: str,
    discord_channel_id: str | None = None,
    channel_topic: str | None = None,
    cron_schedule: str | None = None,
    poll_interval_secs: int | None = None,
) -> int:
    """Insert a schedule and return its id.

    Idempotent: if (user_id, name) already exists, returns the existing id
    without modifying the row. Use update_schedule to change an existing one.
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO schedules
                (user_id, name, task, discord_channel_id, channel_topic, cron_schedule, poll_interval_secs)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (user_id, name) DO UPDATE SET name = EXCLUDED.name
            RETURNING id
            """,
            user_id, name, task, discord_channel_id, channel_topic, cron_schedule, poll_interval_secs,
        )
    return row["id"]


async def get_schedule_by_name(
    pool: asyncpg.Pool,
    *,
    user_id: str,
    name: str,
) -> dict | None:
    """Return the schedule row for (user_id, name), or None."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, user_id, name, task, discord_channel_id, channel_topic,
                   cron_schedule, poll_interval_secs, last_sent_at, enabled, created_at
            FROM schedules
            WHERE user_id = $1 AND name = $2
            """,
            user_id,
            name,
        )
    if not rows:
        return None
    return {k: rows[0][k] for k in _SCHEDULE_COLS}


async def get_schedule(pool: asyncpg.Pool, *, schedule_id: int) -> dict | None:
    """Return the schedule row for the given id, or None."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, user_id, name, task, discord_channel_id, channel_topic,
                   cron_schedule, poll_interval_secs, last_sent_at, enabled, created_at
            FROM schedules
            WHERE id = $1
            """,
            schedule_id,
        )
    if not rows:
        return None
    return {k: rows[0][k] for k in _SCHEDULE_COLS}


async def list_schedules_for_user(pool: asyncpg.Pool, *, user_id: str) -> list[dict]:
    """Return all schedules for user_id, ordered by id."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, user_id, name, task, discord_channel_id, channel_topic,
                   cron_schedule, poll_interval_secs, last_sent_at, enabled, created_at
            FROM schedules
            WHERE user_id = $1
            ORDER BY id ASC
            """,
            user_id,
        )
    return [{k: row[k] for k in _SCHEDULE_COLS} for row in rows]


async def get_enabled_schedules(pool: asyncpg.Pool, *, user_id: str) -> list[dict]:
    """Return all enabled schedules for user_id, ordered by id."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, user_id, name, task, discord_channel_id, channel_topic,
                   cron_schedule, poll_interval_secs, last_sent_at, enabled, created_at
            FROM schedules
            WHERE user_id = $1 AND enabled = true
            ORDER BY id ASC
            """,
            user_id,
        )
    return [{k: row[k] for k in _SCHEDULE_COLS} for row in rows]


async def update_schedule(
    pool: asyncpg.Pool,
    *,
    schedule_id: int,
    task: str | None = None,
    cron_schedule: str | None = None,
    discord_channel_id: str | None = None,
    channel_topic: str | None = None,
    enabled: bool | None = None,
) -> None:
    """Update one or more fields on a schedule row.

    Only fields passed as non-None are updated. Used by Phase 4 orchestrator
    tools when the user edits a schedule via conversation.
    """
    fields: list[str] = []
    values: list = []
    for col, val in [
        ("task", task),
        ("cron_schedule", cron_schedule),
        ("discord_channel_id", discord_channel_id),
        ("channel_topic", channel_topic),
        ("enabled", enabled),
    ]:
        if val is not None:
            fields.append(f"{col} = ${len(values) + 1}")
            values.append(val)
    if not fields:
        return
    values.append(schedule_id)
    async with pool.acquire() as conn:
        await conn.execute(
            f"UPDATE schedules SET {', '.join(fields)} WHERE id = ${len(values)}",
            *values,
        )


async def delete_schedule(pool: asyncpg.Pool, *, schedule_id: int) -> None:
    """Permanently delete a schedule row.

    Used by Phase 4 orchestrator tools when the user removes a schedule via
    conversation. Callers should also remove the corresponding APScheduler job:
      scheduler.remove_job(f"schedule_{schedule_id}")
    """
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM schedules WHERE id = $1", schedule_id)


async def set_schedules_enabled(
    pool: asyncpg.Pool,
    *,
    user_id: str,
    enabled: bool,
) -> None:
    """Enable or disable all schedules for user_id."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE schedules SET enabled = $2 WHERE user_id = $1",
            user_id,
            enabled,
        )


async def update_schedule_last_sent(pool: asyncpg.Pool, *, schedule_id: int) -> None:
    """Set last_sent_at = now() for the given schedule."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE schedules SET last_sent_at = now() WHERE id = $1",
            schedule_id,
        )


# ---------------------------------------------------------------------------
# Pending actions (Phase 4 — medium-risk tool approval)
# ---------------------------------------------------------------------------

_PENDING_ACTION_COLS = (
    "id", "user_id", "tool_name", "tool_args", "status",
    "discord_msg_id", "created_at", "resolved_at", "agent_state",
)


async def insert_pending_action(
    pool: asyncpg.Pool,
    *,
    user_id: str,
    tool_name: str,
    tool_args: dict,
    agent_state: dict | None = None,
) -> int:
    """Insert a pending medium-risk action and return its id."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO pending_actions (user_id, tool_name, tool_args, agent_state)
            VALUES ($1, $2, $3::jsonb, $4::jsonb)
            RETURNING id
            """,
            user_id,
            tool_name,
            json.dumps(tool_args),
            json.dumps(agent_state) if agent_state is not None else None,
        )
    return row["id"]


async def get_pending_action(
    pool: asyncpg.Pool,
    *,
    pending_id: int,
) -> dict | None:
    """Return a pending_actions row by id, or None."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, user_id, tool_name, tool_args, status,
                   discord_msg_id, created_at, resolved_at, agent_state
            FROM pending_actions
            WHERE id = $1
            """,
            pending_id,
        )
    if not rows:
        return None
    row = rows[0]
    return {k: row[k] for k in _PENDING_ACTION_COLS}


async def set_pending_action_discord_msg_id(
    pool: asyncpg.Pool,
    *,
    pending_id: int,
    discord_msg_id: str,
) -> None:
    """Store the Discord message id for the approval embed."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE pending_actions SET discord_msg_id = $2 WHERE id = $1",
            pending_id,
            discord_msg_id,
        )


async def resolve_pending_action(
    pool: asyncpg.Pool,
    *,
    pending_id: int,
    status: str,
) -> None:
    """Mark a pending action approved, rejected, or expired."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE pending_actions
            SET status = $2, resolved_at = now()
            WHERE id = $1
            """,
            pending_id,
            status,
        )


# ---------------------------------------------------------------------------
# Arlo channels (Phase 4 — channel registry for conversation context)
#
# When user sends a message in a channel Arlo manages, handlers.py will look
# up the channel here and inject its topic into the orchestrator system prompt.
# ---------------------------------------------------------------------------

_CHANNEL_COLS = ("id", "user_id", "discord_channel_id", "name", "topic", "enabled", "created_at")


async def insert_channel(
    pool: asyncpg.Pool,
    *,
    user_id: str,
    discord_channel_id: str,
    name: str,
    topic: str,
) -> int:
    """Insert a channel row and return its id. If already exists, return the existing id."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO arlo_channels (user_id, discord_channel_id, name, topic)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (user_id, discord_channel_id) DO UPDATE SET name = arlo_channels.name
            RETURNING id
            """,
            user_id, discord_channel_id, name, topic,
        )
    return row["id"]


async def get_channel_by_discord_id(
    pool: asyncpg.Pool,
    *,
    user_id: str,
    discord_channel_id: str,
) -> dict | None:
    """Return the channel row for (user_id, discord_channel_id), or None."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, user_id, discord_channel_id, name, topic, enabled, created_at
            FROM arlo_channels
            WHERE user_id = $1 AND discord_channel_id = $2
            """,
            user_id, discord_channel_id,
        )
    if not rows:
        return None
    return {k: rows[0][k] for k in _CHANNEL_COLS}


async def get_enabled_channels(pool: asyncpg.Pool, *, user_id: str) -> list[dict]:
    """Return all enabled channels for user_id, ordered by id."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, user_id, discord_channel_id, name, topic, enabled, created_at
            FROM arlo_channels
            WHERE user_id = $1 AND enabled = true
            ORDER BY id ASC
            """,
            user_id,
        )
    return [{k: row[k] for k in _CHANNEL_COLS} for row in rows]


async def set_channels_enabled(
    pool: asyncpg.Pool,
    *,
    user_id: str,
    enabled: bool,
) -> None:
    """Enable or disable all channels for user_id."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE arlo_channels SET enabled = $2 WHERE user_id = $1",
            user_id, enabled,
        )
