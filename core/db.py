"""asyncpg connection pool and database operations.

Owns all direct PostgreSQL access. The rest of the codebase calls functions
here rather than managing connections themselves.

Tables managed here:
  episodic_messages — raw interaction log; source of truth for the context
                      window (handlers.py reads) and the extraction job
                      (extractor.py reads). 30-day retention; pruned weekly.
  digest_config     — APScheduler job state (channel_id, enabled, schedule).
                      Re-read at startup to re-register the job after restarts.

Schema (created by init_tables() at startup):

  episodic_messages:
    id          BIGSERIAL PRIMARY KEY
    user_id     TEXT NOT NULL
    role        TEXT NOT NULL  CHECK (role IN ('user', 'assistant'))
    content     TEXT NOT NULL
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    INDEX ON (user_id, created_at DESC)

  digest_config:
    user_id     TEXT PRIMARY KEY
    channel_id  TEXT NOT NULL
    enabled     BOOLEAN NOT NULL DEFAULT true
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
"""

from __future__ import annotations

import asyncpg

from core.memory.models import EpisodicMessage


async def get_pool(database_url: str) -> asyncpg.Pool:
    """Create and return a shared asyncpg connection pool."""
    return await asyncpg.create_pool(database_url)


async def init_tables(pool: asyncpg.Pool) -> None:
    """Create episodic_messages and digest_config tables if they don't exist."""
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
