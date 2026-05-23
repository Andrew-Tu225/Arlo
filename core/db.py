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

Key functions (to implement):
  get_pool()                            — returns the shared asyncpg pool (created once)
  init_tables(pool)                     — CREATE TABLE IF NOT EXISTS for both tables
  insert_episodic_message(pool, ...)    — called async/non-blocking in handlers.py
  get_recent_messages(pool, user_id, n) — returns last n rows for context window
  prune_old_messages(pool, days=30)     — weekly retention cleanup
  get_digest_config(pool, user_id)      — read job config on restart
  upsert_digest_config(pool, ...)       — persist /digest on|off state
"""
