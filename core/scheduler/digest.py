"""Daily proactive digest — APScheduler job running inside the bot process.

No external cron, queue, or webhook required. APScheduler fires the job
at DIGEST_TIME (default: 09:00) in DIGEST_TIMEZONE (default: America/Toronto).
DIGEST_TIMEZONE must be a valid IANA string and is validated at startup.

Job flow:
  1. mem0.search("interests preferences habits", user_id) → profile snapshot
  2. Build Tavily search queries from profile interests
       e.g. "AI news today", "NBA scores last night", "r/ProgrammerHumor top posts"
  3. Tavily search → news, Reddit posts, trending topics
  4. ONE LLM call composes the digest:
       - 2–4 items max, conversational framing (not a newsletter)
       - References at least one profile fact so it feels personalized
  5. discord.get_channel(channel_id).send(message)

Persistence:
  - Job config (schedule, channel_id, enabled) stored in digest_config table.
  - On bot restart: read config from DB, re-register job with misfire_grace_time=3600
    so a restart within the hour doesn't re-fire the day's digest.
  - /digest off → set enabled=False, pause job
  - /digest on  → set enabled=True, resume job
"""
