"""Scheduled job system — APScheduler running inside the bot process.

OVERVIEW
--------
Arlo proactively reaches out to the user on a schedule. This module owns
that layer: seeding defaults, registering jobs, running them, and composing
messages via LLM.

All schedules — including the default morning DM — live in the `schedules`
DB table. This means the orchestrator can search, edit, and delete them via
conversation in Phase 4, just like any other schedule.

STARTUP SEQUENCE (called from bot.py setup_hook in order):
  await digest.seed_default_schedules(pool, user_id)
      Inserts the morning proactive DM row if it doesn't exist yet. Idempotent.
  digest.scheduler.start()
      Starts AsyncIOScheduler inside the running event loop.
  await digest.register_digest_jobs(bot, pool)
      Reads all enabled schedules from DB and wires up APScheduler jobs.
      Single source of truth — handles both the morning default and any
      user-created schedules. Safe to call at startup; at restart it rebuilds
      from the current DB state, so any edits the user made persist.

RUNTIME
  APScheduler fires schedule_N → run_schedule_job(bot, pool, schedule_id=N)
      Fetches schedule row, determines target (DM or channel), composes
      message via run_schedule_agent, sends.

WHAT THIS MODULE IS NOT RESPONSIBLE FOR
-----------------------------------------
- Responding to user messages — orchestrator.py via handlers.py.
- Adding individual schedules at runtime — Phase 4 orchestrator write tools
  call db.insert_schedule + scheduler.add_job directly. Do not call
  register_digest_jobs at runtime; it removes all jobs before re-adding.

PHASE NOTES
-----------
  Phase 3 (current):
    - One default schedule: morning proactive DM seeded at startup.
    - run_schedule_agent: single LLM call with mem0 profile. No tool use.
    - Only cron-based schedules. Event-driven (poll_interval_secs) is in the
      schema but the polling runner is not implemented.

  Phase 4 (planned):
    - User edits/deletes the morning schedule via conversation; orchestrator
      calls db.update_schedule / db.delete_schedule + scheduler.remove_job.
    - User creates new schedules (channel or DM, cron or poll) conversationally;
      orchestrator calls db.insert_schedule + scheduler.add_job directly.
    - run_schedule_agent becomes a ReAct agent with tools: search_memory,
      web_search, reddit_search, notion_get_tasks, …
    - Morning timing becomes adaptive/spontaneous rather than fixed cron.
    - handlers.py looks up arlo_channels by discord_channel_id and injects the
      channel topic into the orchestrator prompt for channel-aware replies.
    - Event-driven polling runner implemented for poll_interval_secs schedules.
"""

from __future__ import annotations

import logging

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from core import db
from core.settings import get_settings

logger = logging.getLogger(__name__)

# Module-level scheduler instance. Started in setup_hook — never at import time.
scheduler = AsyncIOScheduler()

_MAX_DISCORD_CHARS = 1900


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hhmm_to_cron(hhmm: str) -> str:
    """Convert 'HH:MM' string to a cron expression ('0 9 * * *' for '09:00')."""
    h, m = hhmm.split(":")
    return f"{int(m)} {int(h)} * * *"


# ---------------------------------------------------------------------------
# Startup: seed defaults
# ---------------------------------------------------------------------------

async def seed_default_schedules(pool, user_id: str) -> None:
    """Seed the single built-in default schedule: a morning proactive DM.

    This is the only schedule Arlo creates automatically. It is stored in the
    `schedules` table so the user can view, edit, or delete it via conversation
    just like any schedule they create themselves.

    Idempotent — ON CONFLICT DO NOTHING means restarting the bot never
    duplicates or overwrites the row. If the user edits the task or cron
    via conversation, those changes persist across restarts.

    discord_channel_id = None → DM the user directly (no channel).
    """
    settings = get_settings()
    await db.insert_schedule(
        pool,
        user_id=user_id,
        name="morning-proactive",
        task=(
            "Start a casual, engaging morning conversation with the user. "
            "Based on their interests and profile, open with something relevant — "
            "a question, a thought, something happening in a topic they care about, "
            "or anything they'd enjoy talking about. "
            "Make it feel like a friend reaching out, not a briefing or report. "
            "If nothing stands out, ask how they're doing or bring up something "
            "they've mentioned before. Never send a generic filler message."
        ),
        discord_channel_id=None,
        channel_topic=None,
        cron_schedule=_hhmm_to_cron(settings.digest_time),
    )
    logger.info("Default schedules seeded (or already present)")


# ---------------------------------------------------------------------------
# Startup: job registration
# ---------------------------------------------------------------------------

async def register_digest_jobs(bot, pool) -> None:
    """Register APScheduler jobs for all enabled cron schedules from DB.

    Called once from setup_hook after seed_default_schedules() and
    scheduler.start(). Reads the schedules table and wires up one
    APScheduler CronTrigger job per enabled row with a cron_schedule.

    This is the single source of truth at startup. Clears existing
    schedule_* jobs first so re-registration on restart is always clean.

    Do NOT call at runtime to add a single new schedule — that would briefly
    remove all active jobs. Phase 4 write tools use scheduler.add_job directly.

    Event-driven schedules (poll_interval_secs set, no cron_schedule) are
    skipped — their polling runner is not implemented until Phase 4.
    """
    settings = get_settings()
    user_id = str(settings.discord_user_id)
    schedules = await db.get_enabled_schedules(pool, user_id=user_id)
    tz = pytz.timezone(settings.digest_timezone)

    # Clear existing schedule jobs — no-op at startup (scheduler just started).
    for job in scheduler.get_jobs():
        if job.id.startswith("schedule_"):
            job.remove()

    for sched in schedules:
        if not sched.get("cron_schedule"):
            continue
        try:
            trigger = CronTrigger.from_crontab(sched["cron_schedule"], timezone=tz)
            scheduler.add_job(
                run_schedule_job,
                trigger,
                id=f"schedule_{sched['id']}",
                replace_existing=True,
                misfire_grace_time=3600,
                kwargs={"bot": bot, "pool": pool, "schedule_id": sched["id"]},
            )
            logger.info(
                "Registered schedule_%s (%s @ %s)",
                sched["id"], sched["name"], sched["cron_schedule"],
            )
        except Exception:
            logger.exception("Failed to register job for schedule %s", sched["id"])


# ---------------------------------------------------------------------------
# Runtime: job execution
# ---------------------------------------------------------------------------

async def run_schedule_job(bot, pool, schedule_id: int) -> None:
    """Execute one scheduled job. Called by APScheduler at cron time.

    Handles both DM-based and channel-based schedules:
      - discord_channel_id is None → fetch the user and send a DM
      - discord_channel_id is set → send to that Discord channel

    Fully wrapped in try/except — APScheduler must not receive an exception
    from a job or it may stop re-scheduling it.
    """
    try:
        sched = await db.get_schedule(pool, schedule_id=schedule_id)
        if sched is None:
            logger.warning("Schedule %s not found in DB; skipping", schedule_id)
            return

        from core.agent.proactive import run_proactive_agent  # lazy: breaks digest ↔ proactive cycle
        result = await run_proactive_agent(
            task=sched["task"],
            user_id=sched["user_id"],
            schedule_id=schedule_id,
            channel_topic=sched.get("channel_topic"),
            pool=pool,
        )

        if not result:
            logger.debug("Schedule %s: agent returned empty; nothing sent", schedule_id)
            return

        if len(result) > _MAX_DISCORD_CHARS:
            result = result[:_MAX_DISCORD_CHARS] + "…"

        if sched["discord_channel_id"] is None:
            # DM the user directly
            user = await bot.fetch_user(int(sched["user_id"]))
            dm = await user.create_dm()
            await dm.send(result)
        else:
            # Send to the specified Discord channel
            channel = bot.get_channel(int(sched["discord_channel_id"]))
            if channel is None:
                logger.warning(
                    "Discord channel %s not found for schedule %s — was it deleted?",
                    sched["discord_channel_id"], schedule_id,
                )
                return
            await channel.send(result)

        await db.insert_schedule_run(
            pool,
            schedule_id=schedule_id,
            user_id=sched["user_id"],
            message_preview=result[:150],
        )
        await db.update_schedule_last_sent(pool, schedule_id=schedule_id)

    except Exception:
        logger.exception("run_schedule_job failed for schedule %s", schedule_id)
