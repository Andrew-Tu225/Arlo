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
from core.llm import get_client, get_default_model
from core.memory import store
from core.settings import get_settings

logger = logging.getLogger(__name__)

# Module-level scheduler instance. Started in setup_hook — never at import time.
scheduler = AsyncIOScheduler()

_MAX_DISCORD_CHARS = 1900

# The task description for the default morning proactive DM.
# Stored in the DB so the user can view and edit it via conversation in Phase 4.
_MORNING_PROACTIVE_TASK = (
    "Start a casual, engaging morning conversation with the user. "
    "Based on their interests and profile, open with something relevant — "
    "a question, a thought, something happening in a topic they care about, "
    "or anything they'd enjoy talking about. "
    "Make it feel like a friend reaching out, not a briefing or report. "
    "If nothing stands out, ask how they're doing or bring up something "
    "they've mentioned before. Never send a generic filler message."
)

_MORNING_SCHEDULE_NAME = "morning-proactive"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hhmm_to_cron(hhmm: str) -> str:
    """Convert 'HH:MM' string to a cron expression ('0 9 * * *' for '09:00')."""
    h, m = hhmm.split(":")
    return f"{int(m)} {int(h)} * * *"


def _build_prompt(*, task: str, profile_facts: list[str]) -> str:
    """[Phase 3] Build the system prompt for the single-LLM-call agent.

    Phase 4: removed — the ReAct agent constructs its own prompts per step.
    """
    parts = [f"Your task: {task}"]
    if profile_facts:
        facts = "\n".join(f"- {f}" for f in profile_facts[:10])
        parts.append(f"\nUser profile facts:\n{facts}")
    parts.append(
        "\nIf there is genuinely nothing relevant to say, respond with exactly 'empty'. "
        "Otherwise always send something — never leave the user without a message."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Startup: seed defaults
# ---------------------------------------------------------------------------

async def seed_default_schedules(pool, user_id: str) -> None:
    """Seed the morning proactive DM schedule if it doesn't exist yet.

    Called once from setup_hook after init_tables(), before scheduler.start().
    Idempotent — insert_schedule uses ON CONFLICT DO NOTHING so restarting the
    bot never duplicates or overwrites an existing row. If the user later edits
    the schedule via conversation, the updated DB row persists across restarts.

    discord_channel_id = None means the job will DM the user directly.
    """
    settings = get_settings()
    await db.insert_schedule(
        pool,
        user_id=user_id,
        name=_MORNING_SCHEDULE_NAME,
        task=_MORNING_PROACTIVE_TASK,
        discord_channel_id=None,     #sent to DM by default
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

        result = await run_schedule_agent(
            task=sched["task"],
            user_id=sched["user_id"],
            channel_topic=sched.get("channel_topic"),
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

        await db.update_schedule_last_sent(pool, schedule_id=schedule_id)

    except Exception:
        logger.exception("run_schedule_job failed for schedule %s", schedule_id)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

async def run_schedule_agent(
    *, task: str, user_id: str, channel_topic: str | None = None,
) -> str:
    """Compose a proactive message and return it as a string.

    Called by run_schedule_job for all scheduled sends. Not called for
    user-initiated messages — those go through orchestrator.run() in handlers.py.

    Arguments:
      task: the schedule's instruction for this run. For the morning default,
          this is _MORNING_PROACTIVE_TASK. User can edit this via conversation.
      user_id: used to fetch profile facts from mem0.
      channel_topic: if set, prepended to the prompt so the LLM knows the
          channel's purpose. None for DM-based schedules.

    Returns "" if the agent decides nothing is worth sending. The caller
    (run_schedule_job) skips the send in that case.

    [Phase 3] Single LLM call with mem0 profile facts pre-loaded.

    Phase 4: replace body with a ReAct agent loop (same pattern as
    orchestrator.py). Add tools: search_memory, web_search, reddit_search,
    notion_get_tasks, … channel_topic becomes a system prompt preamble.
    """
    profile_facts: list[str] = []
    try:
        profile_facts = await store.search("interests preferences habits", user_id)
    except Exception:
        logger.warning("store.search failed; proceeding without profile")

    prompt_task = f"Channel context: {channel_topic}\n\n{task}" if channel_topic else task
    system_prompt = _build_prompt(task=prompt_task, profile_facts=profile_facts)

    try:
        response = await get_client().chat.completions.create(
            model=get_default_model(),
            messages=[{"role": "system", "content": system_prompt}],
        )
        content = response.choices[0].message.content or ""
        return "" if content.strip().lower() in ("", "empty", "nothing") else content.strip()
    except Exception:
        logger.exception("LLM call failed in run_schedule_agent")
        return ""
