"""Schedule helpers for Phase 4 orchestrator tools.

Callers pass pool, bot, and scheduler — never call register_digest_jobs at runtime.
Per-job add_job / remove_job only, matching digest.register_digest_jobs job ids.
"""

from __future__ import annotations

import json
import logging
import re

import pytz
from apscheduler.triggers.cron import CronTrigger

from core import db
from core.scheduler.digest import run_schedule_job, scheduler
from core.settings import get_settings

logger = logging.getLogger(__name__)

_HHMM_RE = re.compile(r"^\d{1,2}:\d{2}$")


def parse_cron_schedule(cron_schedule: str) -> str:
    """Accept five-field cron or HH:MM (converted to daily cron)."""
    text = cron_schedule.strip()
    if _HHMM_RE.match(text):
        h, m = text.split(":")
        return f"{int(m)} {int(h)} * * *"
    parts = text.split()
    if len(parts) != 5:
        raise ValueError(
            "cron_schedule must be five cron fields (e.g. '0 9 * * *') or HH:MM"
        )
    return text


async def list_schedules(*, pool, user_id: str) -> str:
    """Return JSON summaries of the user's schedules for agent discovery."""
    rows = await db.list_schedules_for_user(pool, user_id=user_id)
    return json.dumps(
        [
            {
                "name": row["name"],
                "task": row["task"],
                "cron_schedule": row.get("cron_schedule"),
                "enabled": row.get("enabled"),
                "discord_channel_id": row.get("discord_channel_id"),
            }
            for row in rows
        ]
    )


async def create_schedule(
    *,
    pool,
    bot,
    user_id: str,
    name: str,
    task: str,
    cron_schedule: str,
    discord_channel_id: str | None = None,
) -> str:
    """Insert schedule row and register APScheduler job. Returns user-facing message."""
    name = name.strip()
    if not name:
        return "Error: schedule name cannot be empty"
    if not task.strip():
        return "Error: task cannot be empty"

    existing = await db.get_schedule_by_name(pool, user_id=user_id, name=name)
    if existing is not None:
        return f"Error: schedule '{name}' already exists"

    try:
        cron = parse_cron_schedule(cron_schedule)
    except ValueError as exc:
        return f"Error: {exc}"

    schedule_id = await db.insert_schedule(
        pool,
        user_id=user_id,
        name=name,
        task=task.strip(),
        discord_channel_id=discord_channel_id,
        cron_schedule=cron,
    )

    settings = get_settings()
    tz = pytz.timezone(settings.digest_timezone)
    try:
        trigger = CronTrigger.from_crontab(cron, timezone=tz)
        scheduler.add_job(
            run_schedule_job,
            trigger,
            id=f"schedule_{schedule_id}",
            replace_existing=True,
            misfire_grace_time=3600,
            kwargs={"bot": bot, "pool": pool, "schedule_id": schedule_id},
        )
    except Exception as exc:
        await db.delete_schedule(pool, schedule_id=schedule_id)
        logger.exception("Failed to register job for schedule %s", schedule_id)
        return f"Error: invalid cron schedule ({exc})"

    logger.info("Created schedule_%s (%s @ %s)", schedule_id, name, cron)
    return f"Created schedule '{name}' (id={schedule_id})."


async def delete_schedule(
    *,
    pool,
    user_id: str,
    name: str,
) -> str:
    """Delete a schedule by exact name (use list_schedules first)."""
    name = name.strip()
    if not name:
        return "Error: schedule name cannot be empty"

    row = await db.get_schedule_by_name(pool, user_id=user_id, name=name)
    if row is None:
        return (
            f"Error: no schedule named {name!r}. "
            "Call list_schedules for exact names."
        )

    schedule_id = row["id"]
    await db.delete_schedule(pool, schedule_id=schedule_id)

    job_id = f"schedule_{schedule_id}"
    try:
        scheduler.remove_job(job_id)
    except Exception:
        logger.debug("No APScheduler job %s to remove", job_id)

    logger.info("Deleted schedule_%s (%s)", schedule_id, name)
    return f"Deleted schedule '{name}'."
