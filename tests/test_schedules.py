"""Tests for core/tools/schedules.py — schedule write helpers."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.tools.schedules import (
    create_schedule,
    delete_schedule,
    list_schedules,
    parse_cron_schedule,
)


class TestParseCronSchedule:
    def test_hhmm_converts_to_daily_cron(self):
        assert parse_cron_schedule("09:00") == "0 9 * * *"

    def test_five_field_cron_passthrough(self):
        assert parse_cron_schedule("0 7 * * 1-5") == "0 7 * * 1-5"

    def test_invalid_cron_raises(self):
        with pytest.raises(ValueError):
            parse_cron_schedule("not cron")


class TestListSchedules:
    async def test_returns_json_summaries(self):
        pool = MagicMock()
        rows = [
            {
                "name": "morning-proactive",
                "task": "Send a casual morning DM",
                "cron_schedule": "0 9 * * *",
                "enabled": True,
                "discord_channel_id": None,
            }
        ]
        with patch(
            "core.tools.schedules.db.list_schedules_for_user",
            new=AsyncMock(return_value=rows),
        ):
            result = await list_schedules(pool=pool, user_id="u1")

        parsed = json.loads(result)
        assert parsed[0]["name"] == "morning-proactive"
        assert parsed[0]["task"] == "Send a casual morning DM"


class TestCreateSchedule:
    async def test_rejects_duplicate_name(self):
        pool = MagicMock()
        bot = MagicMock()
        existing = {"id": 1, "name": "morning"}

        with patch(
            "core.tools.schedules.db.get_schedule_by_name",
            new=AsyncMock(return_value=existing),
        ):
            result = await create_schedule(
                pool=pool,
                bot=bot,
                user_id="u1",
                name="morning",
                task="say hi",
                cron_schedule="09:00",
            )

        assert "already exists" in result

    async def test_inserts_and_registers_job(self):
        pool = MagicMock()
        bot = MagicMock()

        with (
            patch(
                "core.tools.schedules.db.get_schedule_by_name",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "core.tools.schedules.db.insert_schedule",
                new=AsyncMock(return_value=42),
            ) as mock_insert,
            patch("core.tools.schedules.db.delete_schedule", new=AsyncMock()),
            patch("core.tools.schedules.scheduler.add_job") as mock_add,
            patch("core.tools.schedules.get_settings") as mock_settings,
        ):
            mock_settings.return_value.digest_timezone = "America/Toronto"
            result = await create_schedule(
                pool=pool,
                bot=bot,
                user_id="u1",
                name="gym",
                task="remind about gym",
                cron_schedule="07:00",
                discord_channel_id=None,
            )

        assert "id=42" in result
        mock_insert.assert_awaited_once()
        assert mock_insert.call_args.kwargs["cron_schedule"] == "0 7 * * *"
        mock_add.assert_called_once()
        assert mock_add.call_args.kwargs["id"] == "schedule_42"

    async def test_rolls_back_db_on_scheduler_failure(self):
        pool = MagicMock()
        bot = MagicMock()

        with (
            patch(
                "core.tools.schedules.db.get_schedule_by_name",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "core.tools.schedules.db.insert_schedule",
                new=AsyncMock(return_value=7),
            ),
            patch(
                "core.tools.schedules.db.delete_schedule",
                new=AsyncMock(),
            ) as mock_delete,
            patch(
                "core.tools.schedules.scheduler.add_job",
                side_effect=ValueError("bad cron"),
            ),
            patch("core.tools.schedules.get_settings") as mock_settings,
        ):
            mock_settings.return_value.digest_timezone = "America/Toronto"
            result = await create_schedule(
                pool=pool,
                bot=bot,
                user_id="u1",
                name="bad",
                task="task",
                cron_schedule="99:99",
            )

        assert result.startswith("Error:")
        mock_delete.assert_awaited_once_with(pool, schedule_id=7)


class TestDeleteSchedule:
    async def test_rejects_empty_name(self):
        pool = MagicMock()
        result = await delete_schedule(pool=pool, user_id="u1", name="  ")
        assert "cannot be empty" in result

    async def test_unknown_name_suggests_list(self):
        pool = MagicMock()
        with patch(
            "core.tools.schedules.db.get_schedule_by_name",
            new=AsyncMock(return_value=None),
        ):
            result = await delete_schedule(
                pool=pool, user_id="u1", name="gym-reminder"
            )

        assert "no schedule named" in result
        assert "list_schedules" in result

    async def test_deletes_by_exact_name(self):
        pool = MagicMock()
        row = {
            "id": 5,
            "name": "gym-reminder",
            "task": "remind about gym",
            "cron_schedule": "0 7 * * *",
            "enabled": True,
            "discord_channel_id": None,
        }
        with (
            patch(
                "core.tools.schedules.db.get_schedule_by_name",
                new=AsyncMock(return_value=row),
            ),
            patch("core.tools.schedules.db.delete_schedule", new=AsyncMock()),
            patch("core.tools.schedules.scheduler.remove_job"),
        ):
            result = await delete_schedule(
                pool=pool, user_id="u1", name="gym-reminder"
            )

        assert "Deleted schedule 'gym-reminder'" in result
