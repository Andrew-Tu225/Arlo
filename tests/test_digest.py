"""Tests for core/scheduler/digest.py."""

from unittest.mock import AsyncMock, MagicMock, patch

from core.scheduler.digest import (
    _build_prompt,
    _hhmm_to_cron,
    register_digest_jobs,
    run_schedule_agent,
    run_schedule_job,
    seed_default_schedules,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHhmmToCron:
    def test_converts_09_00(self):
        assert _hhmm_to_cron("09:00") == "0 9 * * *"

    def test_converts_20_30(self):
        assert _hhmm_to_cron("20:30") == "30 20 * * *"

    def test_strips_leading_zeros(self):
        assert _hhmm_to_cron("08:05") == "5 8 * * *"


class TestBuildPrompt:
    def test_includes_task(self):
        assert "Start a morning conversation" in _build_prompt(
            task="Start a morning conversation", profile_facts=[]
        )

    def test_includes_profile_facts(self):
        assert "user likes Python" in _build_prompt(task="task", profile_facts=["user likes Python"])

    def test_empty_sentinel_instruction_present(self):
        assert "empty" in _build_prompt(task="task", profile_facts=[]).lower()

    def test_no_facts_section_when_empty(self):
        assert "profile facts" not in _build_prompt(task="task", profile_facts=[]).lower()


# ---------------------------------------------------------------------------
# seed_default_schedules
# ---------------------------------------------------------------------------

class TestSeedDefaultSchedules:
    async def test_calls_insert_schedule_with_none_channel(self):
        pool = MagicMock()

        with (
            patch("core.scheduler.digest.db.insert_schedule", new=AsyncMock(return_value=1)) as mock_insert,
            patch("core.scheduler.digest.get_settings") as mock_settings,
        ):
            mock_settings.return_value.digest_time = "09:00"
            await seed_default_schedules(pool, user_id="u1")

        call_kwargs = mock_insert.call_args.kwargs
        assert call_kwargs["user_id"] == "u1"
        assert call_kwargs["discord_channel_id"] is None

    async def test_uses_digest_time_for_cron(self):
        pool = MagicMock()

        with (
            patch("core.scheduler.digest.db.insert_schedule", new=AsyncMock(return_value=1)) as mock_insert,
            patch("core.scheduler.digest.get_settings") as mock_settings,
        ):
            mock_settings.return_value.digest_time = "08:30"
            await seed_default_schedules(pool, user_id="u1")

        assert mock_insert.call_args.kwargs["cron_schedule"] == "30 8 * * *"

    async def test_uses_morning_proactive_name(self):
        pool = MagicMock()

        with (
            patch("core.scheduler.digest.db.insert_schedule", new=AsyncMock(return_value=1)) as mock_insert,
            patch("core.scheduler.digest.get_settings") as mock_settings,
        ):
            mock_settings.return_value.digest_time = "09:00"
            await seed_default_schedules(pool, user_id="u1")

        assert mock_insert.call_args.kwargs["name"] == "morning-proactive"


# ---------------------------------------------------------------------------
# run_schedule_agent
# ---------------------------------------------------------------------------

class TestRunScheduleAgent:
    async def test_returns_composed_message(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Good morning!"

        with (
            patch("core.scheduler.digest.store.search", new=AsyncMock(return_value=[])),
            patch("core.scheduler.digest.get_client") as mock_client,
        ):
            mock_client.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
            result = await run_schedule_agent(task="Say good morning", user_id="u1")

        assert result == "Good morning!"

    async def test_returns_empty_on_empty_sentinel(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "empty"

        with (
            patch("core.scheduler.digest.store.search", new=AsyncMock(return_value=[])),
            patch("core.scheduler.digest.get_client") as mock_client,
        ):
            mock_client.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
            result = await run_schedule_agent(task="task", user_id="u1")

        assert result == ""

    async def test_returns_empty_on_llm_error(self):
        with (
            patch("core.scheduler.digest.store.search", new=AsyncMock(return_value=[])),
            patch("core.scheduler.digest.get_client") as mock_client,
        ):
            mock_client.return_value.chat.completions.create = AsyncMock(
                side_effect=RuntimeError("LLM down")
            )
            result = await run_schedule_agent(task="task", user_id="u1")

        assert result == ""

    async def test_continues_when_store_search_fails(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Brief"

        with (
            patch("core.scheduler.digest.store.search", new=AsyncMock(side_effect=RuntimeError("mem0 down"))),
            patch("core.scheduler.digest.get_client") as mock_client,
        ):
            mock_client.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
            result = await run_schedule_agent(task="task", user_id="u1")

        assert result == "Brief"

    async def test_injects_channel_topic_when_provided(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Sports update"
        captured = {}

        async def fake_create(**kwargs):
            captured["content"] = kwargs["messages"][0]["content"]
            return mock_response

        with (
            patch("core.scheduler.digest.store.search", new=AsyncMock(return_value=[])),
            patch("core.scheduler.digest.get_client") as mock_client,
        ):
            mock_client.return_value.chat.completions.create = fake_create
            await run_schedule_agent(task="Find scores", user_id="u1", channel_topic="NBA sports")

        assert "NBA sports" in captured["content"]

    async def test_no_channel_topic_for_dm(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hey!"
        captured = {}

        async def fake_create(**kwargs):
            captured["content"] = kwargs["messages"][0]["content"]
            return mock_response

        with (
            patch("core.scheduler.digest.store.search", new=AsyncMock(return_value=[])),
            patch("core.scheduler.digest.get_client") as mock_client,
        ):
            mock_client.return_value.chat.completions.create = fake_create
            await run_schedule_agent(task="Morning message", user_id="u1", channel_topic=None)

        assert "Channel context" not in captured["content"]


# ---------------------------------------------------------------------------
# run_schedule_job
# ---------------------------------------------------------------------------

def _make_schedule(discord_channel_id=None, user_id="123"):
    return {
        "id": 1, "user_id": user_id, "name": "morning-proactive",
        "task": "Morning task", "discord_channel_id": discord_channel_id,
        "channel_topic": None, "cron_schedule": "0 9 * * *",
        "poll_interval_secs": None, "last_sent_at": None,
        "enabled": True, "created_at": None,
    }


class TestRunScheduleJob:
    async def test_dms_user_when_no_channel_id(self):
        sched = _make_schedule(discord_channel_id=None, user_id="123")
        dm = AsyncMock()
        user = AsyncMock()
        user.create_dm = AsyncMock(return_value=dm)
        bot = AsyncMock()
        bot.fetch_user = AsyncMock(return_value=user)
        pool = MagicMock()

        with (
            patch("core.scheduler.digest.db.get_schedule", new=AsyncMock(return_value=sched)),
            patch("core.scheduler.digest.db.update_schedule_last_sent", new=AsyncMock()),
            patch("core.scheduler.digest.run_schedule_agent", new=AsyncMock(return_value="Hey!")),
        ):
            await run_schedule_job(bot, pool, schedule_id=1)

        dm.send.assert_called_once_with("Hey!")

    async def test_sends_to_channel_when_channel_id_set(self):
        sched = _make_schedule(discord_channel_id="999")
        channel = AsyncMock()
        bot = MagicMock()
        bot.get_channel = MagicMock(return_value=channel)
        pool = MagicMock()

        with (
            patch("core.scheduler.digest.db.get_schedule", new=AsyncMock(return_value=sched)),
            patch("core.scheduler.digest.db.update_schedule_last_sent", new=AsyncMock()),
            patch("core.scheduler.digest.run_schedule_agent", new=AsyncMock(return_value="Update!")),
        ):
            await run_schedule_job(bot, pool, schedule_id=1)

        channel.send.assert_called_once_with("Update!")

    async def test_skips_send_when_agent_returns_empty(self):
        sched = _make_schedule(discord_channel_id=None, user_id="123")
        dm = AsyncMock()
        user = AsyncMock()
        user.create_dm = AsyncMock(return_value=dm)
        bot = AsyncMock()
        bot.fetch_user = AsyncMock(return_value=user)
        pool = MagicMock()

        with (
            patch("core.scheduler.digest.db.get_schedule", new=AsyncMock(return_value=sched)),
            patch("core.scheduler.digest.run_schedule_agent", new=AsyncMock(return_value="")),
        ):
            await run_schedule_job(bot, pool, schedule_id=1)

        dm.send.assert_not_called()

    async def test_skips_when_schedule_not_found(self):
        bot = MagicMock()
        pool = MagicMock()

        with patch("core.scheduler.digest.db.get_schedule", new=AsyncMock(return_value=None)):
            await run_schedule_job(bot, pool, schedule_id=999)

        bot.get_channel.assert_not_called()

    async def test_skips_when_channel_not_found_in_discord(self):
        sched = _make_schedule(discord_channel_id="deleted")
        bot = MagicMock()
        bot.get_channel = MagicMock(return_value=None)
        pool = MagicMock()

        with patch("core.scheduler.digest.db.get_schedule", new=AsyncMock(return_value=sched)):
            await run_schedule_job(bot, pool, schedule_id=1)  # should not raise

    async def test_truncates_long_messages(self):
        sched = _make_schedule(discord_channel_id=None, user_id="123")
        dm = AsyncMock()
        user = AsyncMock()
        user.create_dm = AsyncMock(return_value=dm)
        bot = AsyncMock()
        bot.fetch_user = AsyncMock(return_value=user)
        pool = MagicMock()

        with (
            patch("core.scheduler.digest.db.get_schedule", new=AsyncMock(return_value=sched)),
            patch("core.scheduler.digest.db.update_schedule_last_sent", new=AsyncMock()),
            patch("core.scheduler.digest.run_schedule_agent", new=AsyncMock(return_value="x" * 2000)),
        ):
            await run_schedule_job(bot, pool, schedule_id=1)

        sent = dm.send.call_args.args[0]
        assert len(sent) <= 1901

    async def test_does_not_raise_on_exception(self):
        bot = MagicMock()
        pool = MagicMock()
        with patch("core.scheduler.digest.db.get_schedule", new=AsyncMock(side_effect=RuntimeError("DB down"))):
            await run_schedule_job(bot, pool, schedule_id=1)  # should not raise


# ---------------------------------------------------------------------------
# register_digest_jobs
# ---------------------------------------------------------------------------

class TestRegisterDigestJobs:
    async def test_registers_no_jobs_when_db_empty(self):
        bot = MagicMock()
        pool = MagicMock()

        with (
            patch("core.scheduler.digest.db.get_enabled_schedules", new=AsyncMock(return_value=[])),
            patch("core.scheduler.digest.get_settings") as mock_settings,
            patch("core.scheduler.digest.scheduler") as mock_scheduler,
        ):
            mock_settings.return_value.discord_user_id = "u1"
            mock_settings.return_value.digest_timezone = "America/Toronto"
            mock_scheduler.get_jobs.return_value = []
            await register_digest_jobs(bot, pool)

        mock_scheduler.add_job.assert_not_called()

    async def test_registers_cron_schedule(self):
        schedules = [{
            "id": 1, "name": "morning-proactive", "cron_schedule": "0 9 * * *",
            "discord_channel_id": None, "user_id": "u1",
        }]
        bot = MagicMock()
        pool = MagicMock()

        with (
            patch("core.scheduler.digest.db.get_enabled_schedules", new=AsyncMock(return_value=schedules)),
            patch("core.scheduler.digest.get_settings") as mock_settings,
            patch("core.scheduler.digest.scheduler") as mock_scheduler,
        ):
            mock_settings.return_value.discord_user_id = "u1"
            mock_settings.return_value.digest_timezone = "America/Toronto"
            mock_scheduler.get_jobs.return_value = []
            await register_digest_jobs(bot, pool)

        mock_scheduler.add_job.assert_called_once()

    async def test_skips_event_driven_schedules(self):
        schedules = [{
            "id": 1, "name": "game-alert", "cron_schedule": None,
            "poll_interval_secs": 1800, "discord_channel_id": "111", "user_id": "u1",
        }]
        bot = MagicMock()
        pool = MagicMock()

        with (
            patch("core.scheduler.digest.db.get_enabled_schedules", new=AsyncMock(return_value=schedules)),
            patch("core.scheduler.digest.get_settings") as mock_settings,
            patch("core.scheduler.digest.scheduler") as mock_scheduler,
        ):
            mock_settings.return_value.discord_user_id = "u1"
            mock_settings.return_value.digest_timezone = "America/Toronto"
            mock_scheduler.get_jobs.return_value = []
            await register_digest_jobs(bot, pool)

        mock_scheduler.add_job.assert_not_called()

    async def test_clears_stale_jobs_before_registering(self):
        stale_job = MagicMock()
        stale_job.id = "schedule_99"
        bot = MagicMock()
        pool = MagicMock()

        with (
            patch("core.scheduler.digest.db.get_enabled_schedules", new=AsyncMock(return_value=[])),
            patch("core.scheduler.digest.get_settings") as mock_settings,
            patch("core.scheduler.digest.scheduler") as mock_scheduler,
        ):
            mock_settings.return_value.discord_user_id = "u1"
            mock_settings.return_value.digest_timezone = "America/Toronto"
            mock_scheduler.get_jobs.return_value = [stale_job]
            await register_digest_jobs(bot, pool)

        stale_job.remove.assert_called_once()
