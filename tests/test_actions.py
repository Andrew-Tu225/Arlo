"""Tests for core.agent.actions — medium-risk approval gate."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agent.actions import (
    ScheduleApprovalView,
    format_action_summary,
    request_medium_risk_approval,
)
from core.agent.tools import MEDIUM_RISK_TOOLS, ToolContext, is_medium_risk


class TestRiskTiers:
    def test_medium_risk_tools(self):
        assert MEDIUM_RISK_TOOLS == {
            "create_schedule",
            "edit_schedule",
            "delete_schedule",
        }

    def test_is_medium_risk(self):
        assert is_medium_risk("create_schedule")
        assert is_medium_risk("edit_schedule")
        assert is_medium_risk("delete_schedule")
        assert not is_medium_risk("list_schedules")
        assert not is_medium_risk("web_search")
        assert not is_medium_risk("research")
        assert not is_medium_risk("plan_schedule_change")


class TestFormatActionSummary:
    def test_create_schedule(self):
        text = format_action_summary(
            "create_schedule",
            {
                "name": "gym",
                "task": "workout reminder",
                "cron_schedule": "07:00",
            },
        )
        assert "gym" in text
        assert "07:00" in text

    def test_edit_schedule(self):
        text = format_action_summary(
            "edit_schedule",
            {"name": "gym", "enabled": False},
        )
        assert "gym" in text
        assert "enabled" in text

    def test_delete_schedule(self):
        text = format_action_summary(
            "delete_schedule",
            {"name": "morning-proactive"},
        )
        assert "morning-proactive" in text


@pytest.mark.asyncio
async def test_request_approval_posts_discord_embed():
    pool = MagicMock()
    bot = MagicMock()
    channel = MagicMock()
    channel.send = AsyncMock(return_value=MagicMock(id=555))
    bot.get_channel.return_value = channel

    ctx = ToolContext(
        user_id="u1",
        pool=pool,
        bot=bot,
        discord_channel_id="123",
    )

    with (
        patch(
            "core.agent.actions.db.insert_pending_action",
            new=AsyncMock(return_value=42),
        ),
        patch(
            "core.agent.actions.db.set_pending_action_discord_msg_id",
            new=AsyncMock(),
        ),
    ):
        result = await request_medium_risk_approval(
            "delete_schedule",
            {"name": "gym"},
            ctx,
        )

    assert "Awaiting your confirmation" in result
    channel.send.assert_awaited_once()
    embed = channel.send.call_args.kwargs["embed"]
    assert "gym" in embed.description


@pytest.mark.asyncio
async def test_approval_view_confirm_resumes_orchestrator():
    pool = MagicMock()
    bot = MagicMock()
    interaction = AsyncMock()
    interaction.user.id = 123
    interaction.channel_id = 456

    settings_mock = MagicMock()
    settings_mock.discord_user_id = 123

    pending_row = {
        "id": 42,
        "user_id": "u1",
        "tool_name": "create_schedule",
        "tool_args": {"name": "gym", "task": "reminder", "cron_schedule": "09:00"},
        "status": "pending",
        "agent_state": {
            "thread_id": "thread-abc",
            "tool_call_id": "call_abc",
        }
    }

    get_pending_mock = AsyncMock(return_value=pending_row)
    resolve_mock = AsyncMock()
    orchestrator_resume_mock = AsyncMock(return_value="Friendly reply from agent!")

    view = ScheduleApprovalView(
        pending_id=42,
        pool=pool,
        bot=bot,
        user_id="u1",
    )

    with (
        patch("core.agent.actions.get_settings", return_value=settings_mock),
        patch("core.agent.actions.db.get_pending_action", new=get_pending_mock),
        patch("core.agent.actions.db.resolve_pending_action", new=resolve_mock),
        patch("core.agent.orchestrator.resume", new=orchestrator_resume_mock),
        patch.object(view, "_disable_view", new=AsyncMock()),
    ):
        await view.confirm.callback(interaction)

    get_pending_mock.assert_awaited()
    resolve_mock.assert_awaited_once_with(pool, pending_id=42, status="approved")

    orchestrator_resume_mock.assert_awaited_once_with(
        thread_id="thread-abc",
        approved=True,
        user_id="u1",
        pool=pool,
        bot=bot,
        discord_channel_id="456",
    )

    interaction.response.send_message.assert_awaited_once_with(
        "Friendly reply from agent!",
        ephemeral=False,
    )


@pytest.mark.asyncio
async def test_approval_view_cancel_resumes_orchestrator():
    pool = MagicMock()
    bot = MagicMock()
    interaction = AsyncMock()
    interaction.user.id = 123
    interaction.channel_id = 456

    settings_mock = MagicMock()
    settings_mock.discord_user_id = 123

    pending_row = {
        "id": 42,
        "user_id": "u1",
        "tool_name": "create_schedule",
        "tool_args": {"name": "gym", "task": "reminder", "cron_schedule": "09:00"},
        "status": "pending",
        "agent_state": {
            "thread_id": "thread-abc",
            "tool_call_id": "call_abc",
        }
    }

    get_pending_mock = AsyncMock(return_value=pending_row)
    resolve_mock = AsyncMock()
    orchestrator_resume_mock = AsyncMock(return_value="Agent cancellation reply!")

    view = ScheduleApprovalView(
        pending_id=42,
        pool=pool,
        bot=bot,
        user_id="u1",
    )

    with (
        patch("core.agent.actions.get_settings", return_value=settings_mock),
        patch("core.agent.actions.db.get_pending_action", new=get_pending_mock),
        patch("core.agent.actions.db.resolve_pending_action", new=resolve_mock),
        patch("core.agent.orchestrator.resume", new=orchestrator_resume_mock),
        patch.object(view, "_disable_view", new=AsyncMock()),
    ):
        await view.cancel.callback(interaction)

    resolve_mock.assert_awaited_once_with(pool, pending_id=42, status="rejected")
    orchestrator_resume_mock.assert_awaited_once_with(
        thread_id="thread-abc",
        approved=False,
        user_id="u1",
        pool=pool,
        bot=bot,
        discord_channel_id="456",
    )

    interaction.response.send_message.assert_awaited_once_with(
        "Agent cancellation reply!",
        ephemeral=False,
    )

