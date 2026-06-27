"""Tests for core/agent/schedule_planner.py — schedule planner sub-agent."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agent.schedule_planner import (
    SchedulePlan,
    PLANNER_SYSTEM_PROMPT,
    run_schedule_planner,
)
from core.agent.tools import PLANNER_FALLBACK


# ── SchedulePlan schema tests ──────────────────────────────────────────────


class TestSchedulePlan:
    def test_serialises_create_plan(self):
        plan = SchedulePlan(
            action="create",
            name="gym reminder",
            task="Send the user a gym reminder",
            cron_schedule="0 7 * * 1-5",
            rationale="Set a weekday morning gym reminder at 7:00 AM",
        )
        data = json.loads(plan.model_dump_json())
        assert data["action"] == "create"
        assert data["name"] == "gym reminder"
        assert data["cron_schedule"] == "0 7 * * 1-5"
        assert data["discord_channel_id"] is None
        assert data["enabled"] is True

    def test_serialises_delete_plan(self):
        plan = SchedulePlan(
            action="delete",
            name="morning standup",
            task="",
            cron_schedule="0 9 * * *",
            rationale="Delete the morning standup schedule",
        )
        data = json.loads(plan.model_dump_json())
        assert data["action"] == "delete"
        assert data["name"] == "morning standup"

    def test_discord_channel_id_defaults_to_none(self):
        plan = SchedulePlan(
            action="create",
            name="x",
            task="t",
            cron_schedule="0 8 * * *",
            rationale="r",
        )
        assert plan.discord_channel_id is None

    def test_enabled_defaults_to_true(self):
        plan = SchedulePlan(
            action="create",
            name="x",
            task="t",
            cron_schedule="0 8 * * *",
            rationale="r",
        )
        assert plan.enabled is True

    def test_validates_json_round_trip(self):
        plan = SchedulePlan(
            action="edit",
            name="gym reminder",
            task="Send the user a gym reminder",
            cron_schedule="0 8 * * 1-5",
            discord_channel_id="123456",
            enabled=False,
            rationale="Move gym reminder to 8 AM and pause it",
        )
        restored = SchedulePlan.model_validate_json(plan.model_dump_json())
        assert restored.action == "edit"
        assert restored.name == "gym reminder"
        assert restored.cron_schedule == "0 8 * * 1-5"
        assert restored.discord_channel_id == "123456"
        assert restored.enabled is False

    def test_validate_json_rejects_missing_required_fields(self):
        with pytest.raises(Exception):
            SchedulePlan.model_validate_json('{"action": "create", "name": "x"}')

    def test_validate_json_rejects_invalid_action(self):
        with pytest.raises(Exception):
            SchedulePlan.model_validate_json(json.dumps({
                "action": "rename",
                "name": "x",
                "task": "t",
                "cron_schedule": "0 8 * * *",
                "rationale": "r",
            }))


# ── run_schedule_planner integration tests ────────────────────────────────


def _make_completion(
    *,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    total_tokens: int = 50,
) -> MagicMock:
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    message.model_dump.return_value = {
        "role": "assistant",
        "content": content,
        **({"tool_calls": tool_calls} if tool_calls else {}),
    }
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = MagicMock(total_tokens=total_tokens)
    return completion


def _tool_call(name: str, arguments: dict[str, Any], call_id: str = "c1") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _make_valid_plan_json(
    action: str = "create",
    name: str = "gym reminder",
    task: str = "Send the user a gym reminder",
    cron_schedule: str = "0 7 * * 1-5",
    rationale: str = "Set a weekday morning gym reminder at 7:00 AM",
) -> str:
    return json.dumps({
        "action": action,
        "name": name,
        "task": task,
        "cron_schedule": cron_schedule,
        "discord_channel_id": None,
        "enabled": True,
        "rationale": rationale,
    })


@pytest.mark.asyncio
async def test_run_schedule_planner_returns_valid_plan_json():
    """Sub-agent returns valid JSON → parsed into SchedulePlan and returned."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_make_completion(
            content=_make_valid_plan_json(
                name="gym reminder",
                cron_schedule="0 7 * * 1-5",
                rationale="Weekday morning gym reminder at 7 AM",
            )
        )
    )
    with patch("core.agent.react.get_client", return_value=client):
        result = await run_schedule_planner(
            "remind me to go to the gym on weekdays at 7",
            user_id="u1",
            pool=None,
        )

    plan = SchedulePlan.model_validate_json(result)
    assert plan.action == "create"
    assert plan.cron_schedule == "0 7 * * 1-5"
    assert plan.enabled is True


@pytest.mark.asyncio
async def test_run_schedule_planner_passes_request_as_user_message():
    """The request string is sent as the user message to the sub-agent."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_make_completion(content=_make_valid_plan_json())
    )
    request = "remind me every Sunday at 9 AM to prep for the week"
    with patch("core.agent.react.get_client", return_value=client):
        await run_schedule_planner(request, user_id="u1", pool=None)

    first_call_messages = client.chat.completions.create.await_args_list[0].kwargs["messages"]
    user_msg = next(m for m in first_call_messages if m["role"] == "user")
    assert user_msg["content"] == request


@pytest.mark.asyncio
async def test_run_schedule_planner_calls_list_schedules_before_plan():
    """Sub-agent calls list_schedules then produces a SchedulePlan."""
    list_schedules_call = _tool_call("list_schedules", {})
    valid_plan = _make_valid_plan_json(
        name="gym reminder",
        cron_schedule="0 7 * * 1-5",
        rationale="Weekday morning gym reminder at 7 AM",
    )
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            _make_completion(content=None, tool_calls=[list_schedules_call]),
            _make_completion(content=valid_plan),
        ]
    )
    mock_list = AsyncMock(return_value=json.dumps([]))
    mock_pool = MagicMock()
    with (
        patch("core.agent.react.get_client", return_value=client),
        patch("core.agent.tools.schedules.list_schedules", new=mock_list),
    ):
        result = await run_schedule_planner(
            "add a weekday gym reminder at 7 AM",
            user_id="u1",
            pool=mock_pool,
        )

    mock_list.assert_called_once()
    plan = SchedulePlan.model_validate_json(result)
    assert plan.action == "create"
    assert plan.cron_schedule == "0 7 * * 1-5"


@pytest.mark.asyncio
async def test_run_schedule_planner_clarifying_question_returned_as_plain_string():
    """When the sub-agent returns plain text, it is returned as a clarifying question."""
    clarifying_q = "What time would you like the reminder?"
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_make_completion(content=clarifying_q)
    )
    with patch("core.agent.react.get_client", return_value=client):
        result = await run_schedule_planner(
            "remind me about the gym",
            user_id="u1",
            pool=None,
        )

    assert result == clarifying_q
    # Must not be parseable as SchedulePlan JSON
    with pytest.raises(Exception):
        SchedulePlan.model_validate_json(result)


@pytest.mark.asyncio
async def test_run_schedule_planner_exception_returns_fallback():
    """If ainvoke raises unexpectedly, run_schedule_planner returns PLANNER_FALLBACK."""
    with patch(
        "core.agent.schedule_planner.planner_agent_graph.ainvoke",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        result = await run_schedule_planner("set up a reminder", user_id="u1", pool=None)

    assert result == PLANNER_FALLBACK


@pytest.mark.asyncio
async def test_run_schedule_planner_ceiling_returns_fallback():
    """On iteration ceiling, run_schedule_planner returns PLANNER_FALLBACK."""
    list_schedules_call = _tool_call("list_schedules", {})
    client = MagicMock()
    # Always return a tool_call so the loop never settles
    client.chat.completions.create = AsyncMock(
        return_value=_make_completion(content=None, tool_calls=[list_schedules_call])
    )
    mock_list = AsyncMock(return_value=json.dumps([]))
    mock_pool = MagicMock()
    with (
        patch("core.agent.react.get_client", return_value=client),
        patch("core.agent.tools.schedules.list_schedules", new=mock_list),
    ):
        result = await run_schedule_planner(
            "do something vague",
            user_id="u1",
            pool=mock_pool,
        )

    assert result == PLANNER_FALLBACK


@pytest.mark.asyncio
async def test_run_schedule_planner_empty_response_returns_fallback():
    """An empty response string from the sub-agent returns PLANNER_FALLBACK."""
    with patch(
        "core.agent.schedule_planner.planner_agent_graph.ainvoke",
        new=AsyncMock(return_value={"response": ""}),
    ):
        result = await run_schedule_planner("remind me daily", user_id="u1", pool=None)

    assert result == PLANNER_FALLBACK


@pytest.mark.asyncio
async def test_run_schedule_planner_delete_action():
    """Sub-agent returns a delete plan; result is valid SchedulePlan JSON."""
    delete_plan = _make_valid_plan_json(
        action="delete",
        name="morning standup",
        task="",
        cron_schedule="0 9 * * *",
        rationale="Remove the morning standup schedule",
    )
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_make_completion(content=delete_plan)
    )
    with patch("core.agent.react.get_client", return_value=client):
        result = await run_schedule_planner(
            "delete my morning standup reminder",
            user_id="u1",
            pool=None,
        )

    plan = SchedulePlan.model_validate_json(result)
    assert plan.action == "delete"
    assert plan.name == "morning standup"


def test_planner_system_prompt_includes_output_section():
    """PLANNER_SYSTEM_PROMPT must contain the OUTPUT JSON template."""
    assert "OUTPUT" in PLANNER_SYSTEM_PROMPT
    assert '"action"' in PLANNER_SYSTEM_PROMPT
    assert '"cron_schedule"' in PLANNER_SYSTEM_PROMPT
    assert '"rationale"' in PLANNER_SYSTEM_PROMPT


def test_planner_system_prompt_mentions_list_schedules():
    """PLANNER_SYSTEM_PROMPT must instruct the planner to call list_schedules."""
    assert "list_schedules" in PLANNER_SYSTEM_PROMPT
