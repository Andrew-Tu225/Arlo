"""Tests for core.agent.orchestrator — ReAct graph entrypoint."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.types import Command

from core.agent.orchestrator import FALLBACK_RESPONSE, resume, run


def _completion(
    content: str | None = "Hello there!",
    *,
    tool_calls=None,
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


@pytest.mark.asyncio
async def test_run_returns_string():
    with patch("core.agent.react.get_client") as mock_get:
        mock_get.return_value.chat.completions.create = AsyncMock(
            return_value=_completion("Hi"),
        )
        result = await run([{"role": "user", "content": "Hey"}])
    assert isinstance(result, str)
    assert result == "Hi"


@pytest.mark.asyncio
async def test_run_handles_api_error_gracefully():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=Exception("API down"))
    with patch("core.agent.react.get_client", return_value=client):
        result = await run([{"role": "user", "content": "Hey"}])
    assert result == FALLBACK_RESPONSE


@pytest.mark.asyncio
async def test_run_passes_pool_bot_and_channel_into_graph():
    pool = MagicMock()
    bot = MagicMock()
    with patch(
        "core.agent.orchestrator.run_react_graph",
        new=AsyncMock(return_value="done"),
    ) as mock_run:
        await run(
            [{"role": "user", "content": "Hey"}],
            user_id="u1",
            pool=pool,
            bot=bot,
            discord_channel_id="999",
        )

    _, kwargs = mock_run.call_args
    assert kwargs["user_id"] == "u1"
    assert kwargs["pool"] is pool
    assert kwargs["bot"] is bot
    assert kwargs["discord_channel_id"] == "999"


# ── resume() tests ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resume_approved_passes_command_and_thread_id():
    with patch(
        "core.agent.orchestrator.run_react_graph",
        new=AsyncMock(return_value="Schedule created!"),
    ) as mock_run:
        result = await resume("thread-1", True, user_id="u1")

    assert result == "Schedule created!"
    _, kwargs = mock_run.call_args
    assert isinstance(kwargs["resume_command"], Command)
    assert kwargs["resume_command"].resume is True
    assert kwargs["thread_id"] == "thread-1"


@pytest.mark.asyncio
async def test_resume_rejected_passes_false_command():
    with patch(
        "core.agent.orchestrator.run_react_graph",
        new=AsyncMock(return_value="Nothing was changed."),
    ) as mock_run:
        result = await resume("thread-2", False, user_id="u1")

    assert result == "Nothing was changed."
    _, kwargs = mock_run.call_args
    assert kwargs["resume_command"].resume is False


@pytest.mark.asyncio
async def test_resume_inserts_episodic_message_on_completion():
    pool = MagicMock()
    mock_insert = AsyncMock()
    with (
        patch("core.agent.orchestrator.run_react_graph", new=AsyncMock(return_value="Done!")),
        patch("core.agent.orchestrator.db.insert_episodic_message", new=mock_insert),
        patch("core.agent.orchestrator.extractor.maybe_extract", new=AsyncMock()),
    ):
        await resume("t1", True, user_id="u1", pool=pool)

    mock_insert.assert_awaited_once()
    assert mock_insert.call_args.kwargs["role"] == "assistant"
    assert mock_insert.call_args.kwargs["content"] == "Done!"


@pytest.mark.asyncio
async def test_resume_skips_episodic_insert_when_awaiting_confirmation():
    pool = MagicMock()
    mock_insert = AsyncMock()
    with (
        patch(
            "core.agent.orchestrator.run_react_graph",
            new=AsyncMock(return_value="Awaiting your confirmation before I proceed."),
        ),
        patch("core.agent.orchestrator.db.insert_episodic_message", new=mock_insert),
    ):
        result = await resume("t1", True, user_id="u1", pool=pool)

    mock_insert.assert_not_awaited()
    assert result.startswith("Awaiting")


# ── sub-agent proxy tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_research_tool_delegates_to_run_research():
    """orchestrator research tool calls run_research, not raw web_search."""
    research_call = {
        "id": "call_r1",
        "type": "function",
        "function": {"name": "research", "arguments": '{"task": "latest AI news"}'},
    }
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            _completion(content=None, tool_calls=[research_call]),
            _completion(content="Here's what I found."),
        ]
    )
    brief = '{"summary":"AI advances.","sources":[],"complete":true,"note":null}'
    mock_run_research = AsyncMock(return_value=brief)

    with (
        patch("core.agent.react.get_client", return_value=client),
        patch("core.agent.researcher.run_research", new=mock_run_research),
    ):
        result = await run([{"role": "user", "content": "What's new in AI?"}], user_id="u1")

    mock_run_research.assert_awaited_once()
    assert result == "Here's what I found."


@pytest.mark.asyncio
async def test_run_plan_schedule_change_delegates_to_run_schedule_planner():
    """orchestrator plan_schedule_change tool calls run_schedule_planner."""
    plan_call = {
        "id": "call_p1",
        "type": "function",
        "function": {
            "name": "plan_schedule_change",
            "arguments": '{"request": "gym weekdays at 7am"}',
        },
    }
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            _completion(content=None, tool_calls=[plan_call]),
            _completion(content="Got it — gym reminder set for weekdays at 7 AM."),
        ]
    )
    plan_json = json.dumps({
        "action": "create",
        "name": "gym reminder",
        "task": "Send the user a gym reminder",
        "cron_schedule": "0 7 * * 1-5",
        "discord_channel_id": None,
        "enabled": True,
        "rationale": "Weekday gym reminder at 7 AM",
    })
    mock_run_planner = AsyncMock(return_value=plan_json)

    with (
        patch("core.agent.react.get_client", return_value=client),
        patch("core.agent.schedule_planner.run_schedule_planner", new=mock_run_planner),
    ):
        result = await run(
            [{"role": "user", "content": "Remind me to go to the gym weekdays at 7am"}],
            user_id="u1",
            pool=MagicMock(),
        )

    mock_run_planner.assert_awaited_once()
    assert result == "Got it — gym reminder set for weekdays at 7 AM."
