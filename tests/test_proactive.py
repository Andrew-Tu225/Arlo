"""Tests for core/agent/proactive.py — scheduled task executor."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agent.proactive import run_proactive_agent
from core.agent.tools import PROACTIVE_FALLBACK, get_proactive_schemas


# ---------------------------------------------------------------------------
# Fallback constant
# ---------------------------------------------------------------------------

class TestProactiveFallback:
    def test_proactive_fallback_is_empty_string(self):
        assert PROACTIVE_FALLBACK == ""


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------

class TestProactiveToolSurface:
    def test_schema_names_are_correct(self):
        names = {s["function"]["name"] for s in get_proactive_schemas()}
        assert names == {"search_memory", "run_research", "get_recent_sends"}

    def test_schema_count(self):
        assert len(get_proactive_schemas()) == 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# run_proactive_agent — direct reply (no tool calls)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_returns_message_when_llm_replies_directly():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_make_completion(content="Morning! Saw that GPT-5 dropped today.")
    )
    with patch("core.agent.react.get_client", return_value=client):
        result = await run_proactive_agent(
            "Say good morning", user_id="u1", schedule_id=1
        )
    assert result == "Morning! Saw that GPT-5 dropped today."


@pytest.mark.asyncio
async def test_response_is_stripped():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_make_completion(content="  Hey there!  ")
    )
    with patch("core.agent.react.get_client", return_value=client):
        result = await run_proactive_agent(
            "Morning DM", user_id="u1", schedule_id=1
        )
    assert result == "Hey there!"


# ---------------------------------------------------------------------------
# run_proactive_agent — ainvoke exception (layer 1 fallback)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ainvoke_exception_returns_empty():
    with patch(
        "core.agent.proactive.proactive_agent_graph.ainvoke",
        new=AsyncMock(side_effect=RuntimeError("graph exploded")),
    ):
        result = await run_proactive_agent(
            "Morning task", user_id="u1", schedule_id=1
        )
    assert result == ""


# ---------------------------------------------------------------------------
# run_proactive_agent — tool calls
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_memory_tool_called():
    search_call = _tool_call("search_memory", {"query": "user interests"})
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            _make_completion(content=None, tool_calls=[search_call]),
            _make_completion(content="Based on your love of AI, here's something cool."),
        ]
    )
    mock_search = AsyncMock(return_value=["likes Python", "follows AI news"])
    with (
        patch("core.agent.react.get_client", return_value=client),
        patch("core.agent.tools.store.search", new=mock_search),
    ):
        result = await run_proactive_agent(
            "Morning outreach", user_id="u1", schedule_id=1
        )
    assert result != ""
    mock_search.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_research_tool_called():
    research_call = _tool_call("run_research", {"task": "AI news today"})
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            _make_completion(content=None, tool_calls=[research_call]),
            _make_completion(content="GPT-5 dropped today, pretty big deal."),
        ]
    )
    mock_research = AsyncMock(return_value='{"summary":"GPT-5 released","sources":[],"complete":true}')
    with (
        patch("core.agent.react.get_client", return_value=client),
        patch("core.agent.researcher.run_research", new=mock_research),
    ):
        result = await run_proactive_agent(
            "Find AI news", user_id="u1", schedule_id=1
        )
    assert result != ""


@pytest.mark.asyncio
async def test_get_recent_sends_queries_correct_schedule_id():
    recent_call = _tool_call("get_recent_sends", {"limit": 7})
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            _make_completion(content=None, tool_calls=[recent_call]),
            _make_completion(content="Here's something new today."),
        ]
    )
    mock_get_runs = AsyncMock(return_value=[{"message_preview": "old topic"}])
    pool = MagicMock()

    with (
        patch("core.agent.react.get_client", return_value=client),
        patch("core.agent.tools.db.get_recent_runs", new=mock_get_runs),
    ):
        await run_proactive_agent(
            "Daily topic discovery", user_id="u1", schedule_id=42, pool=pool
        )

    mock_get_runs.assert_awaited_once()
    kwargs = mock_get_runs.call_args.kwargs
    assert kwargs["schedule_id"] == 42
    assert kwargs["limit"] == 7


@pytest.mark.asyncio
async def test_get_recent_sends_empty_history_returns_no_previous_sends():
    recent_call = _tool_call("get_recent_sends", {"limit": 5})
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            _make_completion(content=None, tool_calls=[recent_call]),
            _make_completion(content="Kicking things off fresh."),
        ]
    )
    pool = MagicMock()

    with (
        patch("core.agent.react.get_client", return_value=client),
        patch("core.agent.tools.db.get_recent_runs", new=AsyncMock(return_value=[])),
    ):
        result = await run_proactive_agent(
            "Daily topic discovery", user_id="u1", schedule_id=1, pool=pool
        )

    assert result == "Kicking things off fresh."


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_channel_topic_appears_in_user_message():
    captured: dict[str, Any] = {}

    async def fake_ainvoke(state, config):
        captured["user_msg"] = state["messages"][0]["content"]
        return {"response": "channel message"}

    with patch("core.agent.proactive.proactive_agent_graph.ainvoke", new=fake_ainvoke):
        await run_proactive_agent(
            "Post something", user_id="u1", schedule_id=1, channel_topic="NBA scores"
        )

    assert "NBA scores" in captured["user_msg"]
    assert "Post something" in captured["user_msg"]


@pytest.mark.asyncio
async def test_no_channel_topic_not_in_user_message():
    captured: dict[str, Any] = {}

    async def fake_ainvoke(state, config):
        captured["user_msg"] = state["messages"][0]["content"]
        return {"response": "dm message"}

    with patch("core.agent.proactive.proactive_agent_graph.ainvoke", new=fake_ainvoke):
        await run_proactive_agent(
            "Good morning", user_id="u1", schedule_id=1, channel_topic=None
        )

    assert "channel" not in captured["user_msg"].lower()


# ---------------------------------------------------------------------------
# schedule_id in config
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_schedule_id_passed_in_configurable():
    captured: dict[str, Any] = {}

    async def fake_ainvoke(state, config):
        captured["config"] = config
        return {"response": "ok"}

    with patch("core.agent.proactive.proactive_agent_graph.ainvoke", new=fake_ainvoke):
        await run_proactive_agent(
            "task", user_id="u1", schedule_id=99
        )

    assert captured["config"]["configurable"]["schedule_id"] == 99


@pytest.mark.asyncio
async def test_fresh_thread_id_per_call():
    thread_ids: list[str] = []

    async def fake_ainvoke(state, config):
        thread_ids.append(config["configurable"]["thread_id"])
        return {"response": "ok"}

    with patch("core.agent.proactive.proactive_agent_graph.ainvoke", new=fake_ainvoke):
        await run_proactive_agent("task", user_id="u1", schedule_id=1)
        await run_proactive_agent("task", user_id="u1", schedule_id=1)

    assert thread_ids[0] != thread_ids[1]


# ---------------------------------------------------------------------------
# Ceiling / iteration limit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ceiling_returns_empty():
    search_call = _tool_call("search_memory", {"query": "interests"})
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_make_completion(content=None, tool_calls=[search_call])
    )
    mock_search = AsyncMock(return_value=[])
    with (
        patch("core.agent.react.get_client", return_value=client),
        patch("core.agent.tools.store.search", new=mock_search),
        patch("core.agent.react.get_settings") as mock_settings,
    ):
        mock_settings.return_value.proactive_max_react_iterations = 2
        mock_settings.return_value.proactive_task_token_budget = 99999
        mock_settings.return_value.tool_observation_max_chars = 2000
        result = await run_proactive_agent(
            "Morning task", user_id="u1", schedule_id=1
        )

    assert result == ""
