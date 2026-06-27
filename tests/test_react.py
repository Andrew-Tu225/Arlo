"""Tests for core.agent.react — shared ReAct graph."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from openai.types.chat import ChatCompletionMessageParam

from core.agent.react import (
    LLM_ERROR_FALLBACK,
    ReactGraphConfig,
    build_react_graph,
    run_react_graph,
)
from core.agent.tools import RESEARCH_FALLBACK, build_orchestrator_tools

_USER_MSG: list[ChatCompletionMessageParam] = [
    {"role": "user", "content": "What's the weather?"},
]


def _config(
    *,
    max_iterations: int = 8,
    token_budget: int = 8000,
    tool_schemas: list[dict[str, Any]] | None = None,
) -> ReactGraphConfig:
    schemas = tool_schemas or [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "search",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
    ]
    return ReactGraphConfig(
        build_system_prompt=lambda _state: "You are Arlo.",
        tool_schemas=schemas,
        tool_builder=build_orchestrator_tools,
        max_react_iterations=max_iterations,
        task_token_budget=token_budget,
    )


def _graph(**kwargs):
    """Build a compiled graph with a MemorySaver (required for aget_state)."""
    return build_react_graph(_config(**kwargs), checkpointer=MemorySaver())


def _completion(
    *,
    content: str | None = "Here you go.",
    tool_calls: list[dict[str, Any]] | None = None,
    total_tokens: int = 100,
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


def _tool_call(
    *,
    call_id: str = "call_1",
    name: str = "web_search",
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments or {"query": "weather"}),
        },
    }


@pytest.mark.asyncio
async def test_no_tool_calls_exits_with_assistant_text():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_completion(content="Sunny and warm."),
    )
    graph = _graph()

    with patch("core.agent.react.get_client", return_value=client):
        result = await run_react_graph(
            graph,
            messages=_USER_MSG,
            user_id="u1",
        )

    assert result == "Sunny and warm."
    assert client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_tool_round_trip_then_final_reply():
    tool_calls = [_tool_call()]
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            _completion(content=None, tool_calls=tool_calls),
            _completion(content="Based on search: rain later."),
        ],
    )
    graph = _graph()

    with (
        patch("core.agent.react.get_client", return_value=client),
        patch(
            "core.agent.react.invoke_tool",
            new=AsyncMock(return_value='[{"title":"Rain"}]'),
        ) as mock_invoke,
    ):
        result = await run_react_graph(graph, messages=_USER_MSG, user_id="u1")

    assert result == "Based on search: rain later."
    assert client.chat.completions.create.await_count == 2
    mock_invoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_iteration_ceiling_before_reason_after_max_tool_rounds():
    """After max tool_executor passes, the next reason hits the ceiling."""
    tool_calls = [_tool_call(call_id="c1")]
    client = MagicMock()
    # Never reached if ceiling works — only tool rounds should run
    client.chat.completions.create = AsyncMock(
        return_value=_completion(content=None, tool_calls=tool_calls),
    )
    graph = _graph(max_iterations=2)

    with (
        patch("core.agent.react.get_client", return_value=client),
        patch("core.agent.react.invoke_tool", new=AsyncMock(return_value="[]")),
    ):
        result = await run_react_graph(graph, messages=_USER_MSG, user_id="u1")

    assert result == RESEARCH_FALLBACK
    # 2 reason calls that issued tools, then ceiling on 3rd reason
    assert client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_token_budget_ceiling_before_reason():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_completion(content="ok", total_tokens=200),
    )
    graph = _graph(token_budget=150)

    with patch("core.agent.react.get_client", return_value=client):
        result = await run_react_graph(graph, messages=_USER_MSG, user_id="u1")

    assert result == RESEARCH_FALLBACK


@pytest.mark.asyncio
async def test_llm_error_returns_llm_fallback():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("API down"))
    graph = _graph()

    with patch("core.agent.react.get_client", return_value=client):
        result = await run_react_graph(graph, messages=_USER_MSG, user_id="u1")

    assert result == LLM_ERROR_FALLBACK


@pytest.mark.asyncio
async def test_empty_assistant_content_uses_llm_fallback():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_completion(content=None),
    )
    graph = _graph()

    with patch("core.agent.react.get_client", return_value=client):
        result = await run_react_graph(graph, messages=_USER_MSG, user_id="u1")

    assert result == LLM_ERROR_FALLBACK


@pytest.mark.asyncio
async def test_medium_risk_tool_triggers_interrupt_and_resumes():
    from langgraph.types import Command
    tool_calls = [_tool_call(
        name="create_schedule",
        arguments={"name": "gym", "task": "remind me", "cron_schedule": "07:00"}
    )]

    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            _completion(content=None, tool_calls=tool_calls),
            _completion(content="Gym schedule created!"),
        ],
    )
    
    _create_schedule_schema = [
        {
            "type": "function",
            "function": {
                "name": "create_schedule",
                "description": "create",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "task": {"type": "string"},
                        "cron_schedule": {"type": "string"},
                    },
                    "required": ["name", "task", "cron_schedule"],
                },
            },
        }
    ]
    graph = _graph(tool_schemas=_create_schedule_schema)
    thread_id = "test-thread-interrupt"

    # Mock approval request and schedule tool execution
    mock_approval = AsyncMock(return_value="Awaiting confirmation...")
    mock_create_schedule = AsyncMock(return_value="Created schedule gym")

    # Step 1: Run graph, should hit interrupt and return approval message
    with (
        patch("core.agent.react.get_client", return_value=client),
        patch("core.agent.actions.request_medium_risk_approval", new=mock_approval),
        patch("core.agent.tools.schedules.create_schedule", new=mock_create_schedule),
    ):
        result = await run_react_graph(
            graph,
            messages=_USER_MSG,
            user_id="u1",
            thread_id=thread_id,
            pool=MagicMock(),
            bot=MagicMock(),
        )

    assert result == "Awaiting confirmation..."
    mock_approval.assert_awaited_once()

    # Step 2: Resume graph with Command(resume=True) -> Executes tool and gets final response
    with (
        patch("core.agent.react.get_client", return_value=client),
        patch("core.agent.actions.request_medium_risk_approval", new=mock_approval),
        patch("core.agent.tools.schedules.create_schedule", new=mock_create_schedule),
    ):
        result2 = await run_react_graph(
            graph,
            messages=None,
            user_id="u1",
            thread_id=thread_id,
            resume_command=Command(resume=True),
            pool=MagicMock(),
            bot=MagicMock(),
        )

    assert result2 == "Gym schedule created!"
    mock_create_schedule.assert_awaited_once()


@pytest.mark.asyncio
async def test_medium_risk_tool_cancels_when_rejected():
    tool_calls = [_tool_call(
        name="create_schedule",
        arguments={"name": "gym", "task": "remind me", "cron_schedule": "07:00"}
    )]

    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            _completion(content=None, tool_calls=tool_calls),
            _completion(content="No action was taken."),
        ],
    )
    
    _create_schedule_schema = [
        {
            "type": "function",
            "function": {
                "name": "create_schedule",
                "description": "create",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "task": {"type": "string"},
                        "cron_schedule": {"type": "string"},
                    },
                    "required": ["name", "task", "cron_schedule"],
                },
            },
        }
    ]
    graph = _graph(tool_schemas=_create_schedule_schema)
    thread_id = "test-thread-cancel"

    mock_approval = AsyncMock(return_value="Awaiting confirmation...")
    mock_create_schedule = AsyncMock()

    # Step 1: Run graph to trigger interrupt
    with (
        patch("core.agent.react.get_client", return_value=client),
        patch("core.agent.actions.request_medium_risk_approval", new=mock_approval),
        patch("core.agent.tools.schedules.create_schedule", new=mock_create_schedule),
    ):
        result = await run_react_graph(
            graph,
            messages=_USER_MSG,
            user_id="u1",
            thread_id=thread_id,
            pool=MagicMock(),
            bot=MagicMock(),
        )

    assert result == "Awaiting confirmation..."

    # Step 2: Resume graph with Command(resume=False) -> tool rejected, no execution
    with (
        patch("core.agent.react.get_client", return_value=client),
        patch("core.agent.actions.request_medium_risk_approval", new=mock_approval),
        patch("core.agent.tools.schedules.create_schedule", new=mock_create_schedule),
    ):
        result2 = await run_react_graph(
            graph,
            messages=None,
            user_id="u1",
            thread_id=thread_id,
            resume_command=Command(resume=False),
            pool=MagicMock(),
            bot=MagicMock(),
        )

    assert result2 == "No action was taken."
    mock_create_schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_multiple_tool_calls_in_single_turn():
    """All tool calls within one assistant turn are dispatched before the next reason."""
    tool_calls = [
        _tool_call(call_id="c1", arguments={"query": "news"}),
        _tool_call(call_id="c2", arguments={"query": "weather"}),
    ]
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            _completion(content=None, tool_calls=tool_calls),
            _completion(content="Done."),
        ],
    )
    graph = _graph()

    with (
        patch("core.agent.react.get_client", return_value=client),
        patch("core.agent.react.invoke_tool", new=AsyncMock(return_value="result")) as mock_invoke,
    ):
        result = await run_react_graph(graph, messages=_USER_MSG, user_id="u1")

    assert result == "Done."
    assert mock_invoke.await_count == 2


@pytest.mark.asyncio
async def test_invalid_json_tool_args_falls_back_to_empty_dict():
    """Malformed JSON arguments from the LLM are treated as {} rather than crashing."""
    bad_tool_call = {
        "id": "call_bad",
        "type": "function",
        "function": {"name": "web_search", "arguments": "not-valid-json"},
    }
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            _completion(content=None, tool_calls=[bad_tool_call]),
            _completion(content="Recovered."),
        ],
    )
    graph = _graph()

    with (
        patch("core.agent.react.get_client", return_value=client),
        patch("core.agent.react.invoke_tool", new=AsyncMock(return_value="ok")) as mock_invoke,
    ):
        result = await run_react_graph(graph, messages=_USER_MSG, user_id="u1")

    assert result == "Recovered."
    assert mock_invoke.call_args.args[1] == {}


@pytest.mark.asyncio
async def test_tool_observation_truncated_to_max_chars():
    """Long tool observations are capped at TOOL_OBSERVATION_MAX_CHARS before appending."""
    tool_calls = [_tool_call()]
    long_observation = "x" * 5000

    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            _completion(content=None, tool_calls=tool_calls),
            _completion(content="Got it."),
        ],
    )
    graph = _graph()

    with (
        patch("core.agent.react.get_client", return_value=client),
        patch("core.agent.react.invoke_tool", new=AsyncMock(return_value=long_observation)),
        patch("core.agent.react.get_settings") as mock_settings,
    ):
        mock_settings.return_value.tool_observation_max_chars = 100
        await run_react_graph(graph, messages=_USER_MSG, user_id="u1")

    second_call_messages = client.chat.completions.create.await_args_list[1].kwargs["messages"]
    tool_msg = next(m for m in second_call_messages if m["role"] == "tool")
    assert len(tool_msg["content"]) <= 100 + len(" … [truncated]")
    assert "[truncated]" in tool_msg["content"]

