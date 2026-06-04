"""Tests for core.agent.react — shared ReAct graph."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat import ChatCompletionMessageParam

from core.agent.react import (
    LLM_ERROR_FALLBACK,
    ReactGraphConfig,
    ReActState,
    build_react_graph,
    run_react_graph,
)
from core.agent.tools import RESEARCH_FALLBACK, ToolContext

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
        tool_context_factory=lambda state: ToolContext(
            user_id=state.get("user_id") or "",
        ),
        max_react_iterations=max_iterations,
        task_token_budget=token_budget,
    )


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
    graph = build_react_graph(_config())

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
    graph = build_react_graph(_config())

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
    config = _config(max_iterations=2)
    graph = build_react_graph(config)

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
    config = _config(token_budget=150)
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_completion(content="ok", total_tokens=200),
    )
    graph = build_react_graph(config)

    with patch("core.agent.react.get_client", return_value=client):
        result = await run_react_graph(graph, messages=_USER_MSG, user_id="u1")

    assert result == RESEARCH_FALLBACK


@pytest.mark.asyncio
async def test_llm_error_returns_llm_fallback():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("API down"))
    graph = build_react_graph(_config())

    with patch("core.agent.react.get_client", return_value=client):
        result = await run_react_graph(graph, messages=_USER_MSG, user_id="u1")

    assert result == LLM_ERROR_FALLBACK


@pytest.mark.asyncio
async def test_empty_assistant_content_uses_llm_fallback():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_completion(content=None),
    )
    graph = build_react_graph(_config())

    with patch("core.agent.react.get_client", return_value=client):
        result = await run_react_graph(graph, messages=_USER_MSG, user_id="u1")

    assert result == LLM_ERROR_FALLBACK


@pytest.mark.asyncio
async def test_tool_executor_appends_tool_messages():
    tool_calls = [_tool_call()]

    async def fake_invoke(name: str, args: dict[str, Any], ctx: ToolContext) -> str:
        return "tool-observation"

    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            _completion(content=None, tool_calls=tool_calls),
            _completion(content="done"),
        ],
    )
    graph = build_react_graph(_config())

    # Spy via patched invoke and assert message shape on second LLM call
    with (
        patch("core.agent.react.get_client", return_value=client),
        patch("core.agent.react.invoke_tool", new=AsyncMock(side_effect=fake_invoke)),
    ):
        await run_react_graph(graph, messages=_USER_MSG, user_id="u1")

    second_call_messages = client.chat.completions.create.await_args_list[1].kwargs[
        "messages"
    ]
    roles = [m["role"] for m in second_call_messages]
    assert "tool" in roles
    tool_msg = next(m for m in second_call_messages if m["role"] == "tool")
    assert tool_msg["content"] == "tool-observation"
    assert tool_msg["tool_call_id"] == "call_1"
