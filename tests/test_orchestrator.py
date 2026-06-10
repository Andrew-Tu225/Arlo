"""Tests for core.agent.orchestrator — ReAct graph entrypoint."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agent.orchestrator import FALLBACK_RESPONSE, run


def _completion(content: str | None = "Hello there!", total_tokens: int = 50) -> MagicMock:
    message = MagicMock()
    message.content = content
    message.tool_calls = None
    message.model_dump.return_value = {
        "role": "assistant",
        "content": content,
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
async def test_run_passes_tools_to_llm():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_completion("ok"))
    with patch("core.agent.react.get_client", return_value=client):
        await run([{"role": "user", "content": "Hey"}])

    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs.get("tools")
    assert call_kwargs.get("tool_choice") == "auto"


@pytest.mark.asyncio
async def test_run_system_prompt_includes_arlo_and_tools():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_completion("ok"))
    with patch("core.agent.react.get_client", return_value=client):
        await run([{"role": "user", "content": "Hey"}])

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    system_content = messages[0]["content"]
    assert "Arlo" in system_content
    assert "list_schedules" in system_content


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
