"""Tests for core.agent.orchestrator — LangGraph agent graph."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agent.orchestrator import FALLBACK_RESPONSE, run


def _make_mock_client(content: str | None = "Hello there!") -> MagicMock:
    """Return a mock AsyncOpenAI client whose chat.completions.create returns content."""
    choice = MagicMock()
    choice.message.content = content

    completion = MagicMock()
    completion.choices = [choice]

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=completion)
    return client


@pytest.mark.asyncio
async def test_run_returns_string():
    with patch("core.agent.orchestrator.get_client", return_value=_make_mock_client()):
        result = await run([{"role": "user", "content": "Hey"}])
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_run_returns_llm_content():
    with patch("core.agent.orchestrator.get_client", return_value=_make_mock_client("What's up?")):
        result = await run([{"role": "user", "content": "Hey"}])
    assert result == "What's up?"


@pytest.mark.asyncio
async def test_run_passes_system_prompt_as_first_message():
    client = _make_mock_client()
    with patch("core.agent.orchestrator.get_client", return_value=client):
        await run([{"role": "user", "content": "Hey"}])

    call_args = client.chat.completions.create.call_args
    messages = call_args.kwargs.get("messages") or call_args.args[0]
    assert messages[0]["role"] == "system"
    assert "Arlo" in messages[0]["content"]


@pytest.mark.asyncio
async def test_run_passes_user_messages_after_system():
    client = _make_mock_client()
    user_messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]
    with patch("core.agent.orchestrator.get_client", return_value=client):
        await run(user_messages)

    call_args = client.chat.completions.create.call_args
    messages = call_args.kwargs.get("messages") or call_args.args[0]
    assert messages[1:] == user_messages


@pytest.mark.asyncio
async def test_run_handles_api_error_gracefully():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=Exception("API down"))
    with patch("core.agent.orchestrator.get_client", return_value=client):
        result = await run([{"role": "user", "content": "Hey"}])
    assert result == FALLBACK_RESPONSE


@pytest.mark.asyncio
async def test_run_handles_none_content_from_llm():
    with patch("core.agent.orchestrator.get_client", return_value=_make_mock_client(content=None)):
        result = await run([{"role": "user", "content": "Hey"}])
    assert result == FALLBACK_RESPONSE


@pytest.mark.asyncio
async def test_run_uses_correct_model():
    client = _make_mock_client()
    with (
        patch("core.agent.orchestrator.get_client", return_value=client),
        patch("core.agent.orchestrator.get_default_model", return_value="gpt-test-model"),
    ):
        await run([{"role": "user", "content": "Hey"}])

    call_args = client.chat.completions.create.call_args
    model = call_args.kwargs.get("model") or call_args.args[1]
    assert model == "gpt-test-model"
