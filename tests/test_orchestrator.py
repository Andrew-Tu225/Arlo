"""Tests for core.agent.orchestrator — LangGraph agent graph."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agent.orchestrator import FALLBACK_RESPONSE, run


_USER_MSG = [{"role": "user", "content": "Hey"}]


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


# ---------------------------------------------------------------------------
# Memory injection (Step 8)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_calls_store_search_with_last_user_message():
    with (
        patch("core.agent.orchestrator.store.search", new=AsyncMock(return_value=[])) as mock_search,
        patch("core.agent.orchestrator.get_client", return_value=_make_mock_client()),
    ):
        await run(_USER_MSG, user_id="u42")

    mock_search.assert_awaited_once_with("Hey", "u42")


@pytest.mark.asyncio
async def test_run_injects_memories_into_system_prompt():
    facts = ["diet: vegetarian", "hobby: hiking"]
    client = _make_mock_client()
    with (
        patch("core.agent.orchestrator.store.search", new=AsyncMock(return_value=facts)),
        patch("core.agent.orchestrator.get_client", return_value=client),
    ):
        await run(_USER_MSG, user_id="u42")

    messages_sent = client.chat.completions.create.call_args.kwargs["messages"]
    system_content = messages_sent[0]["content"]
    assert "vegetarian" in system_content
    assert "hiking" in system_content


@pytest.mark.asyncio
async def test_run_handles_store_search_error_gracefully():
    client = _make_mock_client("still works")
    with (
        patch("core.agent.orchestrator.store.search", new=AsyncMock(side_effect=Exception("mem0 down"))),
        patch("core.agent.orchestrator.get_client", return_value=client),
    ):
        result = await run(_USER_MSG, user_id="u42")

    assert result == "still works"


@pytest.mark.asyncio
async def test_run_skips_memory_search_when_no_user_id():
    with (
        patch("core.agent.orchestrator.store.search", new=AsyncMock()) as mock_search,
        patch("core.agent.orchestrator.get_client", return_value=_make_mock_client()),
    ):
        await run(_USER_MSG, user_id=None)

    mock_search.assert_not_awaited()
