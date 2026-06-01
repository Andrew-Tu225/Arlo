"""Tests for core.interfaces.discord.handlers — on_message pipeline."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

import core.interfaces.discord.handlers as handlers_mod
from core.interfaces.discord.handlers import _build_context, handle_message
from core.memory.models import EpisodicMessage


_ALLOWED_USER_ID = 99999  # matches discord_user_id in mock settings


def _make_episodic(role: str, content: str) -> EpisodicMessage:
    return EpisodicMessage(
        id=1,
        user_id=str(_ALLOWED_USER_ID),
        role=role,
        content=content,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _make_message(
    content: str = "hello",
    author_bot: bool = False,
    guild_id: int = 12345,
    author_id: int = _ALLOWED_USER_ID,
) -> MagicMock:
    """Return a minimal mock guild discord.Message."""
    author = MagicMock(spec=discord.Member)
    author.bot = author_bot
    author.id = author_id

    guild = MagicMock()
    guild.id = guild_id

    channel = MagicMock()
    channel.typing = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(), __aexit__=AsyncMock()))
    channel.send = AsyncMock()

    message = MagicMock(spec=discord.Message)
    message.author = author
    message.guild = guild
    message.channel = channel
    message.content = content
    return message


def _make_dm_message(
    content: str = "hello",
    author_id: int = _ALLOWED_USER_ID,
) -> MagicMock:
    """Return a minimal mock DM discord.Message (guild=None).

    Reuses _make_message structure (discord.Member spec works for id/bot attrs)
    then sets guild=None to simulate a DM.
    """
    msg = _make_message(content=content, author_id=author_id)
    msg.guild = None
    return msg


def _make_bot(user_id: int = 11111) -> MagicMock:
    bot = MagicMock()
    bot.user = MagicMock()
    bot.user.id = user_id
    bot.pool = MagicMock()  # asyncpg pool mock
    return bot


def _make_settings() -> MagicMock:
    settings = MagicMock()
    settings.discord_guild_id = 12345
    settings.discord_user_id = _ALLOWED_USER_ID
    settings.context_window_size = 12
    return settings


@pytest.fixture(autouse=True)
def reset_message_counts():
    handlers_mod._message_counts.clear()
    yield
    handlers_mod._message_counts.clear()


def _close_coro(coro):
    """Side effect for create_task mock: close coroutine to avoid 'never awaited' warnings."""
    coro.close()
    return MagicMock()


# ---------------------------------------------------------------------------
# Filter gate tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_message_ignores_bot_messages():
    msg = _make_message(author_bot=True)
    bot = _make_bot()
    with patch("core.interfaces.discord.handlers.get_settings", return_value=_make_settings()):
        await handle_message(bot, msg)
    msg.channel.send.assert_not_called()


@pytest.mark.asyncio
async def test_handle_message_ignores_wrong_guild():
    msg = _make_message(guild_id=99999)
    bot = _make_bot()
    with patch("core.interfaces.discord.handlers.get_settings", return_value=_make_settings()):
        await handle_message(bot, msg)
    msg.channel.send.assert_not_called()


@pytest.mark.asyncio
async def test_handle_message_accepts_dm_from_allowed_user():
    msg = _make_dm_message(author_id=_ALLOWED_USER_ID)
    bot = _make_bot()
    with (
        patch("core.interfaces.discord.handlers.get_settings", return_value=_make_settings()),
        patch("core.interfaces.discord.handlers.orchestrator.run", new=AsyncMock(return_value="hey!")),
        patch("core.interfaces.discord.handlers._build_context", new=AsyncMock(return_value=[])),
        patch("core.interfaces.discord.handlers.db.insert_episodic_message", new=AsyncMock()),
        patch("core.interfaces.discord.handlers.asyncio.create_task", side_effect=_close_coro),
    ):
        await handle_message(bot, msg)
    msg.channel.send.assert_called_once_with("hey!")


@pytest.mark.asyncio
async def test_handle_message_ignores_dm_from_wrong_user():
    msg = _make_dm_message(author_id=55555)  # 55555 != _ALLOWED_USER_ID (99999)
    bot = _make_bot()
    with patch("core.interfaces.discord.handlers.get_settings", return_value=_make_settings()):
        await handle_message(bot, msg)
    msg.channel.send.assert_not_called()


@pytest.mark.asyncio
async def test_handle_message_ignores_wrong_user():
    msg = _make_message(author_id=55555)
    bot = _make_bot()
    with patch("core.interfaces.discord.handlers.get_settings", return_value=_make_settings()):
        await handle_message(bot, msg)
    msg.channel.send.assert_not_called()


@pytest.mark.asyncio
async def test_handle_message_ignores_empty_content():
    msg = _make_message(content="   ")
    bot = _make_bot()
    with patch("core.interfaces.discord.handlers.get_settings", return_value=_make_settings()):
        await handle_message(bot, msg)
    msg.channel.send.assert_not_called()


@pytest.mark.asyncio
async def test_handle_message_skips_when_bot_user_not_ready():
    msg = _make_message()
    bot = _make_bot()
    bot.user = None
    with patch("core.interfaces.discord.handlers.get_settings", return_value=_make_settings()):
        await handle_message(bot, msg)
    msg.channel.send.assert_not_called()


# ---------------------------------------------------------------------------
# Happy path tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_message_sends_reply():
    msg = _make_message()
    bot = _make_bot()
    with (
        patch("core.interfaces.discord.handlers.get_settings", return_value=_make_settings()),
        patch("core.interfaces.discord.handlers.orchestrator.run", new=AsyncMock(return_value="hey!")),
        patch("core.interfaces.discord.handlers._build_context", new=AsyncMock(return_value=[])),
        patch("core.interfaces.discord.handlers.db.insert_episodic_message", new=AsyncMock()),
        patch("core.interfaces.discord.handlers.asyncio.create_task", side_effect=_close_coro),
    ):
        await handle_message(bot, msg)
    msg.channel.send.assert_called_once_with("hey!")


@pytest.mark.asyncio
async def test_handle_message_shows_typing_indicator():
    msg = _make_message()
    bot = _make_bot()
    typing_ctx = AsyncMock()
    typing_ctx.__aenter__ = AsyncMock(return_value=None)
    typing_ctx.__aexit__ = AsyncMock(return_value=False)
    msg.channel.typing = MagicMock(return_value=typing_ctx)

    with (
        patch("core.interfaces.discord.handlers.get_settings", return_value=_make_settings()),
        patch("core.interfaces.discord.handlers.orchestrator.run", new=AsyncMock(return_value="ok")),
        patch("core.interfaces.discord.handlers._build_context", new=AsyncMock(return_value=[])),
        patch("core.interfaces.discord.handlers.db.insert_episodic_message", new=AsyncMock()),
        patch("core.interfaces.discord.handlers.asyncio.create_task", side_effect=_close_coro),
    ):
        await handle_message(bot, msg)

    msg.channel.typing.assert_called_once()
    typing_ctx.__aenter__.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_message_truncates_long_response():
    long_reply = "x" * 2500
    msg = _make_message()
    bot = _make_bot()
    with (
        patch("core.interfaces.discord.handlers.get_settings", return_value=_make_settings()),
        patch("core.interfaces.discord.handlers.orchestrator.run", new=AsyncMock(return_value=long_reply)),
        patch("core.interfaces.discord.handlers._build_context", new=AsyncMock(return_value=[])),
        patch("core.interfaces.discord.handlers.db.insert_episodic_message", new=AsyncMock()),
        patch("core.interfaces.discord.handlers.asyncio.create_task", side_effect=_close_coro),
    ):
        await handle_message(bot, msg)

    sent = msg.channel.send.call_args[0][0]
    assert len(sent) == 2000
    assert sent.endswith("...")


@pytest.mark.asyncio
async def test_handle_message_catches_orchestrator_error():
    msg = _make_message()
    bot = _make_bot()
    with (
        patch("core.interfaces.discord.handlers.get_settings", return_value=_make_settings()),
        patch("core.interfaces.discord.handlers.orchestrator.run", new=AsyncMock(side_effect=Exception("boom"))),
        patch("core.interfaces.discord.handlers._build_context", new=AsyncMock(return_value=[])),
        patch("core.interfaces.discord.handlers.db.insert_episodic_message", new=AsyncMock()),
        patch("core.interfaces.discord.handlers.asyncio.create_task", side_effect=_close_coro),
    ):
        await handle_message(bot, msg)

    msg.channel.send.assert_called_once()
    sent = msg.channel.send.call_args[0][0]
    assert "wrong" in sent.lower() or "error" in sent.lower() or "again" in sent.lower()


# ---------------------------------------------------------------------------
# Memory wiring tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_message_inserts_user_message():
    msg = _make_message(content="I love pizza")
    bot = _make_bot()
    with (
        patch("core.interfaces.discord.handlers.get_settings", return_value=_make_settings()),
        patch("core.interfaces.discord.handlers.orchestrator.run", new=AsyncMock(return_value="yum")),
        patch("core.interfaces.discord.handlers._build_context", new=AsyncMock(return_value=[])),
        patch("core.interfaces.discord.handlers.db.insert_episodic_message", new=AsyncMock()) as mock_insert,
        patch("core.interfaces.discord.handlers.asyncio.create_task", side_effect=_close_coro),
    ):
        await handle_message(bot, msg)

    first_call = mock_insert.call_args_list[0]
    assert first_call.kwargs["role"] == "user"
    assert first_call.kwargs["user_id"] == str(_ALLOWED_USER_ID)
    assert first_call.kwargs["content"] == "I love pizza"


@pytest.mark.asyncio
async def test_handle_message_inserts_assistant_reply():
    msg = _make_message()
    bot = _make_bot()
    with (
        patch("core.interfaces.discord.handlers.get_settings", return_value=_make_settings()),
        patch("core.interfaces.discord.handlers.orchestrator.run", new=AsyncMock(return_value="sure thing")),
        patch("core.interfaces.discord.handlers._build_context", new=AsyncMock(return_value=[])),
        patch("core.interfaces.discord.handlers.db.insert_episodic_message", new=AsyncMock()) as mock_insert,
        patch("core.interfaces.discord.handlers.asyncio.create_task", side_effect=_close_coro),
    ):
        await handle_message(bot, msg)

    second_call = mock_insert.call_args_list[1]
    assert second_call.kwargs["role"] == "assistant"
    assert second_call.kwargs["content"] == "sure thing"


@pytest.mark.asyncio
async def test_handle_message_increments_counter_per_turn():
    bot = _make_bot()
    with (
        patch("core.interfaces.discord.handlers.get_settings", return_value=_make_settings()),
        patch("core.interfaces.discord.handlers.orchestrator.run", new=AsyncMock(return_value="ok")),
        patch("core.interfaces.discord.handlers._build_context", new=AsyncMock(return_value=[])),
        patch("core.interfaces.discord.handlers.db.insert_episodic_message", new=AsyncMock()),
        patch("core.interfaces.discord.handlers.asyncio.create_task", side_effect=_close_coro),
    ):
        await handle_message(bot, _make_message())
        await handle_message(bot, _make_message())

    assert handlers_mod._message_counts[str(_ALLOWED_USER_ID)] == 2


@pytest.mark.asyncio
async def test_handle_message_triggers_extraction_with_correct_count():
    msg = _make_message()
    bot = _make_bot()
    mock_extract = AsyncMock()
    with (
        patch("core.interfaces.discord.handlers.get_settings", return_value=_make_settings()),
        patch("core.interfaces.discord.handlers.orchestrator.run", new=AsyncMock(return_value="hey")),
        patch("core.interfaces.discord.handlers._build_context", new=AsyncMock(return_value=[])),
        patch("core.interfaces.discord.handlers.db.insert_episodic_message", new=AsyncMock()),
        patch("core.interfaces.discord.handlers.extractor.maybe_extract", mock_extract),
        patch("core.interfaces.discord.handlers.asyncio.create_task", side_effect=_close_coro) as mock_ct,
    ):
        await handle_message(bot, msg)

    mock_ct.assert_called_once()
    mock_extract.assert_called_once_with(bot.pool, str(_ALLOWED_USER_ID), 1)


@pytest.mark.asyncio
async def test_handle_message_skips_db_when_pool_none():
    msg = _make_message()
    bot = _make_bot()
    bot.pool = None
    with (
        patch("core.interfaces.discord.handlers.get_settings", return_value=_make_settings()),
        patch("core.interfaces.discord.handlers.orchestrator.run", new=AsyncMock(return_value="ok")),
        patch("core.interfaces.discord.handlers.db.insert_episodic_message", new=AsyncMock()) as mock_insert,
        patch("core.interfaces.discord.handlers.extractor.maybe_extract", new=AsyncMock()) as mock_extract,
        patch("core.interfaces.discord.handlers.asyncio.create_task", side_effect=_close_coro) as mock_ct,
    ):
        await handle_message(bot, msg)

    msg.channel.send.assert_called_once()
    mock_insert.assert_not_called()
    mock_extract.assert_not_called()
    mock_ct.assert_not_called()


@pytest.mark.asyncio
async def test_handle_message_assistant_insert_awaited_before_extraction():
    """Strict ordering: user insert → context → orchestrator → send → assistant insert → create_task."""
    call_order: list[str] = []
    msg = _make_message()
    bot = _make_bot()

    async def track_insert(pool, *, user_id, role, content):
        call_order.append(f"insert:{role}")

    def track_create_task(coro):
        call_order.append("create_task")
        coro.close()
        return MagicMock()

    with (
        patch("core.interfaces.discord.handlers.get_settings", return_value=_make_settings()),
        patch("core.interfaces.discord.handlers.orchestrator.run", new=AsyncMock(return_value="hey")),
        patch("core.interfaces.discord.handlers._build_context", new=AsyncMock(return_value=[])),
        patch("core.interfaces.discord.handlers.db.insert_episodic_message", side_effect=track_insert),
        patch("core.interfaces.discord.handlers.extractor.maybe_extract", new=AsyncMock()),
        patch("core.interfaces.discord.handlers.asyncio.create_task", side_effect=track_create_task),
    ):
        await handle_message(bot, msg)

    assert call_order == ["insert:user", "insert:assistant", "create_task"]


# ---------------------------------------------------------------------------
# _build_context tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_build_context_returns_chronological_order():
    pool = MagicMock()
    messages = [
        _make_episodic("user", "first"),
        _make_episodic("assistant", "second"),
        _make_episodic("user", "third"),
    ]
    with (
        patch("core.interfaces.discord.handlers.get_settings") as mock_settings,
        patch("core.interfaces.discord.handlers.db.get_recent_messages", new=AsyncMock(return_value=messages)),
    ):
        mock_settings.return_value.context_window_size = 12
        result = await _build_context(pool, "u1")

    assert [r["content"] for r in result] == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_build_context_maps_roles_correctly():
    pool = MagicMock()
    messages = [
        _make_episodic("assistant", "bot reply"),
        _make_episodic("user", "user msg"),
    ]
    with (
        patch("core.interfaces.discord.handlers.get_settings") as mock_settings,
        patch("core.interfaces.discord.handlers.db.get_recent_messages", new=AsyncMock(return_value=messages)),
    ):
        mock_settings.return_value.context_window_size = 12
        result = await _build_context(pool, "u1")

    roles = {r["content"]: r["role"] for r in result}
    assert roles["bot reply"] == "assistant"
    assert roles["user msg"] == "user"


@pytest.mark.asyncio
async def test_build_context_skips_empty_messages():
    pool = MagicMock()
    messages = [
        _make_episodic("user", "   "),
        _make_episodic("user", "hello"),
    ]
    with (
        patch("core.interfaces.discord.handlers.get_settings") as mock_settings,
        patch("core.interfaces.discord.handlers.db.get_recent_messages", new=AsyncMock(return_value=messages)),
    ):
        mock_settings.return_value.context_window_size = 12
        result = await _build_context(pool, "u1")

    assert len(result) == 1
    assert result[0]["content"] == "hello"


@pytest.mark.asyncio
async def test_build_context_returns_empty_when_pool_none():
    with patch("core.interfaces.discord.handlers.db.get_recent_messages", new=AsyncMock()) as mock_db:
        result = await _build_context(None, "u1")
    mock_db.assert_not_called()
    assert result == []


@pytest.mark.asyncio
async def test_build_context_uses_context_window_size():
    pool = MagicMock()
    with (
        patch("core.interfaces.discord.handlers.get_settings") as mock_settings,
        patch("core.interfaces.discord.handlers.db.get_recent_messages", new=AsyncMock(return_value=[])) as mock_db,
    ):
        mock_settings.return_value.context_window_size = 7
        await _build_context(pool, "u1")

    mock_db.assert_called_once_with(pool, user_id="u1", n=7)
