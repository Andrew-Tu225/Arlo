"""Tests for core.interfaces.discord.handlers — on_message pipeline."""

from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from core.interfaces.discord.handlers import _build_context, handle_message


_ALLOWED_USER_ID = 99999  # matches discord_user_id in mock settings


def _make_message(
    content: str = "hello",
    author_bot: bool = False,
    guild_id: int = 12345,
    author_id: int = _ALLOWED_USER_ID,
) -> MagicMock:
    """Return a minimal mock discord.Message."""
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


def _make_bot(user_id: int = 11111) -> MagicMock:
    bot = MagicMock()
    bot.user = MagicMock()
    bot.user.id = user_id
    return bot


# --- Filter gate tests (existing behaviour, verified not broken) ---

@pytest.mark.asyncio
async def test_handle_message_ignores_bot_messages():
    msg = _make_message(author_bot=True)
    bot = _make_bot()
    with patch("core.interfaces.discord.handlers.get_settings") as mock_settings:
        mock_settings.return_value.discord_guild_id = 12345
        mock_settings.return_value.discord_user_id = _ALLOWED_USER_ID
        await handle_message(bot, msg)
    msg.channel.send.assert_not_called()


@pytest.mark.asyncio
async def test_handle_message_ignores_wrong_guild():
    msg = _make_message(guild_id=99999)
    bot = _make_bot()
    with patch("core.interfaces.discord.handlers.get_settings") as mock_settings:
        mock_settings.return_value.discord_guild_id = 12345
        mock_settings.return_value.discord_user_id = _ALLOWED_USER_ID
        await handle_message(bot, msg)
    msg.channel.send.assert_not_called()


@pytest.mark.asyncio
async def test_handle_message_ignores_wrong_user():
    msg = _make_message(author_id=55555)  # not the allowed user
    bot = _make_bot()
    with patch("core.interfaces.discord.handlers.get_settings") as mock_settings:
        mock_settings.return_value.discord_guild_id = 12345
        mock_settings.return_value.discord_user_id = _ALLOWED_USER_ID
        await handle_message(bot, msg)
    msg.channel.send.assert_not_called()


@pytest.mark.asyncio
async def test_handle_message_ignores_empty_content():
    msg = _make_message(content="   ")
    bot = _make_bot()
    with patch("core.interfaces.discord.handlers.get_settings") as mock_settings:
        mock_settings.return_value.discord_guild_id = 12345
        mock_settings.return_value.discord_user_id = _ALLOWED_USER_ID
        await handle_message(bot, msg)
    msg.channel.send.assert_not_called()


@pytest.mark.asyncio
async def test_handle_message_skips_when_bot_user_not_ready():
    msg = _make_message()
    bot = _make_bot()
    bot.user = None  # bot not yet logged in
    with patch("core.interfaces.discord.handlers.get_settings") as mock_settings:
        mock_settings.return_value.discord_guild_id = 12345
        mock_settings.return_value.discord_user_id = _ALLOWED_USER_ID
        await handle_message(bot, msg)
    msg.channel.send.assert_not_called()


# --- Happy path ---

@pytest.mark.asyncio
async def test_handle_message_sends_reply():
    msg = _make_message()
    bot = _make_bot()
    with (
        patch("core.interfaces.discord.handlers.get_settings") as mock_settings,
        patch("core.interfaces.discord.handlers.orchestrator.run", new=AsyncMock(return_value="hey!")),
        patch("core.interfaces.discord.handlers._build_context", new=AsyncMock(return_value=[])),
    ):
        mock_settings.return_value.discord_guild_id = 12345
        mock_settings.return_value.discord_user_id = _ALLOWED_USER_ID
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
        patch("core.interfaces.discord.handlers.get_settings") as mock_settings,
        patch("core.interfaces.discord.handlers.orchestrator.run", new=AsyncMock(return_value="ok")),
        patch("core.interfaces.discord.handlers._build_context", new=AsyncMock(return_value=[])),
    ):
        mock_settings.return_value.discord_guild_id = 12345
        mock_settings.return_value.discord_user_id = _ALLOWED_USER_ID
        await handle_message(bot, msg)

    msg.channel.typing.assert_called_once()
    typing_ctx.__aenter__.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_message_truncates_long_response():
    long_reply = "x" * 2500
    msg = _make_message()
    bot = _make_bot()
    with (
        patch("core.interfaces.discord.handlers.get_settings") as mock_settings,
        patch("core.interfaces.discord.handlers.orchestrator.run", new=AsyncMock(return_value=long_reply)),
        patch("core.interfaces.discord.handlers._build_context", new=AsyncMock(return_value=[])),
    ):
        mock_settings.return_value.discord_guild_id = 12345
        mock_settings.return_value.discord_user_id = _ALLOWED_USER_ID
        await handle_message(bot, msg)

    sent = msg.channel.send.call_args[0][0]
    assert len(sent) == 2000
    assert sent.endswith("...")


@pytest.mark.asyncio
async def test_handle_message_catches_orchestrator_error():
    msg = _make_message()
    bot = _make_bot()
    with (
        patch("core.interfaces.discord.handlers.get_settings") as mock_settings,
        patch("core.interfaces.discord.handlers.orchestrator.run", new=AsyncMock(side_effect=Exception("boom"))),
        patch("core.interfaces.discord.handlers._build_context", new=AsyncMock(return_value=[])),
    ):
        mock_settings.return_value.discord_guild_id = 12345
        mock_settings.return_value.discord_user_id = _ALLOWED_USER_ID
        await handle_message(bot, msg)

    msg.channel.send.assert_called_once()
    sent = msg.channel.send.call_args[0][0]
    assert "wrong" in sent.lower() or "error" in sent.lower() or "again" in sent.lower()


# --- _build_context tests ---

@pytest.mark.asyncio
async def test_build_context_returns_chronological_order():
    bot_user = MagicMock()
    bot_user.id = 1

    user = MagicMock()
    user.id = 2

    # discord.py returns newest-first
    m1 = MagicMock(); m1.content = "third"; m1.author = user
    m2 = MagicMock(); m2.content = "second"; m2.author = bot_user
    m3 = MagicMock(); m3.content = "first"; m3.author = user

    async def mock_history(limit):
        for m in [m1, m2, m3]:
            yield m

    channel = MagicMock()
    channel.history = mock_history

    with patch("core.interfaces.discord.handlers.get_settings") as mock_settings:
        mock_settings.return_value.context_window_size = 12
        result = await _build_context(channel, bot_user)

    assert [r["content"] for r in result] == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_build_context_maps_roles_correctly():
    bot_user = MagicMock()
    user = MagicMock()

    m_bot = MagicMock(); m_bot.content = "bot reply"; m_bot.author = bot_user
    m_user = MagicMock(); m_user.content = "user msg"; m_user.author = user

    async def mock_history(limit):
        for m in [m_bot, m_user]:
            yield m

    channel = MagicMock()
    channel.history = mock_history

    with patch("core.interfaces.discord.handlers.get_settings") as mock_settings:
        mock_settings.return_value.context_window_size = 12
        result = await _build_context(channel, bot_user)

    roles = {r["content"]: r["role"] for r in result}
    assert roles["bot reply"] == "assistant"
    assert roles["user msg"] == "user"


@pytest.mark.asyncio
async def test_build_context_skips_empty_messages():
    bot_user = MagicMock()
    user = MagicMock()

    m_empty = MagicMock(); m_empty.content = "   "; m_empty.author = user
    m_valid = MagicMock(); m_valid.content = "hello"; m_valid.author = user

    async def mock_history(limit):
        for m in [m_empty, m_valid]:
            yield m

    channel = MagicMock()
    channel.history = mock_history

    with patch("core.interfaces.discord.handlers.get_settings") as mock_settings:
        mock_settings.return_value.context_window_size = 12
        result = await _build_context(channel, bot_user)

    assert len(result) == 1
    assert result[0]["content"] == "hello"
