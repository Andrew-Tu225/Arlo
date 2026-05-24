"""Unit tests for core/interfaces/discord/bot.py."""

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import discord
from discord.ext import commands

from core.interfaces.discord.bot import bot, create_bot, on_message, on_ready


# --- create_bot ---

def test_create_bot_returns_commands_bot():
    assert isinstance(create_bot(), commands.Bot)


def test_create_bot_message_content_intent_enabled():
    assert create_bot().intents.message_content is True


# --- on_ready ---

async def test_on_ready_logs_each_guild():
    guild1, guild2 = MagicMock(), MagicMock()
    guild1.name, guild2.name = "Server A", "Server B"

    with patch.object(type(bot), "guilds", new_callable=PropertyMock, return_value=[guild1, guild2]):
        with patch("core.interfaces.discord.bot.logger") as mock_log:
            await on_ready()

    assert mock_log.info.call_count == 2
    mock_log.info.assert_any_call("Arlo online in %s", "Server A")
    mock_log.info.assert_any_call("Arlo online in %s", "Server B")


async def test_on_ready_no_guilds_warns_and_skips_info():
    with patch.object(type(bot), "guilds", new_callable=PropertyMock, return_value=[]):
        with patch("core.interfaces.discord.bot.logger") as mock_log:
            await on_ready()

    mock_log.warning.assert_called_once()
    mock_log.info.assert_not_called()


# --- on_message ---

async def test_on_message_calls_handle_message():
    message = MagicMock(spec=discord.Message)

    with patch("core.interfaces.discord.handlers.handle_message", new_callable=AsyncMock) as mock_handle:
        with patch.object(bot, "process_commands", new_callable=AsyncMock):
            await on_message(message)

    mock_handle.assert_called_once_with(bot, message)


async def test_on_message_calls_process_commands():
    message = MagicMock(spec=discord.Message)

    with patch("core.interfaces.discord.handlers.handle_message", new_callable=AsyncMock):
        with patch.object(bot, "process_commands", new_callable=AsyncMock) as mock_process:
            await on_message(message)

    mock_process.assert_called_once_with(message)


async def test_on_message_handle_called_before_process_commands():
    """handle_message must run before process_commands so handlers see every message."""
    call_order = []
    message = MagicMock(spec=discord.Message)

    async def fake_handle(_bot, _msg):
        call_order.append("handle")

    async def fake_process(_msg):
        call_order.append("process")

    with patch("core.interfaces.discord.handlers.handle_message", side_effect=fake_handle):
        with patch.object(bot, "process_commands", side_effect=fake_process):
            await on_message(message)

    assert call_order == ["handle", "process"]
