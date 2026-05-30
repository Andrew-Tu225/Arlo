"""Tests for core.interfaces.discord.commands — /profile and /forget slash commands."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest
from discord.ext import commands

from core.interfaces.discord.commands import MemoryCommands
from core.memory.models import MemoryEntry


def _make_interaction(user_id: int = 99999) -> MagicMock:
    interaction = MagicMock(spec=discord.Interaction)
    interaction.user = MagicMock()
    interaction.user.id = user_id
    interaction.response = MagicMock()
    interaction.response.send_message = AsyncMock()
    return interaction


def _make_entry(content: str) -> MemoryEntry:
    return MemoryEntry(
        id="abc123",
        content=content,
        short_term=False,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def cog() -> MemoryCommands:
    bot = MagicMock(spec=commands.Bot)
    return MemoryCommands(bot)


# ---------------------------------------------------------------------------
# /profile
# ---------------------------------------------------------------------------


class TestProfileCommand:
    @pytest.mark.asyncio
    async def test_profile_sends_fact_summary(self, cog: MemoryCommands) -> None:
        interaction = _make_interaction()
        facts = [_make_entry("user is vegetarian"), _make_entry("user loves jazz")]
        with patch("core.interfaces.discord.commands.store.get_all", new=AsyncMock(return_value=facts)):
            await MemoryCommands.profile.callback(cog, interaction)

        interaction.response.send_message.assert_awaited_once()
        text = interaction.response.send_message.call_args[0][0]
        assert "vegetarian" in text
        assert "jazz" in text

    @pytest.mark.asyncio
    async def test_profile_sends_no_facts_message_when_empty(self, cog: MemoryCommands) -> None:
        interaction = _make_interaction()
        with patch("core.interfaces.discord.commands.store.get_all", new=AsyncMock(return_value=[])):
            await MemoryCommands.profile.callback(cog, interaction)

        text = interaction.response.send_message.call_args[0][0]
        assert "no facts" in text.lower() or "nothing" in text.lower()

    @pytest.mark.asyncio
    async def test_profile_passes_user_id_to_store(self, cog: MemoryCommands) -> None:
        interaction = _make_interaction(user_id=42424242)
        with patch("core.interfaces.discord.commands.store.get_all", new=AsyncMock(return_value=[])) as mock_get_all:
            await MemoryCommands.profile.callback(cog, interaction)

        mock_get_all.assert_awaited_once_with("42424242")

    @pytest.mark.asyncio
    async def test_profile_reply_is_ephemeral(self, cog: MemoryCommands) -> None:
        interaction = _make_interaction()
        with patch("core.interfaces.discord.commands.store.get_all", new=AsyncMock(return_value=[])):
            await MemoryCommands.profile.callback(cog, interaction)

        _, kwargs = interaction.response.send_message.call_args
        assert kwargs.get("ephemeral") is True

    @pytest.mark.asyncio
    async def test_profile_handles_store_error(self, cog: MemoryCommands) -> None:
        interaction = _make_interaction()
        with patch("core.interfaces.discord.commands.store.get_all", new=AsyncMock(side_effect=Exception("mem0 down"))):
            await MemoryCommands.profile.callback(cog, interaction)

        text = interaction.response.send_message.call_args[0][0]
        assert any(word in text.lower() for word in ("error", "wrong", "try"))


# ---------------------------------------------------------------------------
# /forget
# ---------------------------------------------------------------------------


class TestForgetCommand:
    @pytest.mark.asyncio
    async def test_forget_replies_with_count_when_deleted(self, cog: MemoryCommands) -> None:
        interaction = _make_interaction()
        with patch("core.interfaces.discord.commands.store.delete", new=AsyncMock(return_value=2)):
            await MemoryCommands.forget.callback(cog, interaction, topic="my job")

        text = interaction.response.send_message.call_args[0][0]
        assert "2" in text

    @pytest.mark.asyncio
    async def test_forget_replies_when_no_match(self, cog: MemoryCommands) -> None:
        interaction = _make_interaction()
        with patch("core.interfaces.discord.commands.store.delete", new=AsyncMock(return_value=0)):
            await MemoryCommands.forget.callback(cog, interaction, topic="unicorns")

        text = interaction.response.send_message.call_args[0][0]
        assert "0" in text or "nothing" in text.lower() or "no " in text.lower()

    @pytest.mark.asyncio
    async def test_forget_passes_topic_and_user_id_to_store(self, cog: MemoryCommands) -> None:
        interaction = _make_interaction(user_id=77777)
        with patch("core.interfaces.discord.commands.store.delete", new=AsyncMock(return_value=1)) as mock_delete:
            await MemoryCommands.forget.callback(cog, interaction, topic="travel plans")

        mock_delete.assert_awaited_once_with("travel plans", "77777")

    @pytest.mark.asyncio
    async def test_forget_reply_is_ephemeral(self, cog: MemoryCommands) -> None:
        interaction = _make_interaction()
        with patch("core.interfaces.discord.commands.store.delete", new=AsyncMock(return_value=0)):
            await MemoryCommands.forget.callback(cog, interaction, topic="x")

        _, kwargs = interaction.response.send_message.call_args
        assert kwargs.get("ephemeral") is True

    @pytest.mark.asyncio
    async def test_forget_handles_store_error(self, cog: MemoryCommands) -> None:
        interaction = _make_interaction()
        with patch(
            "core.interfaces.discord.commands.store.delete",
            new=AsyncMock(side_effect=Exception("mem0 down")),
        ):
            await MemoryCommands.forget.callback(cog, interaction, topic="my job")

        text = interaction.response.send_message.call_args[0][0]
        assert any(word in text.lower() for word in ("error", "wrong", "try"))
