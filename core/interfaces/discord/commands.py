"""Discord slash commands.

/profile         — mem0.get_all(user_id) → formatted readable summary sent as Discord reply.
/forget <topic>  — mem0 search + delete for facts matching topic; ack on completion.

Both commands are ephemeral (only visible to the invoking user).
"""

import logging

import discord
from discord import app_commands
from discord.ext import commands

from core.memory import store
from core.memory.models import UserProfile

logger = logging.getLogger(__name__)


class MemoryCommands(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @app_commands.command(name="profile", description="Show what Arlo knows about you")
    async def profile(self, interaction: discord.Interaction) -> None:
        user_id = str(interaction.user.id)
        try:
            entries = await store.get_all(user_id)
            summary = UserProfile(user_id=user_id, facts=tuple(entries)).summary()
            await interaction.response.send_message(summary, ephemeral=True)
        except Exception:
            logger.exception("Failed to fetch profile for user %s", user_id)
            await interaction.response.send_message(
                "Something went wrong fetching your profile — try again.", ephemeral=True
            )

    @app_commands.command(name="forget", description="Remove facts Arlo has about a topic")
    @app_commands.describe(topic="What to forget, e.g. 'my job', 'travel plans'")
    async def forget(self, interaction: discord.Interaction, topic: str) -> None:
        user_id = str(interaction.user.id)
        try:
            count = await store.delete(topic, user_id)
            if count == 0:
                reply = f"Nothing found matching '{topic}' — 0 facts removed."
            else:
                noun = "fact" if count == 1 else "facts"
                reply = f"Done — removed {count} {noun} about '{topic}'."
            await interaction.response.send_message(reply, ephemeral=True)
        except Exception:
            logger.exception("Failed to delete facts for user %s, topic=%r", user_id, topic)
            await interaction.response.send_message(
                "Something went wrong — try again.", ephemeral=True
            )


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(MemoryCommands(bot))
