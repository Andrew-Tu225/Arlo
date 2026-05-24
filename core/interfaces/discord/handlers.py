"""Discord message event handlers.

on_message pipeline:
  1. Filter: drop bot's own messages, messages outside DISCORD_GUILD_ID, empty messages.
  2. Build context window: fetch last CONTEXT_WINDOW_SIZE (default: 12) messages.
  3. Start typing indicator.
  4. Classify tone + intent (classifier.py).
  5. Route to chat / task / memory_update flow.
  6. Reply.
  7. Trigger background profile extraction if interval reached.
"""

import logging

import discord
from discord.ext import commands

from core.settings import get_settings

logger = logging.getLogger(__name__)


async def handle_message(_bot: commands.Bot, message: discord.Message) -> None:
    if message.author.bot:
        return
    if message.guild is None or message.guild.id != get_settings().discord_guild_id:
        return
    if not message.content.strip():
        return

    logger.info("Message from %s: %s", message.author, message.content)
