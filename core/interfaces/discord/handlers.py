"""Discord message event handlers.

on_message pipeline:
  1. Filter gate — drop silently if any condition matches:
       - message.author == bot itself
       - message.guild.id != DISCORD_GUILD_ID
       - message.author.id != DISCORD_USER_ID  (single-user enforcement)
       - message.content is empty
  2. INSERT into episodic_messages (async, non-blocking).
  3. Build context window: last CONTEXT_WINDOW_SIZE rows from episodic_messages (default: 12).
  4. Dispatch to the unified LangGraph agent (orchestrator.py).
     The model routes itself via tool-calling — no separate classifier call.
  5. Send reply (plain text).
  6. After reply: if message_count % PROFILE_EXTRACTION_INTERVAL == 0,
     schedule asyncio.create_task(extractor) — never blocks the response path.
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
