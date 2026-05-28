"""Discord message event handlers.

on_message pipeline:
  1. Filter gate — drop silently if any condition matches:
       - message.author is a bot
       - message.guild.id != DISCORD_GUILD_ID
       - message.author.id != DISCORD_USER_ID  (single-user enforcement)
       - message.content is empty
  2. Show typing indicator.
  3. Build context window: last CONTEXT_WINDOW_SIZE messages from channel
     history, mapped to OpenAI-format role/content dicts, chronological order.
  4. Dispatch to the unified LangGraph agent (orchestrator.py).
     The model routes itself via tool-calling — no separate classifier call.
  5. Send reply (plain text, truncated to 2000 chars if needed).
"""

import logging

import discord
from discord.ext import commands

from core.agent import orchestrator
from core.settings import get_settings

logger = logging.getLogger(__name__)

_MAX_DISCORD_LENGTH = 2000


async def _build_context(
    channel: discord.TextChannel,
    bot_user: discord.ClientUser,
) -> list[dict]:
    """Fetch recent channel history and return as OpenAI-format message dicts.

    Fetches the last CONTEXT_WINDOW_SIZE messages, maps the bot's own messages
    to role "assistant" and all others to role "user", and returns them in
    chronological (oldest-first) order. Empty messages are skipped.
    """
    settings = get_settings()
    history = [msg async for msg in channel.history(limit=settings.context_window_size)]
    history.reverse()  # discord.py returns newest-first; we want oldest-first

    context = []
    for msg in history:
        if not msg.content.strip():
            continue
        role = "assistant" if msg.author == bot_user else "user"
        context.append({"role": role, "content": msg.content})

    return context


async def handle_message(bot: commands.Bot, message: discord.Message) -> None:
    """Handle an incoming Discord message through the full agent pipeline."""
    # Filter gate
    if message.author.bot:
        return
    if message.guild is None or message.guild.id != get_settings().discord_guild_id:
        return
    if message.author.id != get_settings().discord_user_id:
        return
    if not message.content.strip():
        return
    if bot.user is None:
        logger.warning("handle_message called before bot.user is available; skipping")
        return

    logger.info("Message from %s: %s", message.author, message.content[:100])

    async with message.channel.typing():
        try:
            context = await _build_context(message.channel, bot.user)
            response = await orchestrator.run(context)

            if len(response) > _MAX_DISCORD_LENGTH:
                response = response[: _MAX_DISCORD_LENGTH - 3] + "..."

            await message.channel.send(response)
        except Exception:
            logger.exception("Agent pipeline failed for message from %s", message.author)
            await message.channel.send("Something went wrong — try again.")
