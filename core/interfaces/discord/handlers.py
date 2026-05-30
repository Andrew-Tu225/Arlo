"""Discord message event handlers.

on_message pipeline:
  1. Filter gate — drop silently if any condition matches:
       - message.author is a bot
       - message.guild.id != DISCORD_GUILD_ID
       - message.author.id != DISCORD_USER_ID  (single-user enforcement)
       - message.content is empty
  2. Show typing indicator.
  3. Insert user message into episodic_messages (awaited — must land before context read).
  4. Build context window: last CONTEXT_WINDOW_SIZE messages from episodic_messages
     (PostgreSQL), returned as OpenAI-format role/content dicts, chronological order.
  5. Dispatch to the unified LangGraph agent (orchestrator.py).
     The model routes itself via tool-calling — no separate classifier call.
  6. Send reply (plain text, truncated to 2000 chars if needed).
  7. Insert assistant reply into episodic_messages (awaited before extraction).
  8. Trigger background profile extraction via asyncio.create_task (non-blocking).

Graceful degradation: if bot.pool is None (pool not yet initialised at startup),
steps 3, 4, 7, and 8 are skipped — context is empty, reply is still sent.
"""

from __future__ import annotations

import asyncio
import logging

import asyncpg
import discord
from discord.ext import commands

from core import db
from core.agent import orchestrator
from core.memory import extractor
from core.settings import get_settings

logger = logging.getLogger(__name__)

_MAX_DISCORD_LENGTH = 2000
_message_counts: dict[str, int] = {}


async def _build_context(
    pool: asyncpg.Pool | None,
    user_id: str,
) -> list[dict]:
    """Fetch recent messages from episodic_messages and return as OpenAI-format dicts.

    Returns an empty list if pool is None (degraded startup state).
    Rows are returned oldest-first by db.get_recent_messages; no reversal needed.
    """
    if pool is None:
        return []
    settings = get_settings()
    messages = await db.get_recent_messages(pool, user_id=user_id, n=settings.context_window_size)
    context = []
    for msg in messages:
        if not msg.content.strip():
            continue
        context.append({"role": msg.role, "content": msg.content})
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

    pool: asyncpg.Pool | None = getattr(bot, "pool", None)
    if pool is None:
        logger.warning("DB pool not yet initialised; replying without memory context")

    user_id = str(message.author.id)
    logger.info("Message from %s: %s", message.author, message.content[:100])

    async with message.channel.typing():
        try:
            if pool is not None:
                await db.insert_episodic_message(
                    pool, user_id=user_id, role="user", content=message.content
                )

            context = await _build_context(pool, user_id)
            response = await orchestrator.run(context)

            if len(response) > _MAX_DISCORD_LENGTH:
                response = response[: _MAX_DISCORD_LENGTH - 3] + "..."

            await message.channel.send(response)

            if pool is not None:
                await db.insert_episodic_message(
                    pool, user_id=user_id, role="assistant", content=response
                )
                _message_counts[user_id] = _message_counts.get(user_id, 0) + 1
                asyncio.create_task(
                    extractor.maybe_extract(pool, user_id, _message_counts[user_id])
                )
        except Exception:
            logger.exception("Agent pipeline failed for message from %s", message.author)
            await message.channel.send("Something went wrong — try again.")
