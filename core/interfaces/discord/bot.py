"""discord.py bot entry point.

Startup sequence:
  1. Validate all required environment variables — halt with a clear error if any are missing:
       DISCORD_BOT_TOKEN, DISCORD_GUILD_ID, DISCORD_USER_ID, DATABASE_URL,
       at least one of OPENAI_API_KEY / OPENROUTER_API_KEY, TAVILY_API_KEY.
       DIGEST_TIMEZONE must be a valid IANA timezone string.
  2. Initialize the asyncpg connection pool (core/db.py).
  3. Run init_tables() to create episodic_messages and digest_config if they don't exist.
  4. Register on_message handler (handlers.py).
  5. Register slash commands (commands.py).
  6. Start APScheduler digest job (scheduler/digest.py).
  7. Connect to Discord and begin the event loop.

Run with:
    python -m core.interfaces.discord.bot
"""

import logging

import discord
from discord.ext import commands

from core.interfaces.discord import handlers
from core.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def create_bot() -> commands.Bot:
    intents = discord.Intents.default()
    intents.message_content = True
    return commands.Bot(command_prefix="!", intents=intents)


bot = create_bot()


@bot.event
async def on_ready() -> None:
    if not bot.guilds:
        logger.warning("on_ready fired but bot is in no guilds — has it been invited?")
        return
    for guild in bot.guilds:
        logger.info("Arlo online in %s", guild.name)


@bot.event
async def on_message(message: discord.Message) -> None:
    await handlers.handle_message(bot, message)
    await bot.process_commands(message)


if __name__ == "__main__":
    settings = get_settings()
    bot.run(settings.discord_bot_token)