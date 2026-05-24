"""discord.py bot entry point.

Sets up the Discord client, registers event handlers from handlers.py,
and registers slash commands from commands.py.

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