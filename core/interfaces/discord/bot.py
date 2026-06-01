"""discord.py bot entry point.

Startup sequence:
  1. Validate all required environment variables — halt with a clear error if any are missing:
       DISCORD_BOT_TOKEN, DISCORD_GUILD_ID, DISCORD_USER_ID, DATABASE_URL,
       at least one of OPENAI_API_KEY / OPENROUTER_API_KEY, TAVILY_API_KEY.
       DIGEST_TIMEZONE must be a valid IANA timezone string.
  2. setup_hook (called by discord.py after login): initialize the asyncpg pool and
     run init_tables() to create all tables if they don't exist.
     Pool is attached as bot.pool so handlers.py can access it via getattr(bot, "pool", None).
  3. Register on_message handler (handlers.py).
  4. Register slash commands (commands.py).
  5. Start APScheduler, register default morning proactive DM job, and register
     any user-created channel schedule jobs from the DB (empty in Phase 3).
  6. Connect to Discord and begin the event loop.

Run with:
    python -m core.interfaces.discord.bot
"""

import logging

import discord
from discord.ext import commands

from core import db
from core.interfaces.discord import commands as discord_commands
from core.interfaces.discord import handlers
from core.scheduler import digest as digest_module
from core.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class ArloBot(commands.Bot):
    async def setup_hook(self) -> None:
        settings = get_settings()
        self.pool = await db.get_pool(settings.database_url)
        await db.init_tables(self.pool)
        logger.info("DB pool and tables initialised")
        await discord_commands.setup(self)
        await self.tree.sync()
        logger.info("Slash commands registered and synced")
        settings = get_settings()
        await digest_module.seed_default_schedules(self.pool, str(settings.discord_user_id))
        digest_module.scheduler.start()
        await digest_module.register_digest_jobs(self, self.pool)
        logger.info("Scheduler started and jobs registered")


def create_bot() -> ArloBot:
    intents = discord.Intents.default()
    intents.message_content = True
    return ArloBot(command_prefix="!", intents=intents)


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