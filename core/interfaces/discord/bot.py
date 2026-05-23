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

if __name__ == "__main__":
    print("Arlo bot — not yet implemented. Check back soon.")
