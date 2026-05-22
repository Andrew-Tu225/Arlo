"""Daily proactive digest engine.

APScheduler job (default: 9am) that:
  1. Reads the user's interest profile from mem0.
  2. Builds Tavily queries from profile interests.
  3. Searches for fresh content (news, Reddit, trending topics).
  4. Composes a casual 2–4 item digest referencing at least one profile fact.
  5. Sends the message to the configured Discord channel.

Job state (schedule, channel, on/off) persisted in PostgreSQL and reloaded on restart.
"""
