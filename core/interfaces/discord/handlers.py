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
