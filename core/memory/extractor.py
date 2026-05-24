"""Passive profile extraction — background task, never blocks the response path.

Triggered via asyncio.create_task() after every PROFILE_EXTRACTION_INTERVAL
messages (default: 10). Reads from the episodic_messages PostgreSQL table
(not the Discord API) to ensure reliable access regardless of message
deletion or API rate limits.

Extraction flow:
  1. SELECT last N rows FROM episodic_messages WHERE user_id = $1
  2. ONE LLM call with an extraction prompt:
       — extract facts as (dimension, value, is_short_term) triples
       — dimensions: interests, preferences, opinions, habits, short-term context
  3. For each extracted fact: store.add(fact, user_id, short_term=is_short_term)
       Contradiction handling: mem0 overwrites the old value on the same dimension.

Short-term vs long-term examples:
  short_term=True   "in Tokyo this week", "has a job interview tomorrow"
  short_term=False  "hates layovers", "vegetarian", "obsessed with mechanical keyboards"

Concurrency note: asyncio.create_task is non-blocking but unbounded. If message
volume spikes, add an async lock or bounded queue to prevent concurrent extractions.
"""