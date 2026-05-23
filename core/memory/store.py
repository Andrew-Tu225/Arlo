"""Memory read/write interface backed by mem0 (self-hosted).

mem0 runs in self-hosted mode against the local PostgreSQL + pgvector instance
defined by DATABASE_URL. No MEM0_API_KEY or cloud dependency required.

Operations:
  add(fact, user_id, short_term)  — mem0.add(); contradiction handling is
                                    delegated to mem0 internals (new fact on
                                    the same dimension overwrites the old one).
  search(query, user_id)          — pgvector similarity search; used by the
                                    search_memory tool and the persona builder.
  delete(topic, user_id)          — find + delete all facts matching topic;
                                    used by the /forget slash command.
  get_all(user_id)                — full memory dump; used by /profile.

Short-term vs long-term tagging:
  short_term=True  — time-bound context ("in Tokyo this week")
  short_term=False — stable trait ("hates layovers", "vegetarian")
  Tag stored in mem0 metadata and used to weight relevance decay.

Fallback: if mem0 introduces blocking issues, this module is the single
swap point — replace with raw asyncpg + pgvector queries without touching
any other file.
"""
