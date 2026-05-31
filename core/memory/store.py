"""Memory read/write interface backed by mem0 cloud.

Requires MEM0_API_KEY in settings. The MemoryClient is initialized lazily on
first use so importing this module does not require env vars to be present.

Operations:
  add(fact, user_id, short_term)  — write a fact to mem0; contradiction
                                    handling is delegated to mem0 internals.
  search(query, user_id, limit)   — semantic similarity search; returns plain
                                    fact strings. Used by search_memory tool
                                    and the persona builder.
  delete(topic, user_id)          — search then delete all matching facts;
                                    used by the /forget slash command.
  get_all(user_id)                — full memory dump as list[MemoryEntry];
                                    used by /profile.

Short-term vs long-term tagging:
  short_term=True  — time-bound context ("in Tokyo this week")
  short_term=False — stable trait ("hates layovers", "vegetarian")
  Stored in mem0 metadata field and available on retrieved MemoryEntry objects.

Swap point: replace this file with raw asyncpg + pgvector queries if mem0
cloud becomes unavailable — no other file needs to change.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from functools import lru_cache

from mem0 import MemoryClient

from core.memory.models import MemoryEntry
from core.settings import get_settings


@lru_cache(maxsize=1)
def _get_client() -> MemoryClient:
    return MemoryClient(api_key=get_settings().mem0_api_key)


async def add(fact: str, user_id: str, short_term: bool) -> None:
    client = _get_client()
    await asyncio.to_thread(
        client.add,
        [{"role": "user", "content": fact}],
        user_id=user_id,
        metadata={"short_term": short_term},
    )


async def search(query: str, user_id: str, limit: int = 5) -> list[str]:
    client = _get_client()
    results = await asyncio.to_thread(client.search, query, user_id=user_id, top_k=limit)
    return [r["memory"] for r in results]


async def delete(topic: str, user_id: str) -> int:
    client = _get_client()
    matches = await asyncio.to_thread(client.search, topic, user_id=user_id)
    for m in matches:
        await asyncio.to_thread(client.delete, m["id"])
    return len(matches)


async def get_all(user_id: str) -> list[MemoryEntry]:
    client = _get_client()
    results = await asyncio.to_thread(client.get_all, user_id=user_id)
    entries = []
    for r in results:
        raw_ts = r.get("created_at")
        if isinstance(raw_ts, str):
            created_at = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        elif isinstance(raw_ts, datetime):
            created_at = raw_ts
        else:
            created_at = datetime.now(timezone.utc)
        short_term = r.get("metadata", {}).get("short_term", False)
        entries.append(MemoryEntry(
            id=r["id"],
            content=r["memory"],
            short_term=short_term,
            created_at=created_at,
        ))
    return entries
